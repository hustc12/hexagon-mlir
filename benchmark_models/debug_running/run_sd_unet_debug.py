#!/usr/bin/env python3
"""Debug SD UNet with reduced channels and no cross-attention blocks.

CrossAttn UNet IR made host Hexagon lowering hang (~6GB RSS). This tiny
topology keeps ResNet blocks only so Phase-4 HVX/HexKL smoke can finish.
Use run_sd_unet.py for published UNet structure hooks.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import torch

_SCRIPT = Path(__file__).resolve().parent.parent / "run_sd_unet.py"
_SPEC = importlib.util.spec_from_file_location("run_sd_unet_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config: dict) -> dict:
    config["block_out_channels"] = [32, 64]
    # No CrossAttn* — attention IR previously stalled host Hexagon lowering.
    config["down_block_types"] = ["DownBlock2D", "DownBlock2D"]
    config["up_block_types"] = ["UpBlock2D", "UpBlock2D"]
    config["mid_block_type"] = "UNetMidBlock2D"
    config["layers_per_block"] = 1
    config["cross_attention_dim"] = 32
    config["attention_head_dim"] = 8
    config["sample_size"] = 8
    config["norm_num_groups"] = 32
    print(
        f"[Debug] tiny UNet (no cross-attn): channels={config['block_out_channels']} "
        f"sample={config['sample_size']} mid={config['mid_block_type']} "
        f"(full harness = published SD UNet)"
    )
    return config


def _tiny_inputs(config: dict):
    sample = int(config.get("sample_size", 8))
    cross = int(config.get("cross_attention_dim", 32))
    latent = torch.rand(1, 4, sample, sample, dtype=torch.float32)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    encoder = torch.rand(1, sample, cross, dtype=torch.float32)
    return latent, timestep, encoder


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_unet_config = _tiny_config
    _MOD.customize_unet_inputs = _tiny_inputs
    _MOD.compare = _loose

    from sd_utils import add_phase4_cli

    parser = argparse.ArgumentParser(description="DEBUG tiny SD UNet")
    add_phase4_cli(parser)
    args = parser.parse_args()
    _MOD.test_unet(
        enablelwp=args.lwp,
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
    )


if __name__ == "__main__":
    main()
