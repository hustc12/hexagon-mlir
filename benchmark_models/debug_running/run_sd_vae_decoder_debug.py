#!/usr/bin/env python3
"""Debug SD VAE decoder with reduced channels + smaller latent grid.

Use run_sd_vae_decoder.py for full published VAE (64×64 latents / full width).
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

_SCRIPT = Path(__file__).resolve().parent.parent / "run_sd_vae_decoder.py"
_SPEC = importlib.util.spec_from_file_location("run_sd_vae_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    # Full VAE produced ~84MB consts.so and hung on device even at 16×16.
    config["block_out_channels"] = [32, 64]
    config["down_block_types"] = ["DownEncoderBlock2D", "DownEncoderBlock2D"]
    config["up_block_types"] = ["UpDecoderBlock2D", "UpDecoderBlock2D"]
    config["layers_per_block"] = 1
    config["norm_num_groups"] = 8  # must divide channel widths
    print(
        f"[Debug] tiny VAE: channels={config['block_out_channels']} "
        f"layers_per_block={config['layers_per_block']} "
        f"(full harness = published [128,256,512,512])"
    )
    return config


def _tiny_latents(_config):
    print("[Debug] VAE latents 1x4x16x16 (full harness = 64x64)")
    return torch.rand(1, 4, 16, 16, dtype=torch.float16)


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_vae_config = _tiny_config
    _MOD.customize_vae_latents = _tiny_latents
    _MOD.compare = _loose

    from sd_utils import add_phase4_cli

    parser = argparse.ArgumentParser(description="DEBUG tiny SD VAE decoder")
    add_phase4_cli(parser)
    parser.add_argument("--no-mid-attn", action="store_true", default=True)
    parser.add_argument("--keep-mid-attn", action="store_true")
    args = parser.parse_args()
    _MOD.test_vae_decoder(
        enablelwp=args.lwp,
        disable_mid_attn=not args.keep_mid_attn,
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
    )


if __name__ == "__main__":
    main()
