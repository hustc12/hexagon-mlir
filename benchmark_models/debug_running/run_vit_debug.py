#!/usr/bin/env python3
"""Debug / DSP-smoke ViT with a *reduced* architecture.

Use run_vit.py for full published ViT-Base (12L / patch16 / 224).
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
from transformers import AutoModelForImageClassification

_SCRIPT = Path(__file__).resolve().parent.parent / "run_vit.py"
_SPEC = importlib.util.spec_from_file_location("run_vit_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    # Larger patches → fewer tokens (DSP heap).
    config.patch_size = 32
    config.hidden_act = "gelu_new"
    print(
        f"[Debug] tiny ViT: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} patch={config.patch_size} "
        f"(full harness = published 12L/patch16)"
    )
    return config


def _load_tiny(_model_name, config):
    return AutoModelForImageClassification.from_config(config).half()


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_model_config = _tiny_config
    _MOD.load_vit_model = _load_tiny
    _MOD.compare = _loose

    from hexkl_utils import add_phase4_args

    parser = argparse.ArgumentParser(description="DEBUG tiny ViT")
    add_phase4_args(parser)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()
    _MOD.vit(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        enable_omnifetch_items_1_7=args.enable_omnifetch_items_1_7,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
