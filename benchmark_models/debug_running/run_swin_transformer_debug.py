#!/usr/bin/env python3
"""Debug / DSP-smoke Swin with a *reduced* architecture.

Use run_swin_transformer.py for full published Swin-Tiny depths=[2,2,6,2].
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from transformers.models.swin.modeling_swin import SwinForImageClassification

_SCRIPT = Path(__file__).resolve().parent.parent / "run_swin_transformer.py"
_SPEC = importlib.util.spec_from_file_location("run_swin_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    config.depths = [1, 1, 1, 1]
    config.embed_dim = 48
    config.num_heads = [3, 6, 12, 24]
    config.hidden_act = "gelu_new"
    print(
        f"[Debug] tiny Swin: depths={config.depths} embed_dim={config.embed_dim} "
        f"(full harness = published [2,2,6,2]/96)"
    )
    return config


def _load_tiny(_model_name, config):
    return SwinForImageClassification(config).half()


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_model_config = _tiny_config
    _MOD.load_swin_model = _load_tiny
    _MOD.compare = _loose
    _MOD.LOWER_CONSTANTS_SEPARATE = False  # tiny ~6MB; separate SO → Bad VA

    from hexkl_utils import add_phase4_args

    parser = argparse.ArgumentParser(description="DEBUG tiny Swin")
    add_phase4_args(parser)
    args = parser.parse_args()
    _MOD.swin_transformer(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        enable_omnifetch_items_1_7=args.enable_omnifetch_items_1_7,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
