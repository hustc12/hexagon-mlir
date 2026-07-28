#!/usr/bin/env python3
"""Debug Real-ESRGAN with a reduced RRDBNet topology and spatial input.

This is a compiler/runtime smoke and three-way A/B runner. Use the parent
``run_real-esrgan.py`` for full-model measurements.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

_SCRIPT = Path(__file__).resolve().parent.parent / "run_real-esrgan.py"
_SPEC = importlib.util.spec_from_file_location("run_esrgan_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _load_tiny_model(device):
    from RealESRGAN.rrdbnet_arch import RRDBNet

    print("[Debug] Real-ESRGAN reduced RRDBNet: feat=16 blocks=2 grow=8")
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=16,
        num_block=2,
        num_grow_ch=8,
    ).to(device).eval()


def _tiny_size(_default=16):
    print("[Debug] Real-ESRGAN input 8x8 (full harness default 64x64)")
    return 8


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_input_size = _tiny_size
    _MOD.load_real_esrgan_model = _load_tiny_model
    _MOD.compare = _loose

    from hexkl_utils import add_phase4_args

    parser = argparse.ArgumentParser(description="DEBUG tiny Real-ESRGAN")
    add_phase4_args(parser)
    parser.add_argument("--input-size", type=int, default=None)
    args = parser.parse_args()
    _MOD.real_esrgan(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        enable_omnifetch_items_1_7=args.enable_omnifetch_items_1_7,
        seq_len=args.seq_len,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
