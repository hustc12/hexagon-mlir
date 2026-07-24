#!/usr/bin/env python3
"""Debug Real-ESRGAN with smaller spatial input (DSP VA).

Main harness keeps full RRDBNet; --input-size is the capacity knob.
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


def _tiny_size(_default=16):
    print("[Debug] Real-ESRGAN input 8x8 (full harness default 64x64; topology=full RRDBNet)")
    return 8


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_input_size = _tiny_size
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
        seq_len=args.seq_len,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
