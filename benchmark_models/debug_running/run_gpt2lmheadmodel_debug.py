#!/usr/bin/env python3
"""Debug / DSP-smoke GPT-2 with n_layer=2.

Use run_gpt2lmheadmodel.py (full 12-layer GPT-2) for fair Phase-4 measurements.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "run_gpt2lmheadmodel.py"
_SPEC = importlib.util.spec_from_file_location("run_gpt2_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _two_layer(config):
    config.n_layer = 2
    print(f"[Debug] GPT-2 n_layer={config.n_layer} (full harness uses 12)")
    return config


def main():
    _MOD.customize_gpt2_config = _two_layer

    parser = argparse.ArgumentParser(
        description="DEBUG 2-layer GPT-2 (not for fair Phase-4 numbers)."
    )
    parser.add_argument("--enable-lwp", action="store_true")
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-vtcm-tiling", action="store_true")
    parser.add_argument("--enable-convert-to-hexagonmem", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    _MOD.gpt2lmheadmodel(
        enablelwp=args.enable_lwp,
        enable_hexkl=args.enable_hexkl,
        enable_vtcm_tiling=args.enable_vtcm_tiling,
        enable_convert_to_hexagonmem=args.enable_convert_to_hexagonmem,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
