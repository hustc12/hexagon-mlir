#!/usr/bin/env python3
"""Debug SD CLIP text encoder with reduced topology."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from transformers import CLIPTextModel

_SCRIPT = Path(__file__).resolve().parent.parent / "run_sd_text_encoder.py"
_SPEC = importlib.util.spec_from_file_location("run_sd_te_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    config.num_hidden_layers = 1
    config.hidden_size = 16
    config.intermediate_size = 16
    config.num_attention_heads = 1
    config.head_dim = 16
    print(
        f"[Debug] tiny CLIP: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} (full = published 12L/768)"
    )
    return config


def _tiny_seq_len(_default: int = 77) -> int:
    # Full CLIP seq=77 can blow Hexagon stack frames on tiny models too.
    print("[Debug] CLIP seq_len=16 (full harness default=77)")
    return 16


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_clip_config = _tiny_config
    _MOD.customize_seq_len = _tiny_seq_len
    _MOD.load_clip = lambda c: CLIPTextModel(c)
    _MOD.compare = _loose

    from sd_utils import add_phase4_cli

    parser = argparse.ArgumentParser(description="DEBUG tiny SD text encoder")
    add_phase4_cli(parser)
    args = parser.parse_args()
    _MOD.test_text_encoder(
        enablelwp=args.lwp,
        enable_hexkl=args.enable_hexkl,
        enable_alps_vdae=args.enable_alps_vdae,
        enable_alps_layout_aware=not args.disable_layout_aware,
        alps_lookahead=args.alps_lookahead,
        enable_alps_adaptive=not args.disable_alps_adaptive,
        enable_alps_items_1_7=args.enable_alps_items_1_7,
    )


if __name__ == "__main__":
    main()
