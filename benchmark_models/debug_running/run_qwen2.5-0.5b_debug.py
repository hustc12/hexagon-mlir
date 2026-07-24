#!/usr/bin/env python3
"""Debug / DSP-smoke Qwen2.5-0.5B with a *reduced* architecture.

Use this for fast compile/device iteration only. Fair Phase-4 measurements
must use run_qwen2.5-0.5b.py (full published architecture).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

_SCRIPT = Path(__file__).resolve().parent.parent / "run_qwen2.5-0.5b.py"
_SPEC = importlib.util.spec_from_file_location("run_qwen_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    config.num_hidden_layers = 2
    config.vocab_size = 4096
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 1
    config.num_key_value_heads = 1
    config.head_dim = 64
    print(
        f"[Debug] tiny Qwen architecture: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} vocab={config.vocab_size}"
    )
    return config


def _load_tiny(_model_name, config):
    # Random weights — full checkpoint does not match shrunk vocab/hidden.
    return AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )


def main():
    _orig_compare = _MOD.compare

    def _compare_loose(*args, **kwargs):
        kwargs["fail_on_mismatch"] = False
        return _orig_compare(*args, **kwargs)

    _MOD.customize_model_config = _tiny_config
    _MOD.load_qwen_model = _load_tiny
    _MOD.compare = _compare_loose

    parser = argparse.ArgumentParser(
        description="DEBUG tiny Qwen2.5-0.5B (not for fair Phase-4 numbers)."
    )
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    _MOD.qwen2_5_0_5b(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
