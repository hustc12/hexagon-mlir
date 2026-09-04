#!/usr/bin/env python3
"""Debug / DSP-smoke TinyLlama-1.1B with a *reduced* architecture.

Use this for fast compile/device iteration only. Fair / full-structure runs
must use run_tinyllama.py (published 22L TinyLlama-1.1B).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

_SCRIPT = Path(__file__).resolve().parent.parent / "run_tinyllama.py"
_SPEC = importlib.util.spec_from_file_location("run_tinyllama_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    # Shrink depth *and* width/vocab so consts.so fits DSP VA.
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.vocab_size = 4096
    print(
        f"[Debug] tiny TinyLlama: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} heads={config.num_attention_heads} "
        f"vocab={config.vocab_size} (full harness = published 22L/2048-d)"
    )
    return config


def _load_tiny(_model_name, config):
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
    _MOD.load_tinyllama_model = _load_tiny
    _MOD.compare = _compare_loose

    parser = argparse.ArgumentParser(
        description="DEBUG tiny TinyLlama-1.1B (not for fair Phase-4 numbers)."
    )
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-hvx-vector", action="store_true")
    parser.add_argument("--enable-alps-activation-multicast",
                        action="store_true")
    parser.add_argument("--enable-alps-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--alps-lookahead", type=int, default=2)
    parser.add_argument("--disable-alps-adaptive", action="store_true")
    parser.add_argument("--enable-alps-items-1-7", action="store_true")
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    _MOD.tinyllama_1_1b(
        enable_hexkl=args.enable_hexkl,
        enable_hvx_vector=args.enable_hvx_vector,
        enable_alps_activation_multicast=(
            args.enable_alps_activation_multicast
        ),
        enable_alps_vdae=args.enable_alps_vdae,
        enable_alps_layout_aware=not args.disable_layout_aware,
        alps_lookahead=args.alps_lookahead,
        enable_alps_adaptive=not args.disable_alps_adaptive,
        enable_alps_items_1_7=args.enable_alps_items_1_7,
        device_iterations=args.device_iterations,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
