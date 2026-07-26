#!/usr/bin/env python3
"""Debug / DSP-smoke Falcon-RW-1B with a *reduced* architecture.

Use this for fast compile/device iteration only. Fair / full-structure runs
must use run_falcon_rw_1b.py (published 24L Falcon-RW-1B).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

_SCRIPT = Path(__file__).resolve().parent.parent / "run_falcon_rw_1b.py"
_SPEC = importlib.util.spec_from_file_location("run_falcon_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    # Shrink depth *and* width/vocab so consts.so fits DSP VA (full hidden=2048
    # + vocab=50304 made a ~389MB consts SO and hung on device).
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.num_attention_heads = 2
    config.num_kv_heads = 2
    config.ffn_hidden_size = 128
    config.vocab_size = 4096
    print(
        f"[Debug] tiny Falcon: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} heads={config.num_attention_heads} "
        f"vocab={config.vocab_size} (full harness = published 24L/2048-d)"
    )
    return config


def _load_tiny(_model_name, config):
    # Keep independently launched ablation cases on byte-identical weights.
    # Without this, every subprocess benchmarks a different random model.
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )


def main():
    _orig_compare = _MOD.compare

    def _compare_loose(*args, **kwargs):
        kwargs["fail_on_mismatch"] = False
        return _orig_compare(*args, **kwargs)

    _MOD.customize_model_config = _tiny_config
    _MOD.load_falcon_model = _load_tiny
    _MOD.compare = _compare_loose

    parser = argparse.ArgumentParser(
        description="DEBUG 2-layer Falcon-RW-1B (not for full-depth numbers)."
    )
    parser.add_argument("--enable-lwp", action="store_true")
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--enable-omnifetch-weight-prepack", action="store_true")
    parser.add_argument("--enable-omnifetch-persistent-wh-cache",
                        action="store_true")
    parser.add_argument("--enable-omnifetch-two-dim-pipeline",
                        action="store_true")
    parser.add_argument("--enable-omnifetch-vtcm-coloring",
                        action="store_true")
    parser.add_argument("--enable-omnifetch-kv-cache-prefetch",
                        action="store_true")
    parser.add_argument("--enable-omnifetch-dequant-reshape",
                        action="store_true")
    parser.add_argument("--omnifetch-kv-cache-page-tokens", type=int,
                        default=32)
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument("--enable-hexkl-persistent-vtcm", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    _MOD.falcon_rw_1b(
        enablelwp=args.enable_lwp,
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        enable_omnifetch_weight_prepack=args.enable_omnifetch_weight_prepack,
        enable_omnifetch_persistent_wh_cache=(
            args.enable_omnifetch_persistent_wh_cache
        ),
        enable_omnifetch_two_dim_pipeline=(
            args.enable_omnifetch_two_dim_pipeline
        ),
        enable_omnifetch_vtcm_coloring=(
            args.enable_omnifetch_vtcm_coloring
        ),
        enable_omnifetch_kv_cache_prefetch=(
            args.enable_omnifetch_kv_cache_prefetch
        ),
        enable_omnifetch_dequant_reshape=(
            args.enable_omnifetch_dequant_reshape
        ),
        omnifetch_kv_cache_page_tokens=(
            args.omnifetch_kv_cache_page_tokens
        ),
        device_iterations=args.device_iterations,
        enable_hexkl_persistent_vtcm=args.enable_hexkl_persistent_vtcm,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
