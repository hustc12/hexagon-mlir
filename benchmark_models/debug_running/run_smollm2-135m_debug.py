#!/usr/bin/env python3
"""Debug SmolLM2-135M structural proxy for Hexagon/OmniFetch experiments.

This intentionally uses deterministic random weights and a reduced topology.
It preserves SmolLM2's Llama-family decoder and 3:1 GQA ratio; it is not a
quality benchmark and must not be reported as the full 135M checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

_SCRIPT = Path(__file__).resolve().parent.parent / "run_tinyllama.py"
_SPEC = importlib.util.spec_from_file_location("run_smollm2_base", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


class _OfflineDebugTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        # Stable content-filled IDs; token quality is irrelevant to a random
        # structural proxy, but non-constant IDs exercise embedding gathers.
        return [3 + (ord(ch) % 251) for ch in text] or [self.eos_token_id]

    def decode(self, ids):
        return f"<debug-token-{int(ids[0])}>"


class _OfflineTokenizerFactory:
    @staticmethod
    def from_pretrained(_model_name):
        return _OfflineDebugTokenizer()


class _OfflineConfigFactory:
    @staticmethod
    def from_pretrained(_model_name):
        return LlamaConfig()


def _smollm2_debug_config(config):
    # Published SmolLM2-135M: 30L, hidden=576, FFN=1536, 9Q/3KV.
    # Keep the decoder/GQA geometry while making device iteration tractable.
    config.num_hidden_layers = 2
    config.hidden_size = 96
    config.intermediate_size = 256
    config.num_attention_heads = 3
    config.num_key_value_heads = 1
    config.head_dim = 32
    config.vocab_size = 4096
    config.max_position_embeddings = 512
    config.tie_word_embeddings = True
    config.use_cache = False
    print(
        "[DebugCandidate] SmolLM2-135M structural proxy: "
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"intermediate={config.intermediate_size} "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
        "(published=30L/576/1536/9Q:3KV; random FP16 weights)"
    )
    return config


def _load_random_fp16(_model_name, config):
    torch.manual_seed(135)
    return AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )


def main():
    original_compare = _MOD.compare

    def _diagnostic_compare(*args, **kwargs):
        # Numerical deltas are still printed; keep candidate sweeps running so
        # all three backends are represented in the matrix.
        kwargs["fail_on_mismatch"] = False
        return original_compare(*args, **kwargs)

    _MOD.customize_model_config = _smollm2_debug_config
    _MOD.load_tinyllama_model = _load_random_fp16
    _MOD.compare = _diagnostic_compare
    _MOD.AutoTokenizer = _OfflineTokenizerFactory
    _MOD.AutoConfig = _OfflineConfigFactory

    parser = argparse.ArgumentParser(
        description="DEBUG SmolLM2-135M structural proxy (random FP16 weights)."
    )
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--enable-omnifetch-items-1-7", action="store_true")
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()
    _MOD.tinyllama_1_1b(
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
