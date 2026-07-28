#!/usr/bin/env python3
"""Offline FP16 Qwen2.5-Coder structural proxy (random weights)."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

_SCRIPT = Path(__file__).resolve().parent.parent / "run_qwen2.5-0.5b.py"
_SPEC = importlib.util.spec_from_file_location("run_qwen_coder_base", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


class _Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + ord(c) % 251 for c in text] or [2]

    def decode(self, ids):
        return f"<coder-token-{int(ids[0])}>"


class _TokenizerFactory:
    from_pretrained = staticmethod(lambda _name: _Tokenizer())


class _ConfigFactory:
    from_pretrained = staticmethod(lambda _name: Qwen2Config())


def _config(config):
    # Published Coder-0.5B uses 14Q/2KV (7:1).  Keep that exact grouping
    # geometry as 7Q/1KV while choosing HexKL-aligned hidden/FFN dimensions.
    config.num_hidden_layers = 2
    config.vocab_size = 4096
    config.hidden_size = 224
    config.intermediate_size = 384
    config.num_attention_heads = 7
    config.num_key_value_heads = 1
    config.head_dim = 32
    config.max_position_embeddings = 512
    config.use_cache = False
    print(
        "[DebugCandidate] Qwen2.5-Coder-0.5B proxy: "
        "layers=2 hidden=224 intermediate=384 heads=7/1 "
        "(published GQA ratio=14/2; random FP16 weights)"
    )
    return config


def _load(_name, config):
    torch.manual_seed(525)
    return AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float16, attn_implementation="eager"
    )


def main():
    original_compare = _MOD.compare

    def diagnostic_compare(*args, **kwargs):
        kwargs["fail_on_mismatch"] = False
        return original_compare(*args, **kwargs)

    _MOD.AutoTokenizer = _TokenizerFactory
    _MOD.AutoConfig = _ConfigFactory
    _MOD.customize_model_config = _config
    _MOD.load_qwen_model = _load
    _MOD.compare = diagnostic_compare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument("--enable-omnifetch-items-1-7", action="store_true")
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()
    _MOD.qwen2_5_0_5b(
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
