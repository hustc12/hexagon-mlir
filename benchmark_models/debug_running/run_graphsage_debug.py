#!/usr/bin/env python3
"""Debug / DSP-smoke GraphSAGE-BERT with a *reduced* architecture.

Use run_graphsage.py for full published 12L GeBERT structure.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import torch
from transformers import AutoModel

_SCRIPT = Path(__file__).resolve().parent.parent / "run_graphsage.py"
_SPEC = importlib.util.spec_from_file_location("run_graphsage_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.vocab_size = 4096
    config.max_position_embeddings = 64
    config.hidden_act = "gelu_new"
    print(
        f"[Debug] tiny GraphSAGE: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} vocab={config.vocab_size} "
        f"(full harness = published 12L/768-d)"
    )
    return config


def _load_tiny(_model_name, config):
    return AutoModel.from_config(
        config, torch_dtype=torch.float16, attn_implementation="eager"
    )


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_model_config = _tiny_config
    _MOD.load_graphsage_model = _load_tiny
    _MOD.compare = _loose

    from hexkl_utils import add_phase4_args, phase4_kwargs_from_args

    parser = argparse.ArgumentParser(description="DEBUG tiny GraphSAGE")
    add_phase4_args(parser)
    args = parser.parse_args()
    _MOD.graphsage_bert(**phase4_kwargs_from_args(args))


if __name__ == "__main__":
    main()
