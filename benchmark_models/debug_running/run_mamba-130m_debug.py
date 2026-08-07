#!/usr/bin/env python3
"""Debug / DSP-smoke Mamba-130M with a *reduced* architecture.

Use run_mamba-130m.py for full published 32L structure.
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
from transformers import AutoModelForCausalLM

_SCRIPT = Path(__file__).resolve().parent.parent / "run_mamba-130m.py"
_SPEC = importlib.util.spec_from_file_location("run_mamba_full", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def _tiny_config(config):
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 128
    config.vocab_size = 4096
    print(
        f"[Debug] tiny Mamba: layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} vocab={config.vocab_size} "
        f"(full harness = published 32L/768-d; 1L for sequential-SSM compile)"
    )
    return config


def _load_tiny(_model_name, config):
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)


def main():
    _orig = _MOD.compare

    def _loose(*a, **k):
        k["fail_on_mismatch"] = False
        return _orig(*a, **k)

    _MOD.customize_model_config = _tiny_config
    _MOD.load_mamba_model = _load_tiny
    _MOD.compare = _loose

    from hexkl_utils import add_phase4_args, phase4_kwargs_from_args

    parser = argparse.ArgumentParser(description="DEBUG tiny Mamba-130M")
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    args = parser.parse_args()
    _MOD.mamba_130m(
        **phase4_kwargs_from_args(args),
        device_iterations=args.device_iterations,
        backend_profile=args.backend_profile,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
    )


if __name__ == "__main__":
    main()
