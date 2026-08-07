#!/usr/bin/env python3
"""Reduced offline SegFormer MiT-B0 candidate for OmniFetch screening."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import SegformerConfig, SegformerForImageClassification

_BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BENCH))
from hexkl_utils import (  # noqa: E402
    add_phase4_args, apply_hexkl_ir_rewrites, compile_to_linalg, hex_execution,
    hexagon_options_phase4, patch_dsp_heap_256mb,
)


class SegformerDebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def run(args):
    patch_dsp_heap_256mb()
    torch.manual_seed(100)
    config = SegformerConfig(
        num_encoder_blocks=2,
        depths=[1, 1],
        hidden_sizes=[32, 64],
        num_attention_heads=[1, 2],
        mlp_ratios=[4, 4],
        sr_ratios=[4, 1],
        patch_sizes=[7, 3],
        strides=[4, 2],
        hidden_act="gelu_new",
        num_labels=1000,
    )
    print(
        "[DebugCandidate] SegFormer MiT-B0 proxy: image=32 stages=2 "
        "depths=[1,1] hidden=[32,64] sr=[4,1] (random FP16 weights)"
    )
    wrapped = SegformerDebugWrapper(
        SegformerForImageClassification(config).half().eval()
    ).eval()
    inputs = [torch.rand(1, 3, 32, 32, dtype=torch.float16)]
    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    patched = None
    if args.enable_hexkl:
        patched, n_bm, n_f16 = apply_hexkl_ir_rewrites(ir)
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")
    options = hexagon_options_phase4(
        args.enable_hexkl, args.enable_omnifetch_vdae,
        not args.disable_layout_aware, args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive, args.enable_omnifetch_items_1_7,
        lower_constants_separate=False,
        backend_profile=args.backend_profile,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
    )
    outputs = hex_execution(
        module, wrapped.__class__.__name__, inputs, options, mlir_text=patched,
        iterations=args.device_iterations,
    )
    with torch.no_grad():
        reference = wrapped(*inputs)
    diff = (outputs[0].float() - reference.float()).abs().max().item()
    print(f"[Compare] max_abs_diff={diff:.4f} top1_match={outputs[0].argmax().item() == reference.argmax().item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
