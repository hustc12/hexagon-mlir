#!/usr/bin/env python3
"""Reduced SwinV2-Tiny structural candidate for Hexagon/OmniFetch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import Swinv2Config, Swinv2ForImageClassification

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_dsp_heap_256mb,
)


class Swinv2DebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def _compare(hex_outputs, reference):
    actual = hex_outputs[0].float()
    expected = reference.float()
    max_diff = (actual - expected).abs().max().item()
    actual_top1 = actual.argmax(dim=-1).item()
    expected_top1 = expected.argmax(dim=-1).item()
    print(
        f"[Compare] max_abs_diff={max_diff:.4f} "
        f"top1_hexagon={actual_top1} top1_x86={expected_top1} "
        f"top1_match={actual_top1 == expected_top1}"
    )


def run(args):
    patch_dsp_heap_256mb()
    torch.manual_seed(202)
    config = Swinv2Config(
        image_size=32,
        patch_size=4,
        num_channels=3,
        embed_dim=32,
        depths=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 8],
        window_size=4,
        mlp_ratio=4.0,
        hidden_act="gelu_new",
        num_labels=1000,
    )
    print(
        "[DebugCandidate] SwinV2-Tiny structural proxy: "
        f"image={config.image_size} depths={config.depths} "
        f"embed={config.embed_dim} heads={config.num_heads} "
        f"window={config.window_size} "
        "(published=256px/[2,2,6,2]/96/window8; random FP16 weights)"
    )
    model = Swinv2ForImageClassification(config).half().eval()
    wrapped = Swinv2DebugWrapper(model).eval()
    pixel_values = torch.rand(1, 3, 32, 32, dtype=torch.float16)

    module = compile_to_linalg(wrapped, (pixel_values,), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )
    mlir_text = None
    if args.enable_hexkl:
        mlir_text, n_bm, n_f16 = apply_hexkl_ir_rewrites(ir)
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")

    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_omnifetch_vdae,
        not args.disable_layout_aware,
        args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,
        args.enable_omnifetch_items_1_7,
        lower_constants_separate=False,
    )
    outputs = hex_execution(
        module,
        wrapped.__class__.__name__,
        [pixel_values],
        options,
        mlir_text=mlir_text,
    )
    print("Successfully ran SwinV2 Debug candidate on Hexagon DSP!")
    with torch.no_grad():
        reference = wrapped(pixel_values)
    _compare(outputs, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    run(parser.parse_args())
