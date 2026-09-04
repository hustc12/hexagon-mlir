#!/usr/bin/env python3
"""Reduced AST AudioSet structural candidate for Hexagon/Alps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import ASTConfig, ASTForAudioClassification

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


class ASTDebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values=input_values).logits


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
    torch.manual_seed(527)
    config = ASTConfig(
        num_mel_bins=32,
        max_length=64,
        patch_size=16,
        frequency_stride=8,
        time_stride=8,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_act="gelu_new",
        num_labels=527,
    )
    print(
        "[DebugCandidate] AST AudioSet structural proxy: "
        f"input={config.max_length}x{config.num_mel_bins} "
        f"patch={config.patch_size} stride="
        f"{config.time_stride}x{config.frequency_stride} "
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} "
        "(AudioSet has 527 labels; random FP16 weights)"
    )
    model = ASTForAudioClassification(config).half().eval()
    wrapped = ASTDebugWrapper(model).eval()
    input_values = torch.rand(
        1, config.max_length, config.num_mel_bins, dtype=torch.float16
    )

    module = compile_to_linalg(wrapped, (input_values,), decomp_pow=False)
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
        args.enable_alps_vdae,
        not args.disable_layout_aware,
        args.alps_lookahead,
        not args.disable_alps_adaptive,
        args.enable_alps_items_1_7,
        lower_constants_separate=False,
    )
    outputs = hex_execution(
        module,
        wrapped.__class__.__name__,
        [input_values],
        options,
        mlir_text=mlir_text,
    )
    print("Successfully ran AST Debug candidate on Hexagon DSP!")
    with torch.no_grad():
        reference = wrapped(input_values)
    _compare(outputs, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    run(parser.parse_args())
