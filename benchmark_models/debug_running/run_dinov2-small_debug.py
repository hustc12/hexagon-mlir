#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import Dinov2Config, Dinov2ForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_dsp_heap_256mb,
)


class Dinov2DebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return self.model(pixel_values=pixels).logits


def run(args):
    patch_dsp_heap_256mb()
    torch.manual_seed(142)
    config = Dinov2Config(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_act="gelu_new",
        num_labels=10,
        use_mask_token=False,
    )
    wrapped = Dinov2DebugWrapper(
        Dinov2ForImageClassification(config).half().eval()
    ).eval()
    inputs = [torch.rand(1, 3, 32, 32, dtype=torch.float16)]
    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        "[DebugCandidate] DINOv2-small proxy: image=32 patch=8 "
        f"batch_matmul={ir.count('linalg.batch_matmul')}"
    )
    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(ir)
        patched = candidate if n_batch or n_f16 else None
        print(f"[HexKL] rewrites={n_batch + n_f16}")
    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_omnifetch_vdae,
        not args.disable_layout_aware,
        args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,
        args.enable_omnifetch_items_1_7,
        lower_constants_separate=False,
    )
    output = hex_execution(
        module, wrapped.__class__.__name__, inputs, options, mlir_text=patched
    )
    with torch.no_grad():
        reference = wrapped(*inputs)
    diff = (output[0].float() - reference.float()).abs().max().item()
    print(
        f"[Compare] max_abs_diff={diff:.4f} "
        f"top1_match={output[0].argmax().item() == reference.argmax().item()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_phase4_args(parser)
    run(parser.parse_args())
