#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dinov2_debug_common import (
    create_dinov2_debug_model_and_input,
    print_dinov2_debug_identity,
)
from hexkl_utils import (
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_dsp_heap_256mb,
)


def run(args):
    patch_dsp_heap_256mb()
    wrapped, pixels = create_dinov2_debug_model_and_input()
    print_dinov2_debug_identity(wrapped, pixels)
    inputs = [pixels]
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
        module,
        wrapped.__class__.__name__,
        inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
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
    parser.add_argument(
        "--device-iterations",
        type=int,
        default=1,
        help="Number of serial in-process executions averaged on the DSP.",
    )
    run(parser.parse_args())
