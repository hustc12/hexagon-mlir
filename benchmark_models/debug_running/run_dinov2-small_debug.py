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
    build_interleave_configs,
    compile_to_linalg,
    hex_execution,
    hex_execution_interleaved,
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

    with torch.no_grad():
        reference = wrapped(*inputs)

    if args.interleave_profiles:
        configs = build_interleave_configs(args, ir)
        results_by_profile = hex_execution_interleaved(
            module,
            wrapped.__class__.__name__,
            inputs,
            configs,
            iterations=args.device_iterations,
            rounds=args.rounds,
        )
        all_ok = True
        for profile, output in results_by_profile.items():
            finite = bool(torch.isfinite(output[0]).all())
            top1_match = output[0].argmax().item() == reference.argmax().item()
            diff = (output[0].float() - reference.float()).abs().max().item()
            print(
                f"[Compare] profile={profile} finite={finite} "
                f"max_abs_diff={diff:.4f} top1_match={top1_match}"
            )
            all_ok = all_ok and finite and top1_match
        if not all_ok:
            raise AssertionError(
                "DINOv2 Debug interleaved result failed correctness gate"
            )
        return

    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(ir)
        patched = candidate if n_batch or n_f16 else None
        print(f"[HexKL] rewrites={n_batch + n_f16}")
    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_alps_vdae,
        not args.disable_layout_aware,
        args.alps_lookahead,
        not args.disable_alps_adaptive,
        args.enable_alps_items_1_7,
        lower_constants_separate=False,
        backend_profile=args.backend_profile,
        enable_lwp=args.enable_lwp,
        lwp_loop_depth=args.lwp_loop_depth,
        disable_lwp_loop=args.disable_lwp_loop,
        alps_items_through=args.alps_items_through,
        enable_alps_kv_vtcm=args.enable_alps_kv_vtcm,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=(
            args.apt_get_hx_manual_candidate_ids
        ),
    )
    output = hex_execution(
        module,
        wrapped.__class__.__name__,
        inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    finite = bool(torch.isfinite(output[0]).all())
    diff = (output[0].float() - reference.float()).abs().max().item()
    top1_match = output[0].argmax().item() == reference.argmax().item()
    print(
        f"[Compare] finite={finite} max_abs_diff={diff:.4f} "
        f"top1_match={top1_match}"
    )
    if not finite or not top1_match:
        raise AssertionError("DINOv2 Debug Hexagon result failed correctness gate")


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
