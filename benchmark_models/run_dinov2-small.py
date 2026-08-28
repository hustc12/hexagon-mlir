#!/usr/bin/env python3
"""DINOv2-small full-structure Hexagon/HexKL/OmniFetch benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov2_full_common import (  # noqa: E402
    create_dinov2_small_full_model_and_input,
    print_dinov2_small_full_identity,
)
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    build_interleave_configs,
    compile_to_linalg,
    hex_execution,
    hex_execution_interleaved,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


def run(args: argparse.Namespace) -> None:
    patch_full_model_dsp_heap()
    wrapped, pixels = create_dinov2_small_full_model_and_input()
    print_dinov2_small_full_identity(wrapped, pixels)
    inputs = [pixels]
    # torch-mlir lifts the non-persistent fixed position-embedding buffer to
    # the first public function argument.  Keep it explicit for the Hexagon
    # ABI; the Python model still exposes only pixel_values to callers.
    device_inputs = [
        wrapped.model.dinov2.embeddings.omnifetch_fixed_position_embeddings,
        pixels,
    ]

    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )

    with torch.no_grad():
        reference = wrapped(*inputs)

    if args.interleave_profiles:
        configs = build_interleave_configs(args, ir)
        results_by_profile = hex_execution_interleaved(
            module,
            wrapped.__class__.__name__,
            device_inputs,
            configs,
            iterations=args.device_iterations,
            rounds=args.rounds,
        )
        all_ok = True
        for profile, output in results_by_profile.items():
            finite = bool(torch.isfinite(output[0]).all())
            top1_match = output[0].argmax().item() == reference.argmax().item()
            abs_diff = (output[0].float() - reference.float()).abs()
            diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            allclose = bool(
                torch.allclose(
                    output[0].float(), reference.float(), rtol=1e-2, atol=1e-2
                )
            )
            print(
                f"[Compare] profile={profile} finite={finite} "
                f"max_abs_diff={diff:.4f} mean_abs_diff={mean_diff:.6f} "
                f"allclose={allclose} top1_match={top1_match}"
            )
            all_ok = all_ok and finite and allclose and top1_match
        if not all_ok:
            raise AssertionError(
                "DINOv2-small interleaved result failed correctness gate"
            )
        return

    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(
            ir, enable_m_pad=args.enable_omnifetch_m_pad_hmx
        )
        patched = candidate if n_batch or n_f16 else None
        print(
            f"[HexKL] batch_matmul→matmul={n_batch}, "
            f"f16-input rewrite={n_f16}"
        )
    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_omnifetch_vdae,
        not args.disable_layout_aware,
        args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,
        args.enable_omnifetch_items_1_7,
        lower_constants_separate=True,
        backend_profile=args.backend_profile,
        enable_lwp=args.enable_lwp,
        lwp_loop_depth=args.lwp_loop_depth,
        disable_lwp_loop=args.disable_lwp_loop,
        instrument_lwp_hexkl_phases=args.lwp_hexkl_phases,
        omnifetch_items_through=args.omnifetch_items_through,
        enable_omnifetch_m_pad_hmx=args.enable_omnifetch_m_pad_hmx,
        enable_out_params=args.enable_out_params,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
        enable_omnifetch_kv_cache_prefetch=(
            args.enable_omnifetch_kv_cache_prefetch
        ),
        disable_omnifetch_persistent_wh_cache=(
            args.disable_omnifetch_persistent_wh_cache
        ),
        alps_p0_mode=args.alps_p0_mode,
    )
    output = hex_execution(
        module,
        wrapped.__class__.__name__,
        device_inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    finite = bool(torch.isfinite(output[0]).all())
    abs_diff = (output[0].float() - reference.float()).abs()
    diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    allclose = bool(
        torch.allclose(output[0].float(), reference.float(), rtol=1e-2, atol=1e-2)
    )
    top1_match = output[0].argmax().item() == reference.argmax().item()
    print(
        f"[Compare] finite={finite} max_abs_diff={diff:.4f} "
        f"mean_abs_diff={mean_diff:.6f} allclose={allclose} "
        f"top1_match={top1_match}"
    )
    if not finite or not allclose or not top1_match:
        raise AssertionError("DINOv2-small Hexagon result failed correctness gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
