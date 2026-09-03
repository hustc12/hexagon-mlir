#!/usr/bin/env python3
"""Aligned non-square GEMM anchor for HMX vs HVX interleaved measurement.

This is the >2x anchor vehicle: a single 32-aligned, non-square linalg.matmul
(M/K/N all divisible by 32, none equal so it is NOT attention-like) that fires
the HexKL/HMX path.  DINOv2 tokens (grid^2+1) are never 32-aligned, so it can
only exercise HVX data-movement items; this GEMM proves the HMX magnitude and
guards that the C1/B1 HVX changes did not regress the HMX path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    build_interleave_configs,
    compile_to_linalg,
    hex_execution,
    hex_execution_interleaved,
    hexagon_options_phase4,
    patch_dsp_heap_256mb,
)


class GemmModel(nn.Module):
    """C = A @ B, f16 inputs / f32 accum (torch-mlir -> linalg.matmul)."""

    def forward(self, a, b):
        return torch.matmul(a, b)


def run(args: argparse.Namespace) -> None:
    assert args.k % 32 == 0 and args.n % 32 == 0, (
        "K/N must be divisible by 32 for the HMX tile; M may be unaligned "
        "when --enable-omnifetch-m-pad-hmx is set"
    )
    assert args.k != args.m and args.n != args.m, (
        "shape must be non-attention-like (K!=M and N!=M) to admit plain HMX"
    )
    patch_dsp_heap_256mb()

    model = GemmModel().half().eval()
    a = torch.randn(args.m, args.k).half()
    b = torch.randn(args.k, args.n).half()
    inputs = [a, b]

    module = compile_to_linalg(model, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] M={args.m} K={args.k} N={args.n} "
        f"matmul={ir.count('linalg.matmul')} "
        f"batch_matmul={ir.count('linalg.batch_matmul')}"
    )

    with torch.no_grad():
        reference = model(*inputs)

    def _compare(label, output):
        actual = output[0]
        if not isinstance(actual, torch.Tensor):
            actual = torch.from_numpy(actual)
        finite = bool(torch.isfinite(actual).all())
        diff = (actual.float() - reference.float()).abs().max().item()
        ok = torch.allclose(actual.float(), reference.float(), atol=0.5, rtol=0.05)
        print(
            f"[Compare] profile={label} finite={finite} "
            f"max_abs_diff={diff:.4f} pass={ok}"
        )
        return ok and finite

    if args.interleave_profiles:
        configs = build_interleave_configs(args, ir)
        results_by_profile = hex_execution_interleaved(
            module,
            model.__class__.__name__,
            inputs,
            configs,
            iterations=args.device_iterations,
            rounds=args.rounds,
        )
        all_ok = True
        for profile, output in results_by_profile.items():
            all_ok = _compare(profile, output) and all_ok
        if not all_ok:
            raise AssertionError("GEMM anchor interleaved correctness gate failed")
        return

    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(ir)
        patched = candidate if n_batch or n_f16 else None
        print(f"[HexKL] batch_matmul->matmul={n_batch}, f16-input rewrite={n_f16}")
        if patched is None:
            print("[HexKL] WARNING: 0 rewrites - no HMX coverage for this shape")
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
        omnifetch_items_through=args.omnifetch_items_through,
        enable_omnifetch_m_pad_hmx=args.enable_omnifetch_m_pad_hmx,
    )
    output = hex_execution(
        module,
        model.__class__.__name__,
        inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    if not _compare(args.backend_profile, output):
        raise AssertionError("GEMM anchor Hexagon correctness gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
