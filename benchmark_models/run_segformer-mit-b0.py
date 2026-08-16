#!/usr/bin/env python3
"""SegFormer MiT-B0 full-structure Hexagon/HexKL/OmniFetch benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import SegformerConfig, SegformerForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


class SegformerWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def run(args: argparse.Namespace) -> None:
    patch_full_model_dsp_heap()
    torch.manual_seed(100)

    # Transformers' SegformerConfig defaults are the published MiT-B0 encoder:
    # depths [2,2,2,2], widths [32,64,160,256], heads [1,2,5,8].
    config = SegformerConfig(
        num_labels=1000,
        hidden_act="gelu_new",
    )
    model = SegformerForImageClassification(config).half().eval()
    wrapped = SegformerWrapper(model).eval()
    pixels = torch.rand(1, 3, 224, 224, dtype=torch.float16)
    inputs = [pixels]
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[FullModel] SegFormer MiT-B0: image=224 "
        f"depths={config.depths} hidden={config.hidden_sizes} "
        f"heads={config.num_attention_heads} params={params} "
        "weights=random_full_structure"
    )

    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )
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
    finite = bool(torch.isfinite(output[0]).all())
    diff = (output[0].float() - reference.float()).abs().max().item()
    top1_match = output[0].argmax().item() == reference.argmax().item()
    print(
        f"[Compare] finite={finite} max_abs_diff={diff:.4f} "
        f"top1_match={top1_match}"
    )
    if not finite or not top1_match:
        raise AssertionError("SegFormer Hexagon result failed correctness gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument(
        "--device-iterations",
        type=int,
        default=1,
        help="Number of serial in-process executions averaged on the DSP.",
    )
    run(parser.parse_args())
