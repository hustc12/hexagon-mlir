#!/usr/bin/env python3
"""BEiT-base patch16-224 full-structure Hexagon benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import BeitConfig, BeitForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


class BeitBaseWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class FixedNativeRelativePositionBias(torch.nn.Module):
    """Export-safe, inference-equivalent BEiT bias for the native 14x14 grid."""

    def __init__(self, source: torch.nn.Module):
        super().__init__()
        # Retain the checkpoint parameter so the model's parameter count and
        # state-dict meaning do not change.  Fixed-resolution inference can
        # capture its evaluated result and avoid the unsupported dynamic
        # meshgrid/interpolate/index construction in Transformers.
        self.relative_position_bias_table = source.relative_position_bias_table
        with torch.no_grad():
            fixed_bias = source(source.window_size).detach()
        # Store as a non-persistent buffer (NOT a plain attribute): torch-mlir
        # lifts buffers into leading public function arguments.  Supplying the
        # bias as a runtime memref arg (as the DINOv2/DeiT runners do for their
        # position embeddings) avoids the on-device NULL/TLBMISS fault seen when
        # a large bias is inlined as a constant and referenced from the async
        # worker threads.  The run() driver supplies every lifted bias, in layer
        # order, ahead of the pixel input.
        self.register_buffer("fixed_bias", fixed_bias, persistent=False)

    def forward(
        self,
        window_size,
        interpolate_pos_encoding: bool = False,
        dim_size=None,
    ) -> torch.Tensor:
        del window_size, interpolate_pos_encoding, dim_size
        return self.fixed_bias


def freeze_native_relative_position_bias(model: torch.nn.Module) -> int:
    """Freeze every per-layer BEiT bias without removing the bias itself."""
    replaced = 0
    for layer in model.beit.encoder.layer:
        attention = layer.attention.attention
        source = attention.relative_position_bias
        if source is not None:
            attention.relative_position_bias = FixedNativeRelativePositionBias(source)
            replaced += 1
    return replaced


def run(args: argparse.Namespace) -> None:
    patch_full_model_dsp_heap()
    torch.manual_seed(871)
    # Matches microsoft/beit-base-patch16-224-pt22k-ft22k. In particular,
    # retain per-layer relative-position bias; the Debug proxy disabled it.
    config = BeitConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        # gelu_new lowers to math.tanh + math.fpowi (integer power), both of
        # which the Hexagon backend supports; plain "gelu" emits math.erf which
        # the backend LLVM translation cannot lower. Matches the DINOv2 runner.
        hidden_act="gelu_new",
        use_relative_position_bias=True,
        use_shared_relative_position_bias=False,
        use_absolute_position_embeddings=False,
        num_labels=1000,
    )
    model = BeitForImageClassification(config).half().eval()
    frozen_biases = freeze_native_relative_position_bias(model)
    wrapped = BeitBaseWrapper(model).eval()
    inputs = [torch.rand(1, 3, 224, 224, dtype=torch.float16)]
    # torch-mlir lifts each frozen relative-position-bias buffer to a leading
    # public function argument, in encoder-layer order, with the pixel input
    # last.  Supply them in exactly that order for the Hexagon ABI.
    lifted_biases = [
        layer.attention.attention.relative_position_bias.fixed_bias
        for layer in model.beit.encoder.layer
    ]
    device_inputs = [*lifted_biases, inputs[0]]
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[FullModel] BEiT-base patch16-224: tokens=197 layers=12 "
        f"hidden=768 heads=12 intermediate=3072 params={params} "
        f"relative_position_bias=per_layer_frozen_native({frozen_biases}) "
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
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(ir)
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
        device_inputs,
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
        raise AssertionError("BEiT-base Hexagon result failed correctness gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
