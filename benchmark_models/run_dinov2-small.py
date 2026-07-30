#!/usr/bin/env python3
"""DINOv2-small full-structure Hexagon/HexKL/OmniFetch benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

import torch
from transformers import Dinov2Config, Dinov2ForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


class Dinov2SmallWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def run(args: argparse.Namespace) -> None:
    patch_full_model_dsp_heap()
    torch.manual_seed(142)
    config = Dinov2Config(
        # The published DINOv2 checkpoint stores 518x518 positional
        # embeddings. ImageNet evaluation supplies a 224x224 crop and
        # interpolates those embeddings.
        image_size=518,
        patch_size=14,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        qkv_bias=True,
        hidden_act="gelu_new",
        layerscale_value=1.0,
        use_mask_token=False,
        num_labels=1000,
    )
    model = Dinov2ForImageClassification(config).half().eval()

    # Precompute the published 518->224 positional interpolation. This retains
    # the checkpoint's full 1370-token position parameter while keeping a
    # fixed-shape bicubic operation out of the device graph.
    embeddings = model.dinov2.embeddings
    with torch.no_grad():
        interpolation_probe = torch.empty(
            1, (224 // config.patch_size) ** 2 + 1, config.hidden_size
        )
        fixed_position_embeddings = embeddings.interpolate_pos_encoding(
            interpolation_probe, 224, 224
        ).to(torch.float16).detach()

    def fixed_position_encoding(self, token_embeddings, height, width):
        del token_embeddings, height, width
        return fixed_position_embeddings

    embeddings.interpolate_pos_encoding = types.MethodType(
        fixed_position_encoding, embeddings
    )

    wrapped = Dinov2SmallWrapper(model).eval()
    inputs = [torch.rand(1, 3, 224, 224, dtype=torch.float16)]
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[FullModel] DINOv2-small patch14: stored_image=518 input=224 "
        "tokens=257 layers=12 "
        f"hidden=384 heads=6 intermediate=1536 params={params} "
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
        raise AssertionError("DINOv2-small Hexagon result failed correctness gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
