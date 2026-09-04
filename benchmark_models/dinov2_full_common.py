#!/usr/bin/env python3
"""Shared deterministic DINOv2-small full workload for all backends."""
from __future__ import annotations

import types

import torch
from transformers import Dinov2Config, Dinov2ForImageClassification


class Dinov2SmallWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def create_dinov2_small_full_model_and_input(num_hidden_layers: int = 12):
    """Return the full-width workload, optionally with fewer encoder blocks."""
    torch.manual_seed(142)
    config = Dinov2Config(
        # The published DINOv2 checkpoint stores 518x518 positional
        # embeddings. ImageNet evaluation supplies a 224x224 crop and
        # interpolates those embeddings.
        image_size=518,
        patch_size=14,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=6,
        intermediate_size=1536,
        qkv_bias=True,
        hidden_act="gelu_new",
        layerscale_value=1.0,
        use_mask_token=False,
        num_labels=1000,
    )
    model = Dinov2ForImageClassification(config).half().eval()

    # This operation is input-shape invariant. Capture the exact native
    # interpolation once so every compiler sees the same fixed-shape graph.
    embeddings = model.dinov2.embeddings
    with torch.no_grad():
        interpolation_probe = torch.empty(
            1, (224 // config.patch_size) ** 2 + 1, config.hidden_size
        )
        fixed_position_embeddings = embeddings.interpolate_pos_encoding(
            interpolation_probe, 224, 224
        ).to(torch.float16).detach()
    embeddings.register_buffer(
        "alps_fixed_position_embeddings",
        fixed_position_embeddings,
        persistent=False,
    )

    def fixed_position_encoding(self, token_embeddings, height, width):
        del token_embeddings, height, width
        return self.alps_fixed_position_embeddings

    embeddings.interpolate_pos_encoding = types.MethodType(
        fixed_position_encoding, embeddings
    )
    wrapped = Dinov2SmallWrapper(model).eval()
    pixels = torch.rand(1, 3, 224, 224, dtype=torch.float16)
    return wrapped, pixels


def print_dinov2_small_full_identity(
    wrapped: Dinov2SmallWrapper, pixels: torch.Tensor
) -> None:
    model = wrapped.model
    config = model.config
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[FullModel] DINOv2-small patch14: stored_image=518 input=224 "
        f"input_shape={tuple(pixels.shape)} input_dtype={pixels.dtype} "
        f"tokens=257 layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} heads={config.num_attention_heads} "
        f"intermediate={config.intermediate_size} params={params} "
        "seed=142 weights=random_full_structure"
    )
