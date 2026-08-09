"""Single source of truth for the deterministic DINOv2 Debug benchmark."""
from __future__ import annotations

import hashlib
import types

import torch
from transformers import Dinov2Config, Dinov2ForImageClassification


class Dinov2DebugWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixels).logits


def create_dinov2_debug_model_and_input(
    *,
    static_position_export: bool = False,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Create the exact model and logical input used by every backend."""
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
    model = Dinov2DebugWrapper(
        Dinov2ForImageClassification(config).half().eval()
    ).eval()

    if static_position_export:
        # Transformers deliberately traces the dynamic bicubic interpolation
        # path even when the image has the configured fixed 32x32 size. Eager
        # execution returns the existing position tensor in this case. Make
        # that equivalent fixed-shape behavior explicit for external export.
        embeddings = model.model.dinov2.embeddings

        def fixed_position_encoding(
            self, token_embeddings: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            del token_embeddings, height, width
            return self.position_embeddings

        embeddings.interpolate_pos_encoding = types.MethodType(
            fixed_position_encoding, embeddings
        )

    pixels = torch.rand(1, 3, 32, 32, dtype=torch.float16)
    return model, pixels


def dinov2_debug_identity(
    model: torch.nn.Module, pixels: torch.Tensor
) -> tuple[str, str]:
    """Return stable hashes proving model parameters and input identity."""
    model_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        model_hash.update(name.encode("utf-8"))
        model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    input_hash = hashlib.sha256(
        pixels.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    return model_hash.hexdigest(), input_hash


def print_dinov2_debug_identity(
    model: torch.nn.Module, pixels: torch.Tensor
) -> None:
    model_hash, input_hash = dinov2_debug_identity(model, pixels)
    print(
        f"[DINOv2 Identity] model_sha256={model_hash} "
        f"input_sha256={input_hash}"
    )
