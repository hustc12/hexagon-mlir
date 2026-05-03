# ===- test_sd_vae_decoder.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Stable Diffusion — VAE Decoder benchmark for Hexagon NPU.
#
# Model: CompVis/stable-diffusion-v1-4  →  vae subfolder
# Architecture: AutoencoderKL (decode path only)
# Input:  latents  (1, 4, 64, 64)  float32
# Output: sample   (1, 3, 512, 512) float32
#
# The VAE Decoder is the most compute-intensive part of the VAE used in SD.
# We wrap vae.decode() so torch-mlir sees a plain tensor→tensor function.
#
# Usage:
#   python benchmark_models/test_sd_vae_decoder.py [--lwp]
#
# ===------------------------------------------------------------------------===

import argparse
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from sd_utils import (
    SD_MODEL_ID,
    compile_to_linalg,
    hex_execution,
    x86_execution,
    compare,
    default_options,
    process_lwp,
)


class GroupNormFP16(torch.nn.Module):
    """Drop-in replacement for torch.nn.GroupNorm that stays in f16.

    torch-mlir's built-in GroupNorm lowering hard-codes a f16→f64 promotion
    for the variance reduction, regardless of the model dtype.  On Hexagon DSP
    there is no f64 hardware unit — every f64 op is software-emulated and
    ~100× slower than f16.  At 512×512 resolution a single GroupNorm reduction
    operates on a 1×32×1048576 f64 tensor (256 MB); with ~12 such reductions
    the total f64 data movement exceeds 3 GB, making execution infeasibly slow.

    This replacement implements the identical mathematical operation using only
    f16 arithmetic so torch-mlir emits pure f16 linalg.generic ops instead of
    the f64-promoting GroupNorm lowering.  The model structure (number of
    groups, affine parameters, eps) is preserved exactly.
    """

    def __init__(self, orig: torch.nn.GroupNorm):
        super().__init__()
        self.num_groups = orig.num_groups
        self.eps = orig.eps
        self.weight = orig.weight  # shared reference — no copy
        self.bias = orig.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, *spatial)  — all f16
        N, C = x.shape[0], x.shape[1]
        G = self.num_groups
        # Reshape to (N, G, -1) so the reduction is over the last dim only.
        # This avoids the large-tensor f64 path that torch.nn.GroupNorm triggers.
        x_grouped = x.reshape(N, G, -1)                          # (N, G, L)
        mean = x_grouped.mean(dim=-1, keepdim=True)              # (N, G, 1)
        var  = ((x_grouped - mean) ** 2).mean(dim=-1, keepdim=True)  # (N, G, 1)
        x_norm = (x_grouped - mean) / (var + self.eps).sqrt()   # (N, G, L)
        x_norm = x_norm.reshape(x.shape)                         # (N, C, *spatial)
        if self.weight is not None:
            # weight/bias are (C,) — broadcast over N and spatial dims
            shape = (1, C) + (1,) * (x.dim() - 2)
            x_norm = x_norm * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x_norm


class VAEDecodeWrapper(torch.nn.Module):
    """Wrap AutoencoderKL.decode() to return the sample tensor directly.

    torch_mlir's fx.export_and_import cannot handle dataclass outputs
    (DecoderOutput).  This wrapper returns a plain tensor so the C++
    MemRefDescriptor interface gets a single concrete output.
    """

    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents).sample


def test_vae_decoder(enablelwp: bool = False):
    print("\n=== Stable Diffusion — VAE Decoder ===")

    # Use from_config (random weights) for fast compilation testing.
    # Swap to AutoencoderKL.from_pretrained(...) for real-weight tests.
    config = AutoencoderKL.load_config(SD_MODEL_ID, subfolder="vae")
    vae = AutoencoderKL.from_config(config)

    # FIX: Replace all GroupNorm layers with GroupNormFP16.
    #
    # torch-mlir's built-in GroupNorm lowering hard-codes a promotion to f64
    # for the variance reduction regardless of model dtype (f32 or f16).
    # GroupNormFP16 implements the same operation with plain f16 arithmetic so
    # torch-mlir emits pure f16 linalg.generic ops instead.
    # The model structure (groups, affine params, eps) is unchanged.
    vae = vae.half()
    for name, module in list(vae.named_modules()):
        if isinstance(module, torch.nn.GroupNorm):
            # Navigate to the parent and replace the child attribute.
            parts = name.rsplit(".", 1)
            parent = vae
            if len(parts) == 2:
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], GroupNormFP16(module))

    model = VAEDecodeWrapper(vae)
    model.eval()

    # Latent space: 64×64 with 4 channels (corresponds to 512×512 image).
    latents = torch.rand(1, 4, 64, 64, dtype=torch.float16)
    print(f"Input latents shape: {latents.shape}  dtype: {latents.dtype}")

    # ---- compile ----
    print("\nCompiling VAE Decoder to linalg …")
    module = compile_to_linalg(model, latents)

    # ---- Hexagon ----
    options = default_options(enablelwp)
    print("Running VAE Decoder on Hexagon NPU …")
    hex_out = hex_execution(module, "VAEDecodeWrapper", [latents], options)

    # ---- x86 reference ----
    print("Running reference on x86 …")
    x86_out = x86_execution(model, latents)

    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()

    print("\nVAE Decoder test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD VAE Decoder Hexagon benchmark")
    parser.add_argument("--lwp", action="store_true", help="Enable lightweight profiling")
    args = parser.parse_args()
    test_vae_decoder(enablelwp=args.lwp)
