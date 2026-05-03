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

    # FIX: Run the entire model in float16 to avoid f64 GroupNorm reductions.
    # torch-mlir's GroupNorm lowering promotes the variance reduction to f64
    # for numerical stability.  On Hexagon DSP there is no f64 hardware — every
    # f64 op is software-emulated and ~100× slower than f32/f16.  At 512×512
    # resolution each GroupNorm reduction operates on a 1×32×1048576 f64 tensor
    # (256 MB), and there are ~12 such reductions, making execution infeasibly
    # slow.
    # Converting the whole model + inputs to f16 gives torch-mlir a pure f16
    # graph so it never promotes to f64.  The Hexagon HVX unit handles f16
    # natively and processes 2× as many elements per cycle vs f32.
    # NOTE: f16 may lose numerical precision; widen atol in compare() if needed.
    vae = vae.half()

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
