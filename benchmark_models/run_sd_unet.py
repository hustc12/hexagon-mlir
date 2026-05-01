# ===- test_sd_unet.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Stable Diffusion — UNet benchmark for Hexagon NPU.
#
# Model: CompVis/stable-diffusion-v1-4  →  unet subfolder
# Architecture: UNet2DConditionModel
# Inputs:
#   latent_model_input   (1, 4, 64, 64)   float32
#   timestep             (1,)              float32
#   encoder_hidden_states (1, 77, 768)    float32
# Output: sample  (1, 4, 64, 64)  float32
#
# Usage:
#   python benchmark_models/test_sd_unet.py [--lwp]
#
# ===------------------------------------------------------------------------===

import argparse
import torch
from diffusers import UNet2DConditionModel

from sd_utils import (
    SD_MODEL_ID,
    compile_to_linalg,
    hex_execution,
    x86_execution,
    compare,
    default_options,
    process_lwp,
)


def test_unet(enablelwp: bool = False):
    print("\n=== Stable Diffusion — UNet ===")

    # Use from_config (random weights) for fast compilation testing.
    # Swap to UNet2DConditionModel.from_pretrained(...) for real-weight tests.
    config = UNet2DConditionModel.load_config(SD_MODEL_ID, subfolder="unet")
    model = UNet2DConditionModel.from_config(config)
    model.eval()

    # Standard SD 512×512 inputs: latent space is 64×64 with 4 channels.
    latent_model_input = torch.rand(1, 4, 64, 64, dtype=torch.float32)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    # Text encoder output: (batch=1, seq_len=77, hidden=768)
    encoder_hidden_states = torch.rand(1, 77, 768, dtype=torch.float32)

    inputs = (latent_model_input, timestep, encoder_hidden_states)
    print(
        f"Inputs — latent: {latent_model_input.shape}, "
        f"timestep: {timestep.shape}, "
        f"encoder_hidden_states: {encoder_hidden_states.shape}"
    )

    # ---- compile ----
    print("\nCompiling UNet to linalg …")
    module = compile_to_linalg(model, *inputs)

    # ---- Hexagon ----
    options = default_options(enablelwp)
    print("Running UNet on Hexagon NPU …")
    hex_out = hex_execution(module, "UNet2DConditionModel", list(inputs), options)

    # ---- x86 reference ----
    print("Running reference on x86 …")
    x86_out = x86_execution(model, *inputs)

    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()

    print("\nUNet test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD UNet Hexagon benchmark")
    parser.add_argument("--lwp", action="store_true", help="Enable lightweight profiling")
    args = parser.parse_args()
    test_unet(enablelwp=args.lwp)
