# ===- test_sd_text_encoder.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Stable Diffusion — Text Encoder (CLIP) benchmark for Hexagon NPU.
#
# Model: CompVis/stable-diffusion-v1-4  →  text_encoder subfolder
# Architecture: CLIPTextModel
# Input:  token ids  (1, 77)  int64
# Output: last_hidden_state  (1, 77, 768)  float32
#
# Usage:
#   python benchmark_models/test_sd_text_encoder.py [--lwp]
#
# ===------------------------------------------------------------------------===

import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from sd_utils import (
    SD_MODEL_ID,
    compile_to_linalg,
    hex_execution,
    x86_execution,
    compare,
    default_options,
    process_lwp,
)


def get_text_inputs(tokenizer, prompt: str):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_input.input_ids


def test_text_encoder(enablelwp: bool = False):
    print("\n=== Stable Diffusion — Text Encoder ===")

    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")

    # Use from_config (random weights) for fast compilation testing.
    # Swap to CLIPTextModel.from_pretrained(...) for real-weight accuracy tests.
    config = CLIPTextModel.config_class.from_pretrained(
        SD_MODEL_ID, subfolder="text_encoder"
    )
    model = CLIPTextModel(config)
    model.eval()

    input_ids = get_text_inputs(tokenizer, "A beautiful picture of a Hexagon NPU")
    print(f"Input shape: {input_ids.shape}  dtype: {input_ids.dtype}")

    # ---- compile ----
    print("\nCompiling Text Encoder to linalg …")
    module = compile_to_linalg(model, input_ids)

    # ---- Hexagon ----
    options = default_options(enablelwp)
    print("Running Text Encoder on Hexagon NPU …")
    hex_out = hex_execution(module, "CLIPTextModel", [input_ids], options)

    # ---- x86 reference ----
    print("Running reference on x86 …")
    x86_out = x86_execution(model, input_ids)

    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()

    print("\nText Encoder test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Text Encoder Hexagon benchmark")
    parser.add_argument("--lwp", action="store_true", help="Enable lightweight profiling")
    args = parser.parse_args()
    test_text_encoder(enablelwp=args.lwp)
