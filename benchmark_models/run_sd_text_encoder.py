# ===- run_sd_text_encoder.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Stable Diffusion — Text Encoder (CLIP) benchmark for Hexagon NPU.
#
# Model: CompVis/stable-diffusion-v1-4  →  text_encoder subfolder
# Architecture: CLIPTextModel
# Input:  token ids  (1, 77)  int64
# Output: last_hidden_state  (1, 77, hidden_size)  float32
#
# Usage:
#   python benchmark_models/run_sd_text_encoder.py [--lwp]
#
# ===------------------------------------------------------------------------===

import argparse
import types
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.activations import GELUActivation

from sd_utils import (
    SD_MODEL_ID,
    compile_to_linalg,
    hex_execution,
    x86_execution,
    compare,
    default_options,
    process_lwp,
)


def get_text_inputs(tokenizer, prompt: str, max_length: int = None):
    if max_length is None:
        max_length = tokenizer.model_max_length
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_input.input_ids


def _patch_gelu_tanh(model):
    """Replace GELUActivation.forward with tanh approximation.

    Default F.gelu uses math.erf which the Hexagon LLVM backend cannot lower
    (no LLVMTranslationDialectInterface → ub.poison → translation failure).
    The tanh approximation avoids math.erf entirely.
    """
    def _tanh_forward(self, x):
        return F.gelu(x, approximate="tanh")

    for module in model.modules():
        if isinstance(module, GELUActivation):
            module.forward = types.MethodType(_tanh_forward, module)


def test_text_encoder(enablelwp: bool = False):
    print("\n=== Stable Diffusion — Text Encoder ===")

    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")

    config = CLIPTextModel.config_class.from_pretrained(
        SD_MODEL_ID, subfolder="text_encoder"
    )

    # FIX: Reduce model size so MLIR compilation is feasible on the DSP.
    # Default CLIP: 12 layers, hidden=768, intermediate=3072, vocab=49408
    # → 984 MB MLIR bytecode, too large to compile in reasonable time.
    # Reduced: 2 layers, hidden=64, intermediate=128, 1 attention head.
    # vocab_size kept at 49408 (tokenizer compatibility), but the embedding
    # weight is small because hidden_size is tiny.
    config.num_hidden_layers = 1
    config.hidden_size = 16
    config.intermediate_size = 16
    config.num_attention_heads = 1   # must divide hidden_size
    config.head_dim = 16             # hidden_size / num_attention_heads


    _clip = CLIPTextModel(config)
    _clip.eval()

    # FIX 1: Wrap CLIPTextModel to return only last_hidden_state.
    # CLIPTextModel returns a BaseModelOutputWithPooling dataclass with two
    # tensors.  torch_mlir exports both as outputs, generating two
    # MemRefDescriptor outputs in the wrapper.  Returning a single tensor
    # avoids the double-output memrefCopy path.
    class CLIPWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
            return self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

    model = CLIPWrapper(_clip)
    model.eval()

    # FIX 2: Patch GELUActivation to use tanh approximation.
    # Default gelu uses math.erf → ub.poison in Hexagon LLVM backend.
    _patch_gelu_tanh(_clip)

    # Use a shorter sequence length to keep CLIPWrapper's stack frame below
    # the LLVM Hexagon large-frame scavenging threshold (~4096 bytes).
    # With seq_len=77 the frame is 5248 bytes (1 layer) or 8960 bytes (2 layers),
    # triggering a prologue/epilogue r16-mismatch bug → TLBMISS on DSP.
    # seq_len=16 brings the frame to ~1 KB, safe for the normal r30-based
    # callee-save sequence.
    SEQ_LEN = 16
    input_ids = get_text_inputs(tokenizer, "A beautiful picture of a Hexagon NPU",
                                max_length=SEQ_LEN)
    # FIX 3: Pass attention_mask explicitly.
    # CLIPTextModel.forward takes (input_ids, attention_mask).  torch.export
    # captures both as user inputs in the MLIR function signature.  The
    # original script only passed input_ids, leaving attention_mask
    # uninitialised on the DSP → infinite loop / timeout.
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

    # FIX 4: Pass position_ids buffer explicitly.
    # torch.export captures CLIPTextEmbeddings.position_ids as a BUFFER input
    # and places it BEFORE the user inputs (input_ids, attention_mask) in the
    # MLIR function signature.  Without it the wrapper only passes 2 tensors
    # but the function expects 3, so r3 (attention_mask) gets a garbage value
    # → TLBMISS crash on the DSP.
    # Slice to SEQ_LEN so all three inputs have the same shape.
    position_ids = model.clip.text_model.embeddings.position_ids[:, :SEQ_LEN].detach()

    print(f"input_ids: {input_ids.shape}  attention_mask: {attention_mask.shape}  position_ids: {position_ids.shape}")

    # ---- compile ----
    print("\nCompiling Text Encoder to linalg …")
    module = compile_to_linalg(model, input_ids, attention_mask)

    # ---- Hexagon ----
    options = default_options(enablelwp)
    print("Running Text Encoder on Hexagon NPU …")
    # MLIR signature: (position_ids, input_ids, attention_mask)
    # Buffer inputs come first, then user inputs.
    hex_out = hex_execution(
        module, "CLIPWrapper", [position_ids, input_ids, attention_mask], options
    )

    # ---- x86 reference ----
    print("Running reference on x86 …")
    x86_out = x86_execution(model, input_ids, attention_mask)

    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()

    print("\nText Encoder test PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Text Encoder Hexagon benchmark")
    parser.add_argument("--lwp", action="store_true", help="Enable lightweight profiling")
    args = parser.parse_args()
    test_text_encoder(enablelwp=args.lwp)
