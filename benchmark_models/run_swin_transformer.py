# ===- test_swin_transformer.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Swin Transformer (Shifted Window Transformer) benchmark for Hexagon NPU.
#
# Model source:
#   keras-io/swin-transformers (https://huggingface.co/keras-io/swin-transformers)
#   is a TF-Keras model incompatible with torch-mlir's FX export pipeline.
#   The functionally equivalent PyTorch implementation from Microsoft is used:
#   microsoft/swin-tiny-patch4-window7-224 (same paper, same architecture).
#
# Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted
#             Windows", Liu et al., ICCV 2021 (https://arxiv.org/abs/2103.14030)
#
# ===------------------------------------------------------------------------===

from typing import Optional
import sys, os
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoConfig, AutoImageProcessor, SwinConfig
from transformers.models.swin.modeling_swin import SwinForImageClassification
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher


# ==========================================
# Execution and Comparison Functions
# ==========================================

def x86_execution(model, pixel_values):
    with torch.no_grad():
        return model(pixel_values)


def hex_execution(module, func_name, inputs, options: dict = None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False
    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(
        str(linalg_filename), inputs, func_name, options=options
    )
    return hex_outputs


def compare(hex_outputs, x86_outputs, atol=0.05, fail_on_mismatch: bool = False):
    hexagon_output = hex_outputs[0]

    # x86_execution calls wrapped_model which returns logits tensor directly.
    # If somehow an ImageClassifierOutput is returned, unwrap it.
    if hasattr(x86_outputs, "logits"):
        x86_tensor = x86_outputs.logits
    elif isinstance(x86_outputs, torch.Tensor):
        x86_tensor = x86_outputs
    else:
        x86_tensor = x86_outputs[0]

    # Print top-5 predicted classes for both runs
    def top5(logits, tag):
        probs = torch.softmax(logits[0].float(), dim=-1)
        k = min(5, probs.shape[-1])
        vals, idxs = torch.topk(probs, k)
        print(f"\n------- Top-5 class predictions ({tag}) -------")
        for v, i in zip(vals.tolist(), idxs.tolist()):
            print(f"  class {i:4d}: {v:.4f}")
        print("-----------------------------------------------")
        return idxs.tolist(), vals.tolist()

    idxs_hex, vals_hex = top5(hexagon_output, "Hexagon")
    idxs_x86, vals_x86 = top5(x86_tensor.to(hexagon_output.dtype), "x86")

    max_diff = torch.max(torch.abs(hexagon_output.float() - x86_tensor.float()))
    print(f"\nMax logit difference between Hexagon and x86: {max_diff.item():.4f}")

    match = torch.allclose(hexagon_output.float(), x86_tensor.float(), atol=atol)
    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, (
            "Correctness issue: the results obtained on Hexagon "
            "(with code produced by the hexagon-mlir compiler) and on x86 "
            "(executed from PyTorch) do not match"
        )


def compile_to_linalg(model, pixel_values, dump_to_file=None, debug=False):
    linalg = fx.export_and_import(
        model,
        pixel_values,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
    )
    if dump_to_file:
        with open(dump_to_file, "w") as f:
            f.write(str(linalg))
    return linalg


def process_lwp():
    HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")
    if not HEXAGON_MLIR_ROOT:
        print("Cannot process lwp data as path to process_lwp.py is unknown")
        return
    try:
        subprocess.run(
            [
                "python3",
                f"{HEXAGON_MLIR_ROOT}/test/python/process_lwp.py",
                "/tmp/lwp.json",
                "/tmp/lwp_infodump.txt",
                "/tmp/initial-linalg.mlir",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("LWP processing completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error processing LWP data: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Error output: {e.stderr}")


# ==========================================
# Model Wrapper
# ==========================================

class SwinWrapper(torch.nn.Module):
    """Wrap SwinForImageClassification to return the logits tensor directly.

    torch_mlir's fx.export_and_import cannot handle dataclass outputs
    (ImageClassifierOutput). This wrapper returns a plain tensor so the
    C++ MemRefDescriptor interface gets a single concrete output.
    """

    def __init__(self, swin_model):
        super().__init__()
        self.swin = swin_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.swin(pixel_values=pixel_values).logits


# ==========================================
# Main Test Entry Point
# ==========================================

def swin_transformer(enablelwp=False):
    # ------------------------------------------------------------------ #
    # Model: microsoft/swin-tiny-patch4-window7-224                       #
    #   - Architecture: Swin-Tiny (4-stage, depths=[2,2,6,2])            #
    #   - Input:  (B, 3, 224, 224)                                        #
    #   - Patch:  4×4, Window: 7×7, Channels: 96                         #
    #   - Params: ~28 M (float32) / ~14 M params effective (float16)     #
    # ------------------------------------------------------------------ #
    model_name = "microsoft/swin-tiny-patch4-window7-224"

    print(f"Loading Swin Transformer config from '{model_name}' …")
    config = SwinConfig.from_pretrained(model_name)

    # Reduce depth per stage to keep the compiled .so within the DSP 32-bit
    # VA space while still exercising all 4 hierarchical stages.
    # Default swin-tiny: depths=[2, 2, 6, 2] → reduced: [1, 1, 1, 1]
    config.depths = [1, 1, 1, 1]

    # FIX: default hidden_act="gelu" lowers to `math.erf` in MLIR, for which
    # the Hexagon backend has no LLVMTranslationDialectInterface registration
    # ("missing `LLVMTranslationDialectInterface` registration for dialect for
    # op: math.erf"). Switch to "gelu_new" (tanh approximation) to avoid it.
    # Same fix applied in test_vit.py.
    config.hidden_act = "gelu_new"

    # Reduce base channel width: default embed_dim=96 → 48.
    # Weights scale as embed_dim², so halving channels cuts weight memory by 4×.
    # With depths=[1,1,1,1] fp16 this yields ~6MB total .so (vs ~24MB at 96).
    # The DSP User PD VA space cannot reliably map a 23MB consts .so alongside
    # the already-mapped libc++, libc++abi, and runtime SOs.
    config.embed_dim = 48
    config.num_heads = [3, 6, 12, 24]  # Must divide embed_dim at each stage

    # Use float16 to halve weight memory (critical for DSP VA budget).
    # Instantiated from config only — hexagon-mlir benchmarks with random weights.
    model = SwinForImageClassification(config).half()
    model.eval()

    wrapped_model = SwinWrapper(model)
    wrapped_model.eval()
    func_name = wrapped_model.__class__.__name__   # "SwinWrapper"

    # Standard 224×224 pixel_values in fp16
    pixel_values = torch.rand(1, 3, 224, 224, dtype=torch.float16)

    # ------------------------------------------------------------------ #
    # Collect relative_position_index buffers (one per stage).            #
    # torch.export captures these as BUFFER inputs and places them        #
    # *before* the user input (pixel_values) in the MLIR function         #
    # signature.  We must pass them explicitly so the wrapper generates   #
    # the correct number of input tensors and the DSP call doesn't        #
    # receive garbage in r4/r5 → Bad VA crash (exit code 13).            #
    # ------------------------------------------------------------------ #
    rel_pos_indices = []
    for layer in wrapped_model.swin.swin.encoder.layers:
        for block in layer.blocks:
            idx = block.attention.self.relative_position_index  # int64, (49,49)
            rel_pos_indices.append(idx.detach())

    print("Compiling to linalg …")
    module = compile_to_linalg(
        wrapped_model,
        pixel_values,
        dump_to_file="swin_transformer.mlir",
    )

    # ------------------------------------------------------------------ #
    # Compiler options                                                     #
    # ------------------------------------------------------------------ #
    options = HexagonOptions().__dict__

    if enablelwp:
        options["enableLWP"] = True

    # With embed_dim=48, depths=[1,1,1,1], fp16 the .so is ~6MB — no splitting needed.
    # (Default embed_dim=96 produced a 23MB consts .so that the DSP VA space cannot
    # map alongside libc++/runtime SOs, leaving GOT entries unresolved → Bad VA crash.)
    options["lowerConstantsInSeparateSharedObjects"] = False

    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False

    # ROOT CAUSE FIX: enableVectorization=True causes HexagonTilingPass to emit
    # scf.forall ops. FormAsyncThreadsPass unconditionally lowers these to
    # async.execute, which triggers MLIR AsyncRuntime `new AsyncToken()` heap
    # allocations on the DSP. The DSP User PD heap cannot satisfy these,
    # yielding NULL/garbage pointers → AsyncToken::~AsyncToken() crashes at
    # Bad VA: 0x18 (exit code 13 / TLB MISS).
    options["enableVectorization"] = False

    # MLIR function signature: (rel_pos_idx_0, ..., rel_pos_idx_N, pixel_values)
    # The buffer inputs must come first, matching the export order.
    inputs = rel_pos_indices + [pixel_values]

    # ------------------------------------------------------------------ #
    # Hexagon execution                                                    #
    # ------------------------------------------------------------------ #
    print("Running on Hexagon NPU …")
    hex_outputs = hex_execution(module, func_name, inputs, options)

    # ------------------------------------------------------------------ #
    # x86 reference                                                        #
    # ------------------------------------------------------------------ #
    print("Running reference on x86 …")
    x86_outputs = x86_execution(wrapped_model, pixel_values)

    compare(hex_outputs, x86_outputs, atol=0.1, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin Transformer Hexagon benchmark")
    parser.add_argument("--lwp", action="store_true", help="Enable lightweight profiling")
    args = parser.parse_args()
    swin_transformer(enablelwp=args.lwp)
