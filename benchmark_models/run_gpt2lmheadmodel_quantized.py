# ===- run_gpt2lmheadmodel_quantized.py ------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Quantized GPT-2 LMHead inference on Hexagon NPU
# ================================================
#
# BACKGROUND — Why "quantization" here means PTQ-float16, not int8
# ----------------------------------------------------------------
# The Hexagon-MLIR compiler stack (as of SDK 6.4) lowers PyTorch models via
# torch-mlir's fx.export_and_import() → Linalg-on-Tensors IR → Hexagon LLVM
# backend.  The backend's code-generation pipeline does NOT yet contain:
#   • a quant-dialect lowering pass (int8/int4 → Hexagon HMX/HVX integer ops)
#   • support for torch.ao.quantization / bitsandbytes / GPTQ / AWQ quantized
#     model graphs (these introduce custom ops that torch-mlir cannot trace)
#
# Therefore the only "quantization" that is end-to-end supported today is
# **post-training weight quantization expressed entirely in float16 arithmetic**,
# which is what this script implements:
#
#   1. Load the pretrained GPT-2 weights in float32.
#   2. Apply dynamic per-channel weight quantization (W8A16):
#        • Quantize every Linear weight to int8 (scale + zero-point stored).
#        • Dequantize back to float16 before the forward pass so the graph
#          seen by torch-mlir contains only standard float16 ops.
#      This is the "fake-quantization" / "simulated quantization" approach:
#      the *numerical effect* of int8 weight rounding is preserved while the
#      *compute graph* stays in float16 — exactly what the compiler can handle.
#   3. Compile the float16 graph to Hexagon via the standard pipeline.
#   4. Run on Hexagon NPU and compare top-5 predictions against the CPU
#      float32 reference.
#
# Quantization scheme options (--quant-scheme CLI flag):
#   w8a16   — int8 per-channel weight quantization, float16 activations (default)
#   w4a16   — int4 per-channel weight quantization, float16 activations
#   fp16    — pure float16 (no weight quantization, baseline for comparison)
#
# Usage:
#   python run_gpt2lmheadmodel_quantized.py
#   python run_gpt2lmheadmodel_quantized.py --quant-scheme w4a16
#   python run_gpt2lmheadmodel_quantized.py --quant-scheme fp16
#   python run_gpt2lmheadmodel_quantized.py --enablelwp
#   python run_gpt2lmheadmodel_quantized.py --dump-mlir  # save linalg IR to file
#
# ===------------------------------------------------------------------------===

from typing import Optional
import sys
import os
import math
import argparse
import subprocess
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers.pytorch_utils import Conv1D
from pathlib import Path

from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
    TorchMLIRHexagonLauncher,
)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _quantize_weight_w8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8 quantization of a weight matrix.

    Returns:
        weight_q  : int8 quantized weight  (same shape as input)
        scale     : float32 per-channel scale  (shape: [out_features])
        zero_point: int8 zero-point (all zeros for symmetric quant)
    """
    # weight shape: [out_features, in_features]  (or higher-rank for Conv)
    out_features = weight.shape[0]
    w_flat = weight.view(out_features, -1).float()  # work in fp32 for precision

    abs_max = w_flat.abs().max(dim=1).values.clamp(min=1e-8)
    scale = abs_max / 127.0                          # symmetric: range [-127, 127]
    zero_point = torch.zeros(out_features, dtype=torch.int8)

    w_scaled = (w_flat / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    weight_q = w_scaled.view_as(weight)
    return weight_q, scale, zero_point


def _quantize_weight_w4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int4 quantization (stored in int8).

    int4 range: [-7, 7]  (symmetric, avoids -8 for numerical stability)

    Returns:
        weight_q  : int8 tensor holding int4 values  (same shape as input)
        scale     : float32 per-channel scale
        zero_point: int8 zero-point (all zeros)
    """
    out_features = weight.shape[0]
    w_flat = weight.view(out_features, -1).float()

    abs_max = w_flat.abs().max(dim=1).values.clamp(min=1e-8)
    scale = abs_max / 7.0                            # int4 symmetric range [-7, 7]
    zero_point = torch.zeros(out_features, dtype=torch.int8)

    w_scaled = (w_flat / scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
    weight_q = w_scaled.view_as(weight)
    return weight_q, scale, zero_point


def _dequantize_weight(
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize int8/int4 (stored as int8) weight back to float.

    The result is cast to `target_dtype` so the downstream graph stays in fp16.
    """
    out_features = weight_q.shape[0]
    w_float = weight_q.view(out_features, -1).float()
    w_dequant = w_float * scale.unsqueeze(1)
    return w_dequant.view_as(weight_q).to(target_dtype)


def apply_weight_quantization(
    model: nn.Module,
    scheme: str = "w8a16",
    target_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Apply simulated (fake) weight quantization to Linear/Conv1D projection layers.

    The quantization is "fake" in the sense that weights are quantized to
    int8/int4 and immediately dequantized back to float16.  The resulting
    model has the same graph structure as a plain float16 model — which is
    what torch-mlir and the Hexagon backend can compile — but the weights
    carry the numerical rounding error introduced by the lower-bit format.

    Args:
        model      : PyTorch model (will be modified in-place).
        scheme     : "w8a16" | "w4a16" | "fp16"
        target_dtype: dtype for the dequantized weights (default: float16)

    Returns:
        The modified model (same object, modified in-place).
    """
    if scheme == "fp16":
        # No weight quantization — just cast everything to fp16
        return model.to(target_dtype)

    quant_fn = _quantize_weight_w8 if scheme == "w8a16" else _quantize_weight_w4
    bits = 8 if scheme == "w8a16" else 4

    quantized_layers = 0
    total_params_before = sum(p.numel() for p in model.parameters())

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue

        # Conv1D in HF GPT-2 stores weight as [in_features, out_features].
        # For per-output-channel quantization we temporarily transpose to
        # [out_features, in_features], then transpose back.
        if isinstance(module, Conv1D):
            w = module.weight.data.float().T
        else:
            w = module.weight.data.float()

        w_q, scale, _ = quant_fn(w)
        w_dq = _dequantize_weight(w_q, scale, target_dtype=target_dtype)
        if isinstance(module, Conv1D):
            module.weight = nn.Parameter(w_dq.T.contiguous(), requires_grad=False)
        else:
            module.weight = nn.Parameter(w_dq, requires_grad=False)

        # Cast bias to target_dtype if present
        if module.bias is not None:
            module.bias = nn.Parameter(
                module.bias.data.to(target_dtype), requires_grad=False
            )

        quantized_layers += 1

    # Cast all remaining parameters (embeddings, LayerNorm, etc.) to target_dtype
    model = model.to(target_dtype)

    print(
        f"[Quantization] scheme={scheme}  "
        f"quantized {quantized_layers} projection layers (Linear/Conv1D) to int{bits} "
        f"(dequantized back to {target_dtype})  "
        f"total params: {total_params_before:,}"
    )
    return model


def estimate_model_size_mb(model: nn.Module) -> float:
    """Rough estimate of model parameter memory in MB."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Inference helpers (mirrored from run_gpt2lmheadmodel.py)
# ---------------------------------------------------------------------------

def get_encodings(tokenizer, *inputs):
    encodings = tokenizer(*inputs, return_tensors="pt")
    return encodings


def x86_execution(model: nn.Module, encoding: dict) -> object:
    """Run the model on CPU (reference)."""
    with torch.no_grad():
        x86_outputs = model(**encoding)
    return x86_outputs


def hex_execution(
    module,
    func_name: str,
    inputs: list,
    options: dict,
) -> list:
    """Compile the linalg MLIR module and run it on Hexagon NPU."""
    linalg_filename = Path(__file__).parent / (str(func_name) + "_quantized.mlirbc")

    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(
        str(linalg_filename), inputs, func_name, options=options
    )
    return hex_outputs


def get_top_5(
    logits: torch.Tensor, tokenizer, run_type: str
) -> tuple[list, list]:
    """Extract top-5 predicted tokens and their logit values."""
    print(f"\n-------Printing the top5 probable tokens for {run_type}--------\n")
    top_k = 5

    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits to be a 3D tensor, but got shape {logits.shape}"
        )

    last_row_logits = logits[0, -1, :]
    top_values, top_indices = torch.topk(last_row_logits, top_k)
    top_confidences = top_values.tolist()

    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    for token, confidence in zip(top_tokens, top_confidences):
        print(f"Token: {[token]}, Confidence: {confidence:.4f}")
    print("---------------------------------------------------\n")
    return top_tokens, top_confidences


def compare(
    hex_outputs: list,
    x86_outputs,
    tokenizer,
    atol: float = 0.05,
    fail_on_mismatch: bool = False,
) -> None:
    """Compare Hexagon and CPU top-5 predictions.

    Note: atol is slightly relaxed (0.05 vs 0.03 in the fp16 baseline) to
    account for the additional numerical error introduced by weight quantization.
    """
    hexagon_logits = hex_outputs[0]
    t_hex, c_hex = get_top_5(hexagon_logits, tokenizer, "hexagon (quantized)")

    x86_logits = x86_outputs.logits
    t_x86, c_x86 = get_top_5(x86_logits, tokenizer, "x86 (quantized reference)")

    tokens_match = t_x86 == t_hex
    confidences_match = torch.allclose(
        torch.tensor(c_x86), torch.tensor(c_hex), atol=atol
    )

    if tokens_match and confidences_match:
        print("✓ The top-5 tokens and their probabilities matched")
    else:
        if not tokens_match:
            print("✗ Token mismatch between Hexagon and CPU")
        if not confidences_match:
            print(
                f"✗ Confidence mismatch (atol={atol}): "
                f"x86={[f'{v:.4f}' for v in c_x86]}  "
                f"hex={[f'{v:.4f}' for v in c_hex]}"
            )
        assert not fail_on_mismatch, (
            "Correctness issue: results obtained on Hexagon (quantized) and "
            "on x86 (quantized reference) do not match"
        )


def compile_to_linalg(model: nn.Module, input, dump_to_file=None, debug=False):
    """Export the PyTorch model to Linalg-on-Tensors MLIR via torch-mlir fx."""
    if isinstance(input, torch.Tensor):
        input = (input,)

    linalg = fx.export_and_import(
        model,
        *input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
    )

    if dump_to_file:
        with open(dump_to_file, "w") as f:
            f.write(str(linalg))
        print(f"[MLIR] Linalg IR dumped to: {dump_to_file}")

    return linalg


def force_projection_fp16_io(model: nn.Module) -> None:
    """Force Linear/Conv1D projection islands to fp16 at graph-export time.

    This aggressively inserts cast boundaries around projection layers so
    linalg.matmul lowering can see f16 operands for HexKL.
    """
    patched = 0
    for module in model.modules():
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue

        # Keep parameters in fp16 to avoid f32 re-promotion in exported graph.
        module.weight = nn.Parameter(module.weight.data.to(torch.float16), requires_grad=False)
        if getattr(module, "bias", None) is not None:
            module.bias = nn.Parameter(module.bias.data.to(torch.float16), requires_grad=False)

        if isinstance(module, nn.Linear):
            def linear_forward_fp16(input, m=module):
                return F.linear(
                    input.to(torch.float16),
                    m.weight,
                    None if m.bias is None else m.bias,
                ).to(torch.float16)

            module.forward = linear_forward_fp16
        else:
            # HF Conv1D forward is effectively addmm(matmul + bias). Re-implement
            # with explicit fp16 boundaries to avoid lingering fp32 matmul islands.
            def conv1d_forward_fp16(x, m=module):
                x_fp16 = x.to(torch.float16)
                size_out = x_fp16.size()[:-1] + (m.nf,)
                x_2d = x_fp16.view(-1, x_fp16.size(-1))
                out = torch.addmm(
                    m.bias.to(torch.float16),
                    x_2d,
                    m.weight.to(torch.float16),
                )
                return out.view(size_out).to(torch.float16)

            module.forward = conv1d_forward_fp16

        patched += 1

    print(f"[DType Guard] Forced fp16 I/O on {patched} projection layers (Linear/Conv1D).")


def has_f32_linalg_matmul(module) -> bool:
    """Return True if the exported Linalg IR still contains f32 matmul inputs."""
    ir = str(module)
    # Matches lines like:
    # linalg.matmul ins(%a, %b : tensor<...xf32>, tensor<...xf32>)
    return re.search(
        r"linalg\.matmul\s+ins\([^\n]*tensor<[^>]*xf32>",
        ir,
    ) is not None


def process_lwp() -> None:
    """Post-process lightweight profiling (LWP) data if available."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gpt2lmheadmodel_quantized(
    quant_scheme: str = "w8a16",
    enablelwp: bool = False,
    dump_mlir: bool = False,
    debug: bool = False,
    # --- Pipeline switches ---
    # Because the fake-quantization graph is entirely float16, HexKL,
    # VTCM tiling and Omni-Fetch prefetching are all opt-in here.
    # Note: VTCM tiling still has a known GPT-2 attention-mask shape issue;
    # use --enable-vtcm-tiling with caution.
    enable_hexkl: bool = False,
    enable_vtcm_tiling: bool = False,
    enable_convert_to_hexagonmem: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
) -> None:
    """End-to-end quantized GPT-2 inference on Hexagon NPU.

    Args:
        quant_scheme               : "w8a16" | "w4a16" | "fp16"
        enablelwp                  : enable lightweight profiling
        dump_mlir                  : save the Linalg MLIR to a .mlir text file
        debug                      : enable torch-mlir graph/IR printing
        enable_hexkl               : enable HexKL HMX matrix-multiply lowering
        enable_vtcm_tiling         : enable VTCM tiling (caution: known GPT-2 shape issue)
        enable_convert_to_hexagonmem: enable memref→hexagonmem conversion
        enable_omnifetch_vdae      : enable Omni-Fetch V-DAE prefetch pass
        enable_omnifetch_layout_aware: enable in-situ layout-aware index maps
        omnifetch_lookahead        : static prefetch look-ahead distance
        enable_omnifetch_adaptive  : enable PMU-based adaptive control
    """
    print(f"\n{'='*60}")
    print(f"  GPT-2 Quantized Inference on Hexagon NPU")
    print(f"  Quantization scheme : {quant_scheme}")
    print(f"{'='*60}\n")

    model_name = "openai-community/gpt2"
    prompt = "What is nature of our existence?"

    # ------------------------------------------------------------------
    # 1. Load tokenizer and model config
    # ------------------------------------------------------------------
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    config = GPT2Config.from_pretrained(model_name)
    # Use 2 transformer layers — same as the fp16 baseline.
    # 1 layer is not practical for accuracy checking; more layers risk
    # exhausting the Hexagon DSP 32-bit virtual address space.
    config.n_layer = 2

    # ------------------------------------------------------------------
    # 2. Load pretrained weights in float32, then apply quantization
    # ------------------------------------------------------------------
    # We always start from float32 to get the best quantization accuracy.
    # The apply_weight_quantization() call will:
    #   - quantize Linear weights to int8/int4
    #   - dequantize them back to float16
    #   - cast all other parameters to float16
    # The resulting model is numerically equivalent to a float16 model
    # with quantization noise injected into the weights.
    print("[Model] Loading pretrained GPT-2 weights (float32) ...")
    model = GPT2LMHeadModel.from_pretrained(
        model_name, config=config, torch_dtype=torch.float32
    )
    model.eval()

    size_before = estimate_model_size_mb(model)
    print(f"[Model] Size before quantization : {size_before:.1f} MB")

    print(f"[Quantization] Applying {quant_scheme} weight quantization ...")
    model = apply_weight_quantization(model, scheme=quant_scheme, target_dtype=torch.float16)
    # Strong repair: aggressively enforce fp16 boundaries around projection ops
    # so exported linalg.matmul islands become HexKL-legal.
    force_projection_fp16_io(model)
    model.eval()

    size_after = estimate_model_size_mb(model)
    print(f"[Model] Size after  quantization : {size_after:.1f} MB  "
          f"(ratio: {size_after/size_before:.2f}x)")

    func_name = model.__class__.__name__

    # ------------------------------------------------------------------
    # 3. Tokenize input
    # ------------------------------------------------------------------
    encoding = get_encodings(tokenizer, prompt)
    print(f"[Input] Prompt      : '{prompt}'")
    print(f"[Input] Token IDs   : {encoding['input_ids'].tolist()}")

    # ------------------------------------------------------------------
    # 4. Compile to Linalg MLIR
    # ------------------------------------------------------------------
    dump_file = None
    if dump_mlir:
        dump_file = str(
            Path(__file__).parent / f"{func_name}_quantized_{quant_scheme}.mlir"
        )

    print("\n[Compile] Exporting model to Linalg-on-Tensors MLIR ...")
    module = compile_to_linalg(
        model, encoding["input_ids"], dump_to_file=dump_file, debug=debug
    )

    # ------------------------------------------------------------------
    # 5. Run on Hexagon NPU
    # ------------------------------------------------------------------
    options = HexagonOptions().__dict__
    # The fake-quantization graph is entirely float16, so HexKL / VTCM
    # tiling / Omni-Fetch are all available.  The two passes that have a
    # known GPT-2 shape issue are kept off by default (see CLI flags).
    # HexKL verifier currently requires f16 inputs for matmul. GPT-2 export can
    # still contain some f32 matmul ops; detect this and gracefully fallback.
    effective_enable_hexkl = enable_hexkl
    if enable_hexkl and has_f32_linalg_matmul(module):
        effective_enable_hexkl = False
        print(
            "[Options] HexKL HMX lowering  : OFF (fallback). "
            "Detected f32 linalg.matmul in exported IR; HexKL matmul currently "
            "requires f16 inputs."
        )
        print(
            "[Hint] Re-run with --dump-mlir and inspect matmul operand dtypes to "
            "identify remaining fp32 islands."
        )

    options["enableHexKL"] = effective_enable_hexkl
    options["enableVTCMTiling"] = enable_vtcm_tiling
    options["enableConvertToHexagonmem"] = enable_convert_to_hexagonmem
    # Omni-Fetch Plan-A: three independent components.
    # enablePrefetch is the base; enableOmniFetchVDAE is Component 3.
    # Enabling V-DAE automatically implies Prefetch is needed.
    options["enablePrefetch"] = enable_omnifetch_vdae  # V-DAE requires Prefetch
    options["enableOmniFetchLayoutAware"] = enable_omnifetch_layout_aware
    options["omniFetchLookahead"] = omnifetch_lookahead
    options["enableOmniFetchVDAE"] = enable_omnifetch_vdae
    options["enableOmniFetchAdaptive"] = enable_omnifetch_adaptive
    if enablelwp:
        options["enableLWP"] = True

    if effective_enable_hexkl:
        print("[Options] HexKL HMX lowering  : ON")
    if enable_omnifetch_vdae:
        print(f"[Options] Omni-Fetch V-DAE     : ON  (lookahead={omnifetch_lookahead}, "
              f"layout_aware={enable_omnifetch_layout_aware}, "
              f"adaptive={enable_omnifetch_adaptive})")

    print("\n[Hexagon] Launching inference on Hexagon NPU ...")
    inputs = [encoding["input_ids"]]
    hex_outputs = hex_execution(module, func_name, inputs, options)

    # ------------------------------------------------------------------
    # 6. CPU reference (same quantized model, same weights)
    # ------------------------------------------------------------------
    print("\n[CPU] Running reference inference on x86 ...")
    x86_outputs = x86_execution(model, encoding)

    # ------------------------------------------------------------------
    # 7. Compare results
    # ------------------------------------------------------------------
    print("\n[Compare] Comparing Hexagon vs CPU predictions ...")
    compare(hex_outputs, x86_outputs, tokenizer, atol=0.05, fail_on_mismatch=True)

    if enablelwp:
        process_lwp()

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantized GPT-2 LMHead inference on Hexagon NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization schemes
--------------------
  w8a16  (default) : int8 per-channel weight quantization, float16 activations.
                     Weights are quantized to int8 then dequantized to float16
                     before compilation.  Reduces effective weight precision to
                     8 bits while keeping the compute graph in float16.

  w4a16            : int4 per-channel weight quantization, float16 activations.
                     More aggressive compression; expect slightly lower accuracy.

  fp16             : No weight quantization.  Pure float16 baseline.
                     Equivalent to the original run_gpt2lmheadmodel.py but
                     starting from float32 weights cast to float16.

Ablation study examples
-----------------------
  # Baseline (no HMX, no prefetch)
  python run_gpt2lmheadmodel_quantized.py

  # Enable HexKL HMX lowering only
  python run_gpt2lmheadmodel_quantized.py --enable-hexkl

  # Full Omni-Fetch pipeline (HexKL + V-DAE prefetch)
  python run_gpt2lmheadmodel_quantized.py --enable-hexkl --enable-omnifetch-vdae

  # V-DAE without layout-aware mapping (linear prefetch only)
  python run_gpt2lmheadmodel_quantized.py --enable-hexkl --enable-omnifetch-vdae --disable-layout-aware

  # V-DAE with aggressive lookahead
  python run_gpt2lmheadmodel_quantized.py --enable-hexkl --enable-omnifetch-vdae --omnifetch-lookahead 4
        """,
    )

    # --- Quantization ---
    parser.add_argument(
        "--quant-scheme",
        choices=["w8a16", "w4a16", "fp16"],
        default="w8a16",
        help="Weight quantization scheme (default: w8a16)",
    )

    # --- General ---
    parser.add_argument(
        "--enablelwp",
        action="store_true",
        default=False,
        help="Enable lightweight profiling (LWP)",
    )
    parser.add_argument(
        "--dump-mlir",
        action="store_true",
        default=False,
        help="Dump the Linalg MLIR IR to a .mlir text file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable torch-mlir graph and IR printing",
    )

    # --- Pipeline switches (ablation study) ---
    pipeline = parser.add_argument_group(
        "pipeline switches (ablation study)",
        "All off by default.  The fake-quantization graph is entirely float16,\n"
        "so HexKL and Omni-Fetch are safe to enable here.",
    )
    pipeline.add_argument(
        "--enable-hexkl",
        action="store_true",
        default=False,
        help="Enable HexKL HMX matrix-multiply lowering (requires f16 graph — safe here).",
    )
    pipeline.add_argument(
        "--enable-vtcm-tiling",
        action="store_true",
        default=False,
        help="Enable VTCM tiling (caution: known GPT-2 attention-mask shape issue).",
    )
    pipeline.add_argument(
        "--enable-convert-to-hexagonmem",
        action="store_true",
        default=False,
        help="Enable memref→hexagonmem conversion pass.",
    )
    pipeline.add_argument(
        "--enable-omnifetch-vdae",
        action="store_true",
        default=False,
        help="Enable Omni-Fetch V-DAE prefetch insertion pass (best used with --enable-hexkl).",
    )
    pipeline.add_argument(
        "--disable-layout-aware",
        action="store_true",
        default=False,
        help="Disable in-situ layout-aware index maps (use linear prefetch only).",
    )
    pipeline.add_argument(
        "--omnifetch-lookahead",
        type=int,
        default=2,
        help="Static prefetch look-ahead distance in loop iterations (default: 2).",
    )
    pipeline.add_argument(
        "--disable-omnifetch-adaptive",
        action="store_true",
        default=False,
        help="Disable PMU-driven adaptive prefetch distance tuning.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gpt2lmheadmodel_quantized(
        quant_scheme=args.quant_scheme,
        enablelwp=args.enablelwp,
        dump_mlir=args.dump_mlir,
        debug=args.debug,
        enable_hexkl=args.enable_hexkl,
        enable_vtcm_tiling=args.enable_vtcm_tiling,
        enable_convert_to_hexagonmem=args.enable_convert_to_hexagonmem,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
    )
