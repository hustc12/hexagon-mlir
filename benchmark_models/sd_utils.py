# ===- sd_utils.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Shared utilities for Stable Diffusion sub-module benchmarks.
# Used by:
#   test_sd_text_encoder.py
#   test_sd_unet.py
#   test_sd_vae_decoder.py
#
# ===------------------------------------------------------------------------===

import os
import subprocess
import torch
from pathlib import Path
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher


# ---------------------------------------------------------------------------
# Default model id
# ---------------------------------------------------------------------------
SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def compile_to_linalg(model, *inputs, dump_to_file=None, debug=False):
    linalg = fx.export_and_import(
        model,
        *inputs,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug,
    )
    if dump_to_file:
        with open(dump_to_file, "w") as f:
            f.write(str(linalg))
    return linalg


# ---------------------------------------------------------------------------
# Hexagon execution
# ---------------------------------------------------------------------------

def hex_execution(module, func_name, inputs, options: dict = None,
                  heap_size_mb: int = 8, iterations: int = 1):
    linalg_filename = Path(__file__).parent / (func_name + ".mlirbc")
    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options["enableVTCMTiling"] = False

    from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb

    _ORIG_SIZE = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
    _NEW_SIZE  = f"unsigned int _QURT_MAX_HEAP_SIZE = {heap_size_mb * 1024 * 1024};    // {heap_size_mb} MB Max Heap Size"

    _orig_base_init = _hlb.WrapperGeneratorStrings.__init__

    def _patched_base_init(self):
        _orig_base_init(self)
        self.code_string = self.code_string.replace(_ORIG_SIZE, _NEW_SIZE)

    _hlb.WrapperGeneratorStrings.__init__ = _patched_base_init
    try:
        hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(linalg_filename),
            inputs,
            func_name,
            iterations=iterations,
            options=options,
        )
    finally:
        _hlb.WrapperGeneratorStrings.__init__ = _orig_base_init

    return hex_outputs


# ---------------------------------------------------------------------------
# x86 reference
# ---------------------------------------------------------------------------

def x86_execution(model, *inputs):
    with torch.no_grad():
        return model(*inputs)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(hex_outputs, x86_outputs, atol=0.05, fail_on_mismatch: bool = False):
    hexagon_tensor = hex_outputs[0]

    if isinstance(x86_outputs, torch.Tensor):
        x86_tensor = x86_outputs
    elif hasattr(x86_outputs, "sample"):
        x86_tensor = x86_outputs.sample
    elif hasattr(x86_outputs, "last_hidden_state"):
        x86_tensor = x86_outputs.last_hidden_state
    else:
        x86_tensor = x86_outputs[0]

    max_diff = torch.max(torch.abs(hexagon_tensor.float() - x86_tensor.float()))
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff.item():.4f}")

    match = torch.allclose(hexagon_tensor.float(), x86_tensor.float(), atol=atol)
    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: results do not match"


# ---------------------------------------------------------------------------
# Default HexagonOptions
# ---------------------------------------------------------------------------

def default_options(
    enablelwp: bool = False,
    enable_hvx_vector: bool = False,
    enable_hexkl: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    enable_omnifetch_items_1_7: bool = False,
) -> dict:
    """Phase-4 options. HexKL off by default for fair HVX baseline."""
    opts = HexagonOptions().__dict__
    opts["lowerConstantsInSeparateSharedObjects"] = True
    opts["enableVTCMTiling"] = False
    opts["enableVectorization"] = bool(enable_hvx_vector)
    opts["enableHexKL"] = bool(enable_hexkl)
    opts["enableConvertToHexagonmem"] = bool(enable_hexkl)
    cumulative = bool(enable_omnifetch_items_1_7)
    opts["enablePrefetch"] = bool(enable_omnifetch_vdae or cumulative)
    opts["enableOmniFetchLayoutAware"] = bool(enable_omnifetch_layout_aware)
    opts["omniFetchLookahead"] = int(omnifetch_lookahead)
    opts["enableOmniFetchVDAE"] = bool(enable_omnifetch_vdae or cumulative)
    opts["enableOmniFetchAdaptive"] = bool(enable_omnifetch_adaptive)
    opts["enableOmniFetchPersistentWhCache"] = cumulative
    opts["enableOmniFetchTwoDimPipeline"] = cumulative
    opts["enableOmniFetchVtcmColoring"] = cumulative
    opts["enableOmniFetchKvCachePrefetch"] = cumulative
    if cumulative:
        print(
            "[OmniFetchItems1To7] enabled: layout/cost/fusion + "
            "persistent-WH + two-dimensional-pipeline + "
            "VTCM-coloring + KV-aware-prefetch"
        )
    if enablelwp:
        opts["enableLWP"] = True
    return opts


def add_phase4_cli(parser):
    parser.add_argument("--lwp", action="store_true")
    parser.add_argument("--enable-hvx-vector", action="store_true")
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch-vdae", action="store_true")
    parser.add_argument("--disable-layout-aware", action="store_true")
    parser.add_argument("--omnifetch-lookahead", type=int, default=2)
    parser.add_argument("--disable-omnifetch-adaptive", action="store_true")
    parser.add_argument(
        "--enable-omnifetch-items-1-7",
        action="store_true",
        help="Enable the cumulative innovation items 1 through 7.",
    )
    parser.add_argument("--device-iterations", type=int, default=1)
    return parser


def options_from_args(args, enablelwp: bool = None):
    return default_options(
        enablelwp=args.lwp if enablelwp is None else enablelwp,
        enable_hvx_vector=getattr(args, "enable_hvx_vector", False),
        enable_hexkl=getattr(args, "enable_hexkl", False),
        enable_omnifetch_vdae=getattr(args, "enable_omnifetch_vdae", False),
        enable_omnifetch_layout_aware=not getattr(args, "disable_layout_aware", False),
        omnifetch_lookahead=getattr(args, "omnifetch_lookahead", 2),
        enable_omnifetch_adaptive=not getattr(args, "disable_omnifetch_adaptive", False),
        enable_omnifetch_items_1_7=getattr(
            args, "enable_omnifetch_items_1_7", False
        ),
    )


# ---------------------------------------------------------------------------
# LWP post-processing
# ---------------------------------------------------------------------------

def process_lwp():
    HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")
    if not HEXAGON_MLIR_ROOT:
        print("Cannot process lwp data: HEXAGON_MLIR_ROOT not set")
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
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
