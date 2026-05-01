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

def hex_execution(module, func_name, inputs, options: dict = None):
    linalg_filename = Path(__file__).parent / (func_name + ".mlirbc")
    bytecode = module.operation.get_asm(binary=True)
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options["enableVTCMTiling"] = False

    # FIX: Reduce _QURT_MAX_HEAP_SIZE from 1 GB to 256 MB.
    #
    # With 1 GB the DSP heap manager allocates memory at addresses like
    # 0xE682BC (~14 MB) which are not mapped in the DSP TLB, causing a
    # TLBMISS_RW crash when the generated code tries to read the
    # MemRefDescriptor stored there.
    # 256 MB keeps all heap allocations within the DSP's mapped region.
    #
    # NOTE: Do NOT reduce below ~64 MB.  CLIPWrapper's bufferized embedding
    # gather can allocate intermediate buffers up to ~12 MB (vocab_size×hidden
    # float32).  With 8 MB the DSP malloc returns NULL, and the subsequent
    # store to NULL+24 produces another TLBMISS_RW crash.
    #
    # The code_string template lives on WrapperGeneratorStrings (base class).
    # TorchMLIRWrapperGeneratorStrings inherits it via super().__init__(),
    # so we patch the base class __init__ to modify the template before the
    # subclass instance is created.
    from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb

    _ORIG_SIZE = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
    _NEW_SIZE  = "unsigned int _QURT_MAX_HEAP_SIZE = 8388608;    // 8 MB Max Heap Size"

    _orig_base_init = _hlb.WrapperGeneratorStrings.__init__

    def _patched_base_init(self):
        _orig_base_init(self)
        self.code_string = self.code_string.replace(_ORIG_SIZE, _NEW_SIZE)

    _hlb.WrapperGeneratorStrings.__init__ = _patched_base_init
    try:
        hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(linalg_filename), inputs, func_name, options=options
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

def default_options(enablelwp: bool = False) -> dict:
    opts = HexagonOptions().__dict__
    opts["lowerConstantsInSeparateSharedObjects"] = True
    opts["enableVTCMTiling"] = False
    opts["enableConvertToHexagonmem"] = False
    if enablelwp:
        opts["enableLWP"] = True
    return opts


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
