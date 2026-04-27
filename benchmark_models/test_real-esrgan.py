from typing import Optional
import sys, os
import torch
import argparse
import subprocess
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from pathlib import Path
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

# Patch huggingface_hub for RealESRGAN compatibility
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from RealESRGAN import RealESRGAN

# ==========================================
# Execution and Comparison Functions
# ==========================================
def x86_execution(model, inputs):
    x86_outputs = model(inputs)
    return x86_outputs

def hex_execution(module, func_name, inputs, options: dict=None):
    linalg_filename = Path(__file__).parent / (str(func_name) + ".mlirbc")

    bytecode = module.operation.get_asm(binary=True)
    # Save the bytecode to a file
    with open(linalg_filename, "wb") as f:
        f.write(bytecode)

    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False 
    hex_outputs = TorchMLIRHexagonLauncher().run_torch_mlir(str(linalg_filename), inputs, func_name, options=options)
    return hex_outputs

def compare(hex_outputs, x86_outputs, atol=0.03, fail_on_mismatch: bool=False):
    hexagon_output = hex_outputs[0]

    max_diff = torch.max(torch.abs(hexagon_output - x86_outputs))
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff.item():.4f}")

    match = torch.allclose(hexagon_output, x86_outputs, atol=atol)

    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: the results obtained on Hexagon (with code produced by the hexagon-mlir compiler) and on x86 (executed from PyTorch) do not match"

def compile_to_linalg(model, input_tensor, dump_to_file=None, debug=False) -> str:
    # Generate linalg-IR using torch-mlir's fx
    linalg = fx.export_and_import(
        model,
        input_tensor,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name=model.__class__.__name__,
        enable_graph_printing=debug,
        enable_ir_printing=debug
    )

    if dump_to_file:
        with open(dump_to_file, "w") as file:
            file.write(str(linalg))

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
                "/tmp/initial-linalg.mlir"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("LWP processing completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error processing LWP data: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Error output: {e.stderr}")

def real_esrgan(enablelwp=False): 

    # Use the official ai-forever/Real-ESRGAN from Hugging Face
    device = torch.device('cpu')
    model_wrapper = RealESRGAN(device, scale=4)
    # Download weights directly via HuggingFace Hub API to avoid legacy cached_download issues in the package
    weights_path = huggingface_hub.hf_hub_download('ai-forever/Real-ESRGAN', 'RealESRGAN_x4.pth')
    model_wrapper.load_weights(weights_path, download=False)
    
    # Extract the underlying PyTorch nn.Module for compilation
    model = model_wrapper.model
    model.eval() # Ensure we are in evaluation mode

    func_name = model.__class__.__name__

    # Real-ESRGAN takes an image tensor of shape (B, C, H, W).
    # Use 16x16 input to reduce intermediate activation memory pressure on the DSP.
    # The full 32x32 input produces a ~65MB monolithic .so that exhausts the DSP 32-bit VA space.
    input_tensor = torch.rand(1, 3, 16, 16)

    module = compile_to_linalg(model, input_tensor)

    options = HexagonOptions().__dict__
    # Split constant weights into a separate shared object to avoid exhausting the DSP 32-bit VA space.
    # Without this, the single lib_mlir_ciface_RRDBNet.so is ~65MB and triggers
    # "qurt_vtlb_mmap: unable to get ANON mapping" followed by TLBMISS crash (exit code 13).
    options['lowerConstantsInSeparateSharedObjects'] = True
    # Disable VTCM tiling/conversion to reduce internal DSP allocation pressure
    options['enableVTCMTiling'] = False
    options['enableConvertToHexagonmem'] = False
    if enablelwp:
        options['enableLWP'] = True
    inputs = [input_tensor]
    
    # Run Hexagon
    hex_outputs = hex_execution(module, func_name, inputs, options)
    
    # Run x86
    with torch.no_grad():
        x86_outputs = x86_execution(model, input_tensor)

    compare(hex_outputs, x86_outputs, atol=0.5, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    real_esrgan()
