import torch
import torch.nn as nn
import torch_mlir.fx as fx_mlir
from pathlib import Path
import subprocess
import os
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(x)

def create_linalg_module(model, inputs, func_name):
    return fx_mlir.export_and_import(
        model,
        *inputs,
        output_type="linalg-on-tensors",
        func_name=func_name,
    )

def write_bytecode_to_file(mlir_module, filename):
    bytecode = mlir_module.operation.get_asm(binary=True)
    with open(filename, "wb") as f:
        f.write(bytecode)

def execute_and_compare(model, inputs, filename, func_name, rtol=1e-05, atol=1e-08, iterations=1, options=None):
    reference = model(*inputs)
    output = TorchMLIRHexagonLauncher().run_torch_mlir(
        str(filename), inputs, func_name, base_dir_for_artifacts=None, iterations=iterations, options=options
    )

    print("\nReference output:\n", reference)
    print("\nHexagon output:\n", output[0])
    assert torch.allclose(output[0], reference, rtol, atol)

def process_lwp():
    HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")
    if not HEXAGON_MLIR_ROOT:
        print("Cannot process lwp data as path to process_lwp.py is unknown")
        return

    subprocess.run(
        [
            "python3",
            f"{HEXAGON_MLIR_ROOT}/test/python/process_lwp.py",
            "/tmp/lwp.json",
            "/tmp/lwp_infodump.txt",
            "/tmp/initial-linalg.mlir"
        ],
        check=True
    )

def test_softmax_torch(enablelwp=False): # Set to True to profile with Light Weight Profiling.
    model = Softmax()
    inp = torch.rand(128, 128)
    func_name = model.__class__.__name__
    linalg_filename = Path(__file__).parent / f"{func_name}.mlirbc"

    options = HexagonOptions().__dict__ if enablelwp else None
    if enablelwp:
        options['enableLWP'] = True

    mlir_module = create_linalg_module(model, [inp], func_name)
    write_bytecode_to_file(mlir_module, linalg_filename)
    execute_and_compare(model, [inp], linalg_filename, func_name, options=options)

    if enablelwp:
        process_lwp()

if __name__ == "__main__":
    test_softmax_torch()
