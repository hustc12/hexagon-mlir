#!/usr/bin/env python3
# ===- test_matmul_benchmark.py ---------------------------------------------===
#
# Matrix Multiplication Benchmark on Hexagon NPU
# Comparing performance across Scalar, HVX, and HMX execution modes
#
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import torch_mlir.fx as fx_mlir
from pathlib import Path
import time
import json
import io
import re
from contextlib import redirect_stdout, redirect_stderr
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher


class MatMul(nn.Module):
    """Simple matrix multiplication module"""
    def __init__(self, transpose_b=False):
        super(MatMul, self).__init__()
        self.transpose_b = transpose_b

    def forward(self, a, b):
        if self.transpose_b:
            b = b.transpose(-2, -1)
        return torch.matmul(a, b)


def create_linalg_module(model, inputs, func_name):
    """Convert PyTorch model to MLIR linalg representation"""
    return fx_mlir.export_and_import(
        model,
        *inputs,
        output_type="linalg-on-tensors",
        func_name=func_name,
    )


def write_bytecode_to_file(mlir_module, filename):
    """Write MLIR bytecode to file"""
    bytecode = mlir_module.operation.get_asm(binary=True)
    with open(filename, "wb") as f:
        f.write(bytecode)


def benchmark_hexagon(model, inputs, filename, func_name, options, iterations=10):
    """
    Run benchmark on Hexagon with specified options
    
    Args:
        model: PyTorch model
        inputs: Input tensors
        filename: MLIR bytecode filename
        func_name: Function name
        options: HexagonOptions dictionary
        iterations: Number of iterations for benchmarking
    
    Returns:
        Dictionary with benchmark results
    """
    # End-to-end wall clock includes: compile + deploy + run + dump/pull tensors.
    start_time = time.time()
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
        output = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(filename),
            inputs,
            func_name,
            base_dir_for_artifacts=None,
            iterations=iterations,
            options=options
        )
    end_time = time.time()
    launcher_logs = log_buffer.getvalue()
    # Keep original launcher output visible in terminal for debugging.
    if launcher_logs:
        print(launcher_logs, end="")

    # Parse on-device kernel perf from Test_Info report.
    perf_match = re.search(r"Perf:\s*([0-9]+(?:\.[0-9]+)?)", launcher_logs)
    units_match = re.search(r"Units:\s*([a-zA-Z]+)", launcher_logs)
    kernel_time_us = float(perf_match.group(1)) if perf_match else None
    perf_units = units_match.group(1) if units_match else None
    
    # Verify correctness
    reference = model(*inputs)
    is_correct = torch.allclose(output[0], reference, rtol=1e-03, atol=1e-03)
    
    return {
        'output': output[0],
        'time': end_time - start_time,
        'e2e_time_s': end_time - start_time,
        'kernel_time_us': kernel_time_us,
        'kernel_time_s': (kernel_time_us / 1e6) if kernel_time_us is not None else None,
        'perf_units': perf_units,
        'is_correct': is_correct,
        'iterations': iterations
    }


def convert_inputs_dtype(inputs, dtype):
    """Create dtype-converted tensor copies for mode-specific compilation."""
    converted = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            converted.append(inp.to(dtype=dtype))
        else:
            converted.append(inp)
    return converted


def run_matmul_benchmark(m, n, k, dtype=torch.float32):
    """
    Benchmark matrix multiplication A(m,k) x B(k,n) = C(m,n)
    
    Args:
        m, n, k: Matrix dimensions
        dtype: Data type for tensors
    
    Returns:
        Dictionary with results for all execution modes
    """
    print(f"\n{'='*80}")
    print(f"Matrix Multiplication Benchmark: ({m}x{k}) @ ({k}x{n})")
    print(f"Data Type: {dtype}")
    print(f"{'='*80}\n")
    
    # Create model and inputs
    model = MatMul()
    a = torch.randn(m, k, dtype=dtype)
    b = torch.randn(k, n, dtype=dtype)
    inputs = [a, b]
    
    func_name = model.__class__.__name__
    linalg_filename = Path(__file__).parent / f"{func_name}_{m}x{k}x{n}.mlirbc"
    
    # Create MLIR module (only once)
    print("Converting to MLIR...")
    mlir_module = create_linalg_module(model, inputs, func_name)
    write_bytecode_to_file(mlir_module, linalg_filename)
    print(f"MLIR bytecode saved to: {linalg_filename}\n")
    
    results = {}
    
    # Configuration 1: Scalar execution (no vectorization, no HexKL)
    print("Running Scalar execution...")
    scalar_options = HexagonOptions(
        enableVectorization=False,
        enableHexKL=False,
        enableVTCMTiling=False,
        enableMultiThreading=False
    ).__dict__
    results['scalar'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, scalar_options)
    print(f"  End-to-End Time: {results['scalar']['e2e_time_s']:.4f}s")
    if results['scalar']['kernel_time_us'] is not None:
        print(f"  Kernel Time (Test_Info): {results['scalar']['kernel_time_us']:.2f} us")
    print(f"  Correct: {results['scalar']['is_correct']}")
    
    # Configuration 2: HVX execution (vectorization enabled, no HexKL)
    print("\nRunning HVX execution...")
    hvx_options = HexagonOptions(
        enableVectorization=True,
        enableHexKL=False,
        enableVTCMTiling=True,
        enableMultiThreading=False
    ).__dict__
    results['hvx'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, hvx_options)
    print(f"  End-to-End Time: {results['hvx']['e2e_time_s']:.4f}s")
    if results['hvx']['kernel_time_us'] is not None:
        print(f"  Kernel Time (Test_Info): {results['hvx']['kernel_time_us']:.2f} us")
    print(f"  Correct: {results['hvx']['is_correct']}")
    
    # Configuration 3: HMX execution (HexKL enabled)
    print("\nRunning HMX execution (HexKL)...")
    try:
        # HexKL matmul lowering currently requires fp16 tensors.
        hmx_dtype = torch.float16
        hmx_inputs = convert_inputs_dtype(inputs, hmx_dtype)
        hmx_linalg_filename = Path(__file__).parent / f"{func_name}_{m}x{k}x{n}_hmx_fp16.mlirbc"

        print(f"  HMX compile dtype: {hmx_dtype}")
        hmx_mlir_module = create_linalg_module(model, hmx_inputs, func_name)
        write_bytecode_to_file(hmx_mlir_module, hmx_linalg_filename)

        hmx_options = HexagonOptions(
            enableVectorization=True,
            enableHexKL=True,
            enableVTCMTiling=True,
            enableMultiThreading=False
        ).__dict__
        results['hmx'] = benchmark_hexagon(model, hmx_inputs, hmx_linalg_filename, func_name, hmx_options)
        print(f"  End-to-End Time: {results['hmx']['e2e_time_s']:.4f}s")
        if results['hmx']['kernel_time_us'] is not None:
            print(f"  Kernel Time (Test_Info): {results['hmx']['kernel_time_us']:.2f} us")
        print(f"  Correct: {results['hmx']['is_correct']}")
    except Exception as e:
        print(f"  HMX execution failed: {e}")
        print("  Note: HexKL may not be available on your system")
        results['hmx'] = None
    
    # Print summary
    print(f"\n{'-'*80}")
    print("Performance Summary:")
    print(f"{'-'*80}")
    print(f"{'Mode':<15} {'Kernel(us)':<15} {'E2E(s)':<15} {'Speedup':<15} {'Correct':<10}")
    print(f"{'-'*80}")
    
    scalar_kernel_us = results['scalar']['kernel_time_us']
    scalar_e2e_s = results['scalar']['e2e_time_s']
    for mode in ['scalar', 'hvx', 'hmx']:
        if results[mode] is not None:
            kernel_us = results[mode]['kernel_time_us']
            e2e_s = results[mode]['e2e_time_s']
            if kernel_us is not None and scalar_kernel_us is not None and kernel_us > 0:
                speedup = scalar_kernel_us / kernel_us
            else:
                speedup = scalar_e2e_s / e2e_s if e2e_s > 0 else 1.0
            kernel_str = f"{kernel_us:.2f}" if kernel_us is not None else "N/A"
            print(f"{mode.upper():<15} {kernel_str:<15} {e2e_s:<15.4f} {speedup:<15.2f}x {str(results[mode]['is_correct']):<10}")
    print(f"{'-'*80}\n")
    
    return results


def main():
    """Run benchmarks with different matrix sizes"""
    
    # Test configurations
    test_configs = [
        (64, 64, 64),      # Small matrices
        (128, 128, 128),   # Medium matrices
        (256, 256, 256),   # Larger matrices
        (512, 512, 512),   # Large matrices
        (1024, 1024, 1024), # Larger matrices
        # (2048, 2048, 2048), # Larger matrices
        # (4096, 4096, 4096), # Larger matrices
    ]
    
    all_results = {}
    
    print("\n" + "="*80)
    print("HEXAGON NPU - MATRIX MULTIPLICATION BENCHMARK")
    print("Testing Scalar vs HVX vs HMX performance")
    print("="*80)
    
    for m, n, k in test_configs:
        try:
            results = run_matmul_benchmark(m, n, k)
            all_results[f"{m}x{k}x{n}"] = results
        except Exception as e:
            print(f"\nError running benchmark for ({m}x{k}x{n}): {e}")
            continue
    
    # Save results to JSON
    results_file = Path(__file__).parent / "matmul_benchmark_results.json"
    print(f"\nSaving results to: {results_file}")
    
    # Convert to JSON-serializable format
    json_results = {}
    for config, results in all_results.items():
        json_results[config] = {}
        for mode, data in results.items():
            if data is not None:
                json_results[config][mode] = {
                    'time': data['time'],
                    'e2e_time_s': data.get('e2e_time_s'),
                    'kernel_time_us': data.get('kernel_time_us'),
                    'kernel_time_s': data.get('kernel_time_s'),
                    'perf_units': data.get('perf_units'),
                    'is_correct': data['is_correct'],
                    'iterations': data['iterations']
                }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
