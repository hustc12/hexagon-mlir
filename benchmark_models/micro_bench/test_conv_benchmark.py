#!/usr/bin/env python3
# ===- test_conv_benchmark.py -----------------------------------------------===
#
# Convolution Benchmark on Hexagon NPU
# Comparing performance across Scalar, HVX, and HMX execution modes
#
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import torch_mlir.fx as fx_mlir
from pathlib import Path
import time
import json
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher


class Conv2D(nn.Module):
    """2D Convolution module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)


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
    start_time = time.time()
    output = TorchMLIRHexagonLauncher().run_torch_mlir(
        str(filename), 
        inputs, 
        func_name, 
        base_dir_for_artifacts=None, 
        iterations=iterations,
        options=options
    )
    end_time = time.time()
    
    # Verify correctness
    with torch.no_grad():
        reference = model(*inputs)
    is_correct = torch.allclose(output[0], reference, rtol=1e-03, atol=1e-03)
    
    return {
        'output': output[0],
        'time': end_time - start_time,
        'is_correct': is_correct,
        'iterations': iterations
    }


def run_conv_benchmark(batch, in_channels, out_channels, height, width, kernel_size, dtype=torch.float32):
    """
    Benchmark 2D convolution
    
    Args:
        batch: Batch size
        in_channels: Number of input channels
        out_channels: Number of output channels
        height, width: Input spatial dimensions
        kernel_size: Convolution kernel size
        dtype: Data type for tensors
    
    Returns:
        Dictionary with results for all execution modes
    """
    print(f"\n{'='*80}")
    print(f"Conv2D Benchmark: B={batch}, IC={in_channels}, OC={out_channels}, H={height}, W={width}, K={kernel_size}")
    print(f"Data Type: {dtype}")
    print(f"{'='*80}\n")
    
    # Create model and inputs
    model = Conv2D(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    model.eval()
    
    x = torch.randn(batch, in_channels, height, width, dtype=dtype)
    inputs = [x]
    
    func_name = model.__class__.__name__
    linalg_filename = Path(__file__).parent / f"{func_name}_b{batch}_ic{in_channels}_oc{out_channels}_h{height}_w{width}_k{kernel_size}.mlirbc"
    
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
        enableConvTiling=False,
        enableMultiThreading=False
    ).__dict__
    results['scalar'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, scalar_options)
    print(f"  Time: {results['scalar']['time']:.4f}s")
    print(f"  Correct: {results['scalar']['is_correct']}")
    
    # Configuration 2: HVX execution (vectorization enabled, no HexKL)
    print("\nRunning HVX execution...")
    hvx_options = HexagonOptions(
        enableVectorization=True,
        enableHexKL=False,
        enableVTCMTiling=True,
        enableConvTiling=True,
        enableMultiThreading=False
    ).__dict__
    results['hvx'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, hvx_options)
    print(f"  Time: {results['hvx']['time']:.4f}s")
    print(f"  Correct: {results['hvx']['is_correct']}")
    
    # Configuration 3: HMX execution (HexKL enabled, with layout conversions for conv2d)
    print("\nRunning HMX execution (HexKL)...")
    try:
        hmx_options = HexagonOptions(
            enableVectorization=True,
            enableHexKL=True,
            enableVTCMTiling=True,
            enableConvTiling=True,
            enableSeedLayoutConversions=True,  # Important for HMX conv2d
            enableMultiThreading=False
        ).__dict__
        results['hmx'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, hmx_options)
        print(f"  Time: {results['hmx']['time']:.4f}s")
        print(f"  Correct: {results['hmx']['is_correct']}")
    except Exception as e:
        print(f"  HMX execution failed: {e}")
        print("  Note: HexKL may not be available on your system")
        results['hmx'] = None
    
    # Print summary
    print(f"\n{'-'*80}")
    print("Performance Summary:")
    print(f"{'-'*80}")
    print(f"{'Mode':<15} {'Time (s)':<15} {'Speedup':<15} {'Correct':<10}")
    print(f"{'-'*80}")
    
    scalar_time = results['scalar']['time']
    for mode in ['scalar', 'hvx', 'hmx']:
        if results[mode] is not None:
            speedup = scalar_time / results[mode]['time']
            print(f"{mode.upper():<15} {results[mode]['time']:<15.4f} {speedup:<15.2f}x {str(results[mode]['is_correct']):<10}")
    print(f"{'-'*80}\n")
    
    return results


def main():
    """Run benchmarks with different convolution configurations"""
    
    # Test configurations: (batch, in_channels, out_channels, height, width, kernel_size)
    test_configs = [
        (1, 3, 16, 32, 32, 3),      # Small input, typical first conv layer
        (1, 16, 32, 32, 32, 3),     # Medium channels
        (1, 32, 64, 16, 16, 3),     # Larger channels, smaller spatial
        (1, 64, 64, 16, 16, 3),     # Square channels
    ]
    
    all_results = {}
    
    print("\n" + "="*80)
    print("HEXAGON NPU - CONVOLUTION BENCHMARK")
    print("Testing Scalar vs HVX vs HMX performance")
    print("="*80)
    
    for batch, ic, oc, h, w, k in test_configs:
        try:
            results = run_conv_benchmark(batch, ic, oc, h, w, k)
            config_name = f"b{batch}_ic{ic}_oc{oc}_h{h}_w{w}_k{k}"
            all_results[config_name] = results
        except Exception as e:
            print(f"\nError running benchmark for config: {e}")
            continue
    
    # Save results to JSON
    results_file = Path(__file__).parent / "conv_benchmark_results.json"
    print(f"\nSaving results to: {results_file}")
    
    # Convert to JSON-serializable format
    json_results = {}
    for config, results in all_results.items():
        json_results[config] = {}
        for mode, data in results.items():
            if data is not None:
                json_results[config][mode] = {
                    'time': data['time'],
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
