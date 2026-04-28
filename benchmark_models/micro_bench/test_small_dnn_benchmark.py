#!/usr/bin/env python3
# ===- test_small_dnn_benchmark.py ------------------------------------------===
#
# Small DNN Model Benchmark on Hexagon NPU
# A mini CNN model with Conv2D + MatMul operations
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


class SmallCNN(nn.Module):
    """
    A small CNN model for benchmarking
    Architecture:
        Conv2D(3, 16, 3x3) -> ReLU -> 
        Conv2D(16, 32, 3x3) -> ReLU -> 
        Flatten -> 
        Linear(32*8*8, 128) -> ReLU ->
        Linear(128, 10)
    """
    def __init__(self, input_size=32, num_classes=10):
        super(SmallCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        
        # Max pooling to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        self.flat_size = 32 * (input_size // 4) * (input_size // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128, bias=False)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, num_classes, bias=False)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC block 1
        x = self.fc1(x)
        x = self.relu3(x)
        
        # FC block 2
        x = self.fc2(x)
        
        return x


class SimpleMLP(nn.Module):
    """
    A simple MLP with multiple matrix multiplications
    Architecture:
        Linear(784, 256) -> ReLU ->
        Linear(256, 128) -> ReLU ->
        Linear(128, 64) -> ReLU ->
        Linear(64, 10)
    """
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], num_classes=10):
        super(SimpleMLP, self).__init__()
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size, bias=False))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, num_classes, bias=False))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


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
    # End-to-end wall clock includes compile, deploy, I/O and kernel execution.
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
    if launcher_logs:
        print(launcher_logs, end="")

    perf_match = re.search(r"Perf:\s*([0-9]+(?:\.[0-9]+)?)", launcher_logs)
    units_match = re.search(r"Units:\s*([a-zA-Z]+)", launcher_logs)
    kernel_time_us = float(perf_match.group(1)) if perf_match else None
    perf_units = units_match.group(1) if units_match else None
    
    # Verify correctness
    with torch.no_grad():
        reference = model(*inputs)
    is_correct = torch.allclose(output[0], reference, rtol=1e-02, atol=1e-02)

    e2e_time_s = end_time - start_time
    kernel_time_s = (kernel_time_us / 1e6) if kernel_time_us is not None else None
    
    return {
        'output': output[0],
        'time': e2e_time_s,
        'e2e_time_s': e2e_time_s,
        'kernel_time_us': kernel_time_us,
        'kernel_time_s': kernel_time_s,
        'perf_units': perf_units,
        'is_correct': is_correct,
        'iterations': iterations,
        'avg_time_per_iter': (e2e_time_s / iterations),
        'avg_kernel_time_per_iter_s': (
            (kernel_time_s / iterations) if kernel_time_s is not None else None
        )
    }


def count_ops(model):
    """Count FLOPs in the model (rough estimate)"""
    total_ops = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Conv2D FLOPs = 2 * kernel_size^2 * in_channels * out_channels * output_h * output_w
            pass
        elif isinstance(module, nn.Linear):
            # Linear FLOPs = 2 * in_features * out_features
            total_ops += 2 * module.in_features * module.out_features
    return total_ops


def run_model_benchmark(model, inputs, model_name, dtype=torch.float32):
    """
    Benchmark a DNN model
    
    Args:
        model: PyTorch model
        inputs: Input tensors
        model_name: Name for the model
        dtype: Data type for tensors
    
    Returns:
        Dictionary with results for all execution modes
    """
    print(f"\n{'='*80}")
    print(f"DNN Model Benchmark: {model_name}")
    print(f"Data Type: {dtype}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*80}\n")
    
    model.eval()
    
    func_name = model.__class__.__name__
    linalg_filename = Path(__file__).parent / f"{func_name}_{model_name}.mlirbc"
    
    # Create MLIR module (only once)
    print("Converting to MLIR...")
    mlir_module = create_linalg_module(model, inputs, func_name)
    write_bytecode_to_file(mlir_module, linalg_filename)
    print(f"MLIR bytecode saved to: {linalg_filename}\n")
    
    results = {}
    
    # Configuration 1: Scalar execution
    print("Running Scalar execution...")
    scalar_options = HexagonOptions(
        enableVectorization=False,
        enableHexKL=False,
        enableVTCMTiling=False,
        enableConvTiling=False,
        enableMultiThreading=False
    ).__dict__
    try:
        results['scalar'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, scalar_options)
        print(f"  End-to-End Time: {results['scalar']['e2e_time_s']:.4f}s ({results['scalar']['avg_time_per_iter']*1000:.2f}ms/iter)")
        if results['scalar']['kernel_time_us'] is not None:
            print(f"  Kernel Time (Test_Info): {results['scalar']['kernel_time_us']:.2f} us")
        print(f"  Correct: {results['scalar']['is_correct']}")
    except Exception as e:
        print(f"  Scalar execution failed: {e}")
        results['scalar'] = None
    
    # Configuration 2: HVX execution
    print("\nRunning HVX execution...")
    hvx_options = HexagonOptions(
        enableVectorization=True,
        enableHexKL=False,
        enableVTCMTiling=True,
        enableConvTiling=True,
        enableMultiThreading=False
    ).__dict__
    try:
        results['hvx'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, hvx_options)
        print(f"  End-to-End Time: {results['hvx']['e2e_time_s']:.4f}s ({results['hvx']['avg_time_per_iter']*1000:.2f}ms/iter)")
        if results['hvx']['kernel_time_us'] is not None:
            print(f"  Kernel Time (Test_Info): {results['hvx']['kernel_time_us']:.2f} us")
        print(f"  Correct: {results['hvx']['is_correct']}")
    except Exception as e:
        print(f"  HVX execution failed: {e}")
        results['hvx'] = None
    
    # Configuration 3: HMX execution
    print("\nRunning HMX execution (HexKL)...")
    try:
        hmx_options = HexagonOptions(
            enableVectorization=True,
            enableHexKL=True,
            enableVTCMTiling=True,
            enableConvTiling=True,
            enableSeedLayoutConversions=True,
            enableMultiThreading=False
        ).__dict__
        results['hmx'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, hmx_options)
        print(f"  End-to-End Time: {results['hmx']['e2e_time_s']:.4f}s ({results['hmx']['avg_time_per_iter']*1000:.2f}ms/iter)")
        if results['hmx']['kernel_time_us'] is not None:
            print(f"  Kernel Time (Test_Info): {results['hmx']['kernel_time_us']:.2f} us")
        print(f"  Correct: {results['hmx']['is_correct']}")
    except Exception as e:
        print(f"  HMX execution failed: {e}")
        print("  Note: HexKL may not be available on your system")
        results['hmx'] = None
    
    # Configuration 4: HVX with multi-threading
    print("\nRunning HVX execution (Multi-threaded)...")
    try:
        hvx_mt_options = HexagonOptions(
            enableVectorization=True,
            enableHexKL=False,
            enableVTCMTiling=True,
            enableConvTiling=True,
            enableMultiThreading=True,
            num_threads=4
        ).__dict__
        results['hvx_mt'] = benchmark_hexagon(model, inputs, linalg_filename, func_name, hvx_mt_options)
        print(f"  End-to-End Time: {results['hvx_mt']['e2e_time_s']:.4f}s ({results['hvx_mt']['avg_time_per_iter']*1000:.2f}ms/iter)")
        if results['hvx_mt']['kernel_time_us'] is not None:
            print(f"  Kernel Time (Test_Info): {results['hvx_mt']['kernel_time_us']:.2f} us")
        print(f"  Correct: {results['hvx_mt']['is_correct']}")
    except Exception as e:
        print(f"  HVX multi-threaded execution failed: {e}")
        results['hvx_mt'] = None
    
    # Print summary
    print(f"\n{'-'*80}")
    print("Performance Summary:")
    print(f"{'-'*80}")
    print(f"{'Mode':<20} {'Kernel(us)':<15} {'E2E(s)':<15} {'Speedup':<15} {'ms/iter':<15} {'Correct':<10}")
    print(f"{'-'*80}")
    
    scalar_kernel_us = (
        results['scalar']['kernel_time_us']
        if results['scalar'] is not None
        else None
    )
    scalar_e2e_s = results['scalar']['e2e_time_s'] if results['scalar'] is not None else None
    for mode in ['scalar', 'hvx', 'hvx_mt', 'hmx']:
        if results.get(mode) is not None:
            kernel_us = results[mode]['kernel_time_us']
            if scalar_kernel_us is not None and kernel_us is not None and kernel_us > 0:
                speedup = scalar_kernel_us / kernel_us
            else:
                speedup = scalar_e2e_s / results[mode]['e2e_time_s'] if scalar_e2e_s else 1.0
            ms_per_iter = results[mode]['avg_time_per_iter'] * 1000
            kernel_str = f"{kernel_us:.2f}" if kernel_us is not None else "N/A"
            print(f"{mode.upper():<20} {kernel_str:<15} {results[mode]['e2e_time_s']:<15.4f} {speedup:<15.2f}x {ms_per_iter:<15.2f} {str(results[mode]['is_correct']):<10}")
    print(f"{'-'*80}\n")
    
    return results


def main():
    """Run benchmarks with different models"""
    
    all_results = {}
    
    print("\n" + "="*80)
    print("HEXAGON NPU - SMALL DNN MODEL BENCHMARK")
    print("Testing Scalar vs HVX vs HMX performance")
    print("="*80)
    
    # Test 1: Simple MLP (mainly matmul operations)
    print("\n" + "="*80)
    print("TEST 1: Simple MLP (Matrix Multiplication Heavy)")
    print("="*80)
    try:
        mlp_model = SimpleMLP(input_size=784, hidden_sizes=[256, 128, 64], num_classes=10)
        mlp_input = torch.randn(1, 784, dtype=torch.float32)
        results_mlp = run_model_benchmark(mlp_model, [mlp_input], "SimpleMLP")
        all_results['mlp'] = results_mlp
    except Exception as e:
        print(f"\nError running MLP benchmark: {e}")
    
    # Test 2: Small CNN (conv + matmul operations)
    print("\n" + "="*80)
    print("TEST 2: Small CNN (Convolution + Matrix Multiplication)")
    print("="*80)
    try:
        cnn_model = SmallCNN(input_size=32, num_classes=10)
        cnn_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        results_cnn = run_model_benchmark(cnn_model, [cnn_input], "SmallCNN")
        all_results['cnn'] = results_cnn
    except Exception as e:
        print(f"\nError running CNN benchmark: {e}")
    
    # Save results to JSON
    results_file = Path(__file__).parent / "dnn_benchmark_results.json"
    print(f"\nSaving results to: {results_file}")
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_type, results in all_results.items():
        json_results[model_type] = {}
        for mode, data in results.items():
            if data is not None:
                json_results[model_type][mode] = {
                    'time': data['time'],
                    'e2e_time_s': data.get('e2e_time_s'),
                    'kernel_time_us': data.get('kernel_time_us'),
                    'kernel_time_s': data.get('kernel_time_s'),
                    'perf_units': data.get('perf_units'),
                    'is_correct': data['is_correct'],
                    'iterations': data['iterations'],
                    'avg_time_per_iter': data['avg_time_per_iter'],
                    'avg_kernel_time_per_iter_s': data.get('avg_kernel_time_per_iter_s')
                }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
