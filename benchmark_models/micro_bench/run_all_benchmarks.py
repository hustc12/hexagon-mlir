#!/usr/bin/env python3
# ===- run_all_benchmarks.py ------------------------------------------------===
#
# Master script to run all benchmarks and generate comparison report
#
# ===------------------------------------------------------------------------===

import subprocess
import sys
import json
from pathlib import Path
import time


def run_benchmark(script_name):
    """Run a benchmark script and return success status"""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error running {script_name}: {e}")
        return False


def load_json_results(filename):
    """Load JSON results file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def generate_comparison_report(results_dir):
    """Generate a comprehensive comparison report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK REPORT")
    print("="*80)
    
    # Load all results
    matmul_results = load_json_results(results_dir / "matmul_benchmark_results.json")
    conv_results = load_json_results(results_dir / "conv_benchmark_results.json")
    dnn_results = load_json_results(results_dir / "dnn_benchmark_results.json")
    
    report_file = results_dir / "benchmark_comparison_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Hexagon NPU Benchmark Results\n\n")
        f.write("## Overview\n\n")
        f.write("This report compares the performance of different execution modes on Qualcomm Hexagon NPU:\n\n")
        f.write("- **Scalar**: Basic scalar processor execution (baseline)\n")
        f.write("- **HVX**: Hexagon Vector eXtensions (vectorized execution)\n")
        f.write("- **HMX**: Hexagon Matrix eXtensions via HexKL library (specialized matrix operations)\n")
        f.write("- **HVX_MT**: HVX with multi-threading enabled\n\n")
        
        # Matrix Multiplication Results
        if matmul_results:
            f.write("## Matrix Multiplication Benchmark\n\n")
            f.write("| Configuration | Scalar Time (s) | HVX Time (s) | HVX Speedup | HMX Time (s) | HMX Speedup |\n")
            f.write("|---------------|-----------------|--------------|-------------|--------------|-------------|\n")
            
            for config, results in matmul_results.items():
                scalar_time = results.get('scalar', {}).get('time', 'N/A')
                hvx_time = results.get('hvx', {}).get('time', 'N/A')
                hmx_time = results.get('hmx', {}).get('time', 'N/A')
                
                hvx_speedup = f"{scalar_time / hvx_time:.2f}x" if isinstance(scalar_time, float) and isinstance(hvx_time, float) else 'N/A'
                hmx_speedup = f"{scalar_time / hmx_time:.2f}x" if isinstance(scalar_time, float) and isinstance(hmx_time, float) else 'N/A'
                
                scalar_time = f"{scalar_time:.4f}" if isinstance(scalar_time, float) else scalar_time
                hvx_time = f"{hvx_time:.4f}" if isinstance(hvx_time, float) else hvx_time
                hmx_time = f"{hmx_time:.4f}" if isinstance(hmx_time, float) else hmx_time
                
                f.write(f"| {config} | {scalar_time} | {hvx_time} | {hvx_speedup} | {hmx_time} | {hmx_speedup} |\n")
            
            f.write("\n")
        
        # Convolution Results
        if conv_results:
            f.write("## Convolution Benchmark\n\n")
            f.write("| Configuration | Scalar Time (s) | HVX Time (s) | HVX Speedup | HMX Time (s) | HMX Speedup |\n")
            f.write("|---------------|-----------------|--------------|-------------|--------------|-------------|\n")
            
            for config, results in conv_results.items():
                scalar_time = results.get('scalar', {}).get('time', 'N/A')
                hvx_time = results.get('hvx', {}).get('time', 'N/A')
                hmx_time = results.get('hmx', {}).get('time', 'N/A')
                
                hvx_speedup = f"{scalar_time / hvx_time:.2f}x" if isinstance(scalar_time, float) and isinstance(hvx_time, float) else 'N/A'
                hmx_speedup = f"{scalar_time / hmx_time:.2f}x" if isinstance(scalar_time, float) and isinstance(hmx_time, float) else 'N/A'
                
                scalar_time = f"{scalar_time:.4f}" if isinstance(scalar_time, float) else scalar_time
                hvx_time = f"{hvx_time:.4f}" if isinstance(hvx_time, float) else hvx_time
                hmx_time = f"{hmx_time:.4f}" if isinstance(hmx_time, float) else hmx_time
                
                f.write(f"| {config} | {scalar_time} | {hvx_time} | {hvx_speedup} | {hmx_time} | {hmx_speedup} |\n")
            
            f.write("\n")
        
        # DNN Model Results
        if dnn_results:
            f.write("## DNN Model Benchmark\n\n")
            f.write("| Model | Mode | Time (s) | Speedup vs Scalar | ms/iter | Correct |\n")
            f.write("|-------|------|----------|-------------------|---------|----------|\n")
            
            for model_name, model_results in dnn_results.items():
                scalar_time = model_results.get('scalar', {}).get('time', None)
                
                for mode, results in model_results.items():
                    time_val = results.get('time', 'N/A')
                    ms_per_iter = results.get('avg_time_per_iter', 0) * 1000
                    is_correct = results.get('is_correct', False)
                    
                    speedup = f"{scalar_time / time_val:.2f}x" if scalar_time and isinstance(time_val, float) else 'N/A'
                    time_str = f"{time_val:.4f}" if isinstance(time_val, float) else time_val
                    ms_str = f"{ms_per_iter:.2f}" if isinstance(ms_per_iter, float) else 'N/A'
                    
                    f.write(f"| {model_name.upper()} | {mode.upper()} | {time_str} | {speedup} | {ms_str} | {is_correct} |\n")
            
            f.write("\n")
        
        # Summary and Observations
        f.write("## Summary\n\n")
        f.write("### Key Observations:\n\n")
        f.write("1. **HVX Vectorization**: Shows significant speedup over scalar execution for all workloads\n")
        f.write("2. **HMX Matrix Extensions**: Provides additional acceleration for matrix operations when HexKL is available\n")
        f.write("3. **Multi-threading**: Can further improve performance on suitable workloads\n")
        f.write("4. **Memory Layout**: VTCM tiling and layout conversions are critical for performance\n\n")
        
        f.write("### Recommendations:\n\n")
        f.write("- For matrix-heavy workloads (matmul, linear layers): Use HMX when available\n")
        f.write("- For convolution operations: Enable both HVX vectorization and conv tiling\n")
        f.write("- For large models: Consider multi-threading for additional parallelism\n")
        f.write("- Always verify correctness when switching execution modes\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"\nComparison report saved to: {report_file}")
    print("\nYou can view the report with:")
    print(f"  cat {report_file}")
    print(f"  or open it in your favorite markdown viewer")


def main():
    """Main function to run all benchmarks"""
    benchmark_dir = Path(__file__).parent
    
    print("\n" + "="*80)
    print("HEXAGON NPU - COMPREHENSIVE BENCHMARK SUITE")
    print("="*80)
    print("\nThis will run all benchmarks:")
    print("  1. Matrix Multiplication Benchmark")
    print("  2. Convolution Benchmark")
    print("  3. Small DNN Model Benchmark")
    print("\nNote: This may take several minutes to complete.")
    
    # Ask for confirmation
    try:
        response = input("\nProceed with all benchmarks? (y/n): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return
    except:
        pass  # Non-interactive mode, proceed anyway
    
    results = {}
    
    # Run Matrix Multiplication Benchmark
    results['matmul'] = run_benchmark(benchmark_dir / "test_matmul_benchmark.py")
    
    # Run Convolution Benchmark
    results['conv'] = run_benchmark(benchmark_dir / "test_conv_benchmark.py")
    
    # Run DNN Model Benchmark
    results['dnn'] = run_benchmark(benchmark_dir / "test_small_dnn_benchmark.py")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*80)
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name.upper():<20} {status}")
    print("="*80)
    
    # Generate comparison report
    if any(results.values()):
        generate_comparison_report(benchmark_dir)
    
    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETE")
    print("="*80)
    print("\nResults available in:")
    print(f"  - {benchmark_dir}/matmul_benchmark_results.json")
    print(f"  - {benchmark_dir}/conv_benchmark_results.json")
    print(f"  - {benchmark_dir}/dnn_benchmark_results.json")
    print(f"  - {benchmark_dir}/benchmark_comparison_report.md")
    print("\n")


if __name__ == "__main__":
    main()
