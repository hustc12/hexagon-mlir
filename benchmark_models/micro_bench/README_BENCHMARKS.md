# Hexagon NPU Benchmark Suite

This directory contains comprehensive benchmarks for evaluating the performance of Matrix Multiplication and Convolution operations on Qualcomm Hexagon NPU across different execution modes.

## Overview

These benchmarks compare three execution modes:

1. **Scalar**: Basic scalar processor execution (baseline)
   - No vectorization
   - No specialized matrix/convolution hardware
   - Serves as performance baseline

2. **HVX**: Hexagon Vector eXtensions
   - SIMD vectorization enabled
   - Optimized for vector operations
   - VTCM tiling for efficient memory access

3. **HMX**: Hexagon Matrix eXtensions (via HexKL library)
   - Specialized matrix multiplication hardware
   - Optimized layout conversions for convolutions
   - Requires HexKL library to be available

## Benchmark Scripts

### 1. Matrix Multiplication Benchmark (`test_matmul_benchmark.py`)

Tests matrix multiplication operations with various sizes:
- 64x64x64
- 128x128x128
- 256x256x256
- 512x512x512

**Usage:**
```bash
cd /home/huzq85/2-working/hexagon_npu/hexagon-mlir/benchmark_models
python3 test_matmul_benchmark.py
```

**Output:**
- Console: Performance comparison for each configuration
- File: `matmul_benchmark_results.json`

### 2. Convolution Benchmark (`test_conv_benchmark.py`)

Tests 2D convolution operations with various configurations:
- Different channel sizes (3→16, 16→32, 32→64, 64→64)
- Various spatial dimensions
- 3x3 kernel size

**Usage:**
```bash
python3 test_conv_benchmark.py
```

**Output:**
- Console: Performance comparison for each configuration
- File: `conv_benchmark_results.json`

### 3. Small DNN Model Benchmark (`test_small_dnn_benchmark.py`)

Tests complete neural network models:

**Model 1: SimpleMLP** (Matrix Multiplication Heavy)
- Architecture: 784 → 256 → 128 → 64 → 10
- Primarily tests linear layers (matrix multiplications)
- No convolutions

**Model 2: SmallCNN** (Convolution + Matrix Multiplication)
- Architecture:
  - Conv2D(3→16) + ReLU + MaxPool
  - Conv2D(16→32) + ReLU + MaxPool
  - Flatten
  - Linear(2048→128) + ReLU
  - Linear(128→10)
- Tests both convolution and matrix multiplication operations

**Usage:**
```bash
python3 test_small_dnn_benchmark.py
```

**Output:**
- Console: Performance comparison for each model
- File: `dnn_benchmark_results.json`

### 4. Master Script (`run_all_benchmarks.py`)

Runs all benchmarks sequentially and generates a comprehensive comparison report.

**Usage:**
```bash
python3 run_all_benchmarks.py
```

**Output:**
- All individual benchmark result files
- `benchmark_comparison_report.md`: Comprehensive markdown report

## Prerequisites

1. **Environment Setup:**
   ```bash
   # Ensure HEXAGON_MLIR_ROOT is set
   export HEXAGON_MLIR_ROOT=/path/to/hexagon-mlir
   
   # Ensure Hexagon SDK paths are set (if running on device)
   export ANDROID_HOST=your_device_ip
   export ANDROID_SERIAL=your_device_serial
   ```

2. **Python Dependencies:**
   - torch
   - torch_mlir
   - triton (with Hexagon backend)

3. **Hardware Requirements:**
   - Qualcomm device with Hexagon NPU (v73, v75, or v79)
   - For HMX benchmarks: HexKL library must be installed

## Understanding the Results

### Performance Metrics

Each benchmark reports:
- **Time (s)**: Total execution time for all iterations
- **ms/iter**: Average time per iteration
- **Speedup**: Relative to scalar baseline
- **Correct**: Whether output matches reference (PyTorch)

### Expected Speedups

Based on typical hardware characteristics:

| Operation Type | Scalar | HVX | HMX |
|----------------|--------|-----|-----|
| Matrix Multiplication | 1.0x | 5-15x | 10-30x |
| Convolution | 1.0x | 8-20x | 15-40x |
| Mixed Workload | 1.0x | 6-18x | 12-35x |

*Actual speedups depend on problem size, data layout, and hardware generation.*

## Configuration Options

Each benchmark uses `HexagonOptions` to configure execution mode:

### Scalar Configuration
```python
HexagonOptions(
    enableVectorization=False,
    enableHexKL=False,
    enableVTCMTiling=False,
    enableMultiThreading=False
)
```

### HVX Configuration
```python
HexagonOptions(
    enableVectorization=True,
    enableHexKL=False,
    enableVTCMTiling=True,
    enableConvTiling=True,
    enableMultiThreading=False
)
```

### HMX Configuration
```python
HexagonOptions(
    enableVectorization=True,
    enableHexKL=True,
    enableVTCMTiling=True,
    enableConvTiling=True,
    enableSeedLayoutConversions=True,  # Important for conv2d
    enableMultiThreading=False
)
```

## Troubleshooting

### HexKL Not Available
If you see "HexKL may not be available on your system":
- HexKL is a proprietary library that may require special access
- Benchmarks will still run Scalar and HVX modes
- Contact Qualcomm for HexKL access

### Correctness Issues
If `is_correct` shows `False`:
- Check tolerance settings (`rtol`, `atol`)
- Verify data types match
- Some numerical differences are expected due to different execution paths

### Memory Issues
If benchmarks fail with out-of-memory errors:
- Reduce matrix/tensor sizes
- Enable `lowerConstantsInSeparateSharedObjects`
- Check VTCM usage

### Device Connection Issues
```bash
# Verify device connection
adb devices

# Check environment variables
echo $ANDROID_HOST
echo $ANDROID_SERIAL
```

## Advanced Usage

### Running Individual Configurations

You can modify the scripts to test specific configurations:

```python
# Example: Test only HVX mode
options = HexagonOptions(
    enableVectorization=True,
    enableHexKL=False,
    enableVTCMTiling=True,
    num_threads=4  # Try multi-threading
).__dict__

results = benchmark_hexagon(model, inputs, filename, func_name, options)
```

### Custom Matrix Sizes

Edit the `test_configs` list in each script:

```python
test_configs = [
    (1024, 1024, 1024),  # Large matrices
    (2048, 2048, 2048),  # Very large matrices
]
```

### Enable Profiling

For detailed profiling with Light Weight Profiling (LWP):

```python
options = HexagonOptions(
    enableLWP=True,
    disableLWPLoop=False,  # Enable loop-level instrumentation
    LWPloopDepth=2
).__dict__
```

Then process LWP data:
```bash
python3 $HEXAGON_MLIR_ROOT/test/python/process_lwp.py \
    /tmp/lwp.json \
    /tmp/lwp_infodump.txt \
    /tmp/initial-linalg.mlir
```

## Interpreting MLIR Output

The benchmarks generate `.mlirbc` files containing the compiled model in MLIR bytecode format. To inspect:

```bash
# Find the linalg-hexagon-opt tool
find . -name linalg-hexagon-opt

# View intermediate IR
linalg-hexagon-opt --help | grep hexagon
```

## Contributing

To add new benchmarks:

1. Create a new Python script following the existing pattern
2. Use `ModelManager` or the helper functions from existing benchmarks
3. Add your script to `run_all_benchmarks.py`
4. Update this README

## References

- [Hexagon-MLIR Documentation](../docs/)
- [Hexagon Architecture Guide](https://developer.qualcomm.com/software/hexagon-dsp-sdk)
- [Torch-MLIR Documentation](https://github.com/llvm/torch-mlir)

## License

See LICENSE.txt in the repository root.
