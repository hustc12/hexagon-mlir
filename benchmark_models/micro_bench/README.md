# Hexagon NPU Benchmark Suite - File Index

## Quick Reference Guide

This document provides a quick reference to all files in the benchmark suite.

## 📋 Main Files

### Benchmark Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `test_matmul_benchmark.py` | Matrix multiplication benchmark | `python3 test_matmul_benchmark.py` |
| `test_conv_benchmark.py` | Convolution benchmark | `python3 test_conv_benchmark.py` |
| `test_small_dnn_benchmark.py` | DNN model benchmark (MLP + CNN) | `python3 test_small_dnn_benchmark.py` |
| `test_quick_validation.py` | Quick environment validation | `python3 test_quick_validation.py` |

### Orchestration Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `run_all_benchmarks.py` | Run all benchmarks and generate report | `python3 run_all_benchmarks.py` |
| `run_benchmarks.sh` | Shell wrapper for easy execution | `./run_benchmarks.sh [mode]` |

### Documentation

| File | Purpose |
|------|---------|
| `README_BENCHMARKS.md` | Comprehensive usage guide |
| `IMPLEMENTATION_SUMMARY.md` | Detailed implementation overview |
| `INDEX.md` | This file - quick reference |

## 🚀 Quick Start Guide

### For First-Time Users

1. **Validate your setup**:
   ```bash
   python3 test_quick_validation.py
   ```

2. **Run all benchmarks**:
   ```bash
   ./run_benchmarks.sh all
   ```
   or
   ```bash
   python3 run_all_benchmarks.py
   ```

3. **View results**:
   - JSON files: `*_benchmark_results.json`
   - Markdown report: `benchmark_comparison_report.md`

### For Advanced Users

Run individual benchmarks:
```bash
./run_benchmarks.sh matmul    # Matrix multiplication only
./run_benchmarks.sh conv      # Convolution only
./run_benchmarks.sh dnn       # DNN models only
```

## 📊 Output Files

After running benchmarks, the following files will be generated:

| File | Content |
|------|---------|
| `matmul_benchmark_results.json` | Matrix multiplication results |
| `conv_benchmark_results.json` | Convolution results |
| `dnn_benchmark_results.json` | DNN model results |
| `benchmark_comparison_report.md` | Comprehensive comparison report |
| `*.mlirbc` | Compiled MLIR bytecode files |

## 🔧 Configuration Files

Each benchmark uses `HexagonOptions` for configuration. Three main modes:

### Mode 1: Scalar (Baseline)
```python
enableVectorization=False
enableHexKL=False
```

### Mode 2: HVX (Vectorized)
```python
enableVectorization=True
enableHexKL=False
enableVTCMTiling=True
```

### Mode 3: HMX (Matrix Extensions)
```python
enableVectorization=True
enableHexKL=True
enableSeedLayoutConversions=True
```

## 📈 Execution Modes Comparison

| Feature | Scalar | HVX | HMX |
|---------|--------|-----|-----|
| Vectorization | ❌ | ✅ | ✅ |
| Matrix Hardware | ❌ | ❌ | ✅ |
| VTCM Tiling | ❌ | ✅ | ✅ |
| Layout Conversion | ❌ | ❌ | ✅ |
| Expected Speedup | 1x | 5-20x | 10-40x |

## 🎯 Benchmark Coverage

### Operations Tested

1. **Matrix Multiplication**
   - Sizes: 64², 128², 256², 512²
   - Pure matmul operations
   - A @ B computation

2. **Convolution**
   - 2D convolution with various channel configurations
   - 3x3 kernel size
   - Padding and stride support

3. **Complete Models**
   - **SimpleMLP**: Multi-layer perceptron (matmul-heavy)
   - **SmallCNN**: Convolutional neural network (mixed ops)

### Hardware Modes

- ✅ Scalar Processor
- ✅ HVX (Vector Extensions)
- ✅ HMX (Matrix Extensions, requires HexKL)
- ✅ HVX Multi-threaded (optional)

## 🔍 File Dependencies

```
run_benchmarks.sh
  └─> run_all_benchmarks.py
       ├─> test_matmul_benchmark.py
       ├─> test_conv_benchmark.py
       └─> test_small_dnn_benchmark.py

test_quick_validation.py (standalone)
```

## 📚 Documentation Structure

```
README_BENCHMARKS.md          # User guide (start here)
  ├─ Prerequisites
  ├─ Usage Instructions
  ├─ Configuration Options
  ├─ Troubleshooting
  └─ Advanced Topics

IMPLEMENTATION_SUMMARY.md     # Technical details
  ├─ Implementation Overview
  ├─ Architecture Details
  ├─ Performance Metrics
  └─ Extension Guide

INDEX.md                      # This file (quick reference)
  ├─ File Listing
  ├─ Quick Start
  └─ Reference Tables
```

## 🛠️ Customization Guide

### Adding New Benchmarks

1. Create new `test_*.py` file
2. Follow existing structure:
   - Use `torch.nn.Module` for model definition
   - Use `fx_mlir.export_and_import()` for MLIR conversion
   - Use `TorchMLIRHexagonLauncher().run_torch_mlir()` for execution
   - Include correctness validation
3. Add to `run_all_benchmarks.py`

### Modifying Test Configurations

Edit `test_configs` in each benchmark file:

```python
# In test_matmul_benchmark.py
test_configs = [
    (M, N, K),  # Add your sizes
]

# In test_conv_benchmark.py
test_configs = [
    (batch, in_ch, out_ch, height, width, kernel),
]
```

## ⚠️ Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| HexKL not available | Skip HMX tests or contact Qualcomm | README section "Troubleshooting" |
| Device connection | Check `adb devices` | README section "Prerequisites" |
| Environment variables | Run validation script first | `test_quick_validation.py` |
| Correctness mismatch | Adjust tolerance settings | Check `rtol`, `atol` parameters |

## 📞 Getting Help

1. **First time?** → Read `README_BENCHMARKS.md`
2. **Quick test?** → Run `test_quick_validation.py`
3. **Technical details?** → Read `IMPLEMENTATION_SUMMARY.md`
4. **Reference?** → This file (`INDEX.md`)

## 🎓 Learning Path

```
Beginner:
  1. Read README_BENCHMARKS.md
  2. Run test_quick_validation.py
  3. Run ./run_benchmarks.sh all
  4. View benchmark_comparison_report.md

Intermediate:
  1. Run individual benchmarks
  2. Modify test configurations
  3. Experiment with HexagonOptions
  4. Analyze JSON results

Advanced:
  1. Read IMPLEMENTATION_SUMMARY.md
  2. Add custom benchmarks
  3. Enable LWP profiling
  4. Optimize specific workloads
```

## 📅 Version Information

- **Created**: 2026-04-28
- **Version**: 1.0
- **Status**: Production Ready
- **Compatibility**: Hexagon NPU v73, v75, v79

---

*For detailed information, please refer to README_BENCHMARKS.md and IMPLEMENTATION_SUMMARY.md*
