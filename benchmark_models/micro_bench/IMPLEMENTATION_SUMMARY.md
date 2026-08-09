# Hexagon NPU Matrix Multiplication & Convolution Benchmark - Implementation Summary

## 概述 (Overview)

本项目实现了在Qualcomm Hexagon NPU上对Matrix Multiplication（矩阵乘法）和Convolution（卷积）算子的全面性能评估，比较了在Scalar Processor、HVX（Hexagon Vector Extensions）和HMX（Hexagon Matrix Extensions）三种执行模式下的性能表现。

This project implements comprehensive performance evaluation of Matrix Multiplication and Convolution operators on Qualcomm Hexagon NPU, comparing performance across three execution modes: Scalar Processor, HVX (Hexagon Vector Extensions), and HMX (Hexagon Matrix Extensions).

## 实现内容 (Implementation)

### 1. 算子基准测试 (Operator Benchmarks)

#### 1.1 矩阵乘法基准测试 (Matrix Multiplication Benchmark)
**文件**: `test_matmul_benchmark.py`

实现了针对矩阵乘法运算的性能测试，测试不同大小的矩阵：
- 64x64x64
- 128x128x128
- 256x256x256
- 512x512x512

**关键特性**:
- 使用PyTorch定义模型
- 通过torch-mlir转换为MLIR表示
- 在三种执行模式下运行
- 自动验证结果正确性
- 输出详细的性能指标

#### 1.2 卷积基准测试 (Convolution Benchmark)
**文件**: `test_conv_benchmark.py`

实现了2D卷积操作的性能测试，测试不同配置：
- 不同通道数配置（3→16, 16→32, 32→64, 64→64）
- 不同空间维度
- 3x3卷积核

**关键特性**:
- 测试真实的Conv2D操作
- 支持padding和stride配置
- 验证输出正确性
- 比较不同执行模式的性能

### 2. 小型DNN模型基准测试 (Small DNN Model Benchmarks)

**文件**: `test_small_dnn_benchmark.py`

实现了两个完整的神经网络模型：

#### 2.1 SimpleMLP (矩阵乘法密集型)
- 架构: 784 → 256 → 128 → 64 → 10
- 主要测试线性层（矩阵乘法）
- 包含ReLU激活函数

#### 2.2 SmallCNN (卷积+矩阵乘法混合)
- 架构:
  - Conv2D(3→16, 3x3) + ReLU + MaxPool
  - Conv2D(16→32, 3x3) + ReLU + MaxPool
  - Flatten
  - Linear(2048→128) + ReLU
  - Linear(128→10)
- 测试卷积和线性层的综合性能

### 3. 三种执行模式配置 (Three Execution Modes)

#### 3.1 Scalar处理器模式
```python
HexagonOptions(
    enableVectorization=False,    # 禁用向量化
    enableHexKL=False,            # 禁用HexKL库
    enableVTCMTiling=False,       # 禁用VTCM tiling
    enableMultiThreading=False    # 禁用多线程
)
```
- 基准性能，使用标量处理器
- 作为性能对比的基线

#### 3.2 HVX模式（向量化执行）
```python
HexagonOptions(
    enableVectorization=True,     # 启用向量化
    enableHexKL=False,           # 不使用HexKL
    enableVTCMTiling=True,       # 启用VTCM tiling
    enableConvTiling=True,       # 启用卷积tiling
    enableMultiThreading=False   # 单线程
)
```
- 使用Hexagon向量扩展单元
- 针对向量操作优化
- VTCM用于高效内存访问

#### 3.3 HMX模式（矩阵扩展）
```python
HexagonOptions(
    enableVectorization=True,
    enableHexKL=True,                    # 启用HexKL库
    enableVTCMTiling=True,
    enableConvTiling=True,
    enableSeedLayoutConversions=True,    # 为conv2d启用layout转换
    enableMultiThreading=False
)
```
- 使用专用矩阵乘法硬件
- 针对卷积优化的layout转换
- 需要HexKL库支持

### 4. 工具和脚本 (Tools and Scripts)

#### 4.1 快速验证脚本
**文件**: `test_quick_validation.py`
- 快速验证环境配置
- 测试基本功能
- 在运行完整benchmark前进行检查

#### 4.2 主运行脚本
**文件**: `run_all_benchmarks.py`
- 顺序运行所有benchmark
- 生成综合对比报告
- 输出JSON格式的结果

#### 4.3 Shell脚本
**文件**: `run_benchmarks.sh`
- 方便的命令行接口
- 支持运行单个或全部测试
- 彩色输出和错误处理

### 5. 文档 (Documentation)

#### 5.1 详细README
**文件**: `README_BENCHMARKS.md`
- 完整的使用说明
- 配置选项说明
- 故障排除指南
- 性能调优建议

#### 5.2 实现总结
**文件**: `IMPLEMENTATION_SUMMARY.md` (本文件)
- 项目概述
- 实现细节
- 使用指南

## 目录结构 (Directory Structure)

```
benchmark_models/
├── test_matmul_benchmark.py         # 矩阵乘法benchmark
├── test_conv_benchmark.py           # 卷积benchmark
├── test_small_dnn_benchmark.py      # DNN模型benchmark
├── test_quick_validation.py         # 快速验证脚本
├── run_all_benchmarks.py            # 主运行脚本
├── run_benchmarks.sh                # Shell脚本
├── README_BENCHMARKS.md             # 详细文档
├── IMPLEMENTATION_SUMMARY.md        # 本文件
└── (生成的结果文件)
    ├── matmul_benchmark_results.json
    ├── conv_benchmark_results.json
    ├── dnn_benchmark_results.json
    └── benchmark_comparison_report.md
```

## 使用方法 (Usage)

### 快速开始 (Quick Start)

1. **环境检查**:
```bash
cd /home/huzq85/2-working/hexagon_npu/hexagon-mlir/benchmark_models
python3 test_quick_validation.py
```

2. **运行所有benchmarks**:
```bash
./run_benchmarks.sh all
# 或
python3 run_all_benchmarks.py
```

3. **运行单个benchmark**:
```bash
# 矩阵乘法
python3 test_matmul_benchmark.py

# 卷积
python3 test_conv_benchmark.py

# DNN模型
python3 test_small_dnn_benchmark.py
```

### 使用Shell脚本 (Using Shell Script)

```bash
# 验证环境
./run_benchmarks.sh validate

# 只运行矩阵乘法
./run_benchmarks.sh matmul

# 只运行卷积
./run_benchmarks.sh conv

# 只运行DNN模型
./run_benchmarks.sh dnn

# 运行所有
./run_benchmarks.sh all
```

## 性能指标 (Performance Metrics)

每个benchmark报告以下指标：
- **Time (s)**: 总执行时间
- **ms/iter**: 每次迭代的平均时间
- **Speedup**: 相对于Scalar基线的加速比
- **Correct**: 输出是否与参考结果匹配

### 预期性能提升 (Expected Performance Gains)

| 操作类型 | Scalar | HVX | HMX |
|---------|--------|-----|-----|
| 矩阵乘法 | 1.0x | 5-15x | 10-30x |
| 卷积 | 1.0x | 8-20x | 15-40x |
| 混合负载 | 1.0x | 6-18x | 12-35x |

*实际加速比取决于问题规模、数据布局和硬件版本*

## 输出文件 (Output Files)

### 1. JSON结果文件
- `matmul_benchmark_results.json`: 矩阵乘法结果
- `conv_benchmark_results.json`: 卷积结果
- `dnn_benchmark_results.json`: DNN模型结果

结构示例:
```json
{
  "64x64x64": {
    "scalar": {
      "time": 0.5234,
      "is_correct": true,
      "iterations": 10
    },
    "hvx": {
      "time": 0.0523,
      "is_correct": true,
      "iterations": 10
    },
    "hmx": {
      "time": 0.0234,
      "is_correct": true,
      "iterations": 10
    }
  }
}
```

### 2. Markdown报告
`benchmark_comparison_report.md`: 综合性能对比报告
- 包含所有测试的表格
- 加速比分析
- 推荐和总结

### 3. MLIR字节码文件
`.mlirbc` 文件包含编译后的模型，用于Hexagon执行。

## 关键技术点 (Key Technical Points)

### 1. MLIR转换流程
```
PyTorch Model 
  → torch_mlir.fx.export_and_import() 
  → Linalg-on-Tensors IR 
  → MLIR Bytecode (.mlirbc)
  → Hexagon Compilation Pipeline
  → Hexagon Execution
```

### 2. HexagonOptions配置
通过`HexagonOptions`类配置编译选项：
- **enableVectorization**: 启用/禁用HVX向量化
- **enableHexKL**: 启用/禁用HexKL矩阵库
- **enableVTCMTiling**: 启用/禁用VTCM内存优化
- **enableConvTiling**: 启用/禁用卷积tiling优化
- **enableSeedLayoutConversions**: 为conv2d启用layout转换
- **enableMultiThreading**: 启用/禁用多线程

### 3. 正确性验证
所有benchmark都包含正确性检查：
```python
reference = model(*inputs)  # PyTorch参考结果
output = run_on_hexagon(...)  # Hexagon执行结果
is_correct = torch.allclose(output, reference, rtol=1e-03, atol=1e-03)
```

## 依赖要求 (Dependencies)

### Python包
- `torch`: PyTorch深度学习框架
- `torch_mlir`: PyTorch到MLIR的转换
- `triton`: Triton编译器（带Hexagon后端）

### 环境变量
- `HEXAGON_MLIR_ROOT`: Hexagon-MLIR根目录
- `TRITON_SHARED_OPT_PATH`: triton-shared-opt工具路径
- `ANDROID_HOST`: Android设备IP（如果在设备上运行）
- `ANDROID_SERIAL`: Android设备序列号

### 硬件要求
- Qualcomm设备，带Hexagon NPU（v73, v75或v79）
- HMX测试需要HexKL库

## 故障排除 (Troubleshooting)

### 常见问题

1. **HexKL不可用**
   - HexKL是专有库，可能需要特殊授权
   - Scalar和HVX模式仍可运行
   - 联系Qualcomm获取HexKL访问权限

2. **正确性问题**
   - 检查容差设置（`rtol`, `atol`）
   - 验证数据类型匹配
   - 不同执行路径可能有数值差异

3. **内存问题**
   - 减小矩阵/张量大小
   - 启用`lowerConstantsInSeparateSharedObjects`
   - 检查VTCM使用情况

4. **设备连接问题**
```bash
# 验证设备连接
adb devices

# 检查环境变量
echo $ANDROID_HOST
echo $ANDROID_SERIAL
```

## 进阶使用 (Advanced Usage)

### 自定义配置

修改脚本以测试特定配置：
```python
# 示例：测试多线程HVX
options = HexagonOptions(
    enableVectorization=True,
    enableHexKL=False,
    enableVTCMTiling=True,
    enableMultiThreading=True,
    num_threads=4
).__dict__
```

### 启用性能分析

使用Light Weight Profiling (LWP)进行详细分析：
```python
options = HexagonOptions(
    enableLWP=True,
    disableLWPLoop=False,
    LWPloopDepth=2
).__dict__
```

### 自定义矩阵大小

编辑`test_configs`列表：
```python
test_configs = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]
```

## 扩展建议 (Extension Suggestions)

### 可以添加的测试
1. **更多算子**:
   - BatchNorm
   - LayerNorm
   - Attention机制
   - 激活函数（GELU, SiLU等）

2. **更大的模型**:
   - ResNet-18/50
   - MobileNetV2/V3
   - 小型Transformer模型

3. **不同数据类型**:
   - FP16
   - INT8量化
   - 混合精度

4. **批处理大小**:
   - 测试不同批处理大小的影响
   - 动态batch size

## 总结 (Summary)

本实现提供了一个完整的框架来评估Hexagon NPU上矩阵乘法和卷积操作的性能：

✅ **已实现**:
- 矩阵乘法算子benchmark
- 卷积算子benchmark
- 包含两种模型的DNN benchmark（MLP和CNN）
- 三种执行模式的性能对比（Scalar, HVX, HMX）
- 自动化测试脚本和工具
- 完整的文档

✅ **关键优势**:
- 易于使用和扩展
- 自动验证正确性
- 详细的性能指标
- 支持多种配置
- 生成可视化报告

✅ **实际应用价值**:
- 帮助理解Hexagon NPU的性能特征
- 指导算子和模型优化
- 为选择执行模式提供数据支持
- 作为进一步研究的基础

## 联系和贡献 (Contact & Contribution)

如需添加新功能或报告问题，请参考主项目的CONTRIBUTING.md文件。

---

**创建日期**: 2026-04-28
**版本**: 1.0
**状态**: 完成并可用
