# Operator-Level Compilation Analysis for Hexagon NPU

## 问题概述

当前的基准测试（`test_matmul_benchmark.py`, `test_conv_benchmark.py`, `test_small_dnn_benchmark.py`）都是在**模型级别**编译支持 Scalar、HVX、HMX 操作的。

**核心问题**：是否可以在**算子级别**分别支持 Scalar、HVX、HMX 的操作？即：
1. 给定一个模型，在编译阶段确定哪些算子在 Scalar Processor 上计算
2. 哪些算子在 HVX Processor 上计算
3. 哪些算子在 HMX 上计算
4. 特别是：能否指定某些 MatMul 算子在 HMX 上计算，其他在 Scalar/HVX 上计算？

## 当前实现分析

### 1. 编译流程概览

```
PyTorch Model
    ↓
torch_mlir.fx.export_and_import() → Linalg-on-Tensors MLIR
    ↓
TorchMLIRHexagonLauncher.run_torch_mlir()
    ↓
translate_linalg_to_obj() [C++ backend]
    ↓
LinalgToLLVMPass (编译管道)
    ↓
Object Code (.o) → Shared Library (.so)
```

### 2. HexagonOptions 控制编译行为

**位置**: `triton/python/triton/backends/qcom_hexagon_backend/hexagon_options.py`

关键选项：
```python
@dataclass(frozen=True)
class HexagonOptions:
    # 向量化控制
    enableVectorization: bool = True      # 启用 HVX 向量化
    
    # HMX 控制
    enableHexKL: bool = False             # 启用 HexKL (HMX 矩阵扩展)
    
    # 其他优化
    enableVTCMTiling: bool = True         # VTCM 内存分块
    enableMultiThreading: bool = False    # 多线程
    enableSeedLayoutConversions: bool = False  # 卷积布局转换
```

**重要发现**：这些选项是**全局的**，应用于整个模型的编译过程。

### 3. 编译管道实现

**位置**: `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`

关键代码段：
```cpp
// Line 212-213
if (enableHexKL)
    pm.addNestedPass<func::FuncOp>(createMatmulToHexKLPass());
```

**分析**：
- `enableHexKL` 是一个**全局开关**
- 如果启用，`MatmulToHexKLPass` 会被添加到编译管道
- 这个 pass 会转换**所有**的 `linalg.matmul` 操作

### 4. MatmulToHexKL Pass 实现

**位置**: `qcom_hexagon_backend/lib/Transforms/MatmulToHexKLPass.cpp`

```cpp
struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];
    // 直接替换为 hexkl.matmul
    rewriter.replaceOpWithNewOp<hexkl::MatmulOp>(op, C.getType(), A, B, C);
    return success();
  }
};
```

**关键发现**：
- 这个 pass 会匹配**所有** `linalg.matmul` 操作
- **没有任何条件判断**来选择性地转换某些 matmul
- 一旦启用 `enableHexKL`，所有 matmul 都会被转换为 `hexkl.matmul`

### 5. HexKL Dialect 定义

**位置**: `qcom_hexagon_backend/include/hexagon/Dialect/HexKL/IR/HexKLOps.td`

```tablegen
def HexKL_MatmulOp : HexKL_Op<"matmul", [...]> {
  let summary = "HexKL Matmul op";
  let description = [{
    Performs matrix multiplication on F16 inputs and F32 output using HexKL APIs.
  }];
  
  let arguments = (ins
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$lhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$rhs,
    AnyTypeOf<[Non0RankedTensorOf<[F32]>, Non0RankedMemRefOf<[F32]>]>:$outs
  );
}
```

**限制**：
- HexKL 只支持 **F16 输入 → F32 输出**
- 不支持其他数据类型组合

## 回答核心问题

### Q1: 能否在算子级别分别支持 Scalar、HVX、HMX？

**答案：理论上可以，但当前实现不支持。**

**原因**：
1. **当前架构是全局控制**：`HexagonOptions` 中的 `enableVectorization` 和 `enableHexKL` 是全局开关
2. **Pass 没有选择性逻辑**：`MatmulToHexKLPass` 会转换所有 matmul，没有条件判断
3. **缺少算子级别的 Annotation 机制**：MLIR IR 中没有标记来指示某个特定算子应该用哪种执行模式

### Q2: 能否指定某些 MatMul 在 HMX 上，其他在 Scalar/HVX 上？

**答案：当前不支持，但可以通过修改实现来支持。**

## 实现算子级别控制的方案

### 方案 1: 基于 Attribute 的选择性转换（推荐）

**思路**：在 MLIR IR 中为每个算子添加 attribute 来指示执行模式。

#### 步骤 1: 在 Python 层添加标记

```python
# 修改 test_matmul_benchmark.py
class AnnotatedMatMul(nn.Module):
    def __init__(self, use_hmx=True):
        super().__init__()
        self.use_hmx = use_hmx
    
    def forward(self, a, b):
        result = torch.matmul(a, b)
        # 添加自定义标记（需要在 torch-mlir 中支持）
        if self.use_hmx:
            result = result  # 标记为 HMX
        return result
```

#### 步骤 2: 修改 MatmulToHexKL Pass

```cpp
// 在 MatmulToHexKLPass.cpp 中添加条件判断
struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    
    // 检查是否有 "use_hmx" attribute
    auto useHMXAttr = op->getAttrOfType<BoolAttr>("use_hmx");
    if (!useHMXAttr || !useHMXAttr.getValue()) {
      // 不转换，保留为 linalg.matmul（后续会被向量化或标量化）
      return failure();
    }
    
    // 检查数据类型是否符合 HexKL 要求
    auto lhsType = op.getDpsInputOperand(0)->get().getType();
    auto rhsType = op.getDpsInputOperand(1)->get().getType();
    auto outType = op.getOutputs()[0].getType();
    
    if (!isF16Type(lhsType) || !isF16Type(rhsType) || !isF32Type(outType)) {
      // 数据类型不匹配，不转换
      return failure();
    }
    
    // 转换为 hexkl.matmul
    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];
    rewriter.replaceOpWithNewOp<hexkl::MatmulOp>(op, C.getType(), A, B, C);
    return success();
  }
};
```

#### 步骤 3: 添加基于启发式的自动选择

```cpp
// 添加启发式规则来自动决定是否使用 HMX
bool shouldUseHMX(linalg::MatmulOp op) {
  auto lhsShape = op.getDpsInputOperand(0)->get().getType().getShape();
  auto rhsShape = op.getDpsInputOperand(1)->get().getType().getShape();
  
  int64_t M = lhsShape[0];
  int64_t K = lhsShape[1];
  int64_t N = rhsShape[1];
  
  // 启发式规则：
  // 1. 矩阵足够大（HMX 有启动开销）
  if (M < 32 || N < 32 || K < 32) {
    return false;  // 太小，用 HVX 或 Scalar 更好
  }
  
  // 2. 数据类型必须是 F16 → F32
  if (!isF16Type(lhsType) || !isF16Type(rhsType) || !isF32Type(outType)) {
    return false;
  }
  
  // 3. 形状必须对齐到 HMX 块大小（32x32）
  // 如果不对齐，可能有性能损失
  
  return true;
}

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    
    // 使用启发式规则
    if (!shouldUseHMX(op)) {
      return failure();  // 不转换，让其他 pass 处理
    }
    
    // 转换为 hexkl.matmul
    // ...
  }
};
```

### 方案 2: 基于 Pass 选项的细粒度控制

**思路**：扩展 `HexagonOptions` 来支持更细粒度的控制。

```python
# 在 hexagon_options.py 中添加
@dataclass(frozen=True)
class HexagonOptions:
    # ... 现有选项 ...
    
    # 新增：算子级别控制
    hmxMatmulMinSize: int = 32          # HMX matmul 最小尺寸
    hmxMatmulMaxSize: int = 4096        # HMX matmul 最大尺寸
    hmxMatmulSizeThreshold: int = 128   # 尺寸阈值
    
    # 新增：选择性启用
    enableHexKLForLargeMatmul: bool = True   # 只对大矩阵启用 HMX
    enableHexKLForSmallMatmul: bool = False  # 小矩阵不用 HMX
```

```cpp
// 在 MatmulToHexKLPass 中使用这些选项
struct MatmulToHexKLPass : public ::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  MatmulToHexKLPass(const MatmulToHexKLOptions &options) : options_(options) {}
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToHexKL>(&getContext(), options_);
    // ...
  }
  
private:
  MatmulToHexKLOptions options_;
};
```

### 方案 3: 多阶段编译（最灵活）

**思路**：将模型分解为多个子图，每个子图使用不同的编译选项。

```python
# 伪代码示例
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = MatMul()  # 大矩阵，用 HMX
        self.matmul2 = MatMul()  # 小矩阵，用 HVX
    
    def forward(self, x):
        # 分别编译
        y1 = self.compile_with_hmx(self.matmul1, x)
        y2 = self.compile_with_hvx(self.matmul2, y1)
        return y2

# 编译函数
def compile_with_hmx(model, inputs):
    options = HexagonOptions(enableHexKL=True, enableVectorization=True)
    return compile_and_run(model, inputs, options)

def compile_with_hvx(model, inputs):
    options = HexagonOptions(enableHexKL=False, enableVectorization=True)
    return compile_and_run(model, inputs, options)
```

## 当前限制和挑战

### 1. HexKL 的限制
- **数据类型**：只支持 F16 输入 → F32 输出
- **块大小**：HMX 硬件块大小是 32x32，不对齐会有性能损失
- **启动开销**：HMX 有配置开销，小矩阵可能不划算

### 2. 编译器架构限制
- **全局 Pass 管道**：当前的 Pass 管道是线性的，所有算子经过相同的 Pass
- **缺少 Cost Model**：没有自动的性能模型来决定使用哪种执行模式
- **缺少 Annotation 机制**：torch-mlir 导出时不保留算子级别的 hint

### 3. 运行时限制
- **单一执行模式**：生成的 `.so` 文件中，所有算子使用相同的执行模式
- **无动态切换**：运行时无法根据输入大小动态选择执行模式

## 实际可行的方案

### 短期方案：基于尺寸的启发式规则

**实现难度**：中等  
**效果**：可以自动为大矩阵使用 HMX，小矩阵使用 HVX

**步骤**：
1. 修改 `MatmulToHexKLPass.cpp`，添加尺寸检查
2. 添加 `HexagonOptions` 中的阈值参数
3. 在 Python 层传递这些参数

**代码示例**：
```python
# 在 test_matmul_benchmark.py 中
def run_matmul_with_selective_hmx(m, n, k):
    model = MatMul()
    inputs = [torch.randn(m, k, dtype=torch.float16),
              torch.randn(k, n, dtype=torch.float16)]
    
    # 根据尺寸选择编译选项
    if m >= 128 and n >= 128 and k >= 128:
        options = HexagonOptions(
            enableHexKL=True,
            enableVectorization=True,
            hmxMatmulMinSize=128  # 新增参数
        )
    else:
        options = HexagonOptions(
            enableHexKL=False,
            enableVectorization=True
        )
    
    return TorchMLIRHexagonLauncher().run_torch_mlir(
        mlir_bytecode_path, inputs, func_name, options=options.__dict__
    )
```

### 中期方案：子图分割 + 多次编译

**实现难度**：高  
**效果**：可以为不同算子使用不同执行模式

**思路**：
1. 分析模型，识别所有 matmul 操作
2. 根据尺寸/数据类型将模型分割为多个子图
3. 每个子图使用不同的 `HexagonOptions` 编译
4. 在 Python 层组合执行结果

### 长期方案：完整的算子级别 Annotation 系统

**实现难度**：很高  
**效果**：完全的算子级别控制

**需要的工作**：
1. 扩展 torch-mlir 以支持自定义 attribute
2. 修改所有相关的 MLIR Pass 以识别这些 attribute
3. 实现 Cost Model 来自动决定执行模式
4. 添加运行时支持来处理混合执行模式

## 结论

### 当前状态
- ✅ **模型级别控制**：完全支持，通过 `HexagonOptions` 全局控制
- ❌ **算子级别控制**：不支持，所有算子使用相同的执行模式
- ❌ **选择性 HMX**：不支持，要么全部用 HMX，要么全部不用

### 可行性评估

| 方案 | 可行性 | 实现难度 | 预期效果 |
|------|--------|----------|----------|
| 基于尺寸的启发式 | ✅ 高 | 中等 | 自动优化大/小矩阵 |
| 子图分割 | ✅ 中 | 高 | 灵活但复杂 |
| 完整 Annotation 系统 | ⚠️ 低 | 很高 | 最优但工程量大 |

### 推荐方案

**对于你的需求**（指定某些 MatMul 在 HMX 上，其他在 Scalar/HVX 上）：

1. **最快实现**：修改 `MatmulToHexKLPass.cpp`，添加基于尺寸的启发式规则
   - 修改 1 个 C++ 文件
   - 添加几个 `HexagonOptions` 参数
   - 可以在 1-2 天内完成

2. **更灵活的方案**：实现子图分割
   - 在 Python 层分析模型
   - 为不同的 matmul 生成不同的 MLIR 模块
   - 分别编译和执行
   - 需要 1-2 周

3. **长期方案**：等待或贡献完整的 Annotation 系统
   - 需要修改 torch-mlir 和 hexagon-mlir
   - 工程量大，需要几个月

## 示例代码：实现基于尺寸的选择性 HMX

### 修改 MatmulToHexKLPass.cpp

```cpp
// 在文件开头添加
#include "llvm/ADT/SmallVector.h"

namespace {

// 添加辅助函数
static bool shouldConvertToHexKL(linalg::MatmulOp op, 
                                  int64_t minSize, 
                                  int64_t maxSize) {
  // 获取输入形状
  auto lhsType = op.getDpsInputOperand(0)->get().getType()
                    .dyn_cast<ShapedType>();
  auto rhsType = op.getDpsInputOperand(1)->get().getType()
                    .dyn_cast<ShapedType>();
  auto outType = op.getOutputs()[0].getType().dyn_cast<ShapedType>();
  
  if (!lhsType || !rhsType || !outType)
    return false;
  
  // 检查数据类型
  if (!lhsType.getElementType().isF16() || 
      !rhsType.getElementType().isF16() ||
      !outType.getElementType().isF32())
    return false;
  
  // 检查形状
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
    return false;  // 动态形状暂不支持
  
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  
  int64_t M = lhsShape[0];
  int64_t K = lhsShape[1];
  int64_t N = rhsShape[1];
  
  // 尺寸检查
  if (M < minSize || N < minSize || K < minSize)
    return false;  // 太小
  
  if (maxSize > 0 && (M > maxSize || N > maxSize || K > maxSize))
    return false;  // 太大
  
  return true;
}

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToHexKL(MLIRContext *ctx, int64_t minSize, int64_t maxSize) 
      : OpRewritePattern(ctx), minSize_(minSize), maxSize_(maxSize) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // 检查是否应该转换
    if (!shouldConvertToHexKL(op, minSize_, maxSize_)) {
      return failure();  // 不转换，保留为 linalg.matmul
    }
    
    // 转换为 hexkl.matmul
    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];
    rewriter.replaceOpWithNewOp<hexkl::MatmulOp>(op, C.getType(), A, B, C);
    return success();
  }

private:
  int64_t minSize_;
  int64_t maxSize_;
};

void populateMatmulToHexKLPatterns(RewritePatternSet &patterns,
                                    int64_t minSize,
                                    int64_t maxSize) {
  patterns.add<MatmulToHexKL>(patterns.getContext(), minSize, maxSize);
}

struct MatmulToHexKLPass : public ::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  // 添加选项
  Option<int64_t> minSize{*this, "min-size",
                          llvm::cl::desc("Minimum matrix size for HexKL"),
                          llvm::cl::init(32)};
  
  Option<int64_t> maxSize{*this, "max-size",
                          llvm::cl::desc("Maximum matrix size for HexKL (-1 for no limit)"),
                          llvm::cl::init(-1)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMatmulToHexKLPatterns(patterns, minSize, maxSize);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
```

### 修改 HexagonOptions

```python
# 在 hexagon_options.py 中添加
@dataclass(frozen=True)
class HexagonOptions:
    # ... 现有选项 ...
    
    # 新增：HexKL 尺寸控制
    hexklMinSize: int = 32      # HMX 最小矩阵尺寸
    hexklMaxSize: int = -1      # HMX 最大矩阵尺寸 (-1 表示无限制)
```

### 使用示例

```python
# 在 test_matmul_benchmark.py 中
def run_matmul_with_selective_hmx(m, n, k):
    model = MatMul()
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    inputs = [a, b]
    
    func_name = "MatMul"
    linalg_filename = Path(__file__).parent / f"MatMul_{m}x{k}x{n}.mlirbc"
    
    # 创建 MLIR 模块
    mlir_module = create_linalg_module(model, inputs, func_name)
    write_bytecode_to_file(mlir_module, linalg_filename)
    
    # 配置选项：只对大矩阵使用 HMX
    options = HexagonOptions(
        enableVectorization=True,
        enableHexKL=True,
        enableVTCMTiling=True,
        hexklMinSize=128,  # 只有 >= 128 的矩阵才用 HMX
        hexklMaxSize=-1    # 无上限
    ).__dict__
    
    # 运行
    output = TorchMLIRHexagonLauncher().run_torch_mlir(
        str(linalg_filename),
        inputs,
        func_name,
        options=options
    )
    
    return output

# 测试不同尺寸
results = {}
for size in [64, 128, 256, 512]:
    print(f"\nTesting {size}x{size}x{size}")
    output = run_matmul_with_selective_hmx(size, size, size)
    results[size] = output
    
    # 64x64 会用 HVX（因为 < 128）
    # 128x128 及以上会用 HMX
```

## 总结

1. **当前不支持算子级别的执行模式控制**，所有算子使用相同的编译选项
2. **可以通过修改 MatmulToHexKLPass 实现基于启发式的选择**，这是最快的方案
3. **HMX 有明确的限制**：F16→F32，32x32 块大小，适合大矩阵
4. **推荐实现基于尺寸的自动选择**，让编译器自动决定哪些 matmul 用 HMX

如果需要更细粒度的控制，需要更大的工程改动，包括修改 torch-mlir 导出、MLIR Pass 管道、以及运行时系统。
