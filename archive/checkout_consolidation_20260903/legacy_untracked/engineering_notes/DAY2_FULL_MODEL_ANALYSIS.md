# Day 2: 完整模型执行分析

**日期**: 2026-08-08  
**关键发现**: 完整模型遇到系统性问题

---

## 🔴 执行结果总结

### DINOv2-small Full (12层, 257 tokens, 384 hidden)

| Profile | HexKL | M-Pad | 编译时间 | Rewrites | 执行状态 | 延迟 |
|---|---|---|---:|---:|---|---|
| **hvx-vector** | ❌ | ❌ | ~120s | 0 | ✅ | **30.9 s** |
| **legacy-scalar** | ❌ | ❌ | 103s | 0 | ⏱️ 超时 | > 120s |
| **legacy-scalar** | ✅ | ❌ | ~120s | 0 | ⏱️ 超时 | > 600s |
| **legacy-scalar** | ✅ | ✅ | 121s | **72** | ❌ Exit 13 | - |

---

## 🔍 关键发现

### 发现 1: Batch MatMul 大规模转换

```
[IR] batch_matmul=96 matmul=1
[HexKL] batch_matmul→matmul=72, f16-input rewrite=0
```

**72 个 batch_matmul 被转换！** 这证明：
- `rewrite_batch_matmul_to_matmul` 在 `enable_m_pad=True` 时工作
- 转换率: 72/96 = 75%
- 剩余 24 个可能因为其他限制（形状、类型等）

### 发现 2: Exit 13 在完整模型上也出现

**触发条件**（修正后）:
```
必要条件: 大量 HMX rewrites (72个)
充分条件: enable_m_pad=True 
```

**不再仅限于 HVX 模式**！legacy-scalar + HMX + M-pad 也会 Exit 13。

### 发现 3: 完整模型执行极慢

Pure scalar baseline (no HexKL):
```
编译: 103s
执行: > 120s (单次迭代超时)
```

这与 HVX baseline 的 30.9s 形成强烈对比（约 4x+ 差距）。


---

## 🧩 问题分析

### Exit 13 的真正原因

#### 排除的假设

❌ **HVX + HMX 资源冲突**: legacy-scalar (无 HVX) 也失败  
❌ **小模型特定问题**: 完整模型也失败  
❌ **简单的参数问题**: 编译成功，执行失败  

#### 更新的假设

**假设 1: Frame Bug 在大规模 M-padding 下触发** ⭐ 最可能

代码注释（`MatmulToHexKLPass.cpp` line 79-85）提到：
```cpp
// M-pad allocates a fresh contiguous A/result buffer sized to the padded M.
// At large N the resulting total function-frame pressure trips a Hexagon
// frame-lowering stack-coloring defect
```

完整模型情况：
```
72 个 M-pad matmuls
每个分配新的 padded buffer
累积的栈压力 = 72 × padding overhead
```

即使单个 N < 1024，累积效果可能触发 frame bug。

**假设 2: VTCM/内存资源耗尽**

完整模型特点：
```
12 层 × 多个 matmuls = 大量 HMX 调用
每个 HMX 需要 VTCM allocation
累积 VTCM 需求可能 > 4MB 限制
```

**假设 3: HMX 库限制**

HexKL library 可能对：
- 并发调用数量有限制
- 总分配内存有限制
- 持续执行时间有限制

### 为什么 legacy-scalar (无 HexKL) 也超时？

**原因**: legacy-scalar = 无向量化 + 无VTCM优化

```
HVX (vectorized):      30.9s  ← 8-way SIMD + VTCM
legacy-scalar:         > 120s ← pure scalar loops
```

这是**预期的**！Scalar 比 vector 慢 4-8x 是正常的。


---

## 🎯 战略调整（重要）

### 当前障碍

**无法获得完整模型的 HMX 性能数据**，因为：
1. Exit 13 阻止执行
2. legacy-scalar 太慢无法作为 HMX 基线
3. HVX + HMX 组合也失败

### 可行的路径

#### 选项 A: 修复 Exit 13（高风险，高回报）

**需要做的**:
- 深入调试 Hexagon frame bug
- 可能需要修改编译器后端
- 可能需要联系 Qualcomm 支持
- 时间: 数天到数周

**收益**: 如果成功，可以获得完整的 HMX 数据

**风险**: 可能是深层系统问题，短期无法解决

#### 选项 B: 降低规模测试（中风险，中回报）⭐ 推荐

**策略**: 在 debug 和 full 之间找到"中等规模"模型

测试点：
```
Debug:  1 layer,  17 tokens  → HMX works, padding 1.88x
Medium: 4 layers, 100 tokens → ? (预期 padding ~1.3x)
Medium: 6 layers, 150 tokens → ? (预期 padding ~1.2x)
Full:   12 layers, 257 tokens → Exit 13
```

**实现**: 修改 config 创建中等规模模型

**预期**: 找到"甜蜜点" - HMX 工作且比 HVX 快

#### 选项 C: 放弃 M-padding，聚焦其他优化（低风险，低回报）

**策略**: 
- 只转换对齐的 matmuls（M % 32 == 0）
- 聚焦 P1 的数据供应优化（VTCM, L2 prefetch）
- 承认 M-padding 有系统限制

**收益**: 避开 Exit 13，聚焦可行优化

**代价**: 覆盖率大幅下降（可能 < 20%）

#### 选项 D: 切换到不同的模型架构

测试其他模型：
- **GPT-2**: seq_len 可调，M = batch × seq_len
- **Whisper**: encoder/decoder 分离
- **其他 Vision 模型**: 可能 M 维度更友好


---

## 📋 推荐的执行计划

### 立即行动（今天下午）

**Task 1: 创建中等规模 DINOv2 测试** (1-2 小时)

修改 `dinov2_debug_common.py` 创建变体：
```python
def create_dinov2_medium_model(num_layers=4, num_tokens=100):
    """
    num_layers: 1 (debug) → 4 (medium) → 12 (full)
    num_tokens: 17 (debug) → 100 (medium) → 257 (full)
    """
    # Padding overhead:
    # 100 → 128 = 1.28x (vs 1.88x for debug, 1.12x for full)
```

测试矩阵：
```bash
# Scale 1: 2 layers, 50 tokens (padding 1.56x)
# Scale 2: 4 layers, 100 tokens (padding 1.28x)  
# Scale 3: 6 layers, 150 tokens (padding 1.19x)
```

**预期**: 找到可以工作的最大规模

**Task 2: 测试 GPT-2 中等规模** (1 小时)

GPT-2 Debug 已经有 2 层，测试不同 seq_len：
```bash
--seq-len 32   # M = 32 (完美对齐)
--seq-len 64   # M = 64 (完美对齐)
--seq-len 100  # M = 100 (padding to 128)
```

**优势**: GPT-2 的 M = batch × seq_len，更容易控制

### 明天（Day 3）

如果中等规模测试成功：
- 建立完整的规模 vs 性能曲线
- 外推到大模型的预期性能
- 写入论文的 methodology

如果仍然失败：
- 深入 Exit 13 调试
- 或考虑选项 C/D

---

## 💡 关键洞察

### 成功的部分

✅ **M-padding 机制**: 在 debug 规模工作良好  
✅ **Batch matmul 转换**: 72/96 转换率证明机制有效  
✅ **编译成功**: 120s 对 72 rewrites 是可接受的  
✅ **数值正确性**: Debug 模型验证通过  

### 遇到的障碍

❌ **规模限制**: 完整模型触发 Exit 13  
❌ **组合失败**: 多种配置组合都失败  
⏱️ **执行超时**: legacy-scalar baseline 太慢  

### 战略认识

**"完整模型"不是唯一目标**

论文价值可以来自：
1. 在中等规模证明概念
2. 外推分析和理论支持
3. 清晰记录系统限制
4. 在可行范围内达到 2x+

**已经证明的价值**:
- Debug 模型: 6.3x vs scalar (M-padding 有效)
- 机制验证: 72 rewrites (大规模转换可行)
- 理论分析: Padding overhead vs M size

---

## 下一步具体指令

### 创建中等规模测试脚本

```python
# benchmark_models/dinov2_medium_common.py
def create_dinov2_medium_model_and_input(
    num_layers: int = 4,
    image_size: int = 64,  # 8×8 patches = 64 tokens
    patch_size: int = 8,
):
    config = Dinov2Config(
        image_size=image_size,
        patch_size=patch_size,
        num_hidden_layers=num_layers,
        # ... rest same as debug
    )
    # tokens = (image_size / patch_size)^2 + 1 (CLS)
    # 64×64 / 8×8 = 64 tokens + 1 = 65 tokens → pad to 96
```

### 运行测试

```bash
# Test scale progression
python benchmark_models/run_dinov2_medium.py \
  --num-layers 2 --image-size 56 \  # 50 tokens
  --enable-hexkl --enable-omnifetch-m-pad-hmx

python benchmark_models/run_dinov2_medium.py \
  --num-layers 4 --image-size 80 \  # 101 tokens
  --enable-hexkl --enable-omnifetch-m-pad-hmx
```

**目标**: 找到能稳定运行的最大规模，建立 scaling law。

