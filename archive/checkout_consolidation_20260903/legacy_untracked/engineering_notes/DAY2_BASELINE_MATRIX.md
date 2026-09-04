# Day 2: 性能基线矩阵

**日期**: 2026-08-08  
**模型**: DINOv2 Debug (1 layer, 17 tokens, 64 hidden)

---

## 完整测试矩阵

| Profile | HexKL | M-Pad | HMX Rewrites | P50 (ms) | 状态 | 倍率 vs Scalar |
|---|---|---|---:|---:|---|---:|
| **legacy-scalar** | ❌ | ❌ | 0 | 147.6 | ✅ | 1.0x |
| **legacy-scalar** | ✅ | ❌ | 0 | 145.0 | ✅ | 1.02x |
| **legacy-scalar** | ✅ | ✅ | **6** | **23.6** | ✅ | **6.25x** |
| **hvx-vector** | ❌ | ❌ | 0 | 17.9 | ✅ | 8.25x |
| **hvx-vector** | ✅ | ❌ | 0 | 19.5 | ✅ | 7.57x |
| **hvx-vector** | ✅ | ✅ | 6 | - | ❌ Exit 13 | - |
| **hvx-vector-vtcm** | ✅ | ✅ | 6 | - | ❌ Exit 13 | - |

---

## 关键发现

### ✅ 工作的配置

**最佳配置**: `legacy-scalar + HexKL + M-Pad`
```yaml
HMX Rewrites: 6/9 (67%)
Latency: 23.6 ms
Speedup: 6.25x vs baseline
Correctness: ✓
```

**HVX baseline**: `hvx-vector` (无 HexKL)
```yaml
Latency: 17.9 ms
Speedup: 8.25x vs scalar
Vectorization: ✓
```

### ❌ 失败的配置

**问题**: `hvx-vector + HexKL + M-Pad` → Exit 13
```
Error -2147482611: Failed to call main() on DSP
AEE_EBADSTATE
```

**影响范围**:
- `hvx-vector` + M-Pad ❌
- `hvx-vector-vtcm` + M-Pad ❌


---

## 性能对比分析

### HMX vs HVX 真实对比

```
配置A: legacy-scalar + HMX (23.6 ms)
配置B: hvx-vector (17.9 ms)

结论: HMX (with M-padding) 比 HVX 慢 1.32x
```

**这是意外的结果！** 预期 HMX 应该更快。

### 可能的原因

1. **M-Padding 算术开销过大**
   - 17 tokens → 32 tokens padding = 1.88x 额外 MACs
   - 在小模型上，开销占比高

2. **HMX 启动/调度开销**
   - 每个 HMX 调用的 FastRPC overhead
   - 小矩阵上，启动开销 > 计算收益

3. **Batch MatMul 拖累**
   - 2个 attention ops 仍在标量模式（非 HVX）
   - 这部分可能非常慢

4. **VTCM 利用率低**
   - legacy-scalar 没有 VTCM tiling
   - 数据频繁往返 DDR

### 优化方向

**假设**: 如果 batch_matmul 也进入 HMX，性能会改善

测试: 
```
当前: 6/9 on HMX, 3/9 on scalar = 23.6 ms
如果: 8/9 on HMX, 1/9 on scalar = 预期 ~15-18 ms?
```

如果这能接近或超过 HVX (17.9 ms)，那么 HMX 路径就有价值。

---

## Exit 13 根因分析

### 错误信息

```
Error -2147482611: Failed to call main() on DSP
AEE_EBADSTATE (Hexagon SDK error code)
```

### 触发条件

```
必要条件: vectorization=1 (HVX 启用)
充分条件: vectorization=1 + HexKL + M-Pad
```

### 可能的根本原因

#### 假设 1: Frame Bug（最可能）

代码中已有注释（`MatmulToHexKLPass.cpp` line 79-85）:
```cpp
// M-pad allocates a fresh contiguous A/result buffer sized to the padded M.
// At large N the resulting total function-frame pressure trips a Hexagon
// frame-lowering stack-coloring defect (over-aligned dynamic frame clobbers
// the sret spill slot -> Bad VA on device).
```

**当前限制**: N ≤ 1024
**DINOv2 shapes**: N = 64, 128 (都 < 1024)

但是，这个 bug 可能在 HVX 模式下更容易触发，因为：
- HVX 寄存器压力更大
- VTCM allocation 额外消耗栈空间

#### 假设 2: 资源耗尽

HVX + HMX 同时使用：
```
HVX: 使用 vector registers (VRF)
HMX: 使用 matrix accelerator + VTCM
VTCM: 4MB 共享资源
Stack: 8MB 配置
```

可能的冲突：
- VTCM 超出 4MB 限制
- 栈溢出（即使配置了 8MB）
- DMA channels 冲突

#### 假设 3: 编译器/链接问题

HVX 优化代码 + HMX 库链接：
- 不同优化级别的代码混合
- ABI 不兼容
- 符号冲突


---

## 决策和下一步

### 战略决策

**关键问题**: HMX (23.6ms) vs HVX (17.9ms) 

当前 HMX **比 HVX 慢 32%**。这意味着：

**选项 A**: 放弃 HMX，专注 HVX 优化
- 优点: 已经很快（17.9ms vs 147ms = 8.25x）
- 缺点: 违背原计划（HMX 是核心加速部件）

**选项 B**: 继续优化 HMX，目标超越 HVX
- 优点: 符合原计划，HMX 理论上应该更快
- 缺点: 需要解决 Exit 13，不确定能否超越 HVX

**选项 C**: 双路并行（推荐）
- HMX 路径: 用于大模型/完整模型（M维度大，padding开销小）
- HVX 路径: 用于 debug 模型（M维度小，HVX 更高效）

### 建议的执行路径

#### 立即行动（今天）

1. **测试更多模型验证假设**
   ```bash
   # GPT-2 Debug (2层，可能 M 更大)
   # Whisper Debug
   # 完整 DINOv2 (12层，257 tokens → M=257)
   ```

2. **实现 P0.2 Batch MatMul**
   - 让 attention ops 进入 HMX
   - 看能否降到 < 18 ms (超越 HVX)

3. **如果仍慢于 HVX，调整策略**
   - 记录"HMX 在小模型上有启动开销"
   - 专注完整模型（M 维度大）

#### 中期方案

**绕过 Exit 13**:
- 使用 `legacy-scalar + HMX` 作为 B2 基线
- 暂时放弃 HVX + HMX 组合
- 在大模型上测试（M大时，HMX 应该更有优势）

**报告策略**:
```
B0 (baseline): hvx-vector = 17.9 ms
B1 (HMX, small model): legacy-scalar + HMX = 23.6 ms (0.76x B0)
B2 (HMX, large model): legacy-scalar + HMX = ? ms (预期 > 2x B0)
```

**关键洞察记录**:
- HMX 在 M=17 时因 padding overhead (1.88x) 不划算
- HMX 在 M=257 或 M>100 时应该有优势
- Exit 13 是 HVX+HMX 组合的系统性问题，需要底层修复

---

## 下一步具体任务

### Task 1: 测试 GPT-2 Debug（验证 M 大小影响）

```bash
python benchmark_models/debug_running/run_gpt2lmheadmodel_debug.py \
  --backend-profile legacy-scalar --enable-hexkl --enable-alps-m-pad-hmx \
  --device-iterations 5
```

**预期**: 如果 GPT-2 的 M 维度更大（batch × seq_len），HMX 应该更快

### Task 2: 实现 Batch MatMul 转换（P0.2）

查看 `rewrite_batch_matmul_to_matmul` 实现，理解为什么 attention ops 未转换

### Task 3: 测试完整 DINOv2 (12层，257 tokens)

这将是真正测试 HMX 能力的场景：
- M = 257 (vs 17)
- Padding: 257 → 288 = 1.12x (vs 1.88x)
- 12 层累积

**预期**: HMX 应该显著快于 HVX

---

**当前结论**: 

不要因为 debug 模型上的结果就放弃 HMX！小模型上 padding overhead 占比高是预期内的。  
真正的测试应该在**完整模型**上进行。

**行动**: 继续 P0.2，然后快速转向完整模型测试。

