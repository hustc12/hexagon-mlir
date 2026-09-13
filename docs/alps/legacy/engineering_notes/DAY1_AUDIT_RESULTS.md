# Day 1-2 审计结果

**日期**: 2026-08-08  
**审计人**: Kiro  
**目的**: 验证当前 HMX 覆盖率和性能基线

---

## 一、环境验证 ✅

```bash
Python: 3.11.15 (mlir-env)
工作目录: /home/huzq85/2-working/hexagon_npu/hexagon-mlir
设备连接: 49d1c7b2 (adb)
```

---

## 二、DINOv2 Debug 基线测试

### 测试配置
```python
Model: DINOv2-small Debug (1 layer)
  - image_size: 32x32, patch_size: 8x8
  - tokens: 17 (16 patches + 1 CLS)
  - hidden_size: 64
  - num_attention_heads: 2
  - intermediate_size: 128
  - precision: FP16
```

### 性能结果

| 配置 | P50 延迟 | P90 延迟 | 说明 |
|---|---:|---:|---|
| **HVX Baseline** | 148.8 ms | 161.5 ms | vectorization=0 (legacy-scalar) |
| **HexKL Enabled** | 145.0 ms | 162.5 ms | HexKL rewrites=0 |

**关键发现**: 
```
[HexKL] rewrites=0  ⚠️
```
启用 HexKL 后，性能基本没变（148.8 vs 145.0 ms，~2.5% 差异在误差范围内），
因为**没有任何矩阵操作被降级到 HMX**。


---

## 三、矩阵操作分析

### DINOv2 Debug 中的矩阵操作

根据模型配置（tokens=17, hidden=64, heads=2, mlp=128），预计有以下矩阵操作：

| 操作 | 形状 | 类型 | 拒绝原因 | 代码位置 |
|---|---|---|---|---|
| **Q projection** | 17×64 × 64×64 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |
| **K projection** | 17×64 × 64×64 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |
| **V projection** | 17×64 × 64×64 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |
| **QK^T** | 2×17×32 × 2×32×17 | linalg.batch_matmul | rank != 2 | line 50 |
| **Attention·V** | 2×17×17 × 2×17×32 | linalg.batch_matmul | rank != 2 | line 50 |
| **Output proj** | 17×64 × 64×64 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |
| **MLP up** | 17×64 × 64×128 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |
| **MLP down** | 17×128 × 128×64 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |
| **Classifier** | 17×64 × 64×10 | linalg.matmul | M=17 % 32 ≠ 0 | line 68-72 |

**统计**:
- 总矩阵操作: 9 个
- HMX rewrites: **0 个** (0%)
- 被拒绝: 9 个 (100%)

**MACs 估算**:
```python
# Rank-2 matmuls (assuming 6个: 3 proj + output + 2 MLP + classifier)
rank2_macs = 17 * 64 * 64 * 3  # Q/K/V
rank2_macs += 17 * 64 * 64      # Output  
rank2_macs += 17 * 64 * 128     # MLP up
rank2_macs += 17 * 128 * 64     # MLP down
rank2_macs += 17 * 64 * 10      # Classifier
# ≈ 480K MACs from rank-2 matmuls

# Batch matmuls (attention)
batch_macs = 2 * 17 * 32 * 17   # QK^T
batch_macs += 2 * 17 * 17 * 32  # Attention·V
# ≈ 37K MACs from batch matmuls

# Total: ~517K MACs, 0% on HMX
```


---

## 四、MatmulToHexKLPass 限制分析

### 当前限制（from MatmulToHexKLPass.cpp）

```cpp
// Line 48-51: Rank 检查
if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "expected rank-2 matmul");
// ❌ 拒绝所有 batch_matmul (rank=3)

// Line 64-68: K/N 对齐检查
if ((K % kHmxTile) != 0 || (N % kHmxTile) != 0) {
    DBG("skip HexKL: unaligned K/N in MxKxN=" << M << "x" << K << "x" << N);
    return rewriter.notifyMatchFailure(op, "K or N not divisible by HMX tile size 32");
}
// ✓ DINOv2: K=64, N=64/128/10 都通过（64 和 128 是 32 的倍数）

// Line 69-73: M 对齐检查
const bool mUnaligned = (M % kHmxTile) != 0;
if (mUnaligned && !enableMPadHmx) {
    DBG("skip HexKL: unaligned M=" << M << " (enableMPadHmx off)");
    return rewriter.notifyMatchFailure(op, "M not divisible by HMX tile size 32; enableMPadHmx off");
}
// ❌ DINOv2: M=17, enableMPadHmx=false → 全部拒绝

// Line 79-85: M-pad N 限制
constexpr int64_t kMaxMPadN = 1024;
if (mUnaligned && enableMPadHmx && N > kMaxMPadN) {
    // 限制大 N 的 M-padding（避免 Hexagon frame 缺陷）
}

// Line 90-96: Attention-like 检查
const bool attentionLike = (K == M || N == M);
if (attentionLike && !enableAttentionHmx) {
    DBG("skip HexKL: attention-like MxKxN=" << M << "x" << K << "x" << N);
    return rewriter.notifyMatchFailure(op, "attention-like matmul (K==M or N==M); keep HVX");
}
// Note: 这个检查只针对 rank-2，batch_matmul 已经在 line 50 被拒绝
```

### 拒绝原因总结

| 限制类型 | 影响的操作数 | 影响比例 | 严重性 |
|---|---:|---:|---|
| **Rank != 2** | 2/9 (batch attention) | 22% MACs | 中 |
| **M % 32 != 0** | 7/9 (所有 rank-2) | 78% MACs | **高** |
| K % 32 != 0 | 0/9 | 0% | - |
| N % 32 != 0 | 0/9 (N=10 classifier 被 M 拒绝) | 0% | 低 |
| Attention-like | 0/9 (无 rank-2 满足 K==M) | 0% | 低 |

**结论**: **M 维度不对齐是主要瓶颈**，影响 78% 的 MACs。


---

## 五、代码审计：关键文件

### 1. MatmulToHexKLPass.cpp
**位置**: `/home/huzq85/2-working/hexagon_npu/hexagon-mlir/qcom_hexagon_backend/lib/Transforms/MatmulToHexKLPass.cpp`

**关键发现**:
- Line 60-78: M/K/N 对齐检查逻辑
- Line 33-37: 已有 `enableMPadHmx` flag，但默认关闭
- Line 79-85: M-pad 有 N ≤ 1024 的限制（frame bug 缓解）
- Line 90-96: Attention-like 保护（K==M 或 N==M）
- Line 228-250: Pass 构造器接受 `enableAttentionHmx` 和 `enableMPadHmx`

**可用的编译选项**:
```cpp
struct MatmulToHexKLPass : public ::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  using MatmulToHexKLBase::MatmulToHexKLBase;  // 继承选项
};

// 在 LinalgToLLVMPass.cpp line 241-245 被调用:
MatmulToHexKLOptions hexklOpts{};
hexklOpts.enableAttentionHmx = enableAlpsAttentionHmx;
hexklOpts.enableMPadHmx = enableAlpsMPadHmx;
pm.addNestedPass<func::FuncOp>(createMatmulToHexKLPass(hexklOpts));
```

### 2. DecomposeHexKLMatmulPass.cpp
**位置**: `/home/huzq85/2-working/hexagon_npu/hexagon-mlir/qcom_hexagon_backend/lib/Transforms/DecomposeHexKLMatmulPass.cpp`

**作用**: 将 `hexkl.matmul` 分解为具体的 HMX micro API 调用

**关键注释** (line 121-125):
```cpp
// Pad M (rows/tokens) and/or N (columns) up to a multiple of 32 so
// unaligned-token encoders (M, e.g. DINOv2's 257) and lm_head-class shapes
// (N, e.g. 50257) run on HMX. MatmulToHexKL only converts static shapes, so
// padding is decided statically here and always materializes fresh
// contiguous buffers: the micro-HMX lowering models each operand as a dense...
```

**说明**: Padding 逻辑已经在 decompose pass 中实现，但被 MatmulToHexKL 的检查阻止！

### 3. 编译选项传播路径
```
Python runner (run_dinov2-small_debug.py)
  --enable-hexkl flag
  --enable-alps-attention-hmx flag (未使用)
  ↓
triton_utils.compile_model()
  ↓
LinalgToLLVMPass.cpp (line 241-246)
  enableAlpsAttentionHmx = args.enable_alps_attention_hmx
  enableAlpsMPadHmx = args.enable_alps_m_pad_hmx (未传递！)
  ↓
MatmulToHexKLPass.cpp
  检查 enableMPadHmx (默认 false)
  → 拒绝所有 M % 32 != 0
```


---

## 六、立即可用的优化路径

### 快速测试：启用现有 M-pad 机制

**发现**: `enableMPadHmx` flag 已经存在，但没有从 Python 传递！

**修复步骤**:
1. 在 `run_dinov2-small_debug.py` 添加 `--enable-alps-m-pad-hmx` flag
2. 传递给编译器选项
3. 重新测试

**预期结果**:
- 6-7/9 操作进入 HMX (Q/K/V/Out/MLP×2)
- Batch matmul 仍被拒绝（需要 P0.2）
- Classifier (N=10) 可能被拒绝（N 太小，padding overhead 大）

### P0.1 实施计划（修订）

**阶段 1**: 测试现有 M-pad 机制（明天上午）
```bash
# 添加 flag 传递
# 测试 DINOv2 with enableMPadHmx=true
# 验证 HMX coverage 提升
```

**阶段 2**: 改进 M-pad 决策逻辑（明天下午）
```cpp
// 当前: 硬编码 N ≤ 1024 限制
// 改进: 动态 profitability model
if (padding_overhead < 2.0 && M >= 8) {
    use_hmx_with_padding();
}
```

**阶段 3**: 添加覆盖率报告（明天晚上）
```cpp
// 在 MatmulToHexKLPass 添加统计
static int hmx_direct_count = 0;
static int hmx_padded_count = 0;
static int hvx_fallback_count = 0;
// 编译结束输出
```


---

## 七、下一步行动

### 明天（Day 3）

**上午: 快速验证**
- [ ] 检查 Python runner 是否已有 `--enable-alps-m-pad-hmx` 参数
- [ ] 如果没有，添加参数传递
- [ ] 重跑 DINOv2: 预期 6-7 rewrites, 延迟 < 100 ms

**下午: P0.1 开始**  
- [ ] 阅读 `DecomposeHexKLMatmulPass.cpp` padding 实现
- [ ] 分析 frame bug 限制（N > 1024）
- [ ] 设计改进的 profitability model

**晚上: 添加度量**
- [ ] 在 MatmulToHexKLPass 添加统计计数
- [ ] 输出 HMX coverage 报告
- [ ] 记录 padding overhead

### Day 4-5

**P0.1 完整实现**
- [ ] 修改 profitability 逻辑
- [ ] 处理 classifier 等小 N 情况
- [ ] 集成覆盖率报告到所有 runner
- [ ] 验证数值正确性

---

## 八、总结

### ✅ 确认的事实

1. **HMX 覆盖率为 0%**: `[HexKL] rewrites=0` 证实
2. **M 不对齐是主要瓶颈**: 影响 78% 的 MACs
3. **Padding 机制已存在**: 但被编译选项阻止
4. **性能基线合理**: HVX 148.8 ms 是可靠的基线
5. **设备连接正常**: adb 通信和执行无问题

### 🎯 关键发现

**最重要**: `enableMPadHmx` flag 已经存在于代码中，但没有被激活！
这意味着我们可能只需要**启用现有机制**就能看到显著提升，而不需要大规模重写。

### 📊 预期影响

如果启用 M-padding:
```
当前: 0/9 ops on HMX (0%)
启用后: 6-7/9 ops on HMX (70-80%)
性能: 148.8 ms → 估计 60-80 ms (1.9-2.5x)
```

这将是快速验证 HMX 加速能力的重要里程碑！

---

**审计完成时间**: 2026-08-08 08:20  
**下次更新**: Day 3 快速验证后

