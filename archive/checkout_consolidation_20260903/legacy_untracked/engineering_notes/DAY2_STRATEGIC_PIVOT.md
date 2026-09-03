# Day 2: 战略调整 - 小模型 vs 大模型

**日期**: 2026-08-08
**关键发现**: HMX 在小模型上不一定比 HVX 快

---

## 🔍 核心发现

### 性能对比（DINOv2 Debug, M=17）

```
HVX (vectorization):        17.9 ms  ← 最快！
HMX (M-padding):            23.6 ms  ← 比 HVX 慢 32%
Scalar (无优化):            147.6 ms ← 基线
```

**结论**: 在 M=17 的小模型上，**HVX > HMX**

### 为什么 HMX 更慢？

**Padding Overhead 分析**:
```python
Original: 17 × 64 × 64 = 69,632 MACs  
Padded:   32 × 64 × 64 = 131,072 MACs
Overhead: 1.88x 额外计算

# 6个 padded matmuls:
Total overhead ≈ 1.88x on 67% of ops
Effective slowdown ≈ 1.25-1.3x
```

**其他开销**:
- HMX FastRPC 调用开销（CPU ↔ DSP）
- 小矩阵上，启动成本 > 计算收益
- Legacy-scalar 模式没有 VTCM tiling（频繁 DDR 访问）

---

## 📊 M 维度的影响

### Padding Overhead vs M

| M | Padded M | Overhead | 场景 |
|---:|---:|---:|---|
| 17 | 32 | 1.88x | DINOv2 Debug (small) |
| 32 | 32 | 1.0x | 完美对齐 |
| 64 | 64 | 1.0x | 完美对齐 |
| 100 | 128 | 1.28x | 中等模型 |
| 257 | 288 | 1.12x | **DINOv2 Full** |
| 512 | 512 | 1.0x | 大 batch |
| 1000 | 1024 | 1.024x | 很大 |

**关键洞察**: 
- M < 32: Overhead > 1.5x（HMX 可能不划算）
- M ≈ 100-200: Overhead ≈ 1.1-1.3x（HMX 开始有优势）
- M > 256: Overhead < 1.15x（HMX 应该明显更快）

### 完整模型的优势

**DINOv2-small Full** (12 layers, 257 tokens):
```python
M = 257 → 288 padding = 1.12x overhead
Layers = 12 (vs 1 in debug)
Total MACs ≈ 12x more

# 预期:
# - Padding overhead 小（1.12x vs 1.88x）
# - HMX 并行度更高
# - 启动开销摊销到更多计算
```

---

## 🎯 战略调整

### 修订后的目标

**原目标**: 在所有模型上用 HMX 超越 HVX

**新目标**: 
1. **小模型（M < 50）**: 承认 HVX 更优，作为参考基线
2. **大模型（M > 100）**: HMX 应该超越 HVX，这是主要战场
3. **完整模型**: 证明 2x+ 加速（这才是用户关心的）

### 优先级调整

**P0（本周）**: 完整模型支持和测试
- 修复所有完整模型 runner 的参数传递
- 测试 DINOv2 Full (12层, M=257)
- 测试 GPT-2 Full (12层)
- 建立完整模型的 B0-B2 基线

**P1（下周）**: 优化 HMX
- P0.2 Batch MatMul（在完整模型上测试）
- 解决 HVX + HMX Exit 13（如果时间允许）
- 添加覆盖率报告

**P2（后续）**: 小模型优化
- 如果有时间，探索小 M 的特殊处理
- 或者承认小模型用 HVX 就好

---

## 📋 立即行动计划

### Task 1: 修复完整 DINOv2 runner（30 分钟）

文件: `benchmark_models/run_dinov2-small.py`

需要添加（like debug 版本）:
1. 传递 `enable_omnifetch_m_pad_hmx` 到 `hexagon_options_phase4()`
2. 传递 `enable_m_pad` 到 `apply_hexkl_ir_rewrites()`

### Task 2: 测试完整 DINOv2（1 小时）

```bash
# B0: HVX baseline
python benchmark_models/run_dinov2-small.py \
  --backend-profile hvx-vector --device-iterations 3

# B1: legacy-scalar (无 HMX)
python benchmark_models/run_dinov2-small.py \
  --backend-profile legacy-scalar --device-iterations 3

# B2: HMX with M-pad
python benchmark_models/run_dinov2-small.py \
  --backend-profile legacy-scalar --enable-hexkl \
  --enable-omnifetch-m-pad-hmx --device-iterations 3
```

**预期**: B2 应该快于 B0（因为 padding overhead 只有 1.12x）

### Task 3: 测试其他完整模型（2 小时）

按优先级:
1. GPT-2 Full (12层) - LLM 代表
2. Whisper Tiny Full - Audio 代表  
3. Swin Transformer - Vision 代表

---

## 🎓 经验教训

### 什么有效

✅ **M-Padding 机制本身**: 代码实现正确，数值稳定  
✅ **在大 M 上的潜力**: 理论分析支持 HMX 优势  
✅ **快速原型**: Day 1 就验证了机制可行性  

### 什么需要注意

⚠️ **Padding Overhead**: 必须考虑 M 维度大小  
⚠️ **HVX + HMX 组合**: Exit 13 是系统性问题  
⚠️ **Debug vs Full**: Debug 模型不代表真实场景  

### 关键认识

**"加速"是相对的**: 
- HMX vs Scalar: 6.3x ✓
- HMX vs HVX: 0.76x ✗
- 但是 HVX vs Scalar: 8.3x

**真正重要的是**: 用户端到端延迟，不是某个技术路线  
**最终目标**: 完整模型 2x+ 加速（vs 任何合理基线）

---

## 下一步

立即开始 Task 1：修复完整 DINOv2 runner，然后测试真实的大模型性能。

**预测**: 完整 DINOv2 (M=257) 上，HMX 会超越 HVX，证明策略正确性。

