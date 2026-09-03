# OmniFetch HMX 加速优化计划

**目标**: 以 HMX 为核心，实现 2 倍以上推理加速，支持完整模型

**更新日期**: 2026-08-08

**关键认识**:
1. HMX 是主要加速部件（与 HVX 共享相似 memory hierarchy）
2. HMX 只能做矩阵乘法，需要最大化 HMX 覆盖率
3. VTCM 降字节率/使用率是次要衡量指标
4. 当前只有 debug 模型（缩减层数），需要补齐完整模型支持

---

## 一、当前问题诊断

### 1.1 HMX 覆盖率严重不足

根据 `omnifetch-design-audit-and-roadmap.md` 和代码分析：

**DINOv2 Debug 案例**:
```
8 个 linalg.batch_matmul 操作
HexKL rewrites = 0  (没有任何操作降级到 HMX)
```

**根本原因**（来自 `MatmulToHexKLPass.cpp`）:
- 当前 HexKL pass 仅接受 static rank-2 `linalg.matmul`
- 要求 M、K、N 维度都是 32 的倍数
- 默认拒绝 attention-like 形状（K==M 或 N==M）
- batch_matmul 需要先折叠到 2D 且满足对齐要求


**典型被拒绝的 DINOv2 矩阵操作**:
```
Q projection:    1x17x64  * 1x64x64    (M=17 不是32倍数)
K projection:    1x17x64  * 1x64x64    (M=17 不是32倍数)  
V projection:    1x17x64  * 1x64x64    (M=17 不是32倍数)
QK^T:           2x17x32  * 2x32x17    (attention-like, batch>1)
AttentionV:     2x17x17  * 2x17x32    (attention-like, batch>1)
Output proj:    1x17x64  * 1x64x64    (M=17 不是32倍数)
MLP up:         1x17x64  * 1x64x256   (M=17 不是32倍数)
MLP down:       1x17x256 * 1x256x64   (M=17 不是32倍数)
```

**结果**: 大约 78% 的矩阵 MACs 无法使用 HMX！

### 1.2 基线混淆导致收益估计不准

从 `engineering_work.md` 的修复记录：

```
DINOv2 Debug 真实结果:
- Legacy scalar (enableVectorization=false):  174.832 ms
- True HVX vector (修复后):                   21.258 ms  (8.224x 基线提升)
- HexKL (0 HMX rewrites):                    22.232 ms  (≈ HVX, 无 HMX)
- Cumulative item 7 scalar:                  60.927 ms  (比真实 HVX 慢 2.866x)
```

**问题**: 早期报告的大倍率主要来自基线配置错误，而非 OmniFetch 本身。


### 1.3 完整模型支持不足

当前状态（来自 `omnifetch-prefetch-insitu-innovation.md` 的完整模型执行计划）:

| 模型 | Debug 版本 | 完整版本状态 | 主要问题 |
|---|---|---|---|
| GPT-2 | 2 层 ✓ | 12 层 - NaN 输出 | FP16 饱和（约第 3-4 层） |
| Falcon-RW-1B | 2 层 ✓ | 24 层 - 缺失 checkpoint | 只有配置无权重 |
| Qwen2.5-0.5B | 2 层 ✓ | 24 层 - 编译超时 | 约 480s 超时 |
| TinyLlama-1.1B | 2 层 ✓ | 22 层 - 待测 | - |
| DINOv2-small | 1 层 ✓ | 12 层 ✓ | 唯一通过的完整模型 |
| Swin-Tiny | 1 层 ✓ | [2,2,6,2] - 基线 exit 13 | 仅组合通过 |
| SegFormer | 2 stage ✓ | 4 stage - 基线 exit 13 | 仅组合通过 |
| Whisper-tiny | 1L enc+dec ✓ | 4L enc+dec - 基线 exit 13 | 仅组合通过 |
| HuBERT-base | 1 层 ✓ | 12 层 ✓ | **严重回退** (2.924x 慢) |

**结论**: 大多数完整模型要么无法编译/执行，要么性能回退。

### 1.4 L2 预取运行时违反 V73 约束

从 `omnifetch_history.md` § 16.2 的审计发现：

1. **32 个独立命令风暴**: 为一个 32x32 FP16 瓦片发出 32 个 64 字节命令
   - V73 规范: 新命令停止旧命令，大多数行永不驻留
   
2. **32 KB 块请求**: 4 倍于手册推荐的 <8 KB
   
3. **无页边界分割**: 跨页地址被丢弃

4. **无全局仲裁**: 多个流同时发出提示，互相覆盖


---

## 二、修订后的优先级和目标

### 2.1 核心目标重新定义

**主目标**: 2 倍以上端到端推理加速
- **必要条件**: HMX 覆盖率显著提升（目标 >60% 的矩阵 MACs）
- **辅助优化**: VTCM 峰值降低 + 有效预取

**次要目标**: VTCM 效率
- 静态峰值降字节率（已有 63.64% 案例）
- VTCM 使用率监控（占用/容量比）

**支撑目标**: 完整模型支持
- 至少 5 个完整模型达到 2x+
- 覆盖 LLM、Vision、Audio 三个领域

### 2.2 策略调整

**From**: "在现有 HexKL 覆盖上优化数据移动"
**To**: "扩大 HMX 覆盖 + 为 HMX 优化数据供应"

这符合文档中的 "Next Paper Idea" 方向，但我们需要将其作为**当前工作的基础设施**，而非未来工作。


---

## 三、实施计划（按优先级排序）

### P0: HMX 覆盖率扩展（基础设施，2-3 周）

**目标**: 将 HMX 覆盖率从 <20% 提升到 >60%

#### P0.1 非对齐 M/N 维度的 HMX 降级（1 周）

**实现**: 扩展 `MatmulToHexKLPass.cpp` 和 `DecomposeHexKLMatmulPass.cpp`

```cpp
// 当前限制
if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return failure();  // 拒绝
}

// 修改为
if (K % 32 != 0) {
    return failure();  // K 对齐是 HMX 硬约束
}
// M 和 N 可以 padding
int64_t M_padded = (M + 31) / 32 * 32;
int64_t N_padded = (N + 31) / 32 * 32;
```

**处理策略**:
1. **M dimension (tokens/batch)**: 
   - Pad 到 32 的倍数
   - 添加 guard 在最后瓦片使用 predication/mask
   - 输出 slice 回原始 M 维度

2. **N dimension (output channels/vocab)**:
   - Pad 到 32 的倍数  
   - 输出 slice 回原始 N 维度
   
3. **盈利性模型**:
   ```python
   padding_overhead = (M_padded * N_padded) / (M * N)
   if padding_overhead < 2.0 and M >= 8:  # 不超过 2x 算术扩展
       use_HMX_with_padding()
   else:
       use_HVX()
   ```


**验收标准**:
- DINOv2 Debug: 6/9 矩阵操作进入 HMX（Q/K/V/Out/MLP-up/MLP-down）
- GPT-2 Debug: lm_head projection 进入 HMX
- 编译器报告: `hmx_direct=X, hmx_padded=Y, hvx_unprofitable=Z`
- 设备测试: 数值正确性（max error < 1e-3）

#### P0.2 Batch MatMul 到 HMX 的降级（1 周）

**实现**: 新增 `BatchMatmulToHexKLPass.cpp`

```cpp
// Attention batch_matmul 处理
// Input: tensor<BxMxK> * tensor<BxKxN>
// Strategy:
1. 检测是否是 attention (QK^T 或 AV)
2. 如果 batch 小且 M/N 可 padding:
   for b in range(B):
       subview_lhs = tensor.extract_slice [b:b+1, :, :]  
       subview_rhs = tensor.extract_slice [b:b+1, :, :]
       collapse to 2D
       apply P0.1 padding logic
       invoke hexkl.matmul
       slice output
3. 盈利性检查:
   - batch <= 8 (head 数量通常 ≤ 12)
   - padding_overhead < 3.0 (attention 可接受更高开销)
```

**特殊优化 - Attention QK^T**:
- 识别 transpose pattern
- 使用 transpose-aware indexing（item 7 已有机制）
- 避免物化 K 的转置

**验收标准**:
- DINOv2 Debug: QK^T 和 AV 进入 HMX（覆盖率 →8/9）
- Falcon Debug: attention 矩阵进入 HMX
- 保持 item 7 的 K/V 语义跟踪能力


#### P0.3 HMX 覆盖率的独立基线建立（3 天）

**重要**: 这是**基础设施工作**，不是 OmniFetch 贡献

创建独立的 HMX 覆盖率基线:
```
B0: HVX (vectorization=true, VTCM=true)
B1: HexKL/HMX (direct-aligned only, no OmniFetch)
B2: HexKL/HMX (with padding, no OmniFetch) <- P0.1+P0.2 的结果
B3: HexKL/HMX (with padding + OmniFetch items 1-7)
```

**度量指标**:
```python
# 编译器自动计算并报告
total_matmul_macs = sum(M*K*N for all matmuls)
hmx_macs = sum(M*K*N for HMX-converted matmuls)
hmx_coverage = hmx_macs / total_matmul_macs

# VTCM 效率
vtcm_peak_bytes = max(VTCM allocation)
vtcm_capacity = 4 * 1024 * 1024  # V73: 4MB
vtcm_utilization = vtcm_peak_bytes / vtcm_capacity

# 报告格式
[HMX Coverage]
  total_matmul_ops: 9
  hmx_direct: 1
  hmx_padded: 7
  hvx_fallback: 1
  hmx_coverage: 88.7%  (789K / 890K MACs)
  
[VTCM Efficiency]
  peak_bytes: 45056 -> 16384 (coloring)
  capacity: 4194304
  utilization: 0.39% -> 0.39%
  saved_bytes: 28672 (63.64%)
```


---

### P1: OmniFetch for HMX（数据供应优化，2 周）

**目标**: 为扩展后的 HMX 优化数据供应，达到 2x+ 目标

#### P1.1 HMX/HVX 边界驻留（N9 机制，5 天）

**当前问题**: HMX 输出写回 DDR，HVX epilogue 再读回

**优化**:
```
HMX matmul output in VTCM (public ABI)
  -> HVX bias/activation/norm in-place
  -> in-situ AH layout for next HMX
  -> avoid DDR round-trip
```

**实现**:
1. `DecomposeHexKLMatmulPass`: 标记 HMX output 为 VTCM 驻留
2. `PrefetchInsert`: 识别 HVX consumer，避免插入冗余加载
3. 添加 lifetime 分析确保下一个 HMX 可复用 VTCM 槽

**预期收益**: 消除 HMX 输出的 DDR 物化（每层节省数十 KB）

#### P1.2 修复 L2 预取运行时（4 天）

根据 V73 规范修复当前运行时:

```c
// Before: 32 个独立命令
for (int i = 0; i < 32; i++) {
    l2fetch(base + i * 64, 64);  // ❌ 每个命令停止前一个
}

// After: 单个 2D 请求
l2fetch_2d(base, width=64, height=32, stride=source_cols*2);  // ✓
```

**关键修复**:
1. 单个 2D 请求替代命令风暴
2. 页边界分割（每个 fragment ≤ 4KB）
3. 全局 L2 仲裁器（单飞行约束）
4. 请求大小 < 8 KB


#### P1.3 权重静止调度（N1 机制，5 天）

**针对 Prefill 场景的优化**:

```python
# Before: position-major (每个 position 重新加载权重)
for pos in positions:
    for out_ch in output_channels:
        load weight_tile[out_ch]  # 重复加载
        compute(activation[pos], weight_tile[out_ch])

# After: weight-stationary (权重驻留在 VRF/VTCM)
for out_ch_tile in output_channel_tiles:
    load weight_tile[out_ch_tile] into VTCM  # 一次加载
    for pos in position_strip:
        compute(activation[pos], weight_tile[out_ch_tile])
```

**收益**:
- 减少权重 VMEM 访问
- 为下一个权重瓦片的预取创建真正的 overlap window
- 与 item 5 的二维流水线组合

**实现位置**: 
- `ScheduleMatmulForHVXPass.cpp`: 调整循环顺序
- 或在 HexKL decompose 阶段应用

---

### P2: 完整模型支持（2-3 周）

#### P2.1 修复 FP16 数值问题（GPT-2）（1 周）

**问题**: FP16 端到端图在第 3-4 层饱和

**解决方案**（来自 `omnifetch-design-audit-and-roadmap.md` § 6）:
```python
# Mixed precision strategy
- HMX matmuls: FP16 (保持计算效率)
- LayerNorm: FP32 (避免累积误差)
- Softmax: FP32 (避免 exp 溢出)
- Residual accumulation: FP32 (避免逐层误差累积)
```

**验证**: 逐层差分验证，定位首个 NaN 层


#### P2.2 大模型编译优化（1 周）

**问题**: Qwen 24 层约 480s 编译超时

**策略**:
1. **增量编译**: 分离常量处理和主图 pass
2. **Pass 优化**: Profile 并优化慢 pass（likely fusion/tiling）
3. **并行编译**: 独立层可并行编译（如果架构允许）
4. **预编译缓存**: 缓存中间 IR

**目标**: 24 层模型编译时间 < 5 分钟

#### P2.3 完整模型执行修复（1 周）

**策略表**（基于 § 三.1.3）:

| 模型 | 当前问题 | 修复策略 | 优先级 |
|---|---|---|---|
| GPT-2 12L | FP16 NaN | P2.1 mixed precision | 高 |
| Falcon-RW-1B | 缺失权重 | 下载 checkpoint | 中 |
| Qwen2.5 24L | 编译超时 | P2.2 编译优化 | 高 |
| TinyLlama 22L | 待测 | 直接尝试 | 高 |
| Swin [2,2,6,2] | 基线 exit 13 | 修复 HVX config | 中 |
| SegFormer 4-stage | 基线 exit 13 | 修复 HVX config | 中 |
| Whisper 4L+4L | 基线 exit 13 | 修复 HVX config | 中 |
| HuBERT 12L | 性能回退 | 重新评估策略选择 | 高 |

**Exit 13 诊断协议**:
1. 保存 DSP log 和 perf 文件
2. Object audit (HVX%, HMX calls, DMA)
3. 不重复盲跑，记录失败原因


---

## 四、验收标准和时间线

### 4.1 三阶段验收

**阶段 1: HMX 基础设施（3 周后）**

门槛要求:
- [ ] 至少 3 个 debug 模型 HMX 覆盖率 > 60%
- [ ] 所有完整模型都有正确的 B0-B2 基线
- [ ] HMX 覆盖率自动报告集成到所有 runner
- [ ] VTCM 效率指标集成

成功标准:
```
DINOv2 Debug:
  B0 (HVX):           21 ms
  B1 (HMX direct):    20 ms  (覆盖率 11%)
  B2 (HMX padded):    8-12 ms (覆盖率 >70%)  <- 目标
```

**阶段 2: OmniFetch + HMX（6 周后）**

门槛要求:
- [ ] 至少 3 个 debug 模型达到 B3 > 2x B2
- [ ] L2 预取运行时符合 V73 规范
- [ ] 所有 OmniFetch 机制与扩展 HMX 兼容

成功标准:
```
典型模型 (如 DINOv2 Debug):
  B2 (HMX padded):           10 ms
  B3 (HMX + OmniFetch):      4-5 ms  (2-2.5x 加速)

指标:
  - HMX 覆盖率: >60%
  - VTCM 峰值降低: >50%
  - L2 请求减少: >90% (vs. 独立 baseline)
```


**阶段 3: 完整模型生产（10 周后）**

门槛要求:
- [ ] 至少 5 个完整模型可执行（3 个 LLM + 1 Vision + 1 Audio）
- [ ] 所有完整模型有 B0-B3 四行结果
- [ ] 数值正确性门槛: max_error < 1e-3, top-1 match

成功标准（至少 5/15 模型达到）:
```
完整模型 2x+ 案例:
  GPT-2 (12L):        B0=150s, B3=60-70s  (2.1-2.5x)
  Falcon (24L):       B0=200s, B3=80-100s (2.0-2.5x)
  Qwen (24L):         B0=180s, B3=70-90s  (2.0-2.6x)
  DINOv2 (12L):       B0=1200s, B3=400-500s (2.4-3.0x)
  Whisper (4L+4L):    B0=350s, B3=120-150s (2.3-2.9x)
```

### 4.2 报告格式

每个模型的标准报告:

```yaml
Model: GPT-2-12L
Configuration: Full (12 layers, 768 hidden, 50257 vocab)
Precision: HMX=FP16, Norm/Residual=FP32

Baselines:
  B0_HVX:
    latency_ms: 152000
    correctness: PASS
  B1_HMX_direct:
    latency_ms: 148000
    hmx_coverage: 12%
    correctness: PASS
  B2_HMX_padded:
    latency_ms: 75000
    hmx_coverage: 68%
    correctness: PASS
  B3_HMX_OmniFetch:
    latency_ms: 65000
    hmx_coverage: 68%
    vtcm_peak_saved: 55%
    l2_requests: 890 (vs 89000 baseline)
    correctness: PASS

Speedup:
  B2_vs_B0: 2.03x  (HMX infrastructure)
  B3_vs_B2: 1.15x  (OmniFetch)
  B3_vs_B0: 2.34x  (Total)

Attribution:
  - HMX coverage expansion: ~50% of speedup
  - OmniFetch data supply: ~15% of speedup  
  - Combined synergy: remaining
```


---

## 五、风险和缓解

### 5.1 技术风险

| 风险 | 可能性 | 影响 | 缓解策略 |
|---|---|---|---|
| Padding 导致算术开销过大 | 中 | 高 | 收益模型动态选择，保留 HVX fallback |
| FP16 数值不稳定 | 中 | 高 | Mixed precision（HMX=FP16, critical ops=FP32） |
| 大模型编译超时持续 | 中 | 中 | 增量编译 + pass 优化 |
| VTCM 容量不足（4MB 限制） | 低 | 中 | Lifetime coloring + 动态分配 |
| HMX 尾部处理错误 | 中 | 高 | 详尽测试边界条件 + mask/predication |

### 5.2 进度风险

**关键路径**: P0.1 (M/N padding) → P0.2 (batch matmul) → P1.1 (HMX boundary)

**应急计划**:
- 如果 P0.2 延迟，先完成 P0.1 + P1.1 组合
- 如果 P2.1 (FP16) 困难，先聚焦 Vision/Audio 模型
- 如果完整模型不足，确保至少 3 个代表性模型达标

### 5.3 度量风险

**避免虚假加速**:
1. ✓ 使用修复后的 true HVX baseline（不是 legacy scalar）
2. ✓ 独立报告 HMX 基础设施收益（B2-B0）
3. ✓ 单独归因 OmniFetch 贡献（B3-B2）
4. ✓ 所有结果包含 HMX 覆盖率报告
5. ✓ 报告 padding 开销和最终 MACs


---

## 六、与现有工作的关系

### 6.1 保留的 OmniFetch 机制

继续使用并优化:
- ✓ Item 1-3: Layout liveness, cost model, fusion
- ✓ Item 4: Persistent WH cache (for decode)
- ✓ Item 5: 二维流水线（与 N1 权重静止组合）
- ✓ Item 6: VTCM lifetime coloring
- ✓ Item 7: Attention K/V（仅 decode，不是 prefill）

### 6.2 修订的机制

**K/V 预取策略**:
```python
if phase == "prefill":
    # Fresh K/V, 使用在线计算 + producer-side layout
    use_online_attention()
    use_producer_direct_layout()
elif phase == "decode":
    # Historical K/V in DDR
    use_kv_stream_prefetch()
    use_page_aware_hints()
```

**L2 预取运行时**: 完全重写以符合 V73 规范

### 6.3 论文故事线调整

**旧故事**: "OmniFetch 是一个预取 + 布局融合系统"

**新故事**:
```
"给定 Hexagon V73 NPU 和 HMX 矩阵加速器，
我们首先扩展 HMX 覆盖率以处理非对齐和批处理矩阵操作，
然后设计 OmniFetch 作为 HMX-aware 的数据供应优化器：
  - 跨 HMX/HVX 边界的 VTCM 驻留
  - L2 单飞行预取调度
  - In-situ 布局形成
  - 权重静止流水线"
```

这样 HMX 扩展是明确的基础设施，OmniFetch 是数据供应层，故事连贯。


---

## 七、立即行动项（本周）

### Day 1-2: 现状审计和环境验证

```bash
# 1. 验证构建环境
cd /home/huzq85/2-working/hexagon_npu/hexagon-mlir
source /home/huzq85/2-working/hexagon_npu/mlir-env/bin/activate
bash scripts/build_hexagon_mlir_incremental.sh --jobs 12

# 2. 运行 DINOv2 Debug baseline
python benchmark_models/debug_running/run_dinov2-small_debug.py \
  --device-iterations 5 > /tmp/dinov2_hvx_baseline.log 2>&1

python benchmark_models/debug_running/run_dinov2-small_debug.py \
  --enable-hexkl --device-iterations 5 > /tmp/dinov2_hexkl_baseline.log 2>&1

# 3. 确认 HMX 覆盖率为 0
grep "HexKL.*rewrite" /tmp/dinov2_hexkl_baseline.log
# 预期: rewrites=0

# 4. 审计现有 MatmulToHexKLPass
cat qcom_hexagon_backend/lib/Transforms/MatmulToHexKLPass.cpp | \
  grep -A 20 "M % 32\|K % 32\|N % 32"
```

### Day 3-5: P0.1 第一阶段实现

**文件**: `qcom_hexagon_backend/lib/Transforms/MatmulToHexKLPass.cpp`

**修改**:
1. 移除 M % 32 和 N % 32 的硬限制
2. 添加 padding 决策逻辑
3. 添加编译器属性记录

**测试**:
```bash
# 编译新版本
bash scripts/build_hexagon_mlir_incremental.sh --jobs 12

# 测试 DINOv2
python benchmark_models/debug_running/run_dinov2-small_debug.py \
  --enable-hexkl --device-iterations 5

# 预期:
# - hmx_direct=1, hmx_padded=6-8
# - 数值正确性 PASS
# - 延迟 < 15 ms (vs. 21 ms baseline)
```


---

## 八、成功案例投影

基于当前数据和 QNN 参考，预测修复后的性能：

### 案例 1: DINOv2 Debug (1 层, 17 tokens, 64 hidden)

| 配置 | 当前实测 | 预期（HMX 扩展后） | 收益来源 |
|---|---:|---:|---|
| B0 (HVX) | 21.3 ms | 21.3 ms | - |
| B1 (HMX direct) | 22.2 ms | 19 ms | 对齐的 classifier |
| B2 (HMX padded) | N/A | **8-10 ms** | 8/9 ops 到 HMX |
| B3 (HMX + OF) | 60.3 ms (scalar) | **4-5 ms** | P1.1-P1.3 优化 |
| QNN HTP (参考) | 0.87 ms | - | 生产级图编译器 |

**B3 收益分解**:
- HMX 覆盖: 11% → 88% (约 2.1-2.4x 来自计算)
- VTCM 驻留: 节省 ~30% DDR 往返
- L2 预取: 请求数 36 vs 122,843 (减少命令开销)
- **总计**: 4.3-5.3x vs HVX, 2.1-2.6x vs HMX-only

### 案例 2: GPT-2 完整 (12 层, 768 hidden)

| 配置 | 当前/历史 | 预期（修复后） | 说明 |
|---|---:|---:|---|
| B0 (HVX) | 24s | 140-160s | 修复 vectorization |
| B2 (HMX) | 12s (病态) | **65-75s** | 正常 HMX 覆盖 |
| B3 (HMX + OF) | 11.7s (FP32 问题) | **60-70s** | Mixed precision |

**收益**: 2.1-2.4x vs HVX baseline

### 案例 3: 完整模型组合目标

| 模型 | 层数 | B0 预期 | B3 目标 | 倍率 | HMX 覆盖 |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 12 | 150s | 65s | 2.3x | 65% |
| Falcon | 24 | 200s | 90s | 2.2x | 60% |
| Qwen | 24 | 180s | 80s | 2.3x | 68% |
| TinyLlama | 22 | 190s | 85s | 2.2x | 62% |
| DINOv2-small | 12 | 1200s | 450s | 2.7x | 70% |
| **Geometric Mean** | - | - | - | **2.34x** | **65%** |


---

## 九、结论和下一步

### 9.1 核心认识

1. **HMX 是关键**: 当前 <20% 覆盖率是主要瓶颈，必须提升到 >60%
2. **基线很重要**: 修复后的 HVX baseline 快 8x，早期倍率被高估
3. **数据供应优先级调整**: 先扩展 HMX，再为 HMX 优化数据移动
4. **完整模型必须**: Debug 结果不能代表生产性能

### 9.2 关键决策

✅ **DO**:
- 将 HMX 覆盖率扩展作为明确的基础设施工作
- 建立独立的 B0-B2 基线，清晰归因
- 修复 L2 预取运行时以符合 V73 规范
- 为 5+ 完整模型建立端到端结果

❌ **DON'T**:
- 不要把 HMX 扩展收益算作 OmniFetch 贡献
- 不要用 legacy scalar baseline 计算倍率
- 不要在 prefill 场景对 fresh K/V 做 L2 预取
- 不要只报告 debug 模型结果

### 9.3 立即启动

**本周**:
- [ ] 完成现状审计（Day 1-2）
- [ ] 开始 P0.1 实现（Day 3-5）
- [ ] 创建跟踪 issue 和 milestone

**下周**:
- [ ] 完成 P0.1，验证 DINOv2
- [ ] 开始 P0.2（batch matmul）
- [ ] 并行修复 L2 预取运行时

**3 周后检查点**:
- HMX 覆盖率 >60% 在至少 3 个 debug 模型
- B0-B2 基线建立
- 至少 1 个模型达到 2x+

---

**文档维护**: 每周更新进展、风险、实测数据到本文档。

