# ALPS 瓶颈、item 7 与 Layout Prefetching 分析

更新时间：2026-08-19
分析对象：以 `baseline_5_upstream_v73` 为基线的新 `alps_v73` 分支、`experimental_data.md` 中最新的完整模型数据，以及 Hexagon V73 / HVX memory hierarchy。

命名约定：从 `alps_v73` 开始，论文、实验配置和新增编译器接口统一使用 **ALPS**。已有 `omni_fetch` dialect、runtime ABI、历史 CLI 和日志字段暂时作为兼容层保留；只有在 ALPS 行为稳定并具备迁移测试后才分阶段更名，避免把命名迁移与性能改动混在同一个因果实验中。本文中的“历史 OmniFetch/item 1–7”特指更名前的实现和数据。

## 0. 核心结论

1. **当前 item 7 不是一个纯粹的 K/V data-prefetch 开关。** 它同时改变 K/V 语义标记、fusion 边界、multi-use fusion、attention slicing、属性传播和最终 prefetch 插入。最新完整模型中，item 7 的 runtime K/V prefetch sites 均为 0；因此这些测得的收益不能归因于实际执行的 `l2fetch`，而主要来自编译拓扑和 code generation 的变化。
2. **普通 data prefetch 很难单独稳定达到 1.8x。** Prefetch 只能隐藏等待，不能消除 compulsory traffic、layout materialization、store-to-load、带宽占用或计算指令。达到 1.8x，需要把优化对象从“cache miss latency”扩大为“表示和数据供应路径”，即先消除搬动，再融合剩余搬动，最后才预取。
3. **可以引入 Layout Prefetching，但它应被严格定义为一个新的图级抽象，而不是简单重命名现有的 layout-aware prefetch。** 它预取的不只是字节，还包括该逻辑值在未来 consumer 所需的 physical layout、tile、memory tier 和 ready time；目标是在不可避免的数据转移中同时完成 pack/transpose/pad/place，或直接让 producer 写出目标布局。
4. **当前代码已有 Layout Prefetching 的局部原型。** `prefetch_in_situ`、`LayoutAwareMapping` 和 `LayoutOpsEliminationPass` 已能处理部分 DDR→VTCM + HMX layout 形成，以及通过静态连续 `collapse_shape` 携带布局请求。但它还没有图级 consumer-demand 分析、版本/alias 身份、multi-consumer 选择、全局驻留计划和可靠的收益模型。
5. 推荐把后续 ALPS 的统一故事线定义为：

   > **Representation-aware Data Supply：预测未来 consumer 对数据“时间、位置、布局”的需求；优先保持在 VRF、不物化，其次在 producer 或不可避免的搬运中直接形成目标布局并复用，最后才用 page-safe L2FETCH 或 DMA/VTCM 隐藏剩余搬运。**

这条故事线仍然以 prefetch 为中心，但不再把 prefetch 狭义地等同于“提前发 cache hint”。

---

## 1. item 7 的真实机制

### 1.1 用户可见配置

item 7 对应：

```text
enableOmniFetchKvCachePrefetch = true
```

在当前严格 item7-only 配置中：

- `enablePrefetch = true`；
- `enableOmniFetchKvCachePrefetch = true`；
- layout-aware、V-DAE、adaptive lookahead、persistent WH、two-dimensional pipeline、VTCM coloring 等 item 1–6 均关闭；
- `kvCacheOnly = true`，不插入普通 loop prefetch。

配置入口主要位于：

- `benchmark_models/hexkl_utils.py`
- `scripts/script_release/internal/layered_hvx_options.py`
- `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`

### 1.2 编译流程

当前 item 7 的实际执行流如下：

```text
attention graph
    |
    v
识别 QK / AV contraction 和 K / V operand
    |  LowerTmTensor: 写入 kv_cache_role / operand / layout
    v
改变 fusion 与 slicing 拓扑
    |  K/V 边界不参与部分 elementwise/reshape fusion
    |  函数存在 K/V 边界时跳过 multi-use fusion
    |  item 7 开启时跳过 full-size attention slicing
    v
tiling / vectorization / bufferization
    |  多次重新识别并传播 K/V 属性
    v
PrefetchInsertPass
    |  只接受函数入口已经存在的 persistent K/V 参数
    |  拒绝本次函数内刚产生的 eager-prefill K/V
    v
page-safe first-demand-line L2 hint（若有合法 site）
```

### 1.3 K/V 语义识别

`LowerTmTensor.cpp` 中的 recognizer 识别：

- QK contraction 的 K operand；
- AV contraction 的 V operand；
- rank-3/rank-4 attention；
- `[B,S,H,D]`、`[S,H,D]`、transpose ancestry、softmax dependency 等模式。

它附加：

- `omni_fetch.kv_cache_role = key | value`
- `omni_fetch.kv_cache_operand`
- `omni_fetch.kv_cache_layout = bshd | shd | sequence_head | ...`

为了使语义跨过 generalize、fusion、tiling、vectorization 和 bufferization，pipeline 会多次重新执行识别或复制属性。这一部分是有价值的基础设施：它使优化基于“逻辑 K/V stream”，而不是依赖某个脆弱的 op 名称。

### 1.4 为什么当前 prefill K/V 不发 runtime prefetch

`PrefetchInsertPass.cpp::insertKvCachePrefetchHints` 当前只允许以下 source：

- static-shape、AS0、floating memref；
- rank 至少为 2；
- 去掉 cast/subview 后，是函数 entry block argument。

函数内部刚产生的 K/V 会计入 `rejectedProducedSites`，不会发 hint。这一限制是合理的：

1. prefill 的 K/V 刚由 projection 产生，通常已经在 L2；
2. 在 producer 之后 prefetch，通常没有新的 DDR miss 可以隐藏；
3. 把 hint 移到 producer 之前违反因果关系，因为数据还不存在；
4. 强制 materialize K/V 会破坏 producer-consumer fusion，增加 store/read；
5. 真正适合 K/V prefetch 的是 autoregressive decode 中在函数调用前已存在、位于 DDR 的 historical `past_key_values`。

当前 L2-only 路径对每个逻辑 stream 只发有限的首需求线提示，以避免跨 4 KiB page 和请求风暴，并对 `(base, role)` 去重。它不是全 K/V stream 搬运。

### 1.5 item 7 同时改变了编译拓扑

这是理解实验结果的关键。

`FusionPass.cpp` 中：

- consumer 或 producer 带 `omni_fetch.kv_cache_role` 时，拒绝部分 elementwise fusion；
- reshape expansion fusion 同样保留该边界；
- 函数中只要存在任意 K/V boundary，就跳过整个 multi-use fusion 阶段。

`LinalgToLLVMPass.cpp` 中：

- 普通路径满足 `enableSlicing` 时执行 full-size attention slicing；
- item 7 开启时，以保护 K/V metadata 为理由跳过该 slicing。

因此，item 7 会改变：

- fusion group 大小；
- intermediate 是否 materialize；
- loop/tile topology；
- vector register pressure 与 spill；
- store-to-load 距离；
- instruction-cache footprint；
- 后续 canonicalization 和 vectorization 的输入形态。

即使一个 runtime prefetch 都没有发，这些变化也足以显著改变 latency，而且可能正向也可能负向。

### 1.6 最新完整模型数据说明了什么

最新严格数据的 matched comparison 是 item7-only 对 `HMLIR HVX (HexKL On)`，而不是 HexKL-off。HexKL-off 与 HexKL-on 即使 HMX rewrite 为 0，也可能进入不同的 lowering/codegen 路径。

| 模型 | Item 7 / HexKL-on 结果 | runtime K/V sites | 可以归因于硬件 K/V prefetch 吗？ |
|---|---:|---:|---|
| DINOv2-small | 1.69x | 0 | 不可以 |
| ViT-Base | 1.42x | 0 | 不可以 |
| DeiT-small | 1.61x | 0 | 不可以 |
| SegFormer MiT-B0 | 1.04x | 0 | 不可以 |
| Swin-Tiny | 1.43x | 0 | 不可以 |
| BEiT-base | 1.01x | 0 | 不可以 |
| Whisper-tiny | 1.66x | 0 | 不可以 |
| HuBERT-base | 1.02x | 0 | 不可以 |
| Wav2Vec2-base | 0.96x | 0 | 不可以；发生回退 |
| UniSpeech-base | 0.95x | 0 | 不可以；发生回退 |
| UniSpeech-SAT-base | 0.93x | 0 | 不可以；发生回退 |
| GPT-2 | 0.97x | 0 | 不可以；轻微回退 |
| SD/CLIP | 1.02x | 0 | 不可以 |
| Qwen2.5-0.5B | **1.87x** | 0 | 不可以；需做 IR/object differential audit |
| TinyLlama-1.1B | 1.60x | 0 | 不可以 |
| SmolLM2-1.7B | 1.39x | 0 | 不可以 |

说明：表中 language 模型采用一致的 staged FP16 device-Perf 求和；它是完整模型各 stage 的 device kernel sum，不包含 host round-trip。数据来自 `experimental_data.md` 最新 strict rows。

这些数据支持以下结论：

- item 7 是目前最强的**编译策略信号**，但不是已经被证明最强的 K/V prefetch；
- 当前只有 Qwen2.5 在 matched HexKL-on 对比中超过 1.8x；
- 15 个模型的正负结果差异很大，说明收益高度依赖 graph topology；
- DINO 的 historical bounded items1–7 达到 1.815x，但减少约 45 倍请求只比未 bounded 组合快约 2%，再次证明“少发 prefetch”重要，却不能自动换来大幅 latency 收益。

### 1.7 必须做的因果拆分

继续实现新方案前，应把 item 7 拆成四个独立开关：

1. `kv-semantic-tracking`：只标记和传播属性；
2. `kv-topology-policy`：fusion boundary / multi-use fusion policy；
3. `kv-slicing-policy`：attention slicing 的选择；
4. `kv-runtime-prefetch`：只插入真正的 L2/DMA 请求。

这样才能回答 Qwen 的 1.87x 到底来自：

- 跳过 slicing；
- 改变 multi-use fusion；
- 改变单 use fusion；
- 属性传播引起的 vectorization 差异；
- 或尚未发现的 codegen side effect。

在没有这一拆分和最终 object/PMU 证据之前，论文中不应写成“item 7 的 K/V prefetch 带来 1.87x”。

---

## 2. 为什么仅靠 data prefetch 难以达到 1.8x

### 2.1 Amdahl 上界

设可优化部分占 baseline 时间的比例为 `f`，该部分被加速 `r` 倍，则：

```text
S = 1 / ((1 - f) + f / r)
```

要得到 1.8x：

- 即使把该部分完全消除，`f` 也必须至少为 **44.44%**；
- 若该部分只加速 2x，`f` 必须至少为 **88.89%**；
- 若该部分加速 4x，`f` 必须至少为 **59.26%**。

要得到 3x，即使完全消除目标部分，目标部分也必须占至少 **66.67%**。

普通 prefetch 往往只是减少一部分 exposed miss latency，`r` 有限，而且它不覆盖所有 data-movement 时间。因此它很难在 15 个结构不同的模型上普遍达到 1.8x。

### 2.2 “数据移动”应拆成五类

用户关于瓶颈仍然与数据移动相关的判断是合理的，但这里的“数据移动”必须比 cache miss 更宽：

1. **必要输入流量**：weights、input activations、persistent K/V 必须从 DDR/L2 读入；
2. **冗余物化流量**：transpose、pack、pad、layout conversion、临时 tensor 的完整读写；
3. **层次往返流量**：producer 写 L2，consumer 立即再从 L2 读；VTCM staging 后又过早写回；
4. **重复读取流量**：相同 weight/tile 被多个 head、consumer 或相邻算子重复读取；
5. **等待时间**：上述流量未及时到达导致 HVX/HMX stall。

普通 prefetch 只直接处理第 5 类，而且可能增加第 1/3 类流量。OmniFetch 要达到 1.8x，应优先处理第 2–4 类，再用 prefetch 处理剩余第 5 类。

### 2.3 V73/HVX memory hierarchy 的限制

根据本仓库中的两份 V73 手册：

#### HVX 不走 scalar L1

HVX 的 VMEM 直接连接 L2，vector load/store 不经过 L1。因此：

- scalar `dcfetch` 不是 HVX tensor path 的主要工具；
- HVX cacheable DDR stream 应使用 L2FETCH；
- 能留在 VRF 的值比 L2/VTCM 都便宜，应先减少 VMEM 指令。

#### L2FETCH 是 best-effort、single-flight 风格资源

手册说明：

- 新 `l2fetch` 在旧请求仍 active 时会停止旧请求；
- 请求优先级低于 demand fetch；
- 跨越起始虚拟页的生成地址会被丢弃；
- 建议每次小于 8 KiB；
- 应在首次使用前数百 cycles 发出；
- 太早会被逐出，太晚无法覆盖 latency；
- row/tile 粒度通常最好。

这意味着数万次 op-local hints 会互相取消或污染 cache；全图必须有统一、page-aware、request-budgeted 的 scheduler。

#### VTCM 有收益条件，不是自动更快的 cache

VTCM non-evictable、比 L2 少 cache-management overhead、降低 L2 pressure，并且是 HVX scatter/gather 的必要空间。但 DDR→VTCM copy、同步、容量和 bank conflict 均有成本。

已有 DINO Debug 数据中，whole-stream VTCM staging 从 21.258 ms 回退至 40.150/41.552 ms，说明没有复用和 overlap window 时，VTCM 只增加一次搬运。

#### store-to-load、对齐与地址局部性

- 相同地址的 VMEM store 后紧跟 VMEM load 会等待 store 完全到达 L2；手册建议约 15 packets 的独立工作；
- VMEMU 会访问多条 L2 cache line，带宽和能耗都更高；
- contiguous、vector-aligned 访问可降低 bank conflict、set aliasing 和 micro-TLB miss；
- final use 可用 `:nt` 帮助 replacement，但必须有可靠的 last-use/alias proof；
- scatter/gather 只在 VTCM 中工作，且同样受 page、冲突和 in-flight 资源限制。

### 2.4 当前 1.8x 的具体阻碍

#### 阻碍 A：覆盖面太窄

item 7 只关注 attention K/V，而模型时间还包括：

- projection、FFN、conv/patch embedding；
- normalization、activation、softmax；
- residual、concat、embedding/head；
- audio feature encoder；
- HMX 不接受或未 rewrite 的 matmul。

DINO Debug 的 LWP 曾显示 patch embedding convolution 占 46.90%，QK/AV/output projection 合计约 33.94%。只优化 K/V miss 无法覆盖第一大热点。

#### 阻碍 B：prefill K/V 没有可隐藏的 miss

fresh K/V 刚被 producer 写入 L2，往往不是 DDR cold stream。真实 decode historical K/V 才有足够长的 persistent stream 和预测性。

#### 阻碍 C：带宽瓶颈不是 latency 瓶颈

当 VMEM 或 DDR/L2 bandwidth 饱和时，prefetch 只是把同样的 bytes 提前搬，不能降低总 traffic，甚至与 demand load 竞争。

#### 阻碍 D：布局转换制造的 bytes 没有消失

对一个大小为 `N` 的显式 physical transform：

```text
producer 写 canonical N
transform 读 N + 写 target N
consumer 再读 target N
```

仅该链条就可能产生约 `4N` 的层次流量；若 producer 直接生成 target layout，并由 consumer 读取，则可降至约 `2N`。不同 cache 命中情况下外部 DDR bytes 不一定严格按这个比例，但 VMEM、store-to-load 和中间 buffer 压力真实存在。

Prefetch canonical tensor 并不会消除 transform 的 read/write。

#### 阻碍 E：fusion、register pressure 和 spill 的非单调关系

融合可以省中间 tensor，但融合过大可能造成 VRF spill 和 code-size 增长；保留边界可能降低 spill，却增加 materialization。item 7 当前正是在无 cost model 地改变这条边界，所以不同模型有 1.87x 到 0.93x 的跨度。

#### 阻碍 F：HMX/HVX compute coverage

未进入 HMX 的 matmul、低效率 vectorization、softmax/norm/conv 等 compute bottleneck 不能被 prefetch 修复。为了正确归因，OmniFetch 实验必须冻结相同的 HVX/HMX mapping，并报告 HMX rewrite 数和最终指令证据。

#### 阻碍 G：测量边界

staged language 数据是各 stage 的 device Perf 求和，不包含 host 往返；这适合当前 matched policy comparison，但不能直接解释 monolithic runtime 中跨 layer residency 的潜在收益。Layout Prefetching 若要利用 inter-layer residency，需要补充一个能保留跨层 buffer ownership 的执行边界。

### 2.5 应收集的硬件证据

后续不能只看“请求数”和 latency，应至少报告：

- 最终 object 中 VMEM load/store、VMEMU、HVX/HMX instruction；
- materialized layout tensor 数和 eliminated bytes；
- DDR/L2/VTCM copy bytes；
- L2 hit/miss、HVX load/store outstanding stall；
- VTCM outstanding、scatter/gather full/conflict；
- prefetch issued/completed/dropped/busy-suppressed/page-split；
- peak live bytes、VTCM occupancy、spill bytes；
- 每个热点 region 的 pcycles 和 call count。

只有“VMEM/bytes/stall 下降 + matched latency 提升”同时成立，才能证明 data-movement 优化是原因。

---

## 3. Layout Prefetching：定义、价值与可实现性

### 3.1 定义

普通 prefetch 的契约近似是：

```text
(address range, destination cache tier, issue time)
```

建议的 **Layout Prefetching（LAP）** 契约是：

```text
(logical value + version,
 target physical layout,
 tile coordinates,
 destination tier,
 consumer set,
 ready time,
 validity/lifetime)
```

它不是“先 transpose，再 prefetch transpose 的结果”，而是：

> 在数据本来就必须被生产或搬运时，预测未来 consumer 的表示需求，一步形成并放置目标 layout；如果能够完全避免 materialization，则不进行搬运。

因此，“prefetch”的对象从 cache line 扩展为 **future-ready representation**。

### 3.2 与现有 layout-aware prefetch 的关系

当前代码已有三个局部构件：

1. `omni_fetch.prefetch_in_situ`：DDR→VTCM 时带 `layout_transform`；
2. `LayoutAwareMapping.cpp`：支持 HMX weight deep-interleaved、HMX activation channel-interleaved 和 custom index map；
3. `LayoutOpsEliminationPass.cpp`：从 layout-aware prefetch 向 producer 反向查找冗余 layout op，并能穿过安全的 `memref.cast` 和静态连续 `collapse_shape`。

这说明 LAP 不是从零开始，但当前实现的范围有限：

- 依赖已经存在的 `prefetch_in_situ`，没有先做全图 future-layout demand 分析；
- 主要面向 HMX 的 weight/activation layout；
- 对 layout-value identity 目前偏分析/标注，尚未形成跨 consumer 的共享物化和 eviction 机制；
- 不能普遍决定“改 indexing map、producer direct-store、VTCM transform-copy、persistent prepack”中的最佳方案；
- 当前 item7-only 明确关闭 layout-aware，因此 item 7 的最新收益与这部分无关。

### 3.3 必须先区分四类“reshape/layout”

1. **descriptor-only view**：`reshape/collapse/expand/cast` 只改 descriptor、不搬数据；应 canonicalize/fold，不能把删除它声称为减少物理 bytes。
2. **consumer indexing-map remap**：transpose 可由 consumer 的 indexing map 吸收；无需物化，优先级最高。
3. **真实 physical conversion**：transpose/pack/pad/convert_layout 读旧 buffer、写新 buffer；是 LAP 的主要目标。
4. **硬件 micro-layout**：HMX WH/AH、HVX contiguous/aligned tile、VTCM bank-friendly layout；应在 producer store 或必要 copy 中直接形成。

任何统计必须只把第 3/4 类记为 eliminated movement。

### 3.4 推荐 compiler pipeline

#### 阶段 A：Future Layout Demand Analysis

对每个 SSA/buffer value 收集 consumer demand：

```text
Demand = {layout, tile shape, access order, engine, reuse count,
          first use, last use, loop depth, alignment, page footprint}
```

Demand 来源包括：

- HMX WH/AH operand contract；
- HVX vector load 的 contiguous/alignment 要求；
- attention QK/AV 的 BSHD/sequence-head 访问；
- conv im2col/patch/tile 访问；
- residual/multi-consumer 分支。

#### 阶段 B：Representation Choice

按以下优先顺序选择：

1. **VRF forwarding / fusion**：不落地；
2. **indexing-map absorption**：逻辑变换、零物理 copy；
3. **producer-direct layout**：producer 直接写 target layout；
4. **fused transform-transfer**：在必要的 DDR→VTCM/L2→VTCM copy 中完成 layout；
5. **persistent prepack**：immutable weight 在 model load/first use 时生成一次，跨 invocation 复用；
6. **普通 physical conversion**：只有前面均不合法时才保留。

#### 阶段 C：Placement and Schedule

- 一次性、连续、cacheable stream：page-safe L2FETCH；
- 有复用且能 overlap：DMA→VTCM ping-pong；
- producer-consumer 紧邻：producer direct-store 或 VRF forwarding；
- immutable weight：bounded persistent target-layout cache；
- 最后一次读取：`:nt` 或 early eviction，保护下一 tile。

#### 阶段 D：Versioning and Ownership

layout-cache key 至少应是：

```text
(allocation identity, producer version, target layout,
 tile coordinates, element type, shape, destination tier)
```

在以下情况 invalidation：

- source 可能被写；
- alias 不可证明；
- dynamic shape/layout 参数变化；
- VTCM slot 被新 tile 复用；
- consumer contract 变化。

这比仅用 pointer 作为 cache key 更安全。

### 3.5 multi-consumer 情况

一个 producer 可能有多个 layout demand。不能为每个 consumer 无条件生成一份副本。cost model 应选择：

- 所有 consumer 共用一个 layout；
- dominant consumer 使用 target layout，其他 consumer 用 indexing remap；
- 保留 canonical + 一个高收益 layout；
- 若两种 layout 都有足够复用，再保留两个版本；
- 当容量或 lifetime 重叠过大时拒绝 LAP。

收益估计可写为：

```text
Benefit = eliminated_transform_bytes
        + eliminated_intermediate_store_load
        + avoided_repeated_reads
        + hidden_residual_latency
        - transform_during_copy_cost
        - synchronization_cost
        - extra_layout_versions
        - spill/VTCM_pressure/cache_pollution
```

只有 `Benefit > threshold` 且存在足够 schedule slack 才 admit。

### 3.6 V73 上的具体 lowering

#### HVX consumer

- 尽量让 producer 直接按 128-byte vector-friendly contiguous order 写出；
- 对临时 tensor 优先保持在 VRF 或 fused region；
- 若必须落地，使用 aligned `.new` VMEM store，并安排足够 store-to-load distance；
- 对一次性输入 tile，L2FETCH 提前数百 cycles，单请求小于 8 KiB、page-contained；
- 对最后一次消费使用经证明安全的 `:nt`。

#### HMX/HexKL consumer

- weight：在 model load 或第一次不可避免的搬运中形成 WH layout，按 immutable identity 复用；
- activation：producer 直接写 AH，或在 DDR/L2→VTCM transfer 中完成 CopySubmatrix + RmToAh；
- 避免 `canonical store -> RmToAh/Wh -> HMX read` 的多次往返；
- 非对齐/不满足当前 HMX lowering 的 matmul 仍属于另一个 codegen 课题，LAP 不应伪装成 HMX coverage 改进。

#### attention

- prefill：优先 online attention、score tile 不物化、K/V producer-direct layout；
- decode：对 historical K/V 做 page-aware layout prefetch，可按 head/group/sequence tile 形成 HVX/HMX 消费布局；
- GQA/MQA：同一 K/V tile 服务多个 query heads，适合 multicast/residency；
- sliding-window attention：只预取下一窗口并及时 `:nt`/evict 已过期页。

### 3.7 为什么 LAP 有机会超过普通 prefetch

LAP 同时作用于三项：

1. **减少 bytes**：删除 physical transform 的中间读写；
2. **提高 locality**：让 HVX/HMX 看到连续、对齐、bank-friendly 的 layout；
3. **隐藏 latency**：目标表示在 consumer 到达前 ready。

普通 prefetch 只做第 3 项。LAP 若覆盖模型中的大规模 activation/weight transform，才可能使可优化比例越过 Amdahl 所需的 44.44%。

### 3.8 风险

- 用 gather/scatter 做 layout transform 可能比连续 VMEM 更慢；V73 gather/scatter 只能访问 VTCM，并存在较高 latency 和冲突限制；
- 过早准备 layout 会增加 live range、VTCM pressure 和 eviction；
- producer-direct layout 可能降低 producer 自身写入连续性；
- multi-layout version 会增加总 bytes；
- staged model 边界会阻断跨层 residency；
- 错误的 alias/version proof 会产生 silent correctness bug；
- HMX target layout 的收益必须包含其实际 rewrite coverage，不能仅看 flag。

### 3.9 实施顺序

#### P0：先拆 item 7 的因果开关

这是新实现的前置条件，否则 LAP 收益仍会被 fusion/slicing side effect 混淆。

#### P1：只做分析和 ledger

输出每个模型：

- physical layout sites；
- logical view sites；
- bytes/read/write estimate；
- consumer layout demands；
- reuse/lifetime/VTCM estimate；
- chosen/rejected reason。

不改变 codegen。

#### P2：零 copy layout elimination

先实现低风险项：

- contiguous reshape/view folding；
- consumer indexing-map absorption；
- dead pack/unpack pair removal；
- final object 验证 VMEM/alloc 下降。

#### P3：producer-direct layout

从 attention K/V、QKV projection 和 HMX activation 三类开始，避免 fresh data 的 store→prefetch→load 反模式。

#### P4：fused transform-transfer

只对有复用、能 overlap、VTCM fit 的 tile 启用 DDR/L2→VTCM + layout formation；保留普通 L2 路径作为 fallback。

#### P5：persistent immutable layout

只对 weight/constant 开启 bounded cache，明确 capacity、generation、invalidated path。不要恢复曾经导致 request/byte storm 的无界 cold/warm 机制。

#### P6：全图 scheduler 与 PMU feedback

统一管理 L2 single-flight request、DMA slots、VTCM slots 和 last-use；以 PMU 和 measured region cycles 校正静态 cost model。

---

## 4. 其他与 prefetch 相关、可减少搬动的方法

### 4.1 IO-aware online attention

FlashAttention 的关键不是“更积极地 prefetch”，而是用 tiling 和 online reduction 避免完整 attention score matrix 在层次间读写。它证明了：降低 memory traffic 往往比只隐藏 memory latency 更有效。

对 OmniFetch 的借鉴：

- QK tile、online softmax、AV 在一个 movement region 内完成；
- score 不落 DDR/L2；
- K/V tile 一次读取服务 QK 和 AV；
- 下一 tile 通过 L2FETCH 或 DMA/VTCM 与当前 tile compute overlap。

这与 LAP 完全一致：先消除 score materialization，再 prefetch 剩余 K/V tile。

参考：[FlashAttention paper](https://arxiv.org/abs/2205.14135)。

### 4.2 Producer-driven prefetch / direct handoff

传统 consumer-side prefetch 从 consumer 向前猜地址；producer-driven 方法在 producer 知道 tile 完成时，直接：

- 保留在 VRF；
- 写入 consumer layout；
- 放入指定 VTCM slot；
- signal 下一 consumer。

这能避免 fresh-prefill K/V 的无效 cache hint，并解决 store-to-load 距离问题。它是“prefetch future representation”，而非“重新读取刚写出的数据”。

### 4.3 Async transform-copy 与 double buffering

硬件/系统中的类似思路是异步把多维 tile 搬到 scratchpad，并在 transfer 中处理目标布局。NVIDIA TMA 的 tensor map、async bulk tensor copy 和 shared-memory swizzle 是概念上的参考：描述多维源/目标布局，让搬运引擎完成 tile movement 和 bank-friendly placement。

Hexagon V73 不应假设拥有完全相同的硬件能力；本项目可用 DMA + VTCM + HVX gather/scatter/producer store 实现软件等价物，并严格计入 transform 和同步成本。

参考：[CUDA Programming Guide — Asynchronous Data Copies / TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

### 4.4 Persistent prepacked weights

immutable weight 的目标 layout 在所有 invocation 中相同。可在 model load/首次使用时：

- 生成 WH/HVX-friendly layout；
- 缓存并按 model/weight/version 标识；
- 后续直接 prefetch target layout tile；
- 避免每次 inference 重复 pack。

这实际上是时间跨度更长的 layout prefetch。必须 bounded，且只对 immutable data 启用。

### 4.5 Multicast / one-fetch-many-consumers

让一次读取服务：

- Q/K/V projection 的共享 input；
- GQA/MQA 中多个 query heads 的 K/V；
- residual 和主分支；
- 相邻 fused op。

实现上优先 VRF reuse，其次 VTCM resident tile；不是简单复制多个 prefetch request。这里节省的是 repeated reads。

### 4.6 Page-aware K/V management

PagedAttention 的主要目标是减少 K/V cache fragmentation 和 redundant duplication，并支持页级共享。对本项目可借鉴：

- K/V 物理 page 与 prefetch scheduling 使用同一 page table/metadata；
- 按 active sequence/window 只预取实际页；
- beam/prefix 共享页只搬一次；
- layout-cache identity 与 page version 绑定。

它更适合真实 decode/serving，不应拿 prefill vision 模型来证明。

参考：[PagedAttention paper](https://arxiv.org/abs/2309.06180)。

### 4.7 Cache-pollution-aware prefetch

Prefetch 不仅要决定“取什么”，还要决定“保护什么”：

- upcoming reused tile 保持 resident；
- last-use stream 使用 `:nt`；
- 避免 prefetch 一次性数据驱逐 weight/K/V hot set；
- page/set-aware placement 降低 set aliasing；
- request coalescing 防止新 L2FETCH 取消旧请求。

这能把 OmniFetch 已表现出的“少发请求”优势转化为更稳定的 latency 收益。

### 4.8 Recompute-versus-materialize

对 cheap elementwise、mask、normalization statistics 或地址变换，重新计算可能比存中间 tensor 再读回更便宜。cost model 比较：

```text
recompute cycles  vs.  store + load + cache/TLB + lifetime pressure
```

该技术与 prefetch 的关系是：通过不物化减少需要预取的数据集合，使有限 L2/DMA 预算集中于真正昂贵的 tensor。

### 4.9 Hierarchical prefetch admission

同一个逻辑 demand 不应同时触发 L2、DMA/VTCM 和普通 copy。统一 admission：

| 数据特征 | 首选 |
|---|---|
| producer/consumer 紧邻，可融合 | VRF/direct handoff |
| 一次性、连续、cacheable | L2FETCH |
| 多次复用、VTCM fit、有 overlap | DMA→VTCM |
| immutable、跨 invocation 复用 | persistent target-layout cache |
| irregular 且必须重排 | VTCM gather/scatter，但需严格 cost gate |

### 4.10 Alignment/page/layout co-design

编译器应联合选择：

- tile shape；
- vector alignment；
- page-contained prefetch rectangle；
- VTCM bank distribution；
- HMX/HVX target layout；
- ping-pong slot 和 store-to-load distance。

仅在 loop 层“提前几次迭代”不能解决这些物理约束。

---

## 5. 统一的论文故事线

可以把原 OmniFetch、item 7、in-situ reshape 和新的 LAP 合成一条逻辑完整的故事线：

### 问题

移动端 NPU 上，模型并非只受 arithmetic 限制；数据在 DDR、L2、VTCM、VRF 之间反复搬动，并以错误布局到达 consumer，造成 layout wall、materialization、store-to-load 和 demand stall。传统 operator-local prefetch 只隐藏部分 miss，而且会产生 command storm 和 cache pollution。

### 观察

编译器能够提前知道 future consumer 的：

- 使用时间；
- tile 和页；
- HVX/HMX engine；
- physical layout；
- reuse 与最后一次使用。

### 方法

OmniFetch 建立一个 **future representation contract**：

1. **Eliminate**：不物化 descriptor-only/layout-intermediate；
2. **Form in situ**：producer 或必要 transfer 直接形成 consumer layout；
3. **Reuse/reside**：一次读取服务多个 consumer，按 lifetime 保持在 VRF/VTCM；
4. **Prefetch remainder**：对不可避免的 DDR stream 使用 page-safe L2FETCH 或 DMA/VTCM overlap；
5. **Protect**：用 request budget、last-use 和 cache/VTCM ownership 避免污染。

### 三个可作为核心贡献的统一抽象

1. **Future Layout Demand / Representation Contract**：把 value identity、version、layout、tile、tier、consumer、ready time 统一表达；
2. **Fusion-preserving Movement Region**：跨 producer/layout/consumer 决定不落地、direct-store、transform-copy 或 ordinary prefetch；
3. **Hierarchy-aware Supply Scheduler**：在 VRF、L2FETCH、DMA/VTCM、persistent layout 之间做 page/capacity/lifetime-aware admission。

### 为什么故事线连贯

- item 7 提供 semantic future-stream tracking 和 topology sensitivity 的经验；
- in-situ reshape 提供“搬运时形成布局”的执行原语；
- request coalescing/单 flight scheduler 解决普通 prefetch command storm；
- Layout Prefetching 把这些局部机制提升为图级 future representation supply；
- 所有机制围绕同一目标：**减少、合并并提前不可避免的数据搬动**。

这不是“为了加速而堆叠无关技巧”。HMX lowering coverage、混合精度等可以作为独立课题，不应混入当前 contribution。

### 关于原创性的边界

“异步 tile copy”“layout transform”“scratchpad double buffering”“IO-aware attention”分别已有大量先例。潜在原创点不应声称单个原语首次出现，而应放在：

- 在 Hexagon HVX/HMX 共存环境中统一 future layout、tier 和 readiness；
- 用 compiler-visible representation identity 跨 producer/consumer 消除 layout materialization；
- 在 single-flight/page-constrained L2FETCH 与显式 VTCM/DMA 之间统一 admission；
- 在真实完整模型上证明减少 VMEM/bytes/stall，而非只证明发出了 prefetch。

正式论文定稿前仍需单独做系统性的相关工作检索和 novelty comparison。

---

## 6. 下一步实验与验收标准

### 6.1 最小可证伪实验

先选三个结构不同、当前信号明确的模型：

- Qwen2.5-0.5B：item 7 现有 1.87x 信号；
- DINOv2-small：vision、0 HMX rewrite、item 7 1.69x；
- UniSpeech-SAT-base：当前 item 7 0.93x，作为防止 cherry-pick 的负例。

对每个模型固定相同 FP16、HVX/HMX mapping、频率和串行协议，运行：

1. HexKL-on control；
2. semantic tracking only；
3. topology policy only；
4. runtime K/V prefetch only；
5. LAP analysis only；
6. zero-copy layout elimination；
7. producer-direct layout；
8. fused transform-transfer；
9. 完整 OmniFetch。

### 6.2 每阶段验收

必须同时满足：

- 正确性通过；
- final object/IR 确认目标机制存在；
- physical layout bytes 或 VMEM 数下降，不能只看 logical ledger；
- 没有 HMX/HVX mapping 漂移；
- runtime request 不发生 storm；
- latency 至少三次稳定测量，报告中位数和离散度；
- 负例不发生不可接受回退。

### 6.3 1.8x 目标的现实路径

要在大多数模型达到 1.8x，单个 L2 hint 不够。更可信的组合是：

```text
online/fused compute
  + eliminate physical layout materialization
  + producer-direct consumer layout
  + one-fetch-many-consumers / residency
  + page-safe prefetch of the remaining compulsory stream
```

个别模型达到 3x，通常要求原 baseline 中确实有占比超过三分之二的可消除 materialization/traffic，或某个主要 kernel 原来严重受 memory topology 限制。应由 profile 选择模型和 region，而不是对所有模型无条件打开同一组 hint。

---

## 7. 本次分析依据

### 仓库代码与实验

- `experimental_data.md`
- `benchmark_models/hexkl_utils.py`
- `scripts/script_release/internal/layered_hvx_options.py`
- `qcom_hexagon_backend/lib/Transforms/LowerTmTensor.cpp`
- `qcom_hexagon_backend/lib/Transforms/PrefetchInsertPass.cpp`
- `qcom_hexagon_backend/lib/Transforms/LayoutOpsEliminationPass.cpp`
- `qcom_hexagon_backend/lib/Dialect/OmniFetch/IR/LayoutAwareMapping.cpp`
- `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/FusionPass.cpp`
- `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`
- `qcom_hexagon_backend/include/hexagon/Dialect/OmniFetch/IR/OmniFetchOps.td`
- `../../archive/engineering_notes/engineering_work.md`
- `../../archive/engineering_notes/omnifetch_history.md`
- `../../archive/engineering_notes/omnifetch-prefetch-insitu-innovation.md`

### V73 手册

- `references/Hexagon_V73_Programmers_Reference_Manual.pdf`，Memory / Cache prefetch / software `l2fetch`；
- `references/Hexagon_V73_HVX_Programmers_Reference_Manual.pdf`，HVX local memory、VMEM/L2、VTCM、scatter/gather、memory performance 与 PMU events。

### 外部概念参照

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [CUDA Programming Guide: Asynchronous Data Copies and Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)

---

## 8. 论文三个贡献点与当前实现的符合程度

论文当前提出：

1. V-DAE 骨架；
2. prefetching + in-situ layout transformation；
3. runtime 中的 PMU monitor 与 traffic control。

这三个贡献点与本文前面提出的 **Representation-aware Data Supply** 故事线在逻辑上是吻合的：第二点负责减少和融合数据搬动，第一点负责把剩余搬动与计算重叠，第三点负责根据硬件反馈控制搬动的时机、层级和流量。问题不在论文叙事，而在当前实现还没有让三个部分形成一个真正闭环。

### 8.1 总体判断

| 论文贡献 | 当前代码状态 | 当前完整模型性能证据 | 判断 |
|---|---|---|---|
| V-DAE 骨架 | 有 pass、dialect op、semaphore、async descriptor ring 和可选 scout thread | 没有可靠的独立 full-model V-DAE 增益；当前 item7-only 关闭 V-DAE | **骨架存在，但尚未成为完整、稳定的 decoupled execution** |
| Prefetch + in-situ layout transformation | 有 `prefetch_in_situ`、HMX WH/AH transform、layout mapping、layout-op elimination、DMA/VTCM 路径 | 历史 micro/Debug 多为噪声或回退；最新 item7 收益与它无关，因为 layout-aware 被关闭 | **三个贡献中实现最完整，但覆盖窄、cost model 和图级消除不足** |
| PMU monitor + traffic control | 有 L2 counters、page clipping、busy/duplicate/budget suppression 和静态 traffic envelope | bounded traffic 显著减少命令，但 DINO latency 只额外改善约 1.99%；没有 PMU 因果证据 | **traffic control 部分真实；PMU monitor 和 closed-loop control 尚未完成** |

因此，最准确的表述是：

> 当前实现已经证明三个贡献点的 IR、lowering 和 runtime 构件可以在 Hexagon-MLIR 中落地，但还没有证明三者形成了有效的完整系统，也没有足够证据支持它们分别或组合后在完整模型上稳定获得显著加速。

### 8.2 贡献一：V-DAE 骨架

#### 已经实现的部分

`VDAEDecouplePass.cpp` 会：

- 找到已有 `omni_fetch.prefetch_in_situ` 的 loop；
- 跳过 fire-and-forget 的 L2 hint；
- 跳过 `lookahead == 0` 的同步-only transfer，避免无意义 wait/signal；
- 为异步 transfer 插入 `create_sem`、loop-entry `wait`、loop-tail `signal`；
- 可选插入 `adaptive_control`。

runtime 已有：

- generation-aware 的 16-slot semaphore pool；
- acquire/release atomic 操作；
- 4-slot async descriptor/staging ring；
- UserDMA start/wait；
- deferred WH transform；
- 可选的单 worker scout thread，能够在计算线程之外完成 DMA wait + WH transform；
- descriptor-full 和 semaphore-timeout error flags。

这些不是空壳代码，说明 V-DAE 的控制骨架和最小执行原语确实存在。

#### 与论文理想 V-DAE 的差距

1. **默认仍是单线程软件流水。** dual-thread scout 默认关闭；单线程模式下 `signal()` 会在计算线程执行 `dma_wait` 和 deferred WH，access/execute 没有真正并行的执行上下文。
2. **只覆盖特定的 async in-situ site。** L2 hint 本来就不应 wait，但普通 HVX stream、跨算子/跨层 movement 和大多数 layout site 没有进入统一 V-DAE schedule。
3. **loop token 与 descriptor ownership 不够显式。** IR 中 semaphore 只表达“有一个 tile ready”，没有携带 tile/version/slot；多个 nested/sibling loop 仍依赖 runtime global ring 和调用顺序。
4. **runtime 状态仍主要是 process-global。** async ring、semaphore pool、adaptive state 和 scout state没有显式 invocation context，不适合重入、并行 invocation 或可靠的多模型场景。
5. **当前 pass 没有构造真正的 access program。** 它是在已有 loop 中插 wait/signal；“上一迭代发下一 tile”是 software pipelining，而不是独立、可调度的 access slice。
6. **缺乏完整模型的独立因果证据。** 最新 item7-only 配置明确关闭 V-DAE，因而不能用 item 7 的 1.87x/1.69x 等结果证明 V-DAE。

#### 结论

论文称其为“V-DAE 骨架”是合理的，前提是明确“骨架”而不是宣称已经实现完整硬件式 DAE。后续需要把它升级为：

```text
Movement plan
  -> explicit tile/version/slot descriptors
  -> access issue / DMA completion / layout-ready states
  -> execute waits only for its exact representation
```

### 8.3 贡献二：prefetching + in-situ layout transformation

#### 已经实现的部分

当前 `PrefetchInsertPass` 的 HexKL micro path 已能：

- plain mode 为 DDR tile 插入 L2 hint；
- layout-aware mode 用 `prefetch_in_situ(HMXWeight)` 替换 `RmToWh`；
- 用 `prefetch_in_situ(HMXActivation)` 替换 `CopySubmatrix + RmToAh`；
- 对 weight 构造 current-tile bootstrap 和 next-tile ping-pong；
- 选择同步、async in-situ、persistent 或 native fallback；
- 可选 DMA 到 DDR staging 或 VTCM staging；
- runtime 调用 HexKL micro API 形成真实 WH/AH layout。

`LayoutOpsEliminationPass` 已能：

- 把 HMX activation layout request 穿过安全的静态连续 `collapse_shape`/cast 携带到 producer；
- 标记 layout-value identity、估计执行次数与 reuse；
- 识别部分 transpose/permute/reshape；
- 在 use/ownership 允许时删除冗余 op。

因此，第二点不仅和统一故事线吻合，也是当前最接近完整 compiler-to-runtime 路径的一项。

#### 为什么当前效果仍不明显

1. **覆盖范围集中在 HexKL Micro HMX WH/AH。** 大量 HVX activation、attention、conv、norm、residual 和跨层 layout 不在该路径内。
2. **当前 item7-only 把 layout-aware 关闭。** 最新完整模型中 item 7 的收益不能作为第二项贡献的实验结果。
3. **activation path 目前通常同步。** `HMXActivation` 会执行 CopySubmatrix + RmToAh，但没有足够 overlap 时只是把两个调用封装进一个 runtime op，不一定减少物理流量。
4. **部分“消除”仍偏语法识别。** generic op 名称含 transpose/permute/reshape 时的判断较宽；最终虽然受 use-empty/safe-remove 限制，但缺少严格的 layout equivalence proof 和 eliminated-physical-bytes 证明。
5. **producer-direct layout 还很有限。** 当前更多是“在 copy 时转换”，尚未普遍做到 producer 直接形成 consumer layout 或让 consumer indexing map 吸收变换。
6. **multi-consumer 与 layout version 没有全局决策。** 同一 logical value 的多个 layout demand 可能造成重复版本、live-range 和 VTCM 压力。
7. **收益模型不包含完整的最终代价。** transform、DMA startup、scatter/gather、同步、spill、VTCM interference 和被破坏的 fusion 尚未统一计入。

#### 结论

第二项应从当前的 **op-local layout-aware prefetch** 升级为本文第 3 节定义的 **Layout Prefetching / Future Representation Contract**。这不是改变论文故事，而是把论文最核心的一点做完整：

```text
不再是：先产生 canonical tensor，再预取/转换
而是：预测 consumer layout，尽量不产生 canonical intermediate；
       若必须搬运，则在该次搬运中一次形成最终 representation
```

### 8.4 贡献三：PMU monitor 与 runtime traffic control

#### 已经真实工作的 traffic control

`OmniFetchRuntime.c` 已实现 V73-aware L2 scheduler：

- `issued`、`busy_suppressed`、`page_clipped`、`unsupported`；
- requested/issued bytes；
- command/byte budget suppression；
- recent-request duplicate suppression；
- 检查 `USR.PFA`，已有 l2fetch active 时不覆盖旧请求；
- 每次请求限制到推荐的 8 KiB 内，并裁剪到起始 4 KiB page；
- launcher 对 OmniFetch 配置静态 envelope：4096 commands、8 MiB、64-entry recent window；
- 运行后把 counters 写入 `perf.txt`。

这部分符合论文中的 traffic control，而且 bounded DINO 实验确实把 runtime issued 从 186,624 降到 4,096、issued bytes 从约 41 MiB 降到约 0.9 MiB。

#### PMU monitor 尚未实现

当前源代码没有读取 V73 PMU events。`__omni_fetch_update_distance` 明确使用 `__omni_fetch_wait` 的 spin counts，而不是：

- HVX L2 load/store outstanding stall；
- L2 miss/hit；
- VTCM outstanding；
- scatter/gather full；
- DDR/MEMNOC bandwidth 或 cycles。

因此，代码/option 中的“PMU-driven”描述目前不准确；最多只能称为 software-wait feedback。

#### 当前 adaptive control 也未真正闭环

还有一个更关键的问题：

- V-DAE pass 在每次 loop iteration 尾部创建一个常量 `initDist`；
- `AdaptiveControlOp` 的返回结果没有成为 `scf.for` iter_arg，也没有更新后续 `prefetch_in_situ` 的 lookahead；
- runtime 虽把计算结果写入 `omni_eff_lookahead`，但当前 prefetch issue 路径没有用它改变 IR 中已经固定的 next-tile address/距离；源码中相关使用主要停留在状态和注释层面。

所以当前控制链实际是：

```text
wait spin -> update_distance() -> 写 runtime global state
                                      X
                                      └─ 未改变后续 tile/distance/tier
```

它不是 closed loop。

#### traffic control 目前也主要是静态/局部策略

- envelope 是固定 4096 commands / 8 MiB，不根据模型 region、cache pressure 或 PMU 调整；
- busy suppression、page clipping 和 duplicate suppression 是正确的安全机制，但不是自适应优化；
- controller 只覆盖 L2 hints，没有统一控制 DMA slots、VTCM occupancy、layout version 和 `:nt` last-use；
- bounded DINO 的 traffic 减少约 45 倍，但 latency 只比未 bounded 组合改善约 1.99%，说明命令效率本身并不是主要 latency bottleneck。

#### 结论

第三项当前应描述为：

> 已实现可观测、V73-aware 的 runtime traffic guard；PMU monitor 和能够实际改变 schedule 的反馈控制仍是待完成工作。

只有完成“监测 → 决策 → actuator → 下一窗口测量”后，才能称为 PMU-guided runtime controller。

### 8.5 三个贡献点如何形成闭环

正确的依赖关系是：

```text
Future Representation Contract / Layout Prefetching
       |  决定搬什么、目标布局、位置、复用和 deadline
       v
V-DAE movement schedule
       |  把 access/transform 与 execute 重叠，维护 tile readiness
       v
Runtime PMU + traffic controller
       |  观察 stall/traffic/occupancy，调节 distance/tier/budget
       +-----------------------------------------------+
                           feedback
```

三者不能各自独立堆叠：

- 没有第二点，V-DAE 可能异步搬运本来不该物化的数据；
- 没有第一点，第二点仍可能同步执行，不能隐藏剩余 transfer；
- 没有第三点，固定 lookahead 和 VTCM/L2 policy 会在不同模型上产生回退；
- PMU controller 不应补救错误的 movement plan，它只能调节已经证明必要的 movement。

这与论文故事线吻合，而且比“items 1–7 的功能列表”更统一。

---

## 9. 综合实施计划

### 9.1 总体原则

1. **论文叙事顺序与工程依赖顺序可以不同。** 论文可以先介绍 V-DAE 骨架；工程上必须先确定合法、值得搬运的 future representation，V-DAE 才知道该调度什么。
2. **先因果拆分，再增加机制。** 当前 item 7 会同时改变 fusion/slicing；不拆开就无法判断任何新收益来源。
3. **先用三个代表模型完成闭环，再扩展 15 个模型。** 每个小改动都跑 15 个完整模型会导致测试无休止；达到阶段 gate 后再全量验证。
4. **不以 hint/site 数量作为成功标准。** 必须证明 final IR/object、physical bytes/VMEM/stall 和 latency 同向改善。
5. **固定 compute mapping。** 同一实验中的 HVX/HMX rewrite、精度、输入、频率、运行顺序和 timing boundary 必须一致。

### 9.2 工作流与依赖

```text
P0 因果与观测基础
  -> P1 Future Representation / movement analysis
      -> P2 Layout Prefetching：消除 + in-situ formation
          -> P3 V-DAE：真实 readiness 与 overlap
              -> P4 PMU/runtime closed loop
                  -> P5 三模型集成 gate
                      -> P6 15 个完整模型验证
```

### P0：拆分 item 7，建立可信 baseline

#### 实现

把 item 7 拆成独立选项：

1. K/V semantic tracking；
2. K/V fusion-boundary policy；
3. attention slicing policy；
4. runtime K/V prefetch。

增加每阶段 IR dump 和 final-object summary：

- fusion groups / slicing sites；
- K/V marked/rejected/runtime sites；
- HMX rewrites；
- HVX/HMX/VMEM/VMEMU 指令数；
- alloc、copy、physical layout bytes。

#### 实验

先跑：

- Qwen2.5-0.5B：当前 1.87x 正信号；
- DINOv2-small：vision、0 HMX rewrite、1.69x；
- UniSpeech-SAT-base：0.93x 负例。

#### Gate P0

- 明确解释 Qwen/DINO 的 item 7 收益来自哪个 topology switch；
- 每个独立开关的 final object 差异可观测；
- 不把 zero-runtime-site 的结果归因于 prefetch。

### P1：Future Representation 与 physical movement ledger

#### 实现

增加 analysis-only pass，记录：

- logical value + producer version；
- consumer layout/engine/tile/deadline；
- descriptor-only view 与 physical transform 的区别；
- read/write/materialization bytes；
- first/last use、reuse、alias、alignment、page footprint；
- VRF/VTCM/DDR residency candidate；
- chosen/rejected reason。

暂不改变 codegen。

#### Gate P1

- ledger 的 physical transform sites 与最终 IR/alloc/copy 一致；
- 不再把 descriptor reshape 计入 saved bytes；
- 能在三个模型上列出 latency top regions 与 movement top regions，识别二者交集。

### P2：完成论文贡献二——Layout Prefetching

按风险从低到高分三步：

#### P2a：零 copy elimination

- contiguous view/collapse canonicalization；
- consumer indexing-map absorption；
- dead pack/unpack/transpose chain elimination；
- 严格 layout equivalence 和 alias proof。

#### P2b：producer-direct layout

- attention Q/K/V projection 直接产生 consumer layout；
- HMX activation producer 直接形成 AH；
- HVX consumer 使用 aligned/contiguous tile；
- multi-consumer 时由 cost model 决定 canonical、target 或双版本。

#### P2c：fused transform-transfer

- 只在确实必须跨 tier 搬动时使用 `prefetch_in_situ`；
- DMA/L2→VTCM 时一次完成 pack/transpose/place；
- immutable weight 使用 bounded persistent target-layout representation；
- page、VTCM capacity、bank 和 lifetime 进入 admission。

#### Gate P2

- 至少一个正例的 physical layout bytes/VMEM 明显下降；
- 负例不产生额外 layout version 或大幅回退；
- 任何 latency 收益都有被删除的 physical op/bytes 证据；
- 不要求此阶段单独达到 1.8x，但必须建立真实的 movement reduction。

### P3：完成论文贡献一——可执行的 V-DAE

#### P3a：显式 descriptor/readiness

引入每次 invocation 的 context 和 descriptor：

```text
(value version, tile, layout, source/destination tier, slot, generation)
FREE -> LOAD_PENDING -> LAYOUT_PENDING -> READY -> CONSUMING -> FREE
```

去除跨 invocation 的隐式 global ownership。

#### P3b：真实 overlap

- access side 只执行已被 P2 证明必要的 movement；
- UserDMA 或 scout 处理 transfer/transform；
- execute 等待 exact tile/version/slot；
- guarded epilogue，不重复预取 final tile；
- descriptor exhaustion、timeout、DMA failure 安全 fallback，而不是继续使用未 ready 数据。

#### P3c：把 lookahead 变为真实数据流

- `AdaptiveControlOp` 结果成为 loop-carried state 或 runtime issue API 的实际参数；
- next tile/address 由 effective distance 决定并有边界证明；
- buffer depth 与 lookahead 一致；
- L2 fire-and-forget 与 DMA readiness 分开建模。

#### Gate P3

- timeline/counters 证明 DMA/transform 与 compute 有重叠；
- wait time 下降且没有错误 slot；
- V-DAE off/on 只改变 schedule，不改变 compute mapping；
- 至少一个 P2 正例获得额外收益，负例可由 admission 自动关闭。

### P4：完成论文贡献三——PMU monitor 与 traffic closed loop

#### P4a：PMU feasibility probe

先在 V73 phone/Unsigned PD 上确认可读取的事件和权限，优先：

- HVX load/store L2 outstanding；
- L2 access/miss；
- VTCM outstanding；
- scatter/gather full；
- cycles、DMA latency/bytes。

若某些硬件 PMU 在当前 PD 不可访问，必须明确记录为 unavailable，并使用 wait cycles/DMA completion cycles 作为 fallback，不能仍称为硬件 PMU。

#### P4b：统一 telemetry window

每个 region/window 记录：

- PMU delta；
- requested/issued/dropped/suppressed bytes/commands；
- DMA queue depth、completion latency；
- VTCM occupancy；
- wait cycles 和 compute cycles；
- effective distance/tier/budget。

#### P4c：真正的 actuator

controller 必须实际改变下一 window：

- lookahead distance；
- L2 command/byte budget；
- L2 vs DMA/VTCM；
- descriptor/buffer depth；
- tile size；
- last-use `:nt`/retention policy。

采用 bounded/hysteresis 控制，避免每个 iteration 抖动。安全约束由 compiler contract 提供，runtime 只能在合法候选间选择。

#### Gate P4

- 日志可展示 `monitor -> decision -> actuator -> next-window response`；
- controller 不再只是写一个未消费的全局变量；
- traffic storm 被抑制，同时 wait/stall 或 latency 至少一项稳定下降；
- fixed policy 已最优时，controller 能保持而不是引入回退。

### P5：三模型集成与论文消融

使用以下 matched matrix：

| ID | 配置 | 证明目标 |
|---|---|---|
| B0 | HexKL-on matched control | 固定 compute mapping |
| E0 | B0 + semantic tracking only | metadata 本身无收益/回退 |
| E1 | B0 + plain necessary-data prefetch | 普通 prefetch 上限 |
| E2 | E1 + Layout Prefetching/in-situ transform | 减少 physical movement |
| E3 | E2 + V-DAE | overlap 剩余 movement |
| E4 | E3 + static traffic guard | page/single-flight/budget 安全性 |
| E5 | E4 + PMU closed loop | 跨模型自适应稳定性 |

每行报告：

- latency p50/p90 和重复波动；
- 正确性；
- HMX rewrite / HVX instruction；
- physical bytes、VMEM/VMEMU；
- L2/DMA/VTCM traffic；
- PMU/wait stalls；
- effective policy。

#### Gate P5

- Qwen 和 DINO 的收益具有明确机制证据；
- UniSpeech-SAT 的原有回退被 admission/controller 消除或显著收窄；
- 三项贡献逐步增加的消融可解释，不能依赖非单调、未知 codegen side effect；
- 目标是正例进入/超过 1.8x，同时负例保持接近 baseline。

### P6：15 个完整模型验证

达到 P5 后才串行运行 15 个完整模型：

- 按 language / vision / speech 分层报告；
- 同时报告 geometric mean、达到 1.8x 的模型数和最差回退；
- 将模型分成 compute-bound、compulsory-bandwidth-bound、layout/materialization-bound 和 latency-bound；
- controller 可以按类别选择不同策略，但所有策略来自同一 representation contract；
- 个别 3x 结果必须能由热点占比和 eliminated traffic 解释。

### 9.3 第一轮应实际执行的任务

综合风险和收益，下一轮不要先重写 V-DAE runtime，也不要立即跑 15 个模型。应按以下顺序开始：

1. 实现 P0 的四个 item 7 独立开关；
2. 为 Qwen/DINO/UniSpeech-SAT 生成 matched IR/object differential report；
3. 建立 P1 analysis-only physical movement ledger；
4. 从 ledger 中选择一个占比最高且可严格证明等价的 layout chain，完成 P2a；
5. 通过三模型 gate 后，再进入 producer-direct layout 和 V-DAE context/descriptor 改造。

这个起点能够最快回答两个关键问题：

- 当前 item 7 的实际收益究竟来自哪里；
- 模型中是否真的存在足以支撑 1.8x 的 physical layout/data-movement 热点。

若第二个问题的答案是否定的，就应及时把 1.8x 目标转向 online/fused attention、producer-consumer residency 等更大 movement region，而不是继续增加 prefetch hint。

---

## 10. 第 4 章能否单独形成一篇论文

### 10.1 结论

**有较强的独立成文潜力，但需要收敛为一个核心抽象，不能把第 4 章的十种方法平铺成十个并列贡献。**

最适合独立成文的核心不是某一种具体 prefetch 技巧，而是：

> **Hierarchical Representation Admission（HRA）/ Hierarchical Prefetch Admission（HPA）**：对每个 future data demand，在 compiler 给出的合法 action 集合中，联合选择“是否物化、由谁产生、以何种 layout、进入哪个 memory tier、何时搬运、保留多久以及服务哪些 consumer”，并由 runtime feedback 在安全边界内调整。

第 4 章其余方法应成为 HRA/HPA 的 **action space**：

- VRF/direct handoff；
- L2FETCH；
- DMA→VTCM；
- transform-copy/double buffering；
- persistent prepacked representation；
- multicast/one-fetch-many-consumers；
- page-aware K/V；
- cache-retention/`:nt`；
- recompute-versus-materialize；
- online/fused region。

这样问题、方法和实验都能统一，而不是一篇“优化技巧合集”。

### 10.2 独立成文能力评估

| 维度 | 评估 | 原因 |
|---|---|---|
| 问题独立性 | 高 | “同一 demand 应进入 L2、VTCM、VRF、persistent layout，还是不物化”本身是独立的 compiler-runtime 决策问题 |
| 核心抽象集中度 | 高，但需重写 | 以 HRA/HPA 为中心时集中；若保留十个并列方法则过散 |
| 与当前论文可分离性 | 中高 | 当前论文回答“如何形成并提前供应 representation”；未来论文回答“在多个合法供应路径中选择哪一个” |
| 新颖性 | 中等、待系统检索 | 单独的 async copy、scratchpad placement、prefetch selection、rematerialization 和 memory allocation 都有相关工作；潜在新意在移动端 Hexagon NPU 上的统一跨层级、跨 representation admission |
| 工程工作量 | 高 | 需要统一 candidate IR、cost model、全局 capacity/lifetime 规划、runtime actuator 和大量硬件计数器证据 |
| 实验工作量 | 高 | 需要多 domain、不同 memory behavior、静态/动态 policy、oracle/heuristic/runtime controller 的全面对比 |
| 论文风险 | 中高 | 若只有规则表和少量模型，容易被认为是 heuristic engineering；必须证明统一模型优于各局部策略 |

因此，该方向适合作为 **下一篇论文**，而不应在当前论文尚未建立三点闭环时立即扩大实现范围。

### 10.3 与相关工作的区别压力

初步相关工作核查表明，以下单点本身不足以构成原创性：

- FlashAttention 已用 IO-aware tiling/online reduction 减少 attention memory traffic；
- PagedAttention 已做 K/V page allocation、sharing 和按页访问；
- NVIDIA TMA 已提供 multi-dimensional async copy 和 shared-memory swizzle；
- COSMA 联合优化 DNN accelerator 的 operator schedule、scratchpad allocation 和 tensor replacement；
- Korch 允许 recomputation/fusion，以降低 kernel launch 和 data movement；
- runtime composite-prefetcher selection 已研究 phase-based 动态启停多个 prefetcher；
- tensor compiler 中也已有 data rearrangement 与 compute schedule 联合优化。

因此，未来论文不能只声称“首次选择 L2 或 VTCM”“首次做 recompute/materialize”或“首次动态开关 prefetch”。更可辩护的差异是：

1. **representation-aware**：选择的不只是地址或 memory tier，而是 `(logical value version, layout, tile, tier, consumer set, deadline)`；
2. **heterogeneous NPU execution-aware**：同时面对 HVX、HMX、UserDMA、VTCM、L2 single-flight 和 producer-direct placement；
3. **compiler-constrained runtime admission**：compiler 证明 legality 和生成候选，runtime 只在安全 action 集合中基于 PMU/traffic 调整；
4. **eliminate-before-prefetch**：action 包含“不物化、direct handoff、recompute”，而不是默认所有 demand 都必须搬运；
5. **one movement, multiple consumers**：把 multicast/reuse/residency 纳入 admission 收益，而不是逐 access 发 hint；
6. **完整模型与真实设备闭环**：用 final VMEM/bytes/PMU/latency 证明决策，而非 simulator-only 或 kernel-only。

上述差异仍需在正式立项时做系统性 literature review，当前只能判断为“有潜力”，不能提前宣称 novelty 已成立。

初步参照：

- [COSMA: Combined Scheduling, Memory Allocation and Tensor Replacement](https://arxiv.org/abs/2311.18246)
- [Korch: Optimal Kernel Orchestration for Tensor Programs](https://arxiv.org/abs/2406.09465)
- [Lightweight ML-based Runtime Prefetcher Selection](https://arxiv.org/abs/2307.08635)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [CUDA TMA / Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

### 10.4 当前论文与未来论文的边界

必须明确边界，否则 Hierarchical Admission 会吞掉当前论文的 PMU controller，或者当前论文提前用完未来论文的贡献。

#### 当前论文：ALPS

核心问题：

> 已知某个 representation/data movement 是必要且已选定的，如何在 Hexagon NPU 上减少 layout materialization，并通过 V-DAE 提前供应，再由 runtime 控制其强度和时机？

保留：

- V-DAE 骨架和 readiness；
- prefetch + in-situ layout transformation；
- Future Representation Contract；
- compiler 的保守、静态 legality/admission；
- runtime 的 **within-path control**：lookahead、command/byte budget、request coalescing、buffer depth和节流；
- PMU/wait feedback 调节已选路径的 intensity/timing。

不把以下内容做成当前论文的主贡献：

- 在 VRF/L2/VTCM/persistent/recompute 之间做全局最优选择；
- 跨多个 movement region 的 VTCM allocation/eviction；
- 动态改变 materialize/recompute/multicast 路径；
- 对所有第 4 章 action 进行联合搜索。

#### 未来论文：Hierarchical Representation Admission

核心问题：

> 对同一个 future representation demand，应该选择哪个 supply action，才能在容量、页、带宽、deadline、layout 和多 consumer 约束下最小化端到端 latency 与 movement？

以当前论文输出的合法 representation contract 和 V-DAE/runtime primitives 为 substrate，贡献集中在：

1. 统一 action/candidate IR；
2. graph-level capacity、lifetime、consumer-sharing 和 path cost model；
3. compiler-safe、PMU-guided hierarchical admission；
4. 静态、oracle、局部 heuristic、runtime adaptive 的系统比较。

### 10.5 为什么当前论文仍然需要“最小 admission”

不能因为 HPA 可能单独成文，就把当前计划中的所有 path selection 删除。当前论文至少需要：

- 拒绝没有收益窗口的 prefetch；
- 区分 fire-and-forget L2 hint 与需要 readiness 的 DMA/VTCM；
- 避免同一 demand 同时触发 L2 和 DMA；
- VTCM capacity/page/alignment legality gate；
- bounded request/byte envelope；
- synchronous fallback。

这些属于机制正确性和合理 baseline，而不是未来论文的全局优化贡献。类比而言，当前论文需要一个正确的 scheduler，但不需要解决所有候选路径的全局最优调度。

### 10.6 对当前 P0–P6 计划的更新

需要更新，但不是推翻原计划。更新原则是：

- 当前论文先完成 **mechanism + conservative admission + within-path feedback**；
- 同时把 future-paper 所需的候选特征和结果记录下来；
- 全局 hierarchical admission 在当前论文稳定之后启动。

#### 更新后的当前论文计划

```text
P0  item 7 因果拆分
  -> P1  Representation/movement ledger + candidate interface
      -> P2  Layout Prefetching mechanisms
          -> P2d Minimal Static Admission
              -> P3  V-DAE readiness/overlap
                  -> P4A PMU monitor + within-path traffic control
                      -> P5 三模型 gate
                          -> P6 15 模型完整验证
```

新增 `P2d Minimal Static Admission`：

- action 仅限 `no-op/native`、L2 hint、in-situ synchronous、DMA/VTCM async；
- compiler 根据 legality、reuse、tile bytes、page、VTCM fit 和最小 overlap window 做保守选择；
- 不做跨 region 全局搜索；
- 不动态选择 recompute/multicast/persistent 等复杂 action；
- 每个决定写入 auditable admission/rejection ledger。

原 `P4` 调整为 `P4A`：

- 读取可用 PMU/wait/DMA telemetry；
- 只调节已选路径的 lookahead、command/byte budget、buffer depth、coalescing 和 throttle；
- 暂不让 runtime 在 L2、VTCM、persistent 和 recompute 之间任意换路；
- controller 返回值必须真实作用于下一 window。

`P1` 增加 future-proof candidate interface：

```text
RepresentationDemand {
  value_version, layout, tile, consumers, deadline,
  legal_actions[], estimated_bytes, reuse, lifetime,
  page_footprint, alignment, capacity_requirement
}
```

当前论文可以只消费其中四种基础 action；未来论文可扩展 action set，而不重写前端分析。

#### 当前论文完成后的未来论文计划

```text
H0  冻结 ALPS substrate、baseline 和 candidate IR
  -> H1  扩展 hierarchical action space
      -> H2  graph-level admission/capacity planner
          -> H3  PMU-calibrated online admission
              -> H4  oracle/heuristic/adaptive 消融
                  -> H5  多 domain 完整模型与跨设备验证
```

##### H1：action space

- VRF/direct handoff；
- L2 lease；
- DMA/VTCM residency；
- persistent target layout；
- multicast；
- recompute；
- page-aware K/V；
- online/fused region；
- `:nt`/retention。

##### H2：全局 planner

- VTCM capacity/lifetime/coloring；
- multi-consumer saved movement；
- representation version 数；
- page/TLB/bank constraints；
- movement 与 compute critical path；
- region 间资源竞争。

##### H3：online admission

- compiler 输出合法 action envelope；
- runtime 根据 PMU/traffic/queue/thermal phase 选择 action；
- bounded/hysteresis，避免抖动；
- action failure 有确定 fallback；
- policy overhead 必须远小于被优化 region。

##### H4/H5：实验

- oracle upper bound；
- always-L2、always-VTCM、static heuristic、local greedy、global planner、adaptive controller；
- physical bytes、VMEM、stall、VTCM occupancy、energy、latency；
- language/vision/speech 三 domain；
- prefill/decode、HVX/HMX coverage 和不同 working-set phase。

### 10.7 对第 4 章各方法的归属建议

| 第 4 章方法 | 当前 ALPS 论文 | 未来 HRA/HPA 论文 |
|---|---|---|
| IO-aware online attention | 作为动机/可组合案例，不作为当前实现主线 | 可作为大 movement-region action/case study |
| Producer-driven direct handoff | Layout Prefetching 的一个必要低风险机制 | 由 admission 决定何时优于 materialization |
| Async transform-copy/double buffer | 当前 V-DAE + in-situ 核心机制 | 作为可选 supply action |
| Persistent prepacked weights | 当前只保留 bounded、静态 candidate 或已有消融 | 纳入跨 invocation admission/capacity |
| Multicast | 当前可保留已有机制，但不扩大为主线 | 作为 one-movement-many-consumers action |
| Page-aware K/V | 当前只保留真实 decode 的合法接口和基础 prefetch | 联合 page sharing/residency/action selection |
| Cache pollution/`:nt` | 当前 traffic guard 可做静态 last-use safety | 纳入全局 retention/admission |
| Recompute/materialize | 当前只在 ledger 中估计，不进入主实验 | 未来论文的正式 action |
| Hierarchical admission | 当前仅 minimal static admission | **未来论文核心贡献** |
| Alignment/page/layout co-design | 当前作为 legality 和 tile gate | 未来进入全局 objective/constraint |

### 10.8 立项 Gate

未来 HRA/HPA 论文应在以下条件满足后正式启动，而不是现在并行大规模实现：

1. 当前 P1 能稳定产生可信 physical movement/candidate ledger；
2. P2/P3 至少提供 L2、in-situ sync、DMA/VTCM async 三种正确 action；
3. P4A 能读取真实 telemetry 并改变下一 window；
4. 三模型中至少两个模型的不同 region 确实偏好不同 action；
5. static always-one-path 策略在至少一个模型/phase 上发生明显回退；
6. policy overhead 和搜索空间可控；
7. 更完整的 related-work review 能支持 representation-aware、compiler-constrained hierarchical admission 的差异性。

若上述第 4、5 条不成立，说明层级选择在当前硬件/模型上没有足够动态性，单独成文会比较薄；此时第 4 章应继续作为 ALPS 的扩展方向，而不是强行拆论文。

### 10.9 最终建议

当前最稳妥的策略是“一条实现主线、两个论文边界”：

- **现在**继续执行更新后的 P0–P6，优先完成当前 ALPS 的三点闭环；
- **实现接口时**为 HRA/HPA 保留 candidate/action/telemetry 扩展性，并积累决策数据；
- **不要现在**同时实现第 4 章所有 action；
- **当前论文完成三模型 gate 后**，依据 action preference 是否出现分化，决定未来论文正式立项；
- 若立项，Hierarchical Prefetch Admission 应是第二篇论文的唯一中心，其他第 4 章方法是其 action 和 case study。

这样既不会削弱当前论文，也能最大化第 4 章的后续研究价值。

---

## 11. ALPS 启动记录与当前实施状态

### 11.1 分支与兼容边界

- 开发分支：`alps_v73`；
- 基线分支：`baseline_5_upstream_v73`；
- 分叉基线：`fc06df1 Add staged FP16 five-way language benchmarks`；
- 新增用户接口、实验名称和论文叙事使用 `ALPS`；
- 历史 `enableOmniFetch*` 选项继续用于复现实验，`enableOmniFetchKvCachePrefetch` 被保留为完整历史 item 7 的 umbrella alias；
- 现有 `omni_fetch` MLIR dialect 和 runtime symbol 暂不做破坏性改名。它们属于内部兼容 ABI，不代表新实验仍把不同机制耦合在一起。

### 11.2 P0 已完成的第一阶段

历史 item 7 已拆成四个后端控制量：

| ALPS 控制量 | 只负责 |
|---|---|
| `enableAlpsKvSemanticTracking` | 识别并传播 K/V 逻辑语义 |
| `enableAlpsKvFusionPolicy` | 保留显式 fusion/topology boundary |
| `enableAlpsKvSlicingPolicy` | 改变 attention slicing policy |
| `enableAlpsKvRuntimePrefetch` | 插入实际 K/V runtime prefetch，并启用所需 lowering/runtime |

关键因果修复：

1. `omni_fetch.kv_cache_role` 现在只表示语义，不再自动禁止 fusion 或 split reduction；
2. topology policy 使用独立的 `alps.kv_fusion_boundary` 属性；
3. slicing 只受独立 slicing policy 控制；
4. runtime K/V 请求只受 runtime-prefetch 控制；
5. 历史 item 7 开关仍同时开启四项，保证旧命令和归档结果可复现；
6. module IR 写入四个 `alps.p0.*` 布尔属性，使每个编译产物的实际配置可审计。

完整模型入口新增：

```text
--alps-p0-mode none
--alps-p0-mode semantic
--alps-p0-mode fusion
--alps-p0-mode slicing
--alps-p0-mode runtime
--alps-p0-mode legacy-all
```

其中 `fusion`、`slicing`、`runtime` 均隐含 semantic tracking，但只额外打开所命名的策略；`legacy-all` 等价于完整历史 item 7。ALPS P0 配置与外部 PK/APT baseline 互斥，且当前严格实验仍要求关闭历史 layout-aware 与 adaptive 组合开关。

### 11.3 已完成验证

增量构建命令：

```bash
bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh
```

构建成功，生成的 `linalg-hexagon-opt` 位于：

```text
triton/build/cmake.linux-x86_64-cpython-3.11/
  third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt
```

`SDPA.mlir` 定向测试已经验证：

- semantic-only 产生 K/V role，但不产生 `alps.kv_fusion_boundary`；
- semantic + fusion policy 恰好对 QK/AV 两个 contraction 产生 boundary；
- 全部关闭时不产生 K/V role；
- Python 六种模式能够准确映射到四个后端控制量；
- Python syntax check 与 `git diff --check` 通过。

构建输出中的 warning 是当前上游 MLIR `OpBuilder::create` API 的既有 deprecation warning，不是本轮 ALPS 改动引入的编译失败。

### 11.4 接下来的严格执行顺序

P0 尚未整体通过 gate；当前只完成了“机制解耦和单元验证”。下一步严格按以下顺序执行：

1. 用同一 FP16、HVX/HexKL mapping、输入和测量边界，为 Qwen2.5-0.5B、DINOv2-small、UniSpeech-SAT-base 串行生成 `control / semantic / fusion / slicing / runtime / legacy-all` 六组产物；
2. 对每组保存最终 IR、编译日志、正确性、latency，并用 `scripts/script_release/internal/audit_hexagon_codegen.sh` 统计 object instruction/HVX/VMEM/HexKL 差异；
3. 补充 fusion group、slicing site、K/V marked/rejected/runtime site、alloc/copy/layout bytes 汇总；
4. 只有明确定位历史 item 7 的正收益和负回退来自哪个独立策略后，才宣布 P0 通过；
5. P0 通过后进入 P1 analysis-only representation/movement ledger，不提前扩展 HRA/HPA 的全局 action search。

这个顺序保持论文的三点主线不变：P1/P2 确定并减少必要 movement，P3 用 V-DAE overlap 剩余 movement，P4A 用 PMU/traffic feedback 调节已选路径；Hierarchical Representation Admission 仍作为未来论文方向保留。

统一脚本已增加 `--alps-p0` 模式，不为每个模型创建新脚本。三模型 gate 的执行形式为：

```bash
OUTPUT_DIR=/path/to/local/results \
REMOTE_RESULTS_DIR=/home/huzq85/2-working/working_set/alps_p0_YYYYMMDD \
scripts/script_release/internal/run_full_hvx_five_way.sh --alps-p0 \
  qwen2.5-0.5b dinov2-small unispeech-sat-base
```

脚本严格串行且不设置 host timeout；每个 case 完成后先生成 object codegen 摘要，再把该 case 的编译模型、日志和结果同步到 nano 的 `working_set`。只有 `rsync` 成功后才删除整个本地 case 目录，本地只保留 compact CSV/Markdown 总表。任何失败同样先移动第一次现场再停止，不自动重复设备运行。

### 11.5 P0 第一轮完整模型结果（2026-08-19）

实验固定条件：FP16、HVX vector、HexKL pipeline on、相同输入/模型结构/测量边界、单次 device measurement、严格串行。三个完整模型分别覆盖 language、vision 和 speech；所有 18 个 case 均通过模型正确性检查。

| 模型 | HexKL control | Semantic | Fusion policy | Slicing policy | Runtime prefetch | Legacy all | Control / legacy | Control / fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 10,769.83 ms | 10,786.03 ms | **5,860.91 ms** | 10,791.45 ms | 10,920.67 ms | **5,788.90 ms** | **1.86x** | **1.84x** |
| DINOv2-small | 9,755.36 ms | 9,841.48 ms | **5,950.36 ms** | 9,794.73 ms | 9,768.02 ms | **5,884.40 ms** | **1.66x** | **1.64x** |
| UniSpeech-SAT-base | 179,131.92 ms | 173,395.10 ms | 179,354.92 ms | 179,634.70 ms | 176,596.15 ms | 170,245.32 ms | 1.05x | 1.00x |

所有 runtime-prefetch case 均为：

```text
kv runtime sites = 0
runtime issued = 0
runtime issued bytes = 0
```

因此：

1. Qwen 的 1.86x 和 DINO 的 1.66x 不能归因于实际 prefetch 请求；
2. semantic-only 与 slicing-only 均接近 control，不能解释正例收益；
3. fusion-policy 单独复现了 legacy-all 的绝大部分收益：Qwen 两者只差约 1.2%，DINO 只差约 1.1%；
4. UniSpeech-SAT 没有获得 fusion-policy 收益，说明该 topology policy 不是“所有模型无条件更快”，必须进入后续 admission；
5. UniSpeech 的 semantic/runtime/legacy 小幅变化来自单次长时运行，当前没有独立机制证据，不应过度解释。

静态 object 汇总如下。`instructions`、`HVX` 和 `vector_mem` 是 disassembly 中的静态计数，不等于动态执行次数；三个模型本轮 HMX rewrite 均为 0，`hmx_mentions` 只是链接产物中的符号文本，不能作为 HMX coverage。

| 模型/配置 | Objects | Static instructions | HVX-like | Vector memory mentions |
|---|---:|---:|---:|---:|
| Qwen control | 26 | 957,545 | 184,291 | 216,786 |
| Qwen semantic | 26 | 957,509 | 184,291 | 216,786 |
| Qwen fusion | 26 | 1,155,962 | 222,331 | 281,298 |
| Qwen legacy-all | 26 | 1,129,630 | 222,331 | 281,298 |
| DINO control | 1 | 281,870 | 16,298 | 21,691 |
| DINO semantic | 1 | 281,873 | 16,298 | 21,691 |
| DINO fusion | 1 | 355,325 | 20,064 | 27,787 |
| DINO legacy-all | 1 | 357,467 | 20,075 | 27,787 |
| UniSpeech control | 1 | 106,307 | 19,136 | 24,963 |
| UniSpeech semantic | 1 | 106,303 | 19,136 | 24,963 |
| UniSpeech fusion | 1 | 116,723 | 21,981 | 29,546 |
| UniSpeech legacy-all | 1 | 117,213 | 21,979 | 29,546 |

正例中 fusion-policy 反而产生更多静态代码和 vector-memory 指令，却显著降低 latency。这并不证明“更多搬动更快”：静态指令数不包含 loop trip count、重复 recomputation、spill 次数或 critical-path stall。它说明原生大 fusion/multi-use/split-reduction 组合形成了较差的动态执行拓扑，而保留边界使 HVX schedule 更有效。下一步必须用细粒度消融和动态 movement/PMU ledger 验证具体是哪一种拓扑变化减少了动态工作或 stall。

完整产物已从本地移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p0_20260819
```

远端共约 48 GB，包含 18 份 `run.log`、18 份 `codegen.csv` 及编译模型；本地只保留约 28 KB 的 `results.csv` 和 `summary.md`。

### 11.6 P0 Gate 判断与 P0b

P0 已完成以下 gate：

- zero-runtime-site 结果不再被归因于 prefetch；
- semantic、slicing、runtime 和 topology 的 final object/latency 差异可观测；
- 已把 Qwen/DINO 的正收益缩小到 fusion topology policy。

但当前 `enableAlpsKvFusionPolicy` 仍同时控制：

1. elementwise/reshape producer-consumer fusion boundary；
2. 函数内存在 K/V boundary 时跳过全部 multi-use fusion；
3. K/V contraction 跳过 split reduction。

因此 P0 尚不能声称已定位到单一 topology switch。进入 P1 前增加一个有界的 `P0b`：把上述三项拆成独立开关，先在 Qwen、DINO、UniSpeech-SAT 上运行 `control + 三个独立项 + fusion-all`。若某一项复现大部分收益，则将其作为 ALPS 的 topology admission candidate；若需要组合，则报告明确 interaction，而不是继续使用含糊的 item7/fusion 标签。P0b 不增加新的优化机制，只完成已有收益的严格因果归因。

### 11.7 P0b 细粒度 topology 归因（2026-08-19）

P0b 把 coarse fusion policy 进一步拆成三个互斥、可独立审计的策略：

| P0b 模式 | 控制范围 |
|---|---|
| `elementwise-fusion` | 阻止标记的 K/V producer/consumer 参与 elementwise fusion，并使第二阶段 fusion driver 同样尊重该局部边界 |
| `multi-use-fusion` | 函数存在标记的 K/V region 时，关闭第二阶段 function-wide multi-use fusion |
| `split-reduction` | 只阻止标记的 K/V contraction 进入 split reduction |
| `fusion` | 保留 P0 coarse umbrella，作为历史收益的复现参考 |

对应完整模型命令为：

```bash
scripts/script_release/internal/run_full_hvx_five_way.sh --alps-p0b \
  --output-dir /home/huzq85/2-working/hexagon_npu/run_artifacts/alps_p0b_20260819 \
  --remote-dir /home/huzq85/2-working/working_set/alps_p0b_20260819 \
  qwen2.5-0.5b dinov2-small unispeech-sat-base
```

实际执行保持严格串行、FP16、HVX vector、HexKL pipeline on、相同完整模型/输入/测量边界、单次 device measurement 且不设置 timeout。15 个 case 均通过正确性检查。每个 case 结束后先生成 object codegen 摘要，再完整移动到 nano；只有传输成功后才删除本地 case。

| 模型 | HexKL control | Elementwise boundary | Multi-use policy | Split-reduction policy | Fusion umbrella | Control / elementwise | Control / umbrella |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 10,977.62 ms | **6,264.10 ms** | 10,853.23 ms | 10,823.07 ms | **5,894.99 ms** | **1.75x** | **1.86x** |
| DINOv2-small | 9,790.36 ms | **5,842.16 ms** | 9,794.75 ms | 9,784.64 ms | **5,908.53 ms** | **1.68x** | **1.66x** |
| UniSpeech-SAT-base | 176,086.71 ms | 173,129.62 ms | 172,366.09 ms | 180,908.01 ms | 180,149.61 ms | 1.02x | 0.98x |

全部 15 个 case 仍然是：

```text
prefetch hints = 0
runtime issued = 0
runtime issued bytes = 0
kv runtime sites = 0
```

因此 P0b 得到以下可证伪结论：

1. **主因已经定位。** DINO 的 elementwise boundary 单独达到 1.68x，且略快于 coarse umbrella；Qwen 单独达到 1.75x，解释了 coarse umbrella 1.86x 的绝大部分收益。
2. **multi-use 和 split-reduction 不是正例主因。** 两者单独在 Qwen/DINO 上均与 control 基本相同；Qwen 剩余约 6% 的组合收益是 interaction，不能再笼统归为整个 item 7。
3. **该收益不是 data prefetch。** 没有 runtime site、issued request 或 issued byte。当前有效机制是保留一个更有利的 representation/layout materialization 边界，避免原生 fusion 形成动态执行效率较差的拓扑。
4. **不是所有 attention-like region 都应保留边界。** UniSpeech-SAT 没有显著收益，coarse umbrella 还出现约 2% 回退；这证明后续必须做静态 admission，而不能全局无条件开启。
5. **论文叙事仍然连贯，但命名必须准确。** 该边界是 Layout Prefetching 的 supply/materialization 决策基础：先确定未来 consumer 所需 representation，在 producer 与 consumer 之间选择“直接生成目标 layout、保留边界后搬运、或融合计算”。在真正实现 transform-transfer overlap 之前，不能把本轮 1.68–1.75x 宣称为 prefetch overlap 收益。

final-object 静态汇总进一步支持这个归因。表中计数来自 disassembly，仍然只是静态代码形状，不等价于动态执行次数或 physical bytes。

| 模型/配置 | Objects | Static instructions | HVX-like | Vector memory mentions |
|---|---:|---:|---:|---:|
| Qwen control | 26 | 957,538 | 184,291 | 216,786 |
| Qwen elementwise | 26 | 1,121,792 | 219,523 | 276,570 |
| Qwen fusion umbrella | 26 | 1,155,978 | 222,331 | 281,298 |
| DINO control | 1 | 281,872 | 16,298 | 21,691 |
| DINO elementwise | 1 | 355,326 | 20,064 | 27,787 |
| DINO fusion umbrella | 1 | 355,327 | 20,064 | 27,787 |
| UniSpeech control | 1 | 106,304 | 19,136 | 24,963 |
| UniSpeech elementwise | 1 | 116,718 | 21,981 | 29,546 |
| UniSpeech fusion umbrella | 1 | 116,720 | 21,981 | 29,546 |

DINO 的 elementwise 与 umbrella final object 几乎完全相同，因此两者 latency 接近具有直接 codegen 证据。UniSpeech 也产生了同方向的静态 code shape 变化，却没有 latency 收益；这正是 P1 不能只看“是否发生变换”，而必须估计动态 trip count、reuse、materialization bytes、working set 和 critical-path stall 的原因。

P0 gate 至此通过。下一步进入 P1 analysis-only movement ledger，并把 `elementwise-fusion` 作为第一个已由完整模型验证的 topology candidate。P1 必须解释 Qwen/DINO 为什么接受该 candidate，而 UniSpeech-SAT 为什么应拒绝；在 ledger 能给出这种可核查差异前，不进入更复杂的 runtime admission。

完整编译产物和日志位于：

```text
nano:/home/huzq85/2-working/working_set/alps_p0b_20260819
```

远端约 42 GB，包含 15 份 `run.log`、15 份 `codegen.csv` 及完整编译模型；本地结果目录约 24 KB，只保留 `results.csv`、`summary.md` 和空的模型目录。

### 11.8 P1 当前状态与 Gate 补全计划（2026-08-20）

当前实施顺序与 10.6 的更新计划一致：`P0 -> P0b -> P1`。其中 P0b 是为完成 P0 因果归因而增加的有界步骤，不是绕过 P1/P2d 的新机制。当前没有提前实现 P2d、P3 或 P4A。

P1 已完成 analysis-only `RepresentationDemand`/movement ledger 的第一版，并在相同 FP16、HVX vector、HexKL pipeline-on、完整模型和串行协议下完成三模型 control/elementwise 对比：

| 模型 | Control | Elementwise boundary | Control / elementwise | 正确性 |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 10,797.93 ms | **5,779.59 ms** | **1.87x** | top-5 match |
| DINOv2-small | 9,751.94 ms | **5,883.58 ms** | **1.66x** | top-1 match |
| UniSpeech-SAT-base | 180,458.29 ms | 172,189.79 ms | 1.05x | last-frame top-1 match |

全部 case 的 runtime prefetch hints、issued requests 和 issued bytes 仍为 0，因此这些收益仍然只能归因于 representation/fusion topology，不是 transfer overlap。

当前 ledger 已满足：

- descriptor reshape/view 记为零 physical bytes；
- copy/pack/unpack/transpose 与 allocation 分开；
- value version、consumer、layout、engine、reuse/lifetime、alias、alignment、page footprint、VTCM fit 和 legal action 可审计；
- physical copy/materialization 汇总与 post-bufferization IR 一致。

但 **P1 gate 尚未完全通过**。当前只有 whole-model latency；Qwen 虽有逐 stage latency，DINO/UniSpeech 仍缺少 region-level latency top list，因而尚不能严格回答 latency-top 与 movement-top 的交集。静态汇总也显示：elementwise policy 在三个模型上都增加候选/alloc 和静态 vector code shape，但收益差异很大；仅凭 candidate/site 数无法做 admission。

因此下一步仍属于 P1，而不是提前进入 P2：

1. ledger site 增加 source-line/region identity，并分别输出 physical-movement top sites 与 representation-candidate top sites；
2. 复用 Hexagon LWP，在 DINO/UniSpeech 上获取 loop/region pcycles；Qwen 使用已有逐 stage device Perf，必要时只对 top stage 补 LWP；
3. 用 source line、contained ops 和 function/stage identity 对齐 latency top 与 movement top；
4. 至少形成可审计的保守 admission 特征：dynamic cycles、physical bytes、reuse、working-set/page footprint、vector-memory density、first-use window；
5. 若无法区分 Qwen/DINO 与 UniSpeech，则 P1 不通过，不得进入 P2d；若能够区分，才进入 P2a 的零-copy elimination。

完整 P1 产物位于：

```text
nano:/home/huzq85/2-working/working_set/alps_p1_20260820
```

### 11.9 P1 Gate 与快速推进策略（2026-08-20）

为补齐 region-level 证据，在 DINOv2-small 的 control 和
`elementwise-fusion` 上启用了 LWP。LWP 使用普通 ABI 的
`lwp_handler(i32)` 直接调用；其 pcycles 只用于热点排序，不作为正式
latency。正式 device latency 与未插桩 P1 结果吻合：control 为
9,749.55 ms，elementwise 为 5,931.64 ms，正确性均通过，收益为 1.64x。

control 中 11 个重复 attention region 同时进入 latency top 与 movement
top：source line 408、778、1129、1480、1831、2182、2533、2884、3235、
3586、3937。每个 region 的 LWP pcycles 约 4.63--4.69 亿，并对应三次
physical copy、合计 4,743,192 B materialization；单个最大 copy 是
`6x257x257xf32`（1,585,176 B）。因此已经识别出可审计的
latency--movement 交集，而不是只按 op 数量猜测。

elementwise 版本仍报告相同的静态 post-bufferization copy bytes，说明
其 1.64x 主要来自更好的动态 topology，而非已经完成 zero-copy。这也
给 P2a 提供了明确目标：attention 中
`expand -> transpose -> collapse -> batch_matmul` 链应由 consumer
indexing map 直接吸收，删除实际 transpose materialization；descriptor-only
expand/collapse 不计入节省 bytes。

完整 profiling 产物位于：

```text
nano:/home/huzq85/2-working/working_set/alps_p1_profile_20260820_retry2
```

P1 gate 据此通过。为避免在单个负例上过度投入，后续 gate 更新为：

1. 每阶段先用一个具有明确热点证据的代表性完整模型验证机制；
2. 机制和正确性成立后直接推进下一阶段，再扩展到结构相近模型；
3. UniSpeech-SAT 保留为负例和 admission 回归检查，但不再要求先解释其
   全部性能差异；
4. 只有 correctness、实际 physical bytes 未下降或代表模型明显回退才
   阻塞阶段推进。

下一步正式进入 P2a，先实现上述 attention transpose 的严格等价
consumer-indexing absorption；不恢复宽泛的字符串匹配式 layout 删除。

### 11.10 P2a：attention zero-copy indexing absorption（2026-08-20）

P2a 第一版只匹配 batch=1 且 shape/permutation 可静态证明的 QK 链：

```text
[1,M,H,K] -> transpose[0,2,1,3] -> collapse -> [H,M,K]
[1,N,H,K] -> transpose[0,2,3,1] -> collapse -> [H,K,N]
                                      batch_matmul -> [H,M,N]
```

pass 用一个四维 reduction `linalg.generic` 的 affine input map 直接访问两
个 producer tensor。batch 维必须严格等于 1，所有 M/N/H/K 和 element
type 必须一致；不满足证明条件时保持原 IR。expand/collapse 仅为 descriptor，
不计入 saved bytes；ledger 只累计两个被删除 transpose result 的物化字节。

DINOv2-small 完整模型结果如下。比较对象是同日、相同 FP16/HVX/HexKL、
相同 elementwise boundary 和输入的 P1 case，未为此重复运行基线：

| 配置 | Latency | 相对 elementwise | Copies | Allocs | Static materialization bytes |
|---|---:|---:|---:|---:|---:|
| Elementwise boundary | 5,931.64 ms | 1.00x | 133 | 298 | 66,290,754 B |
| Elementwise + P2a | **5,889.52 ms** | **1.007x** | **111** | **262** | **31,552,578 B** |

正确性通过（finite、top-1 match，max absolute difference 0.0049）。12 个
attention block 均命中，每个删除 2 个 transpose、报告 394,752 B
materialization，共删除 24 个 physical transpose。整体 materialization
bytes 下降 34,738,176 B，即 **52.40%**。

这证明 P2a 的 zero-copy 机制真实减少了搬动，但 0.72% latency 改善也说明
仅把物化 transpose 改成 strided affine consumer 仍不足：不连续 producer
layout 会把成本转移到 consumer load/addressing，而不是完全消失。因此不在
P2a 继续做无界调参；P2 gate 的“physical bytes 明显下降、正确性通过、无
明显回退”已满足，下一步进入 P2b，让 Q/K/V producer 直接生成 consumer
友好的 contiguous head-major layout，再由 P2a 删除中间表示。

代码提供独立开关 `enableAlpsZeroCopyAttention`，完整模型脚本入口为：

```bash
scripts/script_release/internal/run_full_hvx_five_way.sh --alps-p2a --compile-threads 4 \
  --output-dir /home/huzq85/2-working/hexagon_npu/run_artifacts/alps_p2a_20260820 \
  --remote-dir /home/huzq85/2-working/working_set/alps_p2a_20260820 \
  dinov2-small
```

完整编译产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2a_20260820
```

### 11.11 P2b：producer-direct attention layout（2026-08-20）

P2b 实现了一个独立、默认关闭的严格 pass，匹配 projection bias-add 的
单一使用链：

```text
add [B,M,H*D] -> expand [B,M,H,D] -> transpose
```

新 producer 通过 affine indexing map 直接写出 Q/V 所需的 BHMD，或 K
所需的 BHDM contiguous representation。匹配条件包括 tensor semantics、
静态 rank/shape、rank-1 bias、唯一使用者、严格 `addf + yield` body 以及固定
permutation；证明失败时保持原 IR。

第一次完整模型运行虽然通过正确性且得到 5,833.43 ms，但 P2b 命中数为
0、P2a 命中 12，因此该数值只是一次 P2a 重测，不能作为 P2b 结果。用保存
的 DINO 初始 MLIR 单独诊断后确认 P2b 可命中 36 个 producer；根因是 pass
原先位于 HexKL conversion 之后，前置 rank reduction/canonicalization 已把
producer 形态改写。最终将 P2b 移至 `LowerTmTensor` 后的第一个稳定
tensor-Linalg 边界；它只改变 matmul 的 downstream layout consumer，不改变
上游 named matmul，因而不影响后续 HexKL eligibility。

修正位置后，DINOv2-small 完整 FP16/HVX/HexKL 实验结果为：

| 配置 | Latency | 相对 elementwise | P2b hits | P2a hits | Copies | Allocs | Static materialization bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Elementwise boundary | 5,931.64 ms | 1.000x | 0 | 0 | 133 | 298 | 66,290,754 B |
| Elementwise + P2a | **5,889.52 ms** | **1.007x** | 0 | 12 | **111** | **262** | **31,552,578 B** |
| Elementwise + producer-direct P2b (+ P2a fallback) | 5,937.81 ms | 0.999x | **36** | 0 | 133 | 298 | 66,290,754 B |

P2b 正确性通过（finite、top-1 match，max absolute difference 0.0049）；36
个命中对应每层 Q/K/V 三个 producer，每个 canonical result 为 197,376 B。
但是 post-bufferization ledger 没有出现 physical copy、allocation 或
materialization bytes 的下降，latency 相对 elementwise 回退 0.10%，相对
P2a 回退 0.82%。这说明“删除 canonical add result”并不自动等价于删除
最终 bufferized physical movement；当前 producer-direct generic 也使原有
P2a contraction absorption 不再命中。

因此，**P2b 的实现、流水线定位、完整模型正确性与性能评估已经完成，但
P2b performance gate 未通过**。它保留为独立实验开关，不并入默认 ALPS
候选，也不继续在本阶段做无界调参。下一阶段若复用这一机制，必须让
producer-direct representation 与 contraction consumer/tiling 联合 lowering，
并以 post-bufferization physical bytes 下降为准，而不能只按 tensor IR 中被
删除的中间值计收益。

最终有效运行的完整产物位于：

```text
nano:/home/huzq85/2-working/working_set/alps_p2b_early_20260820
```

原始位置诊断运行保存在：

```text
nano:/home/huzq85/2-working/working_set/alps_p2b_20260820
```

### 11.12 P2c：fused transform-transfer 的实现边界与负结果（2026-08-20）

P2c 增加了独立、默认关闭的 `enableAlpsFusedTransformTransfer` 开关，把
HexKL MicroHMX 中下列相邻操作表示为一个 `prefetch_in_situ`：

```text
weight:     MicroHMXRmToWhF16          -> HMXWeight prefetch_in_situ
activation: CopySubmatrix + RmToAh     -> HMXActivation prefetch_in_situ
```

本阶段只允许同步、`lookahead=0`、HexKL micro-only 路径，不启用 P3 的
异步 readiness/admission，也不启用 persistent cache、two-dimensional DMA、
dequant 或 inter-layer prefetch。P2c 与已通过 movement gate 的 P2a 组合，
不与 P2b 组合。完整模型脚本入口为：

```bash
scripts/script_release/internal/run_full_hvx_five_way.sh --alps-p2c --compile-threads 4 \
  --output-dir /home/huzq85/2-working/hexagon_npu/run_artifacts/alps_p2c_20260820 \
  --remote-dir /home/huzq85/2-working/working_set/alps_p2c_20260820 \
  gpt2
```

实现审计发现一个必须明确记录的边界：当前同步 runtime lowering 虽然减少
了上层 IR op 数，但内部仍调用原有物理 kernel。`HMXWeight` 仍调用
`hexkl_micro_hmx_rm_to_wh_f16`；`HMXActivation` 仍先调用
`hexkl_micro_hmx_copy_submatrix_to_f16`，再调用
`hexkl_micro_hmx_rm_to_ah_f16`。因此当前 P2c 是统一的 transform-transfer
表示和未来异步 lowering 接口，**不是已经消除 intermediate movement 的
fused physical kernel**。pass 明确记录
`alps.p2c.proven_eliminated_physical_bytes = 0`，避免把“少了 IR op”误报为
“少了物理搬动”。

单元测试验证了严格匹配、同步语义和默认关闭行为：一个 weight site 与一个
activation site 共替换 3 个 IR op，且所有 P2c op 的 lookahead 均为 0。
增量构建、P2a 回归测试、脚本语法检查和 GPT-2 12-layer 完整 FP16 模型
正确性均通过。

为避免与历史测量混用，P2c 完成后在相同代码版本、输入、FP16/HVX/HexKL、
P2a 基础和单次 device measurement 下重新运行了匹配的 P2a 对照：

| 配置 | Latency | 相对 P2a | P2a hits | P2c weight sites | P2c activation sites | Replaced IR ops | Proven eliminated physical bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Elementwise + P2a | **3,180.99 ms** | **1.000x** | 12 | 0 | 0 | 0 | 0 B |
| Elementwise + P2a + P2c | 3,225.34 ms | 0.986x | 12 | **49** | **49** | **147** | **0 B** |

两者均通过 12-layer device full compare：finite、last-token top-1 match，
`max_abs=11.875`。P2c latency 比匹配的 P2a 对照回退 **1.39%**；两者的
post-bufferization movement ledger 相同。每个 transformer block 命中 4 个
weight 和 4 个 activation site，head 再各命中 1 个，证明路径确实执行，
但没有产生可验证的 physical-byte reduction。

因此，**P2c 的接口、pass、流水线、测试与完整模型评估已经完成，但 P2c
physical-movement/performance gate 未通过**。该机制保留为默认关闭的研究
开关，不进入当前 ALPS 默认候选，也不能在论文中宣称为 fused transfer 的
性能收益。若后续重新开启，前提是增加真正的 direct DDR/L2-to-AH/WH
lowering 或新的 HexKL tile API，使 copy 与 layout transform 在同一个物理
producer 中完成，并由 PMU/ledger 证明 external traffic 或 materialization
bytes 下降；在此之前不继续对当前 wrapper 做无界调参。

完整产物已移动到远端，本地不保留大文件：

```text
nano:/home/huzq85/2-working/working_set/alps_p2a_gpt2_20260820
nano:/home/huzq85/2-working/working_set/alps_p2c_20260820
```

### 11.13 P2d：Minimal Static Admission（2026-08-20）

P2d 实现为独立、默认关闭的 `alps-minimal-static-admission` pass，并通过
顶层 `enableAlpsMinimalStaticAdmission` 和统一脚本 `--alps-p2d` 接入。它不做
跨 region 全局搜索，也不恢复 P2b/P2c 的失败路径；当前 action set 严格限定为：

```text
no_op/native | l2_hint | in_situ_sync | dma_vtcm_async
```

每个候选都记录 `kind/action/reason/legal_actions/tile_bytes/reuse/pages/
alignment/vtcm_fit/overlap_window/materialize`，函数级 summary 记录各 action
数量、拒绝数量和 planned bytes。选择规则刻意区分“法律上可用”和“当前可以
安全执行”：

- P2a 已消除的 representation 选择 `no_op`，不能再对其预取；
- 只有静态、DDR entry/persistent、达到 byte/window 阈值的 K/V stream 才可
  选择 page-safe L2 hint；
- eager producer 在当前 invocation 内生成的 K/V 选择 native，防止在数据生成
  前发出因果无效的 prefetch；
- `in_situ_sync` 必须有 eliminated physical bytes 证据。P2c 已证明当前 wrapper
  为 0 B，因此拒绝；
- DMA/VTCM 即使满足 capacity、page、alignment 和 overlap 条件，在 P3 建立
  exact descriptor/readiness/slot ownership 前也不能 materialize；
- 一个 demand 只能有一个 chosen action，P2d materialization 模式下
  `PrefetchInsert` 只接受 `alps.p2d.action=l2_hint`，避免同一 demand 同时进入
  L2 和 DMA 路径。

单元测试覆盖了 persistent K/V 的 L2 接受、produced K/V 的拒绝、HMX
sync/DMA 候选拒绝、P2a zero-copy 的 stable `no_op` 传播和 admission-gated
materialization。P1、P2a、P2c 与 P2d 共 4 个回归测试全部通过；增量构建
成功。

DINOv2-small 完整 FP16/HVX/HexKL 运行使用 P2a 作为稳定 movement reduction
基础，结果如下：

| 配置 | Latency | 相对 P2a | P2a zero-copy hits | Runtime hints/issued/bytes | 正确性 |
|---|---:|---:|---:|---:|---|
| Elementwise + P2a | 5,889.52 ms | 1.000x | 12 | 0 / 0 / 0 B | PASS |
| Elementwise + P2a + P2d | **5,863.42 ms** | **1.004x** | 12 | 0 / 0 / 0 B | PASS |

P2d 运行识别 36 个 eager K/V stream、72 个 HMX weight transform 和 72 个
HMX activation transform，共 180 个 transfer candidate，全部选择 native：

- 36 个 K/V：`source_not_entry_persistent`；
- 72 个 2 KiB weight tile：低于保守的 4 KiB DMA threshold；
- 72 个 activation tile：当前 sync wrapper 的 proven byte reduction 为 0 B。

因此没有生成 runtime request，正确性为 finite、top-1 match、
`max_abs_diff=0.0049`。与 P2a 的 0.44% 差异视为测量波动，不能宣称 P2d
性能收益；P2d 本阶段的贡献是阻止不盈利/不安全 action 引入回退和 request
storm，并为 P3 输出可审计的候选及拒绝原因。

第一次完整运行时，P2a 的逐 op 属性在 bufferization 后消失，因此 P2d summary
只列出上述 180 个 transfer candidate，而没有把 12 个已消除 representation
统计成 `no_op`。随后补充函数级稳定的 P2a site/byte contract，并用组合 IR
回归证明 P2d 能消费该 contract；该修复只改变 ledger 属性和日志，不改变
codegen，故没有为此重复运行十分钟级完整模型。后续完整运行的预期 summary
为 12 个 `no_op` 加 180 个 native/rejected transfer candidate。

P2d gate 据此通过：action 互斥、决定可审计、L2 materialization 有独立测试、
错误的 sync/async 路径被保守拒绝、完整正例无明显回退。下一阶段进入 P3，
先为 DMA/VTCM action 建立 invocation-local descriptor 和 exact readiness；
不能直接复用 process-global ring 来绕过 ownership 证明。

完整模型产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2d_20260820
```

### 11.14 P3a：invocation-local descriptor 与 exact readiness（2026-08-20）

P3a 已建立 P3 的安全基础，但刻意没有在本阶段启动 UserDMA 或宣称 overlap
收益。对现有实现的审计确认，legacy V-DAE 的 semaphore 只表达“某个工作完成”，
process-global async ring 只按 FIFO 队头消费；两者都没有携带当前 consumer 所需
的 value version、tile、layout 或 memory tier，因此不能证明“完成的是正确数据”。

新增的 exact-readiness descriptor 保存：

```text
(invocation generation, value version, tile, layout,
 source tier, destination tier, slot, slot generation)
```

其唯一合法生命周期为：

```text
FREE -> LOAD_PENDING -> LAYOUT_PENDING -> READY -> CONSUMING -> FREE
```

runtime 使用 4 个 bounded invocation context、每个 context 8 个 descriptor。
底层 storage 仍是静态 bounded pool，以避免设备端动态分配；但是 ownership 不再
是隐式 process-global ring：每次 invocation 必须显式 begin/end，context handle
和 descriptor handle 均携带 generation。stale context/descriptor、错误 tile/
version/layout/tier、非法状态跳转、未释放 descriptor 的 context end 都会失败并
设置 error bit。descriptor-full 返回负 handle，为 P3b 保留同步 fallback，而
不是覆盖仍在使用的 slot。

编译器新增了独立、默认关闭的 `alps-exact-readiness` pass、顶层
`enableAlpsExactReadiness` 和统一脚本 `--alps-p3a`。P3a 只接受已经由 P2d 选择
`dma_vtcm_async`、处于明确 tile loop 且带 tile operands 的 prefetch；否则记录
拒绝原因。dialect 的 invocation/descriptor ops 已完整 lower 到 runtime ABI。
P3a 不复用 legacy semaphore/ring，也不在没有真实 action 时插入空 descriptor
调用。

验证结果：

- Hexagon v73 runtime bitcode 与完整增量构建成功；
- 6 个 P1/P2/P3 定向 lit 回归全部通过；
- host runtime contract 验证完整状态路径、错误 tile 拒绝和跨 invocation stale
  handle 拒绝；
- DINOv2-small 完整 FP16/HVX/HexKL 正确性通过：finite、top-1 match、
  `max_abs_diff=0.0049`。

| 配置 | Latency | 相对 P2d | P2d async action | P3a exact contract | Runtime hints/issued/bytes |
|---|---:|---:|---:|---:|---:|
| P2a + P2d | 5,863.42 ms | 1.000x | 0 | N/A | 0 / 0 / 0 B |
| P2a + P2d + P3a | 5,871.00 ms | 0.999x | 0 | 0 | 0 / 0 / 0 B |

7.58 ms（0.13%）差异属于测量波动。完整运行中 P2d 正确统计 12 个 P2a no-op
和 180 个 native/rejected transfer candidate；由于没有批准 DMA，P3a 报告
`async_candidates=0, exact_contracts=0`，最终 binary 没有执行 descriptor 调用。
这验证了 P3a 的关键负向 gate：没有 exact identity 的 action 不会暗中进入旧
global ring。

因此 **P3a 的 ownership/readiness correctness gate 已完成**，但它不是性能
结果。下一阶段 P3b 必须选择一个真实正例（首先是满足 VTCM fit、tile identity
和 overlap window 的 HMX weight tile），让 P2d 批准唯一的
`dma_vtcm_async` action，并把 UserDMA token、layout completion 与 scout 全部
绑定到该 exact descriptor。只有 PMU/timeline 证明 DMA 与同一 mapping 的 compute
重叠、consumer 等待的是同一 tile/version、错误/容量 fallback 正确，才能进入
P3c lookahead。

完整产物已移动到远端，本地仅保留 16 KiB 汇总：

```text
nano:/home/huzq85/2-working/working_set/alps_p3a_20260820
```

### 11.15 P3b：descriptor-bound UserDMA/scout overlap 与失败 gate（2026-08-20）

P3b 已实现独立、默认关闭的 `enableAlpsExactOverlap` / `--alps-p3b`
路径。P2d 在 P3 exact-readiness 模式下可把满足静态 tile identity、VTCM fit
和 overlap window 的 2 KiB HMX weight tile 唯一选择为
`dma_vtcm_async`；decomposition 同时为 P3b 预留不与 HMX working set 混叠的
VTCM staging bank。编译器对每个获准的 K-loop 生成：

```text
tile 0: synchronous bootstrap
iteration i: exact kick(tile i+1, descriptor, UserDMA token, VTCM slot)
iteration i+1: consume(exact context/version/tile) -> HMX MM -> release
```

runtime 将 UserDMA token、source/destination、RM tile identity、WH destination、
VTCM stage 和 scout completion 全部绑定到 P3a descriptor，不再通过 legacy
process-global FIFO 查找工作。新增 counter 分别记录 kick、DMA completion、
scout completion、同步 fallback、consumer wait，以及 acquired/consumed/released/
failure。定向验证包括：完整增量构建、7 个 P1/P2/P3 lit 回归、host
exact-readiness contract、Python/bash 语法和 LLVM address-space conversion，均
通过。

第一次 DINOv2-small 完整运行暴露出 pipeline wiring 问题：P3b 同时启用 P2d
时仍被判为 `kvCacheOnly=1`，导致 72 个已准入 HMX weight site 没有进入
PrefetchInsert，runtime exact counter 全为 0。该运行 latency 为 5,799.27 ms，
但它是 P3a 等价 no-op，**不是 P3b 性能结果**。修复后 P3b 明确令
`kvCacheOnly=0`。

第二次完整运行证明真实路径已物化：

| 指标 | 结果 |
|---|---:|
| P2d `dma_vtcm_async` sites | 72 |
| PrefetchInsert candidate loops | 144 |
| P3a exact contracts | 72 |
| Latency | 8,142.30 ms |
| P3a latency | 5,871.00 ms |
| 相对 P3a | 0.721x（回退 38.69%） |
| Exact kicks / completed / scout-completed | 31,552 / 31,552 / 31,552 |
| Sync fallbacks | 143,408 |
| Consume spins | 47,364,853 |
| Descriptor acquired / consumed / released | 31,552 / 31,526 / 31,526 |
| Descriptor failures | 162,468 |
| 输出 | finite、top-1 match，max_abs_diff=0.1093 |

该结果不能被当作 exact-readiness 正确性通过：旧 consumer 在有限自旋耗尽后
仍尝试 consume 非 READY descriptor，26 个 descriptor 未被释放，继而引发
request storm、同步 fallback 和错误累积。代码随后改为 demand-time work
stealing：只有仍处于 `LOAD_PENDING` 的 descriptor 可由 consumer 通过 CAS
领取并完成；已经由 scout 领取的 `LAYOUT_PENDING` descriptor 必须等待 READY。

最终验证中编译于 564.01 s 正常完成，但 DSP kernel 卡在
`LAYOUT_PENDING`，未产生 latency。现场表明 scout 已取得 descriptor，随后
阻塞在 UserDMA completion/WH publication；这也说明当前 UserDMA `wait(token)`
是无限轮询接口，不能满足 P3b 所需的 bounded failure contract。运行已人工
中止，手机端没有残留 `run_main_on_hexagon` 进程。代码增加 device fail-fast
watchdog：layout owner 超时后触发明确失败，绝不继续让 HMX 消费非 READY tile，
也避免再次无限占用 DSP。

因此结论是：**P3b 的 compiler/runtime mechanism、exact identity 和真实正例
物化已经完成，但 device correctness/performance gate 未通过，不能进入 P3c。**
阻塞点不是继续调 lookahead，而是先补齐以下有界 runtime substrate：

1. UserDMA nonblocking completion/status 或可超时 wait，能够区分 complete、
   engine fault 和 token stale；
2. 单 scout/single-flight admission 与 DMA queue credit，禁止 72 个静态站点形成
   completion backlog；
3. timeout 时可安全取消/隔离目标 slot，不能与仍可能写 VTCM 的 owner 竞争；
4. 重新验证 `acquired == consumed == released`、failures=0，再比较 P3a latency；
5. 上述条件满足后才实现 P3c adaptive loop-carried lookahead。

三次现场均已保存到 nano，本地大 artifacts 已删除：

```text
nano:/home/huzq85/2-working/working_set/alps_p3b_20260820
nano:/home/huzq85/2-working/working_set/alps_p3b_fix1_20260820
nano:/home/huzq85/2-working/working_set/alps_p3b_final_20260820
```

### 11.16 P3b bounded runtime 修复、完整模型结果与阶段决策（2026-08-20）

P3b 的设备卡死根因不是模型规模，而是 runtime ownership：UserDMA 由计算
hardware thread 启动后，旧实现把 completion poll/WH publication 交给 scout
thread。V73 UserDMA completion 状态不能被这种跨 hardware-thread owner 模型安全
解释；scout 取得 descriptor 后会停留在 `LAYOUT_PENDING`。因此修复不是扩大
watchdog 或重复运行，而是：

- 新增 bounded nonblocking UserDMA `poll(token)`；
- exact DMA 使用 single-flight credit，禁止 descriptor/queue storm；
- token 0 按合法首个 ring token 处理，由显式 `dma_active` 判断有效性；
- start failure 使用同步 fallback；timeout 标记 `FAILED` 并 fail-fast，绝不让 HMX
  消费未 READY tile；
- P3b 不再隐式启用 dual-thread DAE。kick 与 completion poll 均由 issuing compute
  thread 执行；DMA 仍可在 kick 与下一次 consume 之间同 intervening HMX compute
  重叠。dual-thread DAE 保留为独立开关，只用于 ownership 可证明安全的 action。

按要求不再使用 Debug 模型；修复后直接运行完整 DINOv2-small，结果如下：

| 指标 | 修复后结果 |
|---|---:|
| 完整模型 latency | **10,679.04 ms** |
| P3a matched reference | 5,871.00 ms |
| P3b / P3a speedup | **0.550x**（回退 81.89%） |
| P2d DMA/VTCM async static sites | 72 |
| P3a exact contracts | 72 |
| Exact kicks / completed | 174,960 / 174,960 |
| Acquired / consumed / released | 174,960 / 174,960 / 174,960 |
| Scout completed | 0（设计如此） |
| Sync / credit fallback | 0 / 0 |
| Consume spins / DMA timeout / descriptor failure | 0 / 0 / 0 |
| Correctness | finite、top-1 match、max abs diff 0.0049 |

该结果完成了 P3b 的 device correctness、bounded failure 和 ownership gate：真实
DMA 路径执行了 174,960 次，descriptor 全闭合，没有 timeout、fallback、错误 slot
或 exit 13。但 performance gate 明确失败。当前每个 2 KiB weight tile 单独执行
UserDMA start/poll，再执行 WH transform；虽然 transfer 在逻辑上提前，细粒度 DMA
command/poll 和 transform publication 成本超过了隐藏的等待时间。它不是继续放大
lookahead 就能合理解决的问题。

因此不再对 P3b 做无界局部调参，也不立即进入 P3c。P3b 机制和代码保留为独立、
默认关闭的消融项。下一步直接进入 P4A：用 PMU 可用性探测以及 DMA completion
cycles、issued bytes、queue credit、wait/compute cycles 等 fallback telemetry 建立
within-path traffic control。P4A 的第一项任务就是让 controller 能识别这种
“正确但 command overhead 高于可隐藏 stall”的路径，并在下一 window throttle
或关闭它。这既利用了本轮负结果，也保持论文的 `movement selection -> V-DAE ->
monitor/traffic control` 故事线，而不是为了追逐单点 latency 继续修改调度。

完整编译产物和日志已移动到远端，本地只保留 16 KiB 汇总：

```text
nano:/home/huzq85/2-working/working_set/alps_p3b_repair_full_20260820
```

### 11.17 P4A：PMU feasibility 与 within-path traffic closed loop（2026-08-20）

P4A 新增独立、默认关闭的 `enableAlpsTrafficControl`，统一脚本入口为
`--alps-p4a`。脚本会显式启用其语义依赖 P2d/P3a/P3b，但 P4A 本身可单独关闭，
关闭后生成代码与 P3b 相同。controller 不改变 representation、HMX mapping 或
合法 action 集，只在已经由编译器批准的 exact-DMA 路径内选择下一 window 保持
DMA，还是退回合法同步 transform。

#### PMU feasibility

P4A 按 V73 public event 定义探测以下四项：

- `UDMA_ACTIVE_CYCLES` (`0x812f`)；
- `UDMA_DMPOLL_CYCLES` (`0x8133`)；
- `UDMA_COHERENT_RD_CYCLES` (`0x814b`)；
- `UDMA_VTCM_WR_CYCLES` (`0x8150`)。

当前手机运行于 unsigned PD。Hexagon SDK 6.4 的 `HAP_user_pmu.md` 明确说明
HAP user PMU 只支持 debug-enabled device，且 **unsigned PD 不可访问**。完整
DINOv2-small 也实际返回 `pmu_status=0, pmu_reads=0`。因此当前实验不把软件计数
冒充 hardware PMU；P4A 明确切换到 fallback telemetry：processor pcycles、
DMStart/poll cycles、poll retry、issued/suppressed commands、descriptor credit、
wait/failure/timeout。

#### 第一版 controller 负结果

第一版只在一个 64-completion window 的 poll retry 为 0 时认定 transfer 过早并
throttle。完整模型观测到：

| 指标 | 第一版 P4A |
|---|---:|
| Latency | 10,967.47 ms |
| DMA kicks / completed | 174,960 / 174,960 |
| Windows hold / throttle | 2,733 / 0 |
| Total poll retries | 1,064,159（6.08/tile） |
| Issue / poll pcycles | 29,935,940 / 229,886,261 |
| Correctness | finite、top-1 match、max abs diff 0.0049 |

这说明“没有 consumer semaphore spin”不等于 transfer 没有 demand-time 成本；
P3b 的 retry 位于 exact completion 内部。controller 因判据不完整而全部 hold，
比 P3b 还回退 2.70%。

#### 有界修复与最终闭环

最终固定判据同时抑制两个无收益端点：window retry 为 0 表示全部过早；平均
retry 大于等于 4 表示 demand 仍在持续支付 DMPoll pressure。只有中间区间保持
DMA。该阈值在第二次完整运行前固定，不做 per-model 搜索，也不继续调参。

| 指标 | P3a | P3b | P4A final |
|---|---:|---:|---:|
| Latency | **5,871.00 ms** | 10,679.04 ms | **6,269.39 ms** |
| 相对 P3b 加速 | 1.82x | 1.00x | **1.70x** |
| DMA kicks / completed | 0 / 0 | 174,960 / 174,960 | **64 / 64** |
| Controller windows: throttle / hold | N/A | N/A | **1 / 0** |
| DMA suppressed to sync | N/A | 0 | **174,896** |
| 首 window poll retries | N/A | N/A | 444（6.94/tile） |
| Acquired / consumed / released | N/A | 全闭合 | **174,960 / 174,960 / 174,960** |
| Failure / timeout | 0 / 0 | 0 / 0 | **0 / 0** |
| Correctness | PASS | PASS | **PASS** |

这已经满足最小的 `monitor -> decision -> actuator -> next-window response`：首
window 的真实 fallback telemetry 触发一次 throttle，随后 DMA command 数从预期
174,960 降到 64，latency 相对 P3b 恢复 1.70x。它也证明 traffic controller
不能只调 lookahead；必须能够关闭成本高于被隐藏 stall 的路径。

边界同样必须明确：P4A final 仍比 P3a 慢 6.79%，所以它证明的是动态控制能从
错误静态 DMA 决策中恢复，而不是在当前 DINO 配置上胜过 oracle-like 静态 off。
P3b 和 P4A 均继续保持默认关闭、可独立消融；不再为该模型调 threshold。后续
P5 应把 `static-off/P3b-always/P4A-adaptive` 同列，验证其他结构是否存在 controller
选择 hold 的正例。

两轮完整运行均已移动到 nano，本地各只保留 16 KiB 汇总：

```text
nano:/home/huzq85/2-working/working_set/alps_p4a_full_20260820
nano:/home/huzq85/2-working/working_set/alps_p4a_final_full_20260820
```

### 11.18 P2e：Consumer-Driven In-Situ Layout Formation（2026-08-20）

P2a/P2b 的 attention-only pattern 之后，新增了独立且默认关闭的 P2e：
`enableAlpsConsumerDrivenLayout`，统一脚本入口为 `--alps-p2e`。它不从某个
预取原语反向猜 layout，而是从 physical transpose 的终端 consumer 收集
representation demand，再决定 producer 是否可以直接形成目标 layout。

第一版 legality gate 刻意保守：只接受静态 rank 2--4、唯一 producer chain、
全 parallel tensor `linalg.generic`、identity producer output map、单一已知 engine
consumer，并要求 permutation 保持最内连续维。它同时支持
`producer -> transpose` 和完整模型常见的
`producer -> expand_shape -> transpose`。后者把 expand reassociation 严格展开为
affine source map，使 producer 直接写 transpose output；无法证明、multi-engine、
动态或会破坏最内 unit-stride 的情况保持 native。native op 不携带诊断属性，
contract 只写 function-level summary/日志，避免观测属性影响 canonicalization。

定向测试覆盖了 immediate producer、expanded producer 以及 innermost-stride
负例。完整 DINOv2-small 初始 IR 中共发现 122 个 demand，其中 121 个终端
consumer 为 HVX-bound Linalg；36 个 Q/K/V producer chain 通过 gate，86 个
保持 native。tensor-level contract 估计消除 7,105,536 B canonical
materialization。

同一代码、FP16/HVX/HexKL、输入和设备状态下的完整模型 matched 结果：

| 配置 | Latency | 相对 P2e | P2e direct/native | Pre/post-fusion descriptor sites | 正确性 |
|---|---:|---:|---:|---:|---|
| HexKL-on matched control | 9,794.41 ms | 1.58x | 0 / N/A | 108 / 121 | PASS |
| P2e 第一次 | 6,267.24 ms | 1.01x | 36 / 86 | 48 / 58 | PASS |
| P2e 反向顺序重复 | **6,201.29 ms** | **1.00x** | 36 / 86 | 48 / 58 | PASS |

两次 P2e 相差 1.06%，相对 control 分别为 **1.56x/1.58x**。所有结果均为
finite、top-1 match、`max_abs_diff=0.0049`。final object 的 control/P2e 总指令
为 281,871/282,590，HVX-like 为 16,298/16,590，HMX mentions 为 21/20，DMA
均为 0，因此没有把主要 compute 偷换到另一后端。

边界必须明确：旧 P1 post-bufferization ledger 对两者都报告 133 copies 和
66,290,754 B static materialization，P2e 甚至为 285 allocs、control 为 284。
所以目前不能声称已经由该粗粒度 ledger 证明 external physical bytes 下降。
当前最强的因果证据是 36 个明确 rewrite、tensor layout chain/descriptor sites
显著减少、matched object mapping 以及重复 latency。下一步需要在 layout
generic/vector-transfer 层补充 VMEM bytes、stride、address-generation 和 fusion
region ledger，解释为何相同 copy summary 下 latency 明显变化。

HMX 边界也不能模糊：P2e 当前完成的是 HVX consumer-driven direct formation；
P2c 虽能表达 HMX WH/AH transform-transfer，但底层仍执行原 copy+transform
kernel。真正的相邻 HMX MatMul producer-consumer 还需要让前一 MatMul 保留 AH
output，并让后一 MatMul 直接消费 AH，消除 `AH -> row-major -> AH`；这要求扩展
HexKL op/layout type 与 micro lowering 的语义，不能仅添加属性或删除转换 op。
该 HMX 子阶段应作为 P2e-HMX 独立开关，在 exact layout/version、单 consumer、
tile compatibility 与 fallback 均证明后再做完整模型实验。

#### TODO：P2e-HMX（暂不进入当前 P2e 结论）

- 为 HexKL MatMul 的 AH/WH representation 建立显式、可验证的 layout contract，
  而不是只在 tensor op 上附加提示属性。
- 识别 layout/version、tile shape 和唯一 consumer 均兼容的相邻 HMX MatMul，
  让前一 MatMul 直接保留 AH output、后一 MatMul 直接消费 AH，消除中间的
  `AH -> row-major -> AH` 往返。
- 必要时扩展 HexKL op/type、bufferization interface 与 micro lowering；不允许
  通过删除转换 op 却保留错误物理表示来获得表面加速。
- 以独立且默认关闭的 `P2e-HMX` 开关实现，所有不满足证明条件的路径严格回退；
  最终同时用 post-bufferization movement ledger、final object/PMU 和完整模型
  correctness/latency 验证。

产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2e_dinov2_20260820
nano:/home/huzq85/2-working/working_set/alps_p2e_dinov2_v2_20260820
```

### 11.19 P2f：Consumer Layout Contract 到 HVX Codegen（2026-08-20）

当前 P2e 的准确定位是 **HVX consumer-driven in-situ layout formation**：它把
consumer 要求的表示直接编入 producer 的 indexing maps，并删除显式
expand/transpose materialization。ALPS 的整体范围仍大于 P2e，还包含数据提前
供给、V-DAE 和 runtime traffic control，不能把整个 ALPS 简化成 layout pass。

为验证显式 contract 继续传播到 tiling/vectorization 是否能进一步提速，新增了
独立且默认关闭的 P2f：`enableAlpsConsumerLayoutPropagation`，统一脚本入口为
`--alps-p2f`。P2f 为 P2e admitted op 记录 permutation、连续 loop 和
`hvx_innermost_unit_stride` contract；Hexagon tiling 会验证 identity output 和
最内连续 loop、禁止 padded copy-back，并把仍存活的 contract 传播到 tile loop；
HoistScalarOps 与 vectorizer 也保留并审计该 contract。P2f 必须与 P2e 同时开启，
否则 pipeline 拒绝编译。

完整 DINOv2-small、同一 FP16/HVX/HexKL、输入和设备状态的串行结果为：

| 配置 | Latency | 相对 P2f | P2e direct/contract | 正确性 |
|---|---:|---:|---:|---|
| HexKL-on matched control | 9,805.07 ms | 1.57x | 0 / 0 | PASS |
| P2e | 6,244.10 ms | 1.00x | 36 / 0 | PASS |
| P2e + P2f | 6,248.11 ms | 1.00x | 36 / 36 | PASS |

P2f 相对 P2e 慢 0.06%，属于噪声，没有额外加速。所有配置均 finite、top-1
match、`max_abs_diff=0.0049`。更关键的是，P2f 虽在早期建立 36 个 contract，
但没有带显式 contract metadata 的 Linalg op 到达 Hexagon tiling/vectorization；P2e/P2f final
object 的 HVX-like 指令同为 16,590、vector load/store mentions 同为 22,182，
总指令仅为 282,592/282,587。说明这些 direct producer 在更早的
fusion/canonicalization 中被吸收，或在语义等价的 op 重建中丢失了 metadata；
无论是哪种情况，显式 metadata 都没有改变最终 codegen。当前数据不能把所有
36 个点都进一步归类为“已物理融合”，需要扩展 post-fusion/movement ledger 后
才能逐点区分 discharged contract 与 metadata loss。

因此当前结论不是“contract propagation 普遍无用”，而是：对 DINOv2 的 36 个
P2e rewrite，结构化 indexing-map 改写已经决定最终 codegen；额外 metadata
没有证明存在可优化的剩余对象，也没有新增收益。P2f 保持独立默认关闭，
不进入当前推荐加速组合；它只适用于后续发现“P2e producer 不能被 fusion、且
仍存活到 HVX tiling”的模型。当前更优先的是完善 post-bufferization movement
ledger 和 destination-style bufferization，寻找确实残留的物理 materialization。

结果与完整运行日志已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2f_dinov2_20260820
```

### 11.20 Consumer-Driven In-Situ Layout Formation 与 Prefetch 的统一设计

是的。更准确地说，P2e 的核心逻辑是：

> 从 terminal consumer 的物理访问需求反向推导所需 layout，再判断 producer 能否直接按该 layout 产生结果。

例如原始数据流是：

```text
Producer
  → Expand/Reshape
  → Transpose
  → Consumer
```

P2e 在满足安全条件时改写为：

```text
Producer directly forms consumer layout
  → Consumer
```

它不是简单删除 `transpose`，而是把 transpose 的 permutation 合并到 producer 的 indexing maps 中。当前还要求：

- 静态 shape；
- producer 只有一个结果和唯一使用链；
- producer 是全 parallel `linalg.generic`；
- consumer engine 可以确定为 HVX；
- 不改变最内层连续维；
- 不产生 mixed-engine 或不明确的 layout ownership。

DINOv2-small 中有 36 条链满足条件，显式的 expand/transpose materialization 被消除或为后续 fusion 创造了机会，最终获得约 `1.57x` 加速。

### Prefetch 能与它结合到什么程度？

从设计上结合程度很高，但当前实测中两者还没有真正形成有效组合。

本轮 P2e：

- DMA mentions 为 `0`；
- 没有运行时 prefetch issued bytes；
- `1.57x` 基本来自 in-situ layout formation、数据流变化和后续 fusion；
- 不能声称该收益来自 prefetch 与 layout 的组合。

最自然的组合方式不是“先 prefetch，再单独 transpose”，而是：

```text
Consumer layout demand
        ↓
Representation-aware prefetch planning
        ↓
Prefetch producer inputs into L2/VTCM
        ↓
Producer directly computes into final-layout destination
        ↓
Consumer directly reads
```

这样同时做到：

- 提前搬动 producer 所需输入；
- 不搬动中间 canonical-layout tensor；
- 不生成单独 transpose buffer；
- 不把数据先写 DDR、再读回做 layout conversion；
- consumer 可以直接读取所需表示。

### 最值得实现的三个结合层次

1. Final-layout-aware prefetch

Prefetch planner 不再只记录地址和字节数，还记录：

- consumer 所需 permutation；
- tile shape；
- contiguous dimension；
- memory tier；
- producer/consumer version。

Prefetch 的对象是“未来即将生成或消费的最终表示”，而不是原始 logical tensor。

2. Prefetch inputs + direct-layout production

这是最适合当前 P2e 的方案：

```text
DDR input
  → 提前进入 L2/VTCM
  → HVX producer 使用预取输入
  → 直接写最终 consumer layout
```

UserDMA 本身不会完成任意 transpose，但它可以提前搬入连续 input tile；随后 HVX producer 在计算过程中直接形成目标 layout。这样不会重新引入 layout materialization。

3. Transform-on-arrival

对于无法由 producer 直接形成 layout 的情况，可以采用：

```text
DMA/load source tile
  → 在 VTCM staging 中完成有限 shuffle/layout formation
  → consumer 直接消费
```

这对应真正的“prefetching + in-situ layout transformation”。但只能接纳满足以下条件的转换：

- tile 可放入 VTCM；
- DMA 源访问足够连续；
- HVX shuffle 成本低于 DDR 往返；
- transformed tile 有足够复用；
- 不需要再次写回 canonical layout。

### 一个重要原则

不是所有被 P2e 消除的 transpose 都还需要 prefetch。

如果 producer 和 consumer 已经被融合，中间 representation 边界实际上不存在，此时对该中间 tensor 发起 prefetch 反而会：

- 重新制造中间数据；
- 增加 descriptor 和地址计算；
- 污染 L2/VTCM；
- 抵消 P2e 收益。

因此 runtime admission 应当区分：

```text
Contract discharged by fusion
    → 不 prefetch 中间值

Direct producer remains
    → prefetch producer inputs

Physical transform remains
    → 考虑 transform-on-arrival
```

所以两者可以形成非常统一的故事线：

> Consumer 决定未来需要的 representation；ALPS 提前供应形成该 representation 所需的数据，并让 producer 或 VTCM staging 在数据到达时直接形成最终 layout，从而同时减少“等待数据”和“搬动错误 layout”的成本。

这比单纯的 data prefetching 更接近论文中的 `prefetching + in-situ layout transformation`，也与 V-DAE 和 runtime traffic control 能自然衔接。

### 11.21 实施计划：Representation-Aware Supply + Direct Formation

计划保持每个阶段独立、默认关闭，并以完整模型串行实验，不使用 Debug/GEMM：

1. **P5a：Contract discharge ledger。** 在 P2e rewrite 时分配稳定 contract ID；
   在 pre-fusion、post-fusion、post-tiling 和 post-bufferization 四个边界统计
   `surviving / fused-or-rebuilt / physical-transform-remains`，先区分已消解边界和
   真正仍需供给的数据，禁止根据已丢失的临时 op attribute 盲目 prefetch。
2. **P5b：Representation-aware input supply analysis。** 对仍存活的 direct-layout
   producer 追踪其只读输入，记录 permutation、tile、连续维、memory tier、版本、
   first-use distance 和预计字节；排除 immediate producer result、动态/跨 block、
   非连续源、低复用和容量不满足的输入。本阶段只分析、不改变 codegen。
3. **P5c：L2 input prefetch + direct formation。** 首版仅为静态、连续、只读且
   有真实提前距离的 producer input 发出有界 L2 hint；producer 仍直接写最终
   consumer layout，绝不为 prefetch 重新创建 canonical intermediate。P5c 与
   P2e 独立对比，并记录 hints/issued bytes/correctness/latency。
4. **P5d：VTCM transform-on-arrival。** 只对 P5b 证明 L2 hint 不足且 tile 可容纳的
   residual physical transform，建立 ping-pong VTCM tile；DMA 只搬连续 source，
   HVX 在 staging/compute 中形成目标 layout，consumer 不再读写 canonical DDR。
5. **P5e：Runtime representation admission。** 将 `contract discharged`、实际
   wait/poll、VTCM pressure 和 issued/useful bytes 接入 controller，在
   `no-prefetch / L2 / VTCM-transform-on-arrival` 三种动作间选择。
6. **实验顺序。** 先在完整 DINOv2-small 做
   `HexKL-on control / P2e / P2e+P5b / P2e+P5c`，有正收益后再验证完整 ViT/DeiT；
   P5d 只有在 ledger 证明仍有 residual transform 时才实施，避免为了使用 DMA
   而重新制造已经消失的数据移动。

### 11.22 P5a/P5b 实施结果：先证明供给对象存在（2026-08-20）

P5a 已实现稳定 contract ID、origin-location fingerprint，并在 pre-fusion、
post-fusion、post-tiling、post-bufferization 四个边界运行 analysis-only discharge
ledger。完整 DINOv2-small 的 36 个 P2e direct contract 得到：

| Phase | Explicit | Location carrier | Physical transform | Untraceable |
|---|---:|---:|---:|---:|
| pre-fusion | 0 | 36 | 0 | 0 |
| post-fusion | 0 | 12 | 0 | 24 |
| post-tiling | 0 | 12 | 0 | 24 |
| post-bufferization | 0 | 12 | 0 | 24 |

这第一次把 P2f 的“metadata 未到达 vectorizer”细分出来：24 个 contract 在 fusion
边界后已经消失或无法追踪；12 个仍由等价 location carrier 表示，并持续到
bufferization。P5a 与紧邻 P2e 的 latency 为 6,190.81/6,223.51 ms，差 0.53%，
属于噪声；correctness 均通过。因此 ledger 没有造成性能 regression。

P5b 随后只分析这 12 个 residual carrier。第一版只检查 post-bufferization
Linalg，得到 0 carrier；进一步核对发现该边界上它们已经成为
`vector.transfer_read`，因此分析器下沉到最终 HVX-facing input stream。最终完整
模型结果为：

| 模型 | P2e direct | post-fusion residual | vector input | admitted prefetch | P5b latency | 正确性 |
|---|---:|---:|---:|---:|---:|---|
| DINOv2-small | 36 | 12 | 12 | **0** | 6,198.85 ms | PASS |
| DeiT-small | 36 | 12 | 12 | **0** | 5,690.01 ms | PASS |

两个模型的 12 个 vector input 全部为 contiguous 256 B tile，但从最后可用定义到
读取只有 `lead_ops=1`，且 source 只有一个 use。它们不具备隐藏 latency 的真实
提前距离；发 L2 hint 只会形成 demand-time hint 或重复读取。因此 P5c admission
必须返回 no-prefetch，不能为了展示 prefetch issued count 而把阈值从 4 放宽到 1。

该负结果实际上验证了统一设计中的关键原则：P2e 已消解的 representation 不应
再次被 prefetch。对 DINO/DeiT，当前推荐方案仍是 P2e direct formation；P5c 不
进入这两个模型的 codegen。下一步将 ledger 扩展到 P2e 保留的 86 个 native
layout demand，定位 post-fusion 后真正仍执行物理 transform 的子集，再决定哪些
点可以安全实现 VTCM transform-on-arrival。若 residual transform 也没有提前距离，
则应转向 tile-loop 内的 next-tile supply，而不是 tensor-op 间的 input hint。

完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5a_dinov2_20260820
nano:/home/huzq85/2-working/working_set/alps_p5b_vector_dinov2_20260820
nano:/home/huzq85/2-working/working_set/alps_p5b_deit_20260820
```

### 11.23 P5c：Next-tile prefetch + direct layout formation（2026-08-20）

在 P5b 之后，ledger 又覆盖了 P2e 保留的 86 个 native layout demand。完整
DINOv2-small 在 post-bufferization 的结果为：`physical_transform=0`、
`location_carrier=86`、`untraceable=0`。也就是说，这些 tensor-level native
transpose 在后续 lowering 中已经被索引/融合表达，并没有以独立 transpose 或
非 minor-identity vector transfer 的形式残留。因此当前没有证据支持分配 VTCM
并实施 P5d transform-on-arrival；那样反而可能重新制造一次物理搬动。

由于跨算子供给只有 `lead_ops=1`，P5c 改为实现严格的 loop-local next-tile
supply。该路径默认关闭且独立可控，只有同时满足以下条件才发出 L2 hint：

- 最终 `vector.transfer_read` 可由稳定 P5a contract location 追踪；
- read 来自静态、连续、只读 `memref.subview`；
- backing storage 在被选 loop 外定义，禁止越过 producer 因果边界；
- 恰好一个 subview offset 是 loop IV，静态 step/tile 可证明下一 tile 边界；
- 当前 tile 继续由 P2e producer 直接形成 consumer layout，P5c 不创建 canonical
  intermediate，也不创建 transpose。

定向 IR 测试证明该 pass 能为一个 256 B 的合法 future tile 生成带 bounds guard 的
`omni_fetch.l2_hint`。完整 DINOv2-small 的 matched HexKL-on 实验得到：

| 配置 | Latency | P2e contracts | P5c matched/admitted | Static tile bytes | 正确性 |
|---|---:|---:|---:|---:|---|
| P2e matched control | 6,223.51 ms | 36 | 0 / 0 | 0 | PASS |
| P5b 最近重复 | 6,186.85 ms | 36 | 12 / 0 | 0 | PASS |
| **P2e + P5c** | **6,259.22 ms** | 36 | **12 / 12** | **3,072 B** | PASS |

P5c 相对 P2e 慢 0.57%，相对最近 P5b 重复慢 1.17%；`max_abs_diff=0.0049`、
top-1 match。12 个 site 每个只预取 256 B，且其数据本来就在紧邻的 HVX
producer/consumer tile 中使用。结果说明机制已经真正组合，但这些 hint 没有足够
latency window，命令和 bounds-check 开销反而抵消收益。此前一次 26,589.31 ms
运行是统一脚本漏传 `--enable-hexkl` 的配置错误，已修复并明确排除，不作为性能
数据。

因此当前 gate 结论是：

1. **P2e direct formation 保留为 DINO/DeiT 的推荐路径。**
2. **P5c 实现保留、默认关闭，不进入当前推荐组合。** 它只应在其他完整模型出现
   更大 tile、更长 loop-carried lead 或更高 DDR miss pressure 时由 admission 开启。
3. **暂不实施 P5d/P5e。** 当前没有 residual physical transform，且 P5c 没有正
   收益；直接进入 VTCM staging/runtime controller 不符合“先证明搬动对象和收益”
   的约束。
4. 下一次模型筛选应先运行 P5a/P5b ledger；只有 `physical_transform>0` 或合法
   next-tile 的 byte/lead 显著高于本轮 256 B/1-step 时，才运行 P5c/P5d 实验。

完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5d_native_dinov2_20260820
nano:/home/huzq85/2-working/working_set/alps_p5c_dinov2_hexkl_20260820
```

### 11.24 P5a/P5b 三 domain gate：保持 representation-supply 主线（2026-08-26）

为避免后续实现从“consumer-required representation 的提前供应”发散为任意
prefetch，本轮只对完整 Qwen2.5-0.5B 和完整 UniSpeech-SAT-base 运行 P5a/P5b
analysis-only。两者均使用 HexKL-on matched 配置；没有启用 P5c/P5d，也没有放宽
因果、连续性、同 block 可用性、tile byte 或 lead 条件。

完整 Qwen2.5-0.5B 使用 24 层、FP16、sequence length 32 的 staged full-model
runner。24 个 stable layer 每层有 12 个 layout demand，其中 2 个由 P2e direct
formation 消解、10 个保留 native；embedding 没有 demand，head 有 1 个 native
demand。聚合结果为：

| 指标 | Qwen2.5-0.5B |
|---|---:|
| 完整模型 latency | 10,885.49 ms |
| P2e demand / direct / native | 289 / 48 / 241 |
| P2e eliminated materialization | 1,572,864 B |
| post-buffer direct contract carrier / untraceable | 24 / 24 |
| post-buffer native physical / carrier / untraceable | 0 / 217 / 24 |
| P5b final input / admitted | 24 / **0** |
| 正确性 | PASS（24 layers，finite，top-5 match） |

24 个最终 input 全部是 contiguous 256 B vector tile、`lead_ops=1`、single-use。
它们与 DINOv2 的负例完全同构：在 demand 附近发 hint 不具备隐藏 latency 的窗口。
因此 Qwen 不进入 P5c；又因为 post-bufferization `physical_transform=0`，也不进入
P5d。

完整 UniSpeech-SAT-base 的结果为：

| 指标 | UniSpeech-SAT-base |
|---|---:|
| 完整模型 latency | 184,866.78 ms |
| P2e demand / direct / native | 137 / 48 / 89 |
| P2e eliminated materialization | 4,718,592 B |
| post-buffer direct contract carrier / untraceable | 24 / 24 |
| post-buffer native physical / carrier / untraceable | 0 / 76 / 13 |
| P5b final input / admitted | 36 / **0** |
| 正确性 | PASS（finite，last-frame top-1 match） |

其中 24 个 input 是 256 B、`lead_ops=1`；另外 12 个是 128 B，线性 ordinal
显示较大 `lead_ops`，但其 source 没有通过同 basic block 可用性证明。跨 block 的
ordinal 差不是可执行的 prefetch window，不能越过控制流/producer 边界发 hint。
因此这些点也必须拒绝，而不是为增加 admitted 数量而放宽条件。UniSpeech 同样没有
residual physical transform，P5d 不成立。

#### 后续工作的固定边界（anti-divergence guardrail）

后续 ALPS 工作必须继续满足下面的顺序和语义：

1. consumer contract 指定未来 representation；
2. P2e/P2a/P2b 优先通过 direct formation 或 descriptor/indexing 消除搬动；
3. P5a 证明 contract 在最终物理 IR 中的位置，P5b 证明 source/version/tile/lead；
4. 只有合法的 future tile 才允许 P5c L2 supply；只有仍存在物理 transform、且
   tile 可容纳时才允许 P5d VTCM transform-on-arrival；
5. runtime/PMU 只能在 compiler 已证明合法的 `none/L2/VTCM` 动作之间选择，不能
   创造新地址、新 layout 或跨越 producer 因果边界；
6. 不通过 gate 的模型保留 direct formation，不通过反复调阈值、任意 fusion、
   混合精度或无关 HMX lowering 来制造收益。

本轮三个 domain 的共同结论是：P2e 能消除一部分 tensor-level materialization，
但其 residual HVX-facing tile 都没有证明出值得 prefetch 的供给窗口，且没有最终
物理 transpose。因此当前推荐组合仍是 direct formation，P5c 保留为默认关闭的
合法机制，P5d/P5e 暂不实施。这是由证据收缩实现范围，不是改变论文故事线。

完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5b_qwen2.5-0.5b_20260826
nano:/home/huzq85/2-working/working_set/alps_p5b_unispeech-sat-base_20260826
```

### 11.25 从 strided zero-copy 到连续表示流水线：重新组合 prefetch 与 layout formation

本节回应一个关键判断：普通 data prefetch 单独收益很小，但如果 producer 能按
consumer 所需的连续 layout 产生数据，是否可以在消除 layout 搬动的同时，为
prefetch 创造真正有效的连续 stream 和提前窗口？结论是 **可以，而且这比继续增加
op-local hint 更符合 ALPS 的统一故事线**；但首先必须区分 P2a 与 P2e，不能把两者
都概括成“producer layout 仍不连续”。

#### P2a 与 P2e 的准确区别

| 路径 | 消除的对象 | 代价可能转移到哪里 | 当前证据 |
|---|---|---|---|
| P2a zero-copy attention | 独立 reshape/transpose materialization | consumer contraction 的 affine/strided access、VMEMU、地址计算 | DINO static movement bytes 下降 52.4%，latency 仅改善 0.72% |
| P2e consumer-driven direct formation | producer→expand/transpose→consumer 的整个临时表示边界 | producer 的输入 indexing、loop/vectorization 与最终 buffer ownership | DINO 36 条链改写，约 1.57–1.58x |

P2a 的确可能把一次显式 transpose 变成 consumer 侧的 strided affine access；它
消除了完整中间 buffer，却不保证最终 HVX load 是 unit-stride。P2e 则已经对严格
子集创建 target-shape、row-major destination，并让 `linalg.generic` producer 以
identity output map 直接写入。因此，对 **P2e 已接纳的链**，producer 的目标写出和
consumer 的目标读取本身是连续的。

P2e 当前真正的覆盖缺口是：它要求 transpose permutation 不移动最内层连续维。
若放开该限制，简单 compose indexing map 往往会使 producer 的某个输入变为
strided/gather stream。也就是说，任意全局 transpose 通常只能把不连续访问从
consumer 搬到 producer，不能自动让两端都连续。

因此，下一阶段的优化目标应严格定义为：

> 在 tile 粒度联合选择 physical layout、producer loop order、HVX register
> formation 和 destination ownership，使 producer 的主要输入读取、最终写出及
> consumer 读取尽可能均为连续访问；随后只对该流水线中仍来自 DDR/L2、且具有真实
> 提前距离的连续 source tile 做 prefetch。

这里将该机制暂称为 **Contiguous Representation Pipeline（CRP，连续表示
流水线）**。它不是新的独立故事，而是
`consumer-driven layout contract → in-situ formation → representation-aware supply`
的下一层实现。

#### 为什么不能只要求“producer layout 连续”

对 `A[i,j] -> B[j,i]` 一类 permutation，若只是让 producer 按 `B` 的全局
row-major 顺序写出，则可能需要按列读取 `A`；反过来保持 `A` 的连续读取，又可能
导致 `B` 的 strided store。真正应优化的是区域总成本：

```text
C(region) = producer input VMEM/gather
          + HVX shuffle/permute
          + destination store
          + consumer load
          + materialization/copy
          + spill and synchronization
          + exposed memory stall
```

CRP 只有在该总成本低于 native transpose 链和 P2a strided-consumer 两个候选时才
接纳。不能用“删除了一个 transpose op”或“producer output type 是连续的”代替
最终物理证据。

#### 三种按优先级排列的 formation 方案

1. **Loop-interchanged direct formation（低成本路径）**

   对全 parallel、无依赖 producer，依据 consumer contract 交换 producer loop
   order，并使用 destination-passing style 直接写 target buffer。只有在主要输入和
   output 都能形成 unit-stride vector transfer 时接纳。这是 P2e 的自然扩展，适合
   permutation 虽改变逻辑维顺序、但可通过 loop interchange 保持连续的情况。

2. **HVX register-tile formation（核心路径）**

   对无法同时保持全局输入和输出连续的 transpose，把问题缩小到一个 HVX-aligned
   tile：producer 连续读取 source tile，在 VRF 中完成有限的 transpose/shuffle，
   再连续写入 consumer-layout tile。这样用寄存器内重排替代
   `canonical buffer store → transpose read/write → consumer reload`，并避免把
   strided access 留给 terminal consumer。tile 大小必须由 vector width、VRF
   pressure、shuffle 数和 tail 比例共同选择。

3. **Blocked/tile-major shared layout（扩大复用路径）**

   当 producer 输出被多个相邻 consumer 重用时，选择一个双方都能按连续小块访问的
   blocked physical layout，而不是要求任一方的全局 row-major layout。buffer type、
   subview、vector transfer 和所有 admitted consumer 必须共享同一个 layout/version
   contract。只有收益覆盖 descriptor/indexing 开销且不会为次要 consumer 重新产生
   full-tensor conversion 时使用。

这三条路径的共同点是：layout formation 发生在 producer 计算的 epilogue/VRF 或
原本就不可避免的一次写出中，而不是在 DDR/L2 中再物化一个 canonical tensor。

#### Prefetch 如何与连续 producer 有机结合

正确的预取对象不是刚刚生成的 output，也不是已由 fusion 消失的中间 tensor，而是
形成下一 consumer-layout tile 所必需的 **未来 source tile**。理想的软件流水线为：

```text
tile t+1: page-safe prefetch contiguous producer inputs from DDR into L2
tile t:   HVX producer computes and forms final consumer layout in VRF/destination
tile t-1: consumer reads the already formed contiguous tile
```

若 tile 有足够复用且 VTCM 容量允许，可将中间两级改为 bounded ping-pong：

```text
DDR --DMA/L2 supply--> VTCM source tile
    --HVX compute + register layout formation--> VTCM/final-layout destination tile
    --direct consume--> consumer
```

但这不是无条件使用 VTCM。只有以下条件同时成立时才允许 L2/VTCM supply：

- source 是外部已存在、只读、连续、page-safe 的 future tile；
- 从 issue 到 first use 有可执行的独立工作，而不只是 IR ordinal 差；
- tile 足以摊薄 `l2fetch`/DMA、bounds guard 和同步成本；
- producer 不会在 hint 后重写 source，且 output contract/version 唯一；
- direct formation 不会为了 prefetch 重新创建 canonical intermediate；
- PMU 或静态模型预测为 latency-bound，而不是已经 bandwidth-bound。

如果 source 是紧邻的上游 producer 结果，则不发 `l2fetch`；应使用 producer
scheduling、VRF forwarding 或 VTCM residency 提前“生产”该 representation。这里的
prefetch 被统一解释为 **future representation supply**，硬件 cache hint 只是其中
一种动作。

#### 为什么该设计可能比 P5c 更有效

P5c 的负结果不是“prefetch 永远无用”，而是当前候选只有 256 B、`lead_ops=1`，且
位于已经被 P2e 大幅消解的局部边界。CRP 改变两个前提：

1. 先将 strided/gather 工作变成较大的连续 source/destination tile，降低 VMEMU、
   cache-line amplification 和 micro-TLB 压力；
2. 在 tile loop 上把 supply 提前一轮或多轮，使 prefetch 与 producer compute、HVX
   register formation、consumer compute 真正重叠。

因此预期收益不应被描述为“prefetch latency + layout speedup 简单相加”，而应来自
一个联合机制：**连续化扩大有效搬运粒度，direct formation 消除冗余物化，software
pipeline 提供隐藏剩余 compulsory traffic 的时间窗口。**

#### 建议的新实施阶段：P2g / P5f

所有改动继续使用独立、默认关闭的开关，并保持完整模型测试：

1. **P2g-a：Continuity audit（只分析）。** 对 P2a absorbed、P2e admitted/rejected
   链在 post-bufferization/vector IR 统计 producer input、output、consumer input 的
   unit-stride/VMEMU、tile bytes、vector width、reuse、materialized bytes；首先证明
   stride 究竟落在哪一端。
2. **P2g-b：Loop-interchanged direct formation。** 只处理能证明主要 input/output
   均连续的全 parallel producer；用 final buffer ownership 和最终
   `vector.transfer_{read,write}` 证明，而不是 tensor-level estimate。
3. **P2g-c：Register-tile direct formation。** 覆盖 P2e 当前因移动 innermost dim
   而拒绝、且 tile shuffle 有界的链；先实现 HVX register tile，再考虑 blocked
   shared layout。记录 eliminated VMEM bytes、VMEMU、shuffle、spill 和 latency。
4. **P5f-a：CRP supply analysis（只分析）。** 对 P2g 形成的 tile 找外部连续 source、
   page footprint 和真实 loop-carried lead；不满足 gate 时保持 no-prefetch。
5. **P5f-b：Prefetch + CRP software pipeline。** 只对 P5f-a 接纳的 tile 做
   `prefetch(t+1) / form(t) / consume(t-1)`；先用 L2，只有 reuse/容量/overlap 证据
   支持时才启用 VTCM ping-pong。
6. **消融顺序。** 在完整 DINOv2-small 上运行
   `HexKL-on / P2a / P2e / P2g / P2g+P5f`；随后用 Qwen2.5-0.5B 和
   UniSpeech-SAT-base 检查跨 domain。必须冻结相同 HexKL/HVX mapping，并报告最终
   VMEM/VMEMU、物理 copy bytes、prefetch issued/useful bytes、correctness 和 latency。

进入实现前最有价值的第一步不是再发 hint，而是 P2g-a。若 audit 显示 P2a 的收益
确实被 consumer strided VMEM 抵消，且 P2e rejected 链占据足够热点比例，P2g-b/c
才有达到显著增量收益的空间；若这些链不在热点，则应保持现有 P2e，不为理论上的
连续 layout 扩大代码复杂度。

### 11.26 P2g-a 实施与完整 DINOv2 continuity gate（2026-08-26）

P2g-a 已按 11.25 的计划实现为独立、默认关闭、analysis-only 的
`enableAlpsContinuityAudit` 开关。它在 vectorization 和 one-shot bufferization 之后
运行，并同时使用：

- P2e direct/native contract 的稳定 ID、transpose origin、terminal-consumer origins、
  permutation 和 `moves_innermost`；
- P2a zero-copy contraction 的稳定 function-level origin；
- 最终 `vector.transfer_read/write` 的 memref stride 与 permutation map。

只有 memref 最内层 stride 为 1 且 transfer permutation 是 minor identity 时才记为
unit-stride；其余记为 VMEMU risk。该 pass 不插入 hint、不分配 buffer，也不修改
codegen。定向 LIT 覆盖了 producer read、target write 和 consumer read 都连续的正例，
增量 v73 构建及测试通过。

完整 DINOv2-small 使用 FP16、真实 HVX vectorization、HexKL-on、P2e+P2g-a，得到：

| 指标 | 结果 |
|---|---:|
| Latency | 6,255.62 ms |
| Correctness | PASS，finite，top-1 match，max abs diff 0.0049 |
| P2e demand / direct / native | 122 / 36 / 86 |
| Tensor-level eliminated materialization | 7,105,536 B |
| P2g contracts / observed | 122 / 50 |
| Moves-innermost contracts | **86** |
| Producer reads / unit-stride | 37 / 12 |
| Consumer reads / unit-stride | 122 / 85 |
| VMEMU-risk transfers | **62** |
| Observed static vector tile bytes | 23,720 B |

按 contract 类别进一步聚合：

| Contract | Total | Observed | Moves inner | Risk contracts | Risk transfers |
|---|---:|---:|---:|---:|---:|
| P2e direct | 36 | 36 | 0 | 24 | 24 |
| P2e native | 86 | 14 | **86** | 14 | **38** |

这给出三个比 tensor-level ledger 更强的结论：

1. P2e 拒绝的 86 条链全部移动最内层连续维，不是随机 unsupported case；
2. 其中最终可追踪的 14 条全部仍有 VMEMU risk，并以 12 层同构模式重复出现；
3. 即使 P2e direct contract 删除了显式 materialization，terminal consumer 的 fused
   location 中仍存在 mixed transfer，说明“direct formation”不能仅凭 tensor rewrite
   数声称整条 region 已连续化。后续必须以 buffer identity/operand role 进一步去除
   fused-location 的保守过计数，但它不改变 native 链已存在 stride 的正证据。

P2g-a latency 与紧邻的 P2e 重复值 6,259.31 ms 相差 0.06%，符合 analysis-only
预期。第一次 6,259.31 ms 编译因 phase4 runner 漏传新 P2g option，实际是纯 P2e；
该配置被保留为 matched repeat，明确不作为 audit 结果。接线已同时补到 layered 和
phase4 runner。

Gate 判定为 **通过**：DINO 中存在跨 12 层重复、移动 innermost dimension、并在最终
vector IR 中留下 strided/VMEMU-risk 的热点候选。因此进入 P2g-b；P2g-b 只接纳通过
loop interchange 后所有主要 producer input 为 unit-stride 或 loop-invariant 的链。
不能满足该证明的 attention K transpose 等候选不得强行 direct-map，而应进入 P2g-c
HVX register-tile formation。

有效完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2g_dinov2_20260826_v2
```

### 11.27 P2g-b 严格 loop-interchange gate（2026-08-26）

P2g-b 已实现为独立、默认关闭的 `enableAlpsLoopInterchangedDirectFormation`
开关。它只在 permutation 移动最内层维时额外尝试 direct formation，并对 producer
的每个 input indexing map 与目标 loop order 做 compose；只有每个 input 对新的最内层
loop **完全 invariant**，或只在 indexing map 的最后一个 result 上以该 loop 的裸
`AffineDimExpr` 出现时才接纳。output 仍必须是 target-layout identity map。该条件的
目的不是保守地减少 rewrite 数，而是避免把 terminal consumer 的 stride 转移成
producer 的 gather/strided VMEM。

定向 LIT 同时证明：

- 一个 input map 在 interchange 后恢复 identity/unit-stride 的正例被接纳；
- 原有 identity-input、`[0,2,1]` transpose 反例仍被拒绝。

增量 v73 构建和 LIT 均通过。随后在与 P2g-a 完全 matched 的完整
DINOv2-small（FP16、HVX vectorization、HexKL-on）上运行 P2g-b：

| 指标 | P2g-a | P2g-b |
|---|---:|---:|
| Latency | 6,255.62 ms | 6,293.49 ms |
| Correctness | PASS | PASS，finite，top-1 match，max abs diff 0.0049 |
| P2e demand / direct / native | 122 / 36 / 86 | 122 / 36 / 86 |
| Loop-interchanged direct | 0 | **0** |
| Moves-innermost contracts | 86 | 86 |
| Producer reads / unit-stride | 37 / 12 | 37 / 12 |
| Consumer reads / unit-stride | 122 / 85 | 122 / 85 |
| VMEMU-risk transfers | 62 | 62 |

P2g-b 与 P2g-a 的 0.61% latency 差异没有对应任何 codegen rewrite，属于单次设备
测量波动，不能归因为 P2g-b。更重要的 gate 结果是：DINO 的 86 个真实
moves-innermost 候选没有一个能在全局 loop interchange 后同时证明 producer input
连续。严格证明正确拒绝了它们，不能为了得到非零 rewrite 数而放松条件。

因此 P2g-b 保留为低成本合法路径，但不再针对 DINO 调阈值；下一步进入 P2g-c。
P2g-c 必须在 tile/VRF 粒度用连续 source read + bounded HVX shuffle/transpose + 连续
target write 替代这些 native 链，而不是再次生成全局 strided producer。实现和评估仍
以 post-vectorization 的实际 transfer、shuffle、spill、VMEMU 和 latency 为准。

完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2gb_dinov2_20260826
```

### 11.28 P2g-c HVX register-tile direct formation（2026-08-26）

P2g-c 已实现为独立、默认关闭的 `enableAlpsRegisterTileFormation` 开关。它只覆盖
P2e 因 permutation 移动 innermost dimension 而拒绝、且满足严格 cyclic permutation
与静态 affine legality 的链。实现不是把 stride 从 consumer 搬到 producer，而是：

1. 将 `d1 * extent + d2` 这类可证明的 row-major flattened coordinate 分解为
   descriptor-only `tensor.expand_shape`；任意 `mod/floordiv`、非线性或重复 loop dim
   仍拒绝；
2. 把 producer input map 变成 projected permutation，保持连续 source tile；
3. 由 Hexagon tiler 创建二维 register tile，vectorizer 在 VRF 中形成 consumer layout；
4. 将 register-tile contract 穿过 unit-dim folding、fusion、tiling 和 vectorization；
   feature 关闭时完全保留 upstream pipeline；
5. 以 v73 的 1024-bit（128 B）HVX vector 为硬预算按元素位宽选择 tile：FP32 为
   `2×16`，FP16 为 `4×16`，而不是固定 `8×16`。

定向 LIT、增量 v73 构建和 DINOv2 Debug 均通过。Debug 中命中 1 条
register-tile direct，主 tile 的二维 HVX vectorization 成功，producer 观测到 2 个
unit-stride reads；设备结果 finite、top-1 match，最大绝对误差 0.0005。

第一次完整 DINOv2 实验使用固定 `8×16` tile：编译成功，但设备返回 exit 13。该
配置没有重复运行。只读诊断发现 pass 运行时候选仍可为 FP32，`8×16×4 = 512 B`，
而 Debug 的实际 outer extent 仅为 2，恰好被 clamp 为 128 B；完整模型则超出单个
native HVX vector。随后将 tile 改为上述 128 B 位宽预算，并使用新的结果目录重新
验证，完整模型成功运行。

完整 DINOv2-small（FP16 execution、真实 HVX vectorization、HexKL-on）结果：

| 指标 | P2g-a | P2g-b | P2g-c |
|---|---:|---:|---:|
| Latency | 6,255.62 ms | 6,293.49 ms | **6,352.15 ms** |
| Correctness | PASS | PASS | PASS，finite、top-1 match、max abs diff 0.0044 |
| P2e demand / direct / native | 122 / 36 / 86 | 122 / 36 / 86 | **122 / 48 / 74** |
| Register-tile direct | 0 | 0 | **12** |
| Eliminated tensor materialization | 7,105,536 B | 7,105,536 B | **9,474,048 B** |
| Producer reads / unit-stride | 37 / 12 | 37 / 12 | **37 / 36** |
| Consumer reads / unit-stride | 122 / 85 | 122 / 85 | 122 / 85 |
| VMEMU-risk transfers | 62 | 62 | **38** |
| Main-tile vectorization | NA | NA | **12 / 12 succeeded** |

P2g-c 将 12 条跨层重复链真正变成了连续 producer reads，并将 VMEMU-risk 从 62
降至 38，说明 CRP formation 的代码生成目标已实现；但延迟相对 P2g-a 慢 1.54%，
相对 P2g-b 慢 0.93%。因此当前结论是：**连续化本身是必要的供给条件，但不是端到端
收益的充分条件**。额外二维 tiling、tail（每个逻辑 op 形成 main/remainder 两个
candidate，其中主 tile 12/12 vectorize、remainder 12/12 保持 scalar）以及未被隐藏
的 compulsory source traffic 抵消了 movement/VMEMU 改善。

这并不改变原故事线，反而给出进入 P5f 的物理证据：下一步只分析这 12 个已证明
连续的 register-tile contract，寻找外部只读 source、真实 loop-carried lead、page
safety 和 tile reuse。只有 P5f-a admission 通过的 source 才允许进入
`prefetch(t+1) / form(t) / consume(t-1)`；不得恢复全图或 ordinal-only prefetch。

有效完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p2gc_dinov2_20260826_v2
```

首次 exit-13 证据保留在：

```text
nano:/home/huzq85/2-working/working_set/alps_p2gc_dinov2_20260826
```

### 11.29 P5f-a CRP future-supply admission（2026-08-26）

P5f-a 已实现为独立、默认关闭、analysis-only 的
`enableAlpsCrpSupplyAnalysis` 开关。它不重新扫描全图，也不根据 ordinal 或文本
location 猜测候选；唯一入口是 P2g-c vectorizer 显式产生的
`alps.p2g.register_tile` provenance。由于 vector lowering 与 one-shot bufferization 会
重建 `vector.transfer_read` 并丢弃非语义属性，实现增加了 exact `LocationAttr`
ledger：在 vectorization 后记录已标记 transfer 的编译器 provenance，bufferization
后只对同一 provenance 的 read 恢复 marker。该桥接不会扩大候选集合。

对每个显式 marker，P5f-a 只在下列条件同时成立时准入：

1. 最终 memref 最内层 stride 为 1，transfer permutation 为 minor identity；
2. backing root 定义在循环外，并且循环内没有对同一 root 的 write/copy/DPS init；
3. `memref.subview` 恰有一个 offset 依赖 induction variable，所有 size/stride 均为
   静态正值/单位步长，因此 `t+1` 地址可精确构造；
4. 静态 trip count 大于 lookahead（当前为 1），tile byte size 可知；
5. 同时报告 4 KiB page 下的 worst-case page footprint 和 root reuse，但 P5f-a 本身
   不插入 hint、不分配 VTCM、也不改变 codegen。

定向 LIT 使用 128 B、只读、loop-carried unit-stride register tile，证明
`matched=1 / admitted=1`。DINOv2 Debug 的真实流水线进一步证明 marker bridge
恢复 2 条 read，P5f-a 准入 2/2（128 B main tile 与 8 B remainder），设备正确性
通过，最大绝对误差 0.0005。

完整 DINOv2-small（FP16 execution、真实 HVX vectorization、HexKL-on、P2g-c）结果：

| 指标 | P2g-c | P5f-a |
|---|---:|---:|
| Latency | 6,352.15 ms | **6,345.65 ms** |
| Correctness | PASS | PASS，finite、top-1 match、max abs diff 0.0044 |
| Register-tile direct | 12 | 12 |
| Marker bridge restored | NA | **24** |
| P5f-a matched / admitted | NA | **24 / 12** |
| Admitted bytes（每个静态 site 的 tile bytes 之和） | NA | **1,536 B** |
| Main tile | 12 × 128 B | **12/12 admitted** |
| Remainder tile | 12 × 8 B | **0/12 admitted** |
| Reject reason | NA | **12 address-not-loop-carried** |
| Reject stride / loop / causal / read-only | NA | **0 / 0 / 0 / 0** |

P5f-a 相对 P2g-c 快 0.10%，但它是 analysis-only，差值属于设备测量噪声，不能
宣称为加速。真正的 gate 结论是：12 个跨 Transformer block 重复的 128 B 主 tile
同时满足连续、外部只读、精确 `t+1` 地址和 16 次循环 lead；12 个 remainder read
虽连续，但没有 loop-carried 地址，因此被严格拒绝。这正好避免了为 tail 发出低价值
或错误预取。

Gate 判定为 **通过**。下一步 P5f-b 只允许对这 12 个 admitted main-tile source
生成 `prefetch(t+1)`，并保留 `form(t) / consume(t-1)` 的 CRP 因果关系；实现前必须
加入 loop bound/page 安全，且通过独立默认关闭开关进行 P2g-c matched A/B。若实际
hint 不能在 lowering 后保留或引起 latency regression，应关闭 P5f-b，不得通过放宽
admission 到 remainder/全图来制造命中数。

有效完整产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5fa_dinov2_20260826
```

### 11.30 P5f-b：CRP supply prefetch、因果隔离与物理页问题（2026-08-26）

P5f-b 已实现为独立、默认关闭的 `enableAlpsCrpSupplyPrefetch` 开关。它只消费
P2g-c 显式恢复到 post-bufferization vector read 的
`alps.p2g.register_tile` provenance，并要求 P5f-a 的外部只读 root、静态循环、精确
`t+1` 地址、unit-stride inner dimension、future bound、tile byte limit 与 page-footprint
条件。P5f-b 不接纳 remainder、global constant 或一般 affine 地址猜测。

恢复断线现场时发现两类工程问题，均已修复：

1. 仅重建 `linalg-hexagon-opt` 不会更新完整模型由 Python backend 加载的
   `triton/_C/libtriton.so`。因此四轮 `6460.28 / 6452.31 / 6775.98 / 6806.68 ms`
   实际使用 stale plugin，均为 `P5f-b admitted=0`，不能作为 P5f-b 性能数据。
2. P5f-b 最初被加入统一 `alpsPrefetchPipeline` gate，意外激活通用
   `PrefetchInsert`。有效 plugin 下的第一轮虽然静态 `P5f-b hints=12`，runtime 却
   `issued=4096` 并命中命令预算上限；日志还出现 144 个 loop 的 HexKL L2 hint。
   该轮约 `6404.82 ms` 是混合策略结果，不能归因给 P5f-b。现已将“是否运行通用
   PrefetchInsert”和“是否运行 OmniFetch dialect lowering”拆开：P5f-b 只触发后者。

同时，wrapper 的 L2 scheduler 白名单原本没有
`enableAlpsCrpSupplyPrefetch`，导致 runtime counter 不可见且未配置 traffic envelope。
现已把 P5f-b 接入相同的 4096-command、8 MiB、64-entry duplicate window 配置和
`OmniFetchL2Scheduler` 报告。增量构建脚本也会在登录环境仍指向已删除
无效的旧路径时自动回退到规范的 `LLVM_DIR`。迁移期间保留的
`LLVM_DIR_upstream` 仅是兼容已有 CMake/Ninja 绝对路径的软链接，不再是第二份 LLVM。

隔离后的完整 DINOv2-small（FP16、真实 HVX vectorization、HexKL-on、P2g-c +
P5f-b）结果为：

| 指标 | P2g-c | P5f-a | P5f-b isolated |
|---|---:|---:|---:|
| Latency | 6,352.15 ms | 6,345.65 ms | **6,698.09 ms** |
| Correctness | PASS | PASS | PASS，finite、top-1 match、max abs diff 0.0044 |
| P5f matched / admitted | NA | 24 / 12 | **24 / 12** |
| Static hints | 0 | 0 | **12** |
| Generic `PrefetchInsert` log sites | 0 | 0 | **0** |
| Runtime requested / issued calls | 0 / 0 | 0 / 0 | **17,280 / 1,080** |
| Runtime requested / issued bytes | 0 / 0 | 0 / 0 | **2,211,840 / 27,360 B** |
| Duplicate suppressed | 0 | 0 | **16,200** |
| Page clipped | 0 | 0 | **17,280** |
| Budget / busy / unsupported | 0 | 0 | **0 / 0 / 0** |

P5f-b 相对 P2g-c 慢约 5.45%，因此当前实现不能进入最终加速组合。这里的核心问题
不是 admission 错误，也不是再次发生全图 prefetch：12 个 admitted main tile 都只有
一个 IV-dependent subview；另外 12 个 8 B read 来自
`expand_shape -> subview -> expand_shape -> get_global` 且没有 loop-carried address，
被正确拒绝。

真正瓶颈是“128 B logical CRP tile”不等价于“128 B physical contiguous region”。
当前 `L2HintOp` lowering 将最后一维编码为 width、最近的非 1 外层维编码为 height，
并保留物理 row stride。V73 的 `l2fetch` 只保证起始 4 KiB 页内的生成地址；本轮所有
动态 2-D 请求都触发 page clipping，最终每个 issued command 平均只有 25.33 B，未能
为完整 CRP tile 提供有效供给。高 duplicate 数又说明同一 future row 在内层复用中被
反复请求，当前插入层级仍偏内。

下一步更新为 **P5f-c page-safe segmented CRP supply**，仍保持原统一故事线：

1. 先证明整个 tile 的物理 strides 满足真正 row-major contiguous；只有这种情况才
   lower 为 `width=128, height=1` 的单请求。
2. 若只能证明每一物理 row 连续，则按真实 row address 拆为若干单行 page-safe
   request，不能把逻辑 vector tile 强行 flatten。
3. 将 hint hoist 到最外层仍改变 future tile address 的循环，目标是每个 future tile
   每次只请求一次；duplicate suppression 只作为运行时保护，不替代编译期 placement。
4. gate 先看 `page_clipped/requested`、`issued_bytes/requested_bytes` 和动态请求数，再看
   latency。若覆盖率修复后仍相对 P2g-c 回归，则默认关闭 P5f-c，保留 P2g-c/P5f-a
   作为论文中的连续表示与供给分析结果，不通过扩大候选集合制造收益。

有效隔离实验产物已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5fb_dinov2_20260826_isolated
```

### 11.31 P5f-c：物理分段供给的设备安全 gate（2026-08-26）

P5f-c 已实现为独立、默认关闭的 `enableAlpsCrpSegmentedSupply` 开关。它不扩展
P5f-b 候选集，仍只消费 12 个具有 P2g-c provenance 的 admitted main tile。新增静态
诊断确认 DINOv2 的 12 个 tile 全部不是整体连续区域，而是：

```text
shape=memref<16x4xf16, strided<[384, 1], offset: ?>>
row_bytes=8, physical_rows=16, tile_bytes=128
```

因此 P5f-b 的“最内层 stride=1”只能证明每行 8 B 连续，不能把逻辑 128 B 当作
整体连续区。P5f-c 的定向 pass/lowering 测试覆盖了整体连续与分行两条路径，并且
增量构建和 Python plugin 重链接均通过。

#### V73 runtime 审计发现的独立错误

检查 V73 Programmer's Reference Manual 后发现，runtime 原来用 `USR` bit 3 判断
prefetch active，而 V73 的 `USR:PFA` 实际是 **bit 31**。这解释了 P5f-b telemetry
中 `busy_suppressed=0`：代码声称的 single-flight 实际从未生效。现已把
`OMNI_USR_PFA_BIT` 修正为 31，并在 configure 时重置新增的 segmented-site cursor。
这是一项独立的 V73 正确性修复，但下面的设备结果证明它不是 P5f-c exception 的
唯一根因。

#### 三轮完整 DINOv2 gate（取证前暂定判断，已由 11.31.1 修正）

三轮均使用完整 DINOv2-small、HVX vector、HexKL off、P2e + P2g-c + P5f-a/b/c，
静态候选均为 `matched=24 / admitted=12 / tile bytes=1536`，且均完成 MLIR-to-SO
编译；失败发生在手机 DSP 执行，FastRPC 报 non-recoverable user-PD exception
`0x8000040d`，没有产生输出 tensor 或有效 latency：

| 实现 | PFA 位 | lowering | 设备结果 |
|---|---:|---|---|
| P5f-c static rows | 3（旧错误） | 每 tile 展开 16 个 GEP/call | FAIL，`0x8000040d` |
| P5f-c static rows | 31（已修） | 每 tile 展开 16 个 GEP/call | FAIL，`0x8000040d` |
| P5f-c temporal rows | 31（已修） | 每逻辑 hint 单 call，按 site 轮转物理行 | FAIL，`0x8000040d` |

对应现场均已直接移至 nano：

```text
nano:/home/huzq85/2-working/working_set/alps_p5fc_dinov2_20260826
nano:/home/huzq85/2-working/working_set/alps_p5fc_dinov2_20260826_pfa31
nano:/home/huzq85/2-working/working_set/alps_p5fc_dinov2_20260826_temporal
```

这个对照排除了 stale plugin、通用 PrefetchInsert、错误 PFA 位和静态 call storm
作为唯一原因。三轮共同的新行为是尝试触及 P5f-b 因 4 KiB page clipping 而没有
覆盖的后续物理行。V73 手册还明确指出：`dcfetch` 遇到无效地址可作为 NOP，但
`l2fetch` 遇到 virtual-address translation/protection error 会触发 processor
exception。因此当时的暂定假设是：P5f-a/b 的 `inner stride=1 + static subview +
read-only root` admission 尚未证明每个生成行地址都属于可 L2-fetch 的合法 DDR
allocation；descriptor/type 层面的 strided shape 不是充分的硬件地址安全证明。
11.31.1 的 FARF 取证随后证明三轮均在 SO 装载阶段失败，未执行到这些地址，故该
假设不能解释本次 `0x8000040d`。

#### Gate 判定与 P5f-d 更新计划

取证前 P5f-c 设备安全 gate 被暂记为 **失败**，开关保持默认关闭。为避免
无休止设备试验，下一阶段改为 analysis-only 的 **P5f-d physical fetchability
proof**，在任何新 hint 发射前必须同时证明：

1. 沿完整 view/cast/expand/collapse 链解析到真实 allocation 或函数参数，并证明
   memory tier 是 DDR/L2-cacheable，而不是 VTCM、scratch 或未知 address space；
2. 用 root 静态 extent、descriptor offset、future subview offset、每维 physical
   stride 和 element bytes 计算每个将被 prefetch 的 `[begin,end)`；
3. 所有行范围均落在 root allocation byte extent 内，且 future loop bound 对地址
   计算是充分条件；带 mask、动态 extent、未知 alias 或无法证明 tier 的候选拒绝；
4. analysis-only 在完整 DINOv2 上得到 12 个 site 的逐项证明账本后，才能重新开启
   一个 physical-row hint；若 admitted 变成 0，则记录为该模型不适合这一 CRP
   prefetch，而不是放宽安全条件。

这仍与 ALPS 的统一故事线一致：consumer-driven layout formation 给出真实消费
形状，prefetch admission 必须进一步由物理 memory hierarchy 和 traffic-control
证据约束；“提前搬动”不能绕过“可安全搬动”的证明。

#### 11.31.1 异常取证后的根因修正与最终 gate

上面的“后续物理行可能不可 fetch”只是当时缺少 DSP dump 权限条件下的假设，不能
作为最终根因。后续在原失败二进制上启用了 FastRPC PD dump 与 FARF（PD 创建日志
确认 `PD dump: (Config:Y, Debug:Y)`）；量产手机不允许 `adb root`，但 FARF 已给出
足够明确的装载器证据：主模型 SO 的三个符号
`hexkl_micro_hmx_rm_to_wh_f16`、`hexkl_micro_hmx_rm_to_ah_f16` 和
`hexkl_micro_hmx_copy_submatrix_to_f16` 未解析，`dlopen_ex` 随后失败。未开启 FARF
时，这个装载失败被 FastRPC 外显为 `0x8000040d`，因此三轮失败实际上都没有进入
segmented `l2fetch` 执行，不能用于证明物理行地址非法。

根因是 `LinkRuntimeModules` 会在 ALPS/OmniFetch 路径中链接
`OmniFetchRuntime`，而该 runtime 同时包含可选 HMX layout helper，对
`libhexkl_micro.a` 有 native link 依赖；旧 `HexagonExecutor` 却只在
`enableHexKL=true` 时链接 HexKL archive。于是“HVX + ALPS、HexKL off”可以生成
含未解析 HexKL micro PLT 的 SO。修复后 executor 不再从 pass 开关猜测 native
依赖，而是用 `hexagon-nm -u` 检查即将链接的 kernel object；只要实际出现
`hexkl_micro_*` 未定义符号，就自动链接 `libhexkl_micro.a`，缺少 archive 时在 host
link 阶段明确失败。使用原 1.5 MB kernel object 重链接的回归验证确认修复 SO 不再
含上述未解析符号，并在同一完整 DINOv2 设备目录成功执行。

真正进入设备执行后，temporal-row 版本为 `26,370.60 / 28,253.53 ms`，并报告
`issued=4096`、`issued_bytes=32768`。这揭示了与地址安全不同的实际问题：12 个
候选均为 `row_bytes=8 / rows=16 / tile_bytes=128`；每行 8 B 仍至少占用一个
128 B L2 cache line，整 tile 的 useful-byte utilization 只有
`128 / (16 * 128) = 6.25%`。跨调用轮转行还消除了 P5f-b 对首行请求的大量 duplicate
suppression，导致低价值命令耗尽 command budget。

P5f-c admission 因此新增 cache-line traffic proof：默认按 V73 128 B line 估算每个
segmented tile 的物理 line-fill bytes，要求 useful-byte utilization 至少 50%；连续
tile 不受该门槛影响。阈值与 line size均为 pass option，功能仍独立、默认关闭。
定向测试覆盖连续准入和 `8 B x 16` 稀疏拒绝。完整 DINOv2 得到
`matched=24 / reject_view=12 / reject_segment_utilization=12 / admitted=0`，不再生成
任何 hint。

同时还发现统一实验脚本的 HexKL-on ALPS scheme 列表漏掉了
`alps-crp-segmented-supply`。这解释了为什么早期 P2g-c/P5f-a/P5f-b matched control
约为 6.35--6.70 s，而 P5f-c 修复后即使 zero hint 仍约为 28.28 s：两者分别是
HexKL on 与 HexKL off，并非 matched comparison。脚本已修正，最终完整模型结果为：

| 配置 | HexKL | Latency | Static hints | Runtime issued | 正确性 |
|---|---:|---:|---:|---:|---|
| P2g-c 历史 matched control | on | 6,352.15 ms | 0 | 0 | PASS |
| P5f-a 历史 analysis-only | on | 6,345.65 ms | 0 | 0 | PASS |
| P5f-c（错误配置，仅供诊断） | off | 28,279.79 ms | 0 | 0 | PASS |
| **P5f-c cache-line admission（最终 matched gate）** | **on** | **6,383.04 ms** | **0** | **0** | **PASS，top-1 match，max abs diff 0.0044** |

最终 P5f-c 相对 P2g-c 仅慢 0.49%，属于单次设备测量量级，原先 4.45x 的表面回归已
消除。结论必须分开表述：**工程异常已经解决，P5f-c 的安全/流量 admission 也已
闭环；但 DINOv2 的 12 个 CRP source 物理行过稀，不适合 segmented L2 prefetch，
所以该模型上 P5f-c 为 zero-hint、无加速，开关继续默认关闭。** 这正是 ALPS
hierarchical admission 的必要性：只有 layout continuity、地址可证明和 cache-line
利用率同时成立，prefetch 才能进入 runtime traffic control。

最终有效产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5fc_dinov2_20260826_hexkl_matched_fix
```

HexKL-off 诊断对照保存在：

```text
nano:/home/huzq85/2-working/working_set/alps_p5fc_dinov2_20260826_admission_fix
```

### 11.32 P5g：从稀疏 L2 hint 转向连续 VTCM supply（2026-08-27）

P5f-c 的最终 gate 证明，DINOv2 的 CRP main tile 虽然逻辑大小为 128 B，但物理上是
`16 x 8 B`、行 stride 768 B 的稀疏区域；按 128 B cache line 计算的 useful-byte
utilization 只有 6.25%。继续调低 50% admission 门槛只会重新引入低效 L2 traffic，
不能修复 producer 物理布局。因此 P5g 改为验证 V73 memory hierarchy 下更合适的
落点：把 consumer 所需表示形成在 VTCM，并让 HVX 从连续 VTCM tile 消费。

V73 Programmer's Reference Manual 与 HVX Programmer's Reference Manual 对该设计
给出的直接约束是：VTCM 是面向 vector/DMA 的片上 tightly coupled memory；L2
prefetch 只改变 cache residency，并不会把 `stride=768 B` 的 8 B 行变成连续 vector
供给。因此 VTCM 不是简单替换 L2 hint 的另一种 cache，而是 CRP 中 `form` 与
`consume` 之间的显式物理表示层。

#### P5g-a：exact-tile VTCM formation 的失败证据

P5g-a 首先把每个 `16x4xf16` exact tile 同步 copy 到 VTCM。静态 lowering 确认生成
12 个 `dma2d_start + dma_wait` call site，参数对应：

```text
row width = 8 B, height = 16, source stride = 768 B, destination stride = 8 B
```

该完整 DINOv2 二进制在设备上退出 13。按照实验约定没有重复运行同一配置。更重要
的是，即使忽略设备失败，每次只搬 8 B 的二维 DMA 行也重复了 P5f-c 已否决的
cache-line/命令低利用率问题。P5g-a 因而增加保守的 `row_bytes >= 64` legality gate；
64 B 是当前安全筛选和后续 coalescing 目标，不宣称是硬件全局最优值。

#### P5g-b：coalesced VTCM supply window

P5g-b 是独立、默认关闭的 `enableAlpsCrpVtcmWindow` 开关。它不再为每个 4-channel
consumer tile 单独搬运，而是在固定 attention head 下把 8 个相邻 channel tile 合成
一个 `memref<256x32xf16, 1>` 的 16 KiB VTCM window：

```text
DMA logical elements = 8192 x f16
row width = 32 x f16 = 64 B
height = 256
source stride = 384 x f16 = 768 B
destination stride = 32 x f16 = 64 B
```

window 在 channel 0 和 32 处同步更新，并跨 8 个相邻 4-channel consumer iteration
复用。consumer 不再恢复原稀疏四维 base，而是直接读取
`memref<16x4xf16, strided<[32,1]>, 1>`。对于 tensor/vector 语义中的 singleton-expanded
rank，pass 构造 projected permutation map：非 1 维度依次映射 token/channel，其余
维度映射常量 0。因此 `vector<16x4>`、`vector<16x1x4>` 和
`vector<1x16x1x4>` 都共享同一连续二维物理供给，而不改变逻辑结果 shape。

定向 pass + DMA lowering gate 已验证一个 16 KiB window、64 B 行宽二维 DMA 和
连续 VTCM consumer。完整 DINOv2-small（FP16、HVX vector、HexKL on、P2g-c +
P5g-b）结果为：

| 指标 | P2g-c matched control | P5f-c matched gate | P5g-b VTCM window |
|---|---:|---:|---:|
| Latency | 6,352.15 ms | 6,383.04 ms | **6,383.66 ms** |
| Correctness | PASS | PASS | **PASS，finite、top-1 match、max abs diff 0.0044** |
| P5g-b matched / formed | NA | NA | **24 / 12** |
| Static VTCM windows | 0 | 0 | **12 x 16 KiB** |
| Static DMA start/wait call sites（object relocation） | 0 | 0 | **12 / 12** |
| L2 hints / runtime issued | 0 / 0 | 0 / 0 | **0 / 0** |

P5g-b 相对 P2g-c 慢 0.50%，与 P5f-c 只差 0.01%，属于单次设备测量量级，不能宣称
加速。但这个 gate 得到了两个明确结论：

1. **物理连续性问题已真正解决。** HVX consumer 最终读取 stride `[32,1]` 的 VTCM
   tile；不是 tensor IR 中的预测消除，也不是 zero-hint/no-op matched control。
2. **同步 staging 没有减少总搬动。** 它把原 DDR strided load 变成 DDR→VTCM DMA
   加 VTCM→HVX read，并在每个 window 立即 wait。更好的 DMA row utilization 和连续
   HVX access 被额外 copy、启动/wait 与缺少 overlap 抵消。

因此不能直接从 P5g-b 跳到“多发 DMA”。从减少总搬动的优先级看，下一步原计划是
producer-direct VTCM supply：沿 12 个 source root 找到真正 producer，证明其输出能否
直接以 32-channel window layout 写入 VTCM，从而删除 DDR 中间 materialization。
随后为了回答“同步搬运能否异步化、从而真正体现 prefetch”这一问题，P5g-c 被保留给
双 buffer 异步 DMA gate；producer-direct 分析顺延为 **P5g-d**。这不改变优先级判断：
“消除搬动”仍高于“重叠额外搬动”，P5g-c 只是先把 overlap 机制和收益边界测清。

有效 P5g-b 产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5gb_dinov2_20260827
```

其中早期 `formed=0` 的 6,332.33 / 6,418.67 ms 是 consumer vector-rank 门禁调试轮，
不是 P5g-b 有效数据，已明确排除；只有 `formed=12` 的 6,383.66 ms 用于上表。

### 11.33 P5g-c：late-formed asynchronous VTCM prefetch（2026-08-27）

P5g-c 把 P5g-b 的同步 `DDR -> VTCM -> wait -> HVX` 改成双 VTCM buffer 的异步
ping-pong。动态执行顺序为：

```text
DMA start(window 0)
wait(window 0)
DMA start(window 1) -> HVX consumes window 0
loop backedge
wait(window 1)
DMA start(window 0) -> HVX consumes window 1
```

因此这里的 DMA 不再只是同步 staging：下一份 consumer-required contiguous layout
在当前 HVX 计算期间提前形成，正是 prefetch 与 in-situ layout formation 的结合。
两个 VTCM window 是独立的 `memref<256x32xf16, 1>` allocation，并通过
`memref.distinct_objects` 明确 ping/pong 及 opposing select 后的无别名关系；tag table
使用 `memref<2xi32>`，每个 slot 的 wait 只约束对应的 DMA transaction。

#### 中断前排除的伪异步路径

实现过程中得到的 6,392.09 ms、6,387.78 ms 和 6,304.90 ms 均不计入 P5g-c
有效结果。前两版最终 object 的 `dma_start` 与 `dma_wait` 仍相邻；后一版虽然 latency
较低，但检查 object 后仍是旧同步序列，因而只是设备测量波动，不能宣称异步收益。
根因有两层：

1. 单 allocation 的动态 ping/pong subview 无法向后端证明当前 HVX source 与下一 DMA
   destination 不重叠；改为两个 allocation 后，这个 alias 问题才被消除。
2. 更关键的是，旧 P5g-c 在 buffer ownership/deallocation 之前用普通 store/copy
   marker 表示异步意图；canonicalization 与 ownership 会重建 copy 并丢失 marker。
   后续通用 `HexmemCpyToDMA` 对这些 copy 采用的正确保守语义本来就是立即
   `dma_start + dma_wait`，所以最终仍被串行化。

最终修复不再依赖脆弱 marker：同步 P5g-b 仍在原阶段运行；异步 P5g-c 则移动到
buffer ownership/deallocation 和 `ConvertBufferizationToMemRef` 之后、
`ConvertToHexagonmem` 之前，直接形成 `memref.dma_start/dma_wait`。window 的显式
dealloc 也在该阶段由 P5g-c 自己插入。

#### 最终 object 与完整模型 gate

最终 Hexagon object 静态保留 14 个 `dma2d_start` relocation 和 7 个 `dma_wait`
relocation（pass 形成 12 个 source site；此处统计的是后续优化后的静态 loop site）。
首个存活 site 的反汇编顺序为：prologue start `0x2e38`、当前窗口 wait `0x2eb8`、
下一窗口 start `0x2f30`，随后从约 `0x4da0` 开始执行 HVX vector load/compute，直到
`0x3cc8` 的 loop backedge 回到 `0x2e80`，下一次动态 wait 才再次到达 `0x2eb8`。
这证明 next-window DMA 与 current-window HVX work 之间确有真实 overlap，而不是仅在
高层 IR 中看起来异步。

完整 DINOv2-small（FP16、HVX vector、HexKL on、P2g-c + P5g-c）结果：

| 指标 | P2g-c matched control | P5g-b synchronous VTCM | P5g-c asynchronous VTCM |
|---|---:|---:|---:|
| Latency | 6,352.15 ms | 6,383.66 ms | **6,375.10 ms** |
| 相对 P2g-c | 1.00x | 慢 0.50% | **慢 0.36%** |
| 相对 P5g-b | NA | 1.00x | **快 0.13%** |
| Correctness | PASS | PASS | **PASS，finite、top-1 match、max abs diff 0.0044** |
| matched / formed | 24 / 12 | 24 / 12 | **24 / 12** |
| formed window bytes | 0 | 196,608 B | **393,216 B（双 buffer）** |

P5g-c 的结论需要严格区分“机制成立”和“性能成立”：**真实异步 prefetch 已经实现并由
最终 object 证明；但它在完整 DINOv2 上只比同步 staging 快约 0.13%，相对 P2g-c
仍慢约 0.36%，没有形成有意义的模型级加速。** 这说明当前可隐藏的 DMA transfer
占比或有效 overlap site 数量不足，而且 DDR 中间 materialization 仍未删除。异步化
消除了“立即 wait”的主要结构问题，却不能自动抵消额外 staging 的总搬动成本。

下一 gate 因此回到 **P5g-d producer-direct VTCM supply analysis**：先证明 producer
能否直接生成 consumer 所需的 32-channel contiguous VTCM representation，删除 DDR
中间写回与再次 DMA；只有在无法删除搬动时，P5g-c 才作为有计算距离的 fallback。
这个顺序维持 ALPS 的统一故事线：consumer contract 决定 layout formation，优先消除
搬动，其次才通过 memory-hierarchy-aware asynchronous prefetch 隐藏剩余搬动。

最终完整模型产物与日志已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5gc_lateasync_dinov2_20260827
```

### 11.34 P5g-d：producer-direct gate 与 HMX/HVX VTCM layout contract（2026-08-27）

P5g-c 证明了异步 DMA 与 HVX 计算能够真实重叠，但没有删除 DDR
materialization。P5g-d 因此新增独立、默认关闭的
`alps-crp-producer-direct-analysis`，在 post-bufferization 阶段沿 P2g-c 的 exact
consumer tile 追踪 root allocation、alias、writer、reader、footprint 与覆盖关系。定向
测试中，一个完整 `memref.copy` producer、唯一 CRP reader、196,608 B root 能得到
`rewrite_ready=1`，且 analysis-only 不修改 IR。

这里必须修正一个过度简化的表述：**V73 上 HVX 与 HMX 都能使用 VTCM，并不代表两者
能直接消费 VTCM 中相同的字节排布。** HexKL micro API 明确要求 HMX FP16 activation
使用 AH layout、weight 使用 WH layout，alignment 分别为 2,048 B 与 128 B；HMX
accumulator read 也先把 AH tile 写入 VTCM。HVX 则通过普通/vector address contract
读取 VTCM，最适合 consumer 所需的连续、对齐布局。因此跨引擎路径必须分三类：

1. HMX -> HMX 且 layout contract 相同：可以保留 AH/WH；
2. HMX -> HVX 且索引映射与 AH 物理映射恰好等价：经过证明后才能零转换；
3. 一般 HMX -> HVX：应在 VTCM 内完成 `AH -> consumer-required layout`，再由 HVX
   消费，目标是删除 DDR round-trip，而不是错误地删除必要的 layout formation。

当前 HexKL micro lowering 的实际输出链为：

```text
HMX acc_read -> AH@VTCM -> AH-to-RM@VTCM -> copy-to-DDR
```

所以将来若发现真正的 HMX-output/HVX-consumer pair，ALPS 的目标应是：

```text
HMX acc_read -> AH@VTCM -> consumer-layout@VTCM -> HVX
```

并把下一 tile 的 VTCM layout formation 与当前 HVX/HMX compute 流水化。这与论文的
consumer-driven in-situ layout formation、prefetch 和 traffic control 故事线一致；
但不能把 AH 与 HVX layout 混为一谈。

#### 11.34.1 完整 DINOv2 的 root-level 结果与 lifetime 去混淆

完整 DINOv2-small（FP16、HVX vector、HexKL on）第一次精确 operand 分类得到：

```text
root_type=memref<257x6x64xf16>
root_bytes=197376
writers=24 (vector.transfer_write)
readers=36
hmx_roles=hexkl.matmul:lhs
reader map=(d0,d1,d2,d3)->(0,d3,d1,d2)
rewrite_ready=0
```

这说明该物理 root 在整函数范围内既作为 HexKL matmul 的 RM lhs，又承载带换序映射的
HVX/`linalg.generic` reader；它**不是 HMX accumulator output**。但是该 root 是 buffer
planner 跨层复用的 allocation，root-level 的 HMX/HVX 混合不能证明同一个逻辑 value
同时服务两个 engine。直接据此实现双 layout 写入同样是不安全的推断。

P5g-d 随后加入保守的 top-level writer epoch 分区：每个 CRP reader 只与同 block 中
最近的前驱 writer 配对，位于同一 top-level op、跨 block 或次序无法证明的情况记为
ambiguous。最终完整模型结果为：

| 指标 | 结果 |
|---|---:|
| Latency | 6,305.05 ms |
| Correctness | PASS，finite、top-1 match、max abs diff 0.0044 |
| root-level HMX-input/HVX mixed allocation | 1 |
| HVX-only logical writer epochs | **12** |
| same-epoch HMX+HVX consumers | **0** |
| ambiguous epochs | **0** |
| root-level rewrite-ready | 0 |

因此 DINOv2 的结论不是“HMX AH 结果应由 HVX 直接读取”，也不是“同一个 value 必须
同时生成 HMX/HVX 两种 layout”。真实结论是：**同一个 197 KB allocation 被不同逻辑
生命周期复用；当前 12 个 CRP 候选全部属于可区分的 HVX-only writer epoch，HMX lhs
出现在别的 epoch。** 下一 gate 应对这 12 个 epoch 做 writer coverage 与 alias
重定向证明，然后只在相应生命周期把 producer 直接放入 VTCM/consumer layout；不能
把整块共享 allocation 粗暴改成 AH 或 head-major，也不能影响其他 HMX epoch。

两轮有效产物保存在：

```text
nano:/home/huzq85/2-working/working_set/alps_p5gd_contract_dinov2_20260827
nano:/home/huzq85/2-working/working_set/alps_p5gd_epoch_dinov2_20260827
```

第一轮（operand contract）为 6,378.86 ms，第二轮（epoch partition）为 6,305.05 ms；
两者均是 analysis-only，latency 差异属于设备波动，不能宣称 P5g-d 加速。

#### 11.34.2 producer writer coverage 与 redirect gate

在 lifetime 去混淆之后，P5g-d 继续对每个 HVX-only epoch 检查 writer 和同 epoch
reader。完整 DINOv2 的 12 个 epoch 结构完全一致：

```text
writer_count=1
writer_op=vector.transfer_write
writer_vector_type=vector<64xf16>
writer_base_type=memref<64xf16, strided<[1], offset:?>>
writer_map=(d0)->(d0)
writer_masked=0
writer_loops=for(0,6,1);for(0,257,1)
subview offsets=[token_iv, head_iv, 0]
subview sizes=[1,1,64]
subview strides=[1,1,1]
epoch readers=2 (CRP=1, other HVX=1, HMX=0)
```

这里的 loop contract 是从 writer 向外打印，因此实际嵌套为 token `0..257`、head
`0..6`；每个迭代无 mask、identity-map 地写满一个 64-channel row。P5g-d 新增的完整
覆盖证明要求：root 为静态 shape；末维 vector 宽度等于 root 末维；末维 offset=0、
size=full、stride=1；每个前导维恰由一个 `lb=0 / ub=shape[d] / step=1` 的 induction
variable 作为 subview offset，且 size=1。未知动态 bound、mask、非 identity map、缺失
维度或非单位 stride 均拒绝。

与完整模型 writer 结构相同的定向测试已得到 `complete_coverage=1`。最终完整
DINOv2-small proof gate 得到：

| 指标 | 结果 |
|---|---:|
| Latency | 6,392.25 ms |
| Correctness | PASS，finite、top-1 match、max abs diff 0.0044 |
| HVX-only epochs | 12 |
| single-writer epochs | 12 |
| legal vector-writer epochs | 12 |
| **coverage-proven epochs** | **12** |
| **epoch redirect candidates** | **12** |
| HMX/mixed/ambiguous epochs | 0 / 0 / 0 |

root-level `rewrite_ready=0` 仍然是正确结果，因为该 197 KB allocation 在整函数中有
24 个 writer 并跨 HMX/HVX 生命周期复用；真正可改写的粒度是上述 12 个逻辑 epoch，
而不是整个 root。该轮仍为 analysis-only，6,392.25 ms 不能解释为优化收益。

下一阶段应新增独立、默认关闭的 producer-direct rewrite：为每个已证明 epoch 建立
consumer-layout VTCM representation，重定向该 epoch 的唯一 producer writer 与两个
已枚举 reader，并保证其他 HMX epoch 仍使用原 allocation。先完成 alias/view type
conversion 和 correctness gate，再考虑将当前 token-outer/head-inner producer loop
交换为 head-outer/token-inner，以使 head-major VTCM formation 也保持连续写入。

最终 proof 产物保存在：

```text
nano:/home/huzq85/2-working/working_set/alps_p5gd_proof_dinov2_20260827
```

#### 11.34.3 P5g-e：按 logical epoch 直接形成 VTCM representation

P5g-e 把 11.34.2 已证明的 12 个 HVX-only epoch 从 analysis 升级为独立、默认关闭的
rewrite。每个 epoch 在 writer 前建立一个 197,376 B VTCM object，仅把该 epoch 的
唯一 `vector.transfer_write` 和两个已枚举 HVX reader 重定向到新 object，并在最后一个
reader 后释放；其他 HMX epoch 仍使用原 DDR allocation。它不插入 DDR→VTCM copy、
DMA 或 prefetch hint，因此真正实现的是 producer-direct placement，而非增加一次 staging。

完整 DINOv2-small gate：

| 指标 | P5g-e |
|---|---:|
| Latency | **6,289.04 ms** |
| Correctness | PASS，finite、top-1 match、max abs diff 0.0044 |
| rewritten logical epochs | **12** |
| prefetch hint / DMA issued | 0 / 0 |
| compile time | 319.7415 s |

相对 P5g-d proof 的 6,392.25 ms，单样本快 1.64%；但两者不是同轮 matched control，
目前只能确认机制与正确性，不能把 1.64% 当作稳健加速结论。产物位于：

```text
nano:/home/huzq85/2-working/working_set/alps_p5ge_vtcm_dinov2_20260827
```

#### 11.34.4 P5g-f：head-major VTCM 与 strided alloc lowering 修复

P5g-f 保持逻辑索引 `[token, head, channel]`，但令 VTCM root 的物理 stride 为
`[64, 16448, 1]`，即物理字节顺序 `[head, token, channel]`。这不是 tensor transpose；
producer 和 consumer 仍使用同一逻辑坐标，只通过 memref descriptor 表达物理布局。

第一次完整编译暴露了一个独立的 upstream backend 缺口：HexagonMem→LLVM 对静态
多维 `StridedLayoutAttr` 调用 `normalizeMemRefType`，把一结果 linear map 展平为 rank-1
descriptor，随后遗留无法 reconcile 的 rank-1→rank-3 cast。修复后，lowering 对静态
非负 stride VTCM memref：

1. 按 `offset + sum((dim-1)*stride) + 1` 计算真实物理 span；
2. 直接构造原 rank 的 LLVM memref descriptor；
3. 显式写入原 shape、offset 和 stride，不再依赖 rank cast。

定向测试覆盖 `memref<257x6x64xf16, strided<[64,16448,1]>,1>`，验证分配
197,376 B、rank-3 descriptor 以及不存在 unrealized cast。完整模型在系统中断后复用
已生成 `.so` 恢复设备阶段，结果为：

| 指标 | P5g-d analysis | P5g-e identity VTCM | P5g-f head-major VTCM |
|---|---:|---:|---:|
| Latency | 6,392.25 ms | **6,289.04 ms** | 6,348.58 ms |
| Device result | PASS | PASS | PASS |
| Correctness | PASS | PASS | **输入和输出均与 P5g-e SHA-256 完全一致** |
| rewritten epochs | 0 | 12 | 12 |

P5g-f 相对 P5g-d 单样本快 0.68%，但比 P5g-e 慢 0.95%，差异不足以形成性能结论。
当前 head-major 仅改变物理布局，producer 仍按 token-outer/head-inner 遍历，因而在六个
head region 间跳写。下一 gate 是在严格 dependence proof 下形成
head-outer/token-inner producer traversal，使写入顺序与 head-major 物理连续方向一致；
之后必须用同轮 matched control 判断 layout formation 是否真正降低访问成本。

最终产物位于：

```text
nano:/home/huzq85/2-working/working_set/alps_p5gf_headmajor_backendfix_dinov2_20260827
```

#### 11.34.5 P5g-g：连续 producer traversal 的完整模型结论

P5g-g 在 P5g-f 上增加独立、默认关闭的严格 loop-formation gate。它只接受无 iter
argument/result 的 perfect two-loop producer nest；P5g-d 已证明唯一 writer 对
`[token,head,channel]` row 的完整、互斥覆盖。rewrite 将原 token-outer/head-inner
访问顺序改为 head-outer/token-inner，但仍使用逻辑下标 `[token,head,0]`，因此只是让
producer 的动态遍历顺序与 head-major `[64,16448,1]` 物理 stride 一致，不改变模型
语义，也不插入 prefetch、DMA 或额外 copy。

完整 DINOv2-small gate 得到：

| 指标 | P5g-e identity VTCM | P5g-f head-major | P5g-g head-major + continuous producer |
|---|---:|---:|---:|
| Latency | **6,289.04 ms** | 6,348.58 ms | 6,394.35 ms |
| Correctness | PASS | PASS，bit-identical | **PASS，max abs diff 0.0044、top-1 match** |
| rewritten / interchanged epochs | 12 / 0 | 12 / 0 | **12 / 12** |
| prefetch hints / runtime DMA | 0 / 0 | 0 / 0 | **0 / 0** |
| compile time | 319.7415 s | 366.7355 s | 320.7634 s |

三者来自不同单次设备运行，1% 左右差异可能属于系统波动，不能用来声称 slowdown；
但 P5g-g 显然没有呈现足以继续深挖该局部 loop 的模型级收益。最终 object 仍有
297,411 条静态 instruction、16,015 个 HVX-like instruction（5.38%）和 21,822 个
vector load/store mention。当前 12 个约 197 KB logical epoch 即使形成方向正确，也只占
整图很小一部分，改善其 producer 写序不足以改变约 6.3 s 总延迟。

这一步把下一优先级从“继续微调同一 attention buffer”改为：

1. 在 post-bufferization allocation/copy/view 图上扩大 consumer-contract discharge，
   真正减少 66.29 MB movement ledger，而不是只优化约 2.37 MB epoch footprint；
2. 用 region-level cycles、HVX active、HMX utilization、L2 miss/DDR traffic、VTCM stall
   和 DMA overlap counter 定量区分 compute/codegen 与 memory traffic；
3. 对能跨多个 consumer 或 layer 复用的 representation 做 persistence + VTCM lifetime
   coloring；仅对无法消除且有足够计算距离的剩余搬动做异步 prefetch。

产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5gg_contiguous_headmajor_dinov2_20260827
```

### 11.35 P5h：高覆盖 attention destination formation（2026-08-27）

P5g-e 至 P5g-g 只覆盖 12 个约 197 KB 的 Q/K/V logical epoch，合计约
2.37 MB，因此即使 producer-direct、VTCM 和连续遍历机制均成立，也不足以显著改变
完整 DINOv2 的总延迟。P5h 不再继续微调该小 buffer，而是回到 P1 的完整模型 LWP
与 post-bufferization movement ledger：11 个重复 attention region 各含三次约 1.58 MB
的 physical copy，合计约 52.18 MB，是 66.29 MB 静态 materialization 的主要部分。

为避免仅按 shape 猜测，P1 ledger 新增 source/destination version、alias root、defining
op、layout、memory space、direct users、same-type 和 distinct-storage 字段。完整 DINOv2
显示每层均为同一严格结构：

```text
source.root [6,257,257xf32]
  -> source.active [6,257,256xf32, strided]
  -> copy 到 identity temporary
  -> consumer 在 temporary 上原地计算
source.root -> whole copy 到 destination.root
temporary -> copy 回 destination.active
```

这不是三个独立 transpose，而是 padded attention row 的 bufferization materialization
链。P5h 因而实现为独立、默认关闭的
`enableAlpsAttentionDestinationFormation` gate：

1. 严格要求静态 rank-3、相同 active subview、唯一 seed/writeback/whole-root copy、
   相同 root type、同一 block 且顺序可证明；
2. 提前创建最终 destination active subview，以原 seed copy 初始化它；
3. 保留原 rank-reduction，令所有 consumer subview 直接派生自 destination active，
   即 consumer-driven in-situ formation；
4. 删除 temporary、writeback 和 whole-root copy；
5. whole-root 中没有被 active consumer 覆盖的最后一列仍从 source 复制，保持严格语义。

完整 DINOv2-small（FP16、HVX vector、HexKL on，累积至 P5g-g）结果为：

| 指标 | P5g-g / profiling control | P5h |
|---|---:|---:|
| matched / rewritten chains | 0 / 0 | **11 / 11** |
| pass-estimated eliminated copy bytes | 0 | **34,738,176 B** |
| residual tail copy bytes | NA | **67,848 B** |
| post-bufferization copy sites | 133 | 134 |
| post-bufferization materialization ledger | 66,290,754 B | **33,911,874 B** |
| ledger net reduction | NA | **32,378,880 B（48.85%）** |
| runtime prefetch / DMA | 0 / 0 | **0 / 0** |
| latency | 6,323.76--6,431.61 ms（不同轮 P5g-g） | **6,145.01 ms** |
| correctness | PASS | **PASS，max abs diff 0.0044，top-1 match** |
| compile time | 287.83--320.76 s | 343.92 s |

P5h 相对两次最近 P5g-g 单样本快约 2.9%--4.7%；由于不是同轮 matched control，
只能判断方向明确，不能把该范围当成稳定正式加速比。更重要的是，P5h 第一次让 tensor
IR 的“预测删除”转化成 final post-bufferization ledger 的大幅下降，证明应优先扩大
物理 contract discharge 覆盖，而不是继续优化小 footprint 或无选择地插入 prefetch。

同时，约 49% 的静态 materialization 下降只对应数个百分点的模型 latency，说明
`static copy bytes != critical-path DDR bytes`：这些 copy 可能由高带宽顺序访问完成，且
attention/MLP 计算仍占主导。下一 gate 不应宣称仅靠 copy 删除已解决瓶颈，而应按以下
顺序推进：

1. 用同轮 matched control 重复一次 P5h，稳定其 latency 因果量级；
2. 追踪剩余约 33.91 MB movement top sites，优先选择跨 11/12 层重复的大链；
3. 分析 P5h 保留的 active seed 是否可由 upstream producer/HMX 直接写入最终
   destination，从而进一步删除约 17.37 MB，而不是对它立即做同步搬运；
4. 只有 seed 无法消除且 producer 到 consumer 有实际计算距离时，才把它作为异步
   DMA/VTCM 或 L2 prefetch 候选，并由 PMU/traffic control admission 控制。

有效完整产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5h_attention_destination_v2_dinov2_20260827
```

### 11.36 DINOv2：LWP critical path 与 sysMon PMU 物理流量联合归因（2026-08-27）

本轮不再根据静态 copy 数量推测瓶颈，而是把三类证据放在同一个完整
DINOv2-small、FP16、HVX vector、HexKL-on、P5h 配置中采集：

1. post-bufferization movement ledger：回答“编译器最终显式物化了多少字节”；
2. LWP：回答“哪些编译后 region 位于 critical path”；
3. SDK sysMon hardware PMU：回答“执行窗口实际向 L2 外发出了多少 AXI line
   request，以及 HVX/HMX 在何时活跃”。

#### 11.36.1 P5h 的同轮因果量级与正确性

P5h 的 legality 已收紧为 distinct source/destination root、唯一 whole-root copy、完整
temporary/destination 生命周期检查、禁止 destination 提前读取及 temporary 晚期使用；
新增正反例测试覆盖了这些条件。与同配置、仅关闭 P5h 的 P5g-g matched control 比较：

| 指标 | P5g-g matched control | P5h |
|---|---:|---:|
| Latency | 6,315.88 ms | **6,097.97 ms** |
| Speedup | 1.00x | **1.0357x** |
| Latency reduction | 0% | **3.45%** |
| post-bufferization materialization | 66,290,754 B | **33,911,874 B** |
| 物化量下降 | 0 B | **32,378,880 B（48.85%）** |
| 输出 | reference | **SHA-256 与 control 完全相同** |

因此实现本身是正确且确实命中了物理 copy 的；收益不大不是 rewrite 未发生，而是被删除
的 copy 只覆盖了整图真实物理访问中的一小部分。

#### 11.36.2 修正后的 LWP：用 exclusive cycles 排名

旧汇总把 parent 和 child 的 inclusive cycles 同时相加，会重复计算嵌套 region。当前脚本
改为 `exclusive = inclusive - direct-child inclusive`，并只把
`phase=post-bufferization` 的 logical-access record 与 region 连接。联合 profile 的
exclusive-cycle 分解显示：

| 类别 / region | Exclusive-cycle share | 解释 |
|---|---:|---|
| patch embedding `linalg.conv_2d_nchw_fchw`，source line 259 | **39.09%** | 单一最大热点；57,802,752 logical iterations，非 HMX |
| 全部 HMX microkernel + packing/unpacking region | **28.42%** | 216 个 region；包含 `rm_to_wh`、MM、acc read、`ah_to_rm` 和 f16→f32 submatrix copy |
| root / 当前未归因开销 | 9.28% | 需要 marker 或更细 region 才能继续拆分 |
| 显式 materialization region | 4.13% | 与 P5h 只有数个百分点收益一致 |
| 其他已插桩 region | 1.51% | 非主要目标 |

line 259 的 post-bufferization logical upper bound 是 462,422,016 B read 和
231,211,008 B write，但 unique operands 只有 1,145,856 B，logical/unique 约 605x。
这不是 693.63 MB 的物理 DDR 流量证明，而是说明当前 convolution lowering 在极小
operand footprint 上执行了大量循环访问和算术，优先应检查 tiling/vectorization、复用
以及是否发生重复扩展/累加，而不是给它盲目增加 L2 prefetch。

#### 11.36.3 sysMon：模型确有大量真实物理流量，但不等于带宽饱和

SDK 6.4.0.2 的 `sysMonApp profiler` 命令行工具已在手机上直接运行成功，并识别到
Q6 v73。联合 profile 在 6.563 s 的 host kernel process 窗口内得到：

| Hardware PMU 指标 | 数值 |
|---|---:|
| AXI cached read（L2 miss line request） | **791,315,200 B** |
| AXI cached write | **337,076,864 B** |
| AXI total | **1,128,392,064 B（约 1.13 GB）** |
| 平均 read / write bandwidth | 120.57 / 51.36 MB/s |
| 1 ms AXI bytes p50 / p90 / p99 | 38,144 / 359,424 / 2,649,856 B |
| HVX packet event | 199,757,981 |
| HMX active event | 9,687,670 |
| explicit L2fetch miss | 0 |

按 1 ms sample 的计算部件活动分组，HVX+HMX 同时活跃的窗口贡献 493,339,392 B
（43.72%）AXI 流量，HVX-only 贡献 279,558,016 B（24.77%），二者都不活跃的窗口仍
贡献 355,354,752 B（31.49%）。HMX-only 几乎不存在，说明当前 HexKL/HMX 路径会同时
伴随 HVX-side preparation/packing，而不是一个完全独立、只做矩阵计算的阶段。

P5h 实际删除的 32.38 MB 只相当于该 1.13 GB 窗口流量的约 **2.87%**，与同轮
3.45% latency reduction 同量级。这解释了“静态 materialization 降低 48.85%，模型却
只快数个百分点”的表面矛盾：48.85% 的分母只是显式 copy ledger，不是模型的全部物理
流量。另一方面，约 172 MB/s 的窗口平均 AXI 带宽远低于平台峰值，且流量高度 bursty，
因此当前也不能把 DINO 简化为持续 DDR bandwidth saturation；更可能是计算、低效访问、
cache-line miss、packing/unpacking 和间歇性流量共同组成瓶颈。

空闲 2 s 对照只产生约 0.58 MB AXI，总量远小于模型窗口，故 1.13 GB 信号不是后台噪声。
但当前 default sysMon PMU 是 CDSP system-domain 统计，host process 窗口还比 wrapper
内部 Perf 多约 0.47 s；该数值适合作为物理流量量级和 matched-delta 指标，不能直接把
每一字节归属于某一个 LWP region。

#### 11.36.4 LWP 与 PMU 的区别，以及 profiler 安装结论

- **LWP 是软件 region instrumentation。** 它在编译器选定的 region 入口/出口读取
  cycle，能映射回 source line/operator，适合 critical-path 归因；但会改变执行，并且
  不测 DDR/L2/VTCM 流量。
- **sysMon PMU 是硬件事件采样。** 它测真实 processor/HVX/HMX/AXI 事件，适合判断
  compute-versus-traffic；但默认缺少算子级来源映射。两者互补，不能互相替代。
- SDK 的 sysMon marker API 本可给指定 code region 做 PMU 归属，但手册明确说明它
  **不支持 unsigned PD**；当前 Hexagon-MLIR FastRPC runner 正是 unsigned PD，所以
  不能把 marker 当作现阶段必需方案。STID 也必须在线程创建时设置，而当前执行线程并非
  runner 自己创建，不能通过一个脚本开关可靠补上。
- 当前实验所需的 `sysMonApp` CLI 和 parser 都已包含在 SDK 中，已经可以自动 push、
  采集、pull 和解析；**不需要用户额外安装 sysMon APK**。APK 只在需要 GUI 交互探索时
  才有价值，不是可复现实验依赖。

完整联合 profile 已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_dinov2_bottleneck_joint_20260827
```

#### 11.36.5 收紧后的下一步（避免继续在小 buffer 上来回试验）

1. **先攻 line 259 patch convolution。** 它独占约 39.09% exclusive cycles；检查最终
   HVX 指令、vector width、tile shape、load reuse 和 f16→f32 accumulation。若它主要是
   compute/codegen 低效，就优化 vectorization/tiling；若 sysMon matched delta 显示其
   AXI line request 异常，再做 tile-local VTCM supply/prefetch。
2. **再攻重复 HMX representation conversion。** 12 层反复出现 WH/AH formation、
   accumulator readback 和 f16/f32 submatrix copy，合计约 28.42% cycles，并与 43.72%
   的 HVX+HMX active-window AXI 流量相伴。这里应做 consumer-contract-driven persistent
   representation：producer 直接形成下一 HMX consumer 所需布局，并让能跨 consumer
   复用的 weight/tile 保持在适合 HMX 的表示中，避免每次 MM 前后重新 pack/unpack。
3. **只有无法消除的 critical transfer 才 prefetch。** admission 同时要求：region 位于
   LWP top critical path、sysMon matched delta 能降低 AXI/stall、producer-consumer 有
   足够 overlap distance、VTCM lifetime 不与其他 tile 冲突。这样 prefetch 是 layout
   formation 的异步供给机制，而不是独立撒 hint。
4. 对每个改动只做同轮 control/treatment：固定 mapping、频率和 correctness，比较
   latency、exclusive cycles、AXI bytes、HVX/HMX events 及最终 materialization；没有
   覆盖大数据量或 critical cycles 的候选不再进入完整模型编译。

#### 11.36.6 line 259 最终机器码核验：确认是 vectorization 缺口

对归档的 v73 object 按 LWP region ID 反汇编后，region 3（source line 259）对应
`0x228--0x7bc`。虽然整图配置是 HVX vector on，该区间没有 HVX vector arithmetic；
14-wide kernel-width loop 被标量展开，并且每个位置执行：

1. `memuh` 读取一个 f16 input 和一个 f16 weight；
2. 对二者分别 `call __extendhfsf2`；
3. 用 scalar `sfmpy` 累加到 f32。

因此每个 14-wide inner reduction 有 **28 次 half-to-float helper call**。外层循环又覆盖
384 output channels、16x16 output patches、3 input channels，解释了该 region 的
3.493 billion exclusive pcycles。当前通用 Hexagon tiling/vectorization 只接受最内层
恰好达到 native data tile，或能把纯 parallel loop interchange 到最内层；这个 NCHW/FCHW
convolution 的最内层 reduction 是 14，小于 f16 HVX tile 64，而且 convolution 含 reduction
loop，现有 interchange legality 会拒绝它。现有 `ConvTilingPass` 又只支持
`conv_2d_nhwc_fhwc`，因此没有覆盖 DINO 的 `conv_2d_nchw_fchw`。

这使下一步决策更明确：不能把 line 259 当作 data-prefetch 候选。应先用独立 gate 做
patch-convolution reduction vectorization（优先把连续的 kernel-width f16 input/weight
批量扩展并向量 FMA，再做 horizontal reduction），或形成适合已有 NHWC/HMX convolution
路径且不会引入更大 layout materialization 的 producer representation。只有消除 scalar
conversion helper 后，matched sysMon 仍显示 line-request/stall 主导，才为其增加
VTCM tile supply 或异步 prefetch。这个 codegen 修复是 DINO baseline correctness 的必要
工程项，但论文贡献仍应表述为“profile-guided admission 避免对 compute-bound region
错误 prefetch”，而不是把普通 convolution vectorization 包装成 ALPS 创新点。

### 11.37 P5i：consumer-driven patch formation 消除 DINO 最大热点（2026-08-27）

#### 11.37.1 设计与 legality

P5i 没有给 line 259 继续添加 data-prefetch hint，而是把其后继 consumer 的 token
layout contract 传播回 patch producer。该 pass 独立、默认关闭，仅在以下条件全部成立时
改写：静态 FP16 NCHW/FCHW、FP32 accumulation、stride 等于 kernel、dilation 为 1、
输入空间恰好被 non-overlapping patch 覆盖、output channel 是 64 的整数倍、filter 是
编译期常量，并且 `conv -> f32-to-f16 -> collapse -> [0,2,1] transpose` 全链均为唯一 use。
overlapping convolution、dynamic shape、非唯一 use 或无法证明的 bias topology 均保留原生
lowering。

改写后的 reduction loop 为：

```text
[N, OH, OW, IC, KH, KW, OC]
```

其中 `OC` 是最内层连续 parallel 维。FCHW filter 在编译期折叠并转置为
`[IC*KH*KW, OC]`，再以 `[IC,KH,KW,OC]` view 消费；输出直接形成 NHWC，随后只做
descriptor-only collapse 得到 `[N,OH*OW,OC]` token。因此它既没有生成完整 im2col，
也没有在运行时转置 384x3x14x14 权重。完整 DINO IR 中只命中 1 个目标，报告消除
196,608 B patch-output transpose；常量 fold 后可见单一 `588x384xf16` dense resource，
运行时 filter transpose 已消失。正例及 overlapping-window 反例均有 FileCheck 覆盖。

#### 11.37.2 完整 DINO matched 结果

固定完整 DINOv2-small、FP16、HVX vector、HexKL on，并保持 P5h 及其此前 gate 相同：

| 指标 | P5h matched candidate | P5h + P5i |
|---|---:|---:|
| Latency | 6,097.97 ms | **3,740.29 ms** |
| P5i incremental speedup | 1.00x | **1.6303x** |
| 相对 P5g-g matched control（6,315.88 ms） | 1.0357x | **1.6886x** |
| Correctness | PASS | **PASS；max abs diff 0.0046，top-1 match** |
| Runtime prefetch hints / issued bytes | 0 / 0 | **0 / 0** |
| Post-bufferization materialization | 33,911,874 B | **33,911,874 B** |

独立 LWP build 的设备 latency 为 3,699.41 ms，正确性同样通过。两次 P5i 结果相近，且
最终显式 materialization 完全不变，所以这次 1.63x 增量收益不是由新增 prefetch 或
copy ledger 下降造成，而是由 consumer-required continuity 使原 scalar reduction 进入
真正的 HVX codegen。P5i object 在该区间出现连续 `vmem/vmpy/vadd`，不再是每个 f16
元素调用转换 helper 后做 scalar `sfmpy`。

#### 11.37.3 LWP 因果闭环与下一 gate

采用修正后的 exclusive-cycle 汇总，P5i LWP 得到：

| 类别 | P5h share | P5i share | 判断 |
|---|---:|---:|---|
| patch embedding line 259 | 39.09% | **0.80%** | 最大热点已退出 critical path |
| HMX microkernel + representation preparation | 28.42% | **45.25%** | 成为当前第一优化对象 |
| attention arithmetic（div/exp/reduction 等） | 未单列 | **21.58%** | 当前第二优化对象 |
| root / 未归因 | 9.28% | **15.39%** | 仍需更细 marker/region 分解 |

这给出一个重要的 profile-guided admission 结论：**不能因为 P5i 的 layout 已连续，就继续
给 patch 路径加 VTCM/异步 prefetch。** 它当前只剩 0.80% exclusive cycles，即使完全
消除也不可能带来有意义的模型级收益。下一步先细分 45.25% 中真正的 HMX compute、
`rm_to_wh/ah_to_rm` packing、accumulator readback 和 f16-to-f32 submatrix copy；再分析
21.58% attention arithmetic 的 producer/consumer layout 与 vector reduction。只有被
证明仍处于关键路径、无法通过 direct formation 消除、且存在 producer-consumer overlap
distance 的连续 supply，才进入 VTCM async-prefetch gate。

实现、完整模型、LWP 与机器码证据已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5i_patch_conv_dinov2_20260827
nano:/home/huzq85/2-working/working_set/alps_p5i_lwp_dinov2_20260827
```

### 11.38 P5j：consumer-driven HMX FP16 epilogue formation（2026-08-27）

P5i 消除 patch embedding 的 scalar codegen 后，HMX microkernel 及其 representation
preparation 升至 45.25% exclusive cycles。细粒度 HexKL LWP 又显示，原 HMX 输出路径
每次先把 accumulator readout 从 VTCM 中的 FP16 tile 扩展为 DDR FP32 submatrix，紧接着
唯一 consumer 再做 identity-layout FP32→FP16 truncation。这条链既增加输出流量，也让
同一批元素经历一次无用的 widen+narrow。

P5j 因而在 `matmul-to-hexkl` 中加入默认关闭、严格 legality 的 consumer contract：
只有 matmul 结果为 FP32、唯一 consumer 是逐元素 identity-map truncf、无其他 use 时，
才把 HexKL result contract 改为 FP16，并让 decomposition 使用新的
`hexkl.micro_hmx_copy_f16_to_submatrix`。这不是引入混合精度：HexKL/HMX 原路径本来
就是 FP16 输入与 FP16 accumulator readout；P5j 只是按最终 consumer 所需表示直接落盘，
不再人为扩展到 FP32 后立即截断。非唯一 use、非 identity layout 和 FP32 consumer 均
保持原路径。

完整 DINOv2-small 共形成 **72 个** FP16 epilogue，结果如下：

| 指标 | P5i | P5j |
|---|---:|---:|
| Latency | 3,740.29 ms | **3,346.73 ms** |
| P5j incremental speedup | 1.00x | **1.1176x** |
| 相对 P5g-g（6,315.88 ms） | 1.6886x | **1.8872x** |
| Correctness | PASS | **PASS；max abs diff 0.0046、allclose、top-1 match** |
| Runtime prefetch hints / issued bytes | 0 / 0 | **0 / 0** |
| Post-bufferization movement ledger | 33,911,874 B | **33,911,874 B** |

ledger 不变是预期行为：它在 HexKL decomposition 之前采集，无法看到 epilogue runtime
API 内部的数据宽度变化。细粒度 LWP 给出了物理执行侧证据：

| HexKL phase | P5i | P5j | 判断 |
|---|---:|---:|---|
| FP16→FP32 output copy | 537,747,174 cycles（9.90%） | **0** | widen 路径已消失 |
| direct FP16 output copy | 0 | **337,446,121 cycles（6.77%）** | consumer 所需表示直接形成 |
| RM→WH | 88,411,786 cycles（1.63%） | 91,960,430 cycles（1.84%） | 基本不变 |
| input submatrix→FP16 | 52,975,269 cycles（0.98%） | 51,263,802 cycles（1.03%） | 基本不变 |

原 9.90% FP32 output-copy phase 被 6.77% FP16 direct-copy phase 替代，减少约
200.30 M sampled cycles；这与完整模型 10.52% latency reduction 方向一致。HMX MM
是异步启动，LWP 对单个 `micro_hmx_mm` call 的极小数值不能解释为计算免费；完成等待
可能落在后续 read/copy phase，因此 phase 归因必须按整条 producer-consumer chain 解读。

产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5j_hmx_f16_epilogue_dinov2_20260827
nano:/home/huzq85/2-working/working_set/alps_p5j_hexkl_phase_lwp_dinov2_20260827
```

### 11.39 P5k：HMX non-aligned result 直接形成（2026-08-27）

P5j 的最终 IR 仍暴露出一条重复 72 次的高覆盖搬运链。DINO token M=257 不满足 HMX
32-row tile 对齐，所以 decomposition 同时为输入和输出创建 288-row padded buffer：

```text
HMX/VTCM epilogue -> padded [288,N] FP16 DDR result
                  -> subview [257,N]
                  -> memref.copy -> final [257,N] destination
```

输入补齐是 HMX tile contract 所需，但输出补齐并非必要。现有 HexKL
`copy_f16_to_submatrix` API 已接收独立的 `output_rows/output_cols` valid bounds，
可在边界 tile 内裁剪 store。P5k 因而增加独立、默认关闭的
`enableAlpsHmxDirectOutputFormation` gate：HMX 循环与输入 padded representation
保持不变，epilogue 直接以原始 M/N bounds 写 caller destination，不再分配 padded
result，也不再生成 output subview、copy 和 dealloc。默认关闭时旧 IR 完全保持；正反向
FileCheck 同时验证了这一点。

完整 DINOv2-small（P5i+P5j 累积）结果为：

| 指标 | P5j | P5k |
|---|---:|---:|
| Latency | 3,346.73 ms | **3,278.35 ms** |
| P5k incremental speedup | 1.00x | **1.0209x** |
| Latency reduction | 0% | **2.04%** |
| 相对 P5i（3,740.29 ms） | 1.1176x | **1.1409x** |
| 相对 P5g-g（6,315.88 ms） | 1.8872x | **1.9265x** |
| P5j formed epilogues | 72 | **72** |
| Correctness | PASS | **PASS；max abs diff 0.0046、allclose、top-1 match** |
| Runtime prefetch hints / issued bytes | 0 / 0 | **0 / 0** |

因此 P5k 的约 2% 增益不是 cache hint 或 DMA 波动，而是 direct formation 在 HMX
non-aligned output 上继续删除关键路径搬运。它也让当前论文故事更统一：
consumer contract 决定最终 representation；能直接形成的就消除搬运，不能消除且有
overlap distance 的才进入 VTCM/DMA prefetch admission。

P5j+P5k 后，下一优先级不应再修改已退出热点的 patch path，也不应无选择地 prefetch。
应重新按 LWP 排名剩余 HMX outer regions 与 attention arithmetic：先判断 RM→WH、
input tile formation、accumulator completion/readout 中哪些是重复 representation
formation，哪些是 HMX completion critical path；然后只对无法跨 consumer 保留、但可与
当前 HMX compute 重叠的下一 tile 做异步 VTCM supply。

完整产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5k_hmx_direct_output_dinov2_20260827
```

#### 11.39.1 P5k 后的 critical-path 复核

独立完整模型 HexKL-phase LWP build 为 3,360.38 ms，相对 P5j LWP 的 3,416.99 ms
同样快约 1.7%，正确性保持。最新 exclusive-cycle 分解为：

| 类别 | P5j | P5k | 判断 |
|---|---:|---:|---|
| HMX outer pipeline | 35.88% | **33.66%** | padded-result copy 从 outer critical path 消失 |
| attention arithmetic | 16.81% | **17.40%** | 基本不变，比例随总周期变化 |
| root / 未归因 | 15.99% | **15.75%** | 基本不变 |
| direct FP16 output drain | 6.77% | **6.78%** | 仍是无法忽略的 residual transfer |
| RM→WH | 1.84% | **1.85%** | 基本不变 |
| input submatrix→FP16 | 1.03% | **1.03%** | 基本不变 |
| patch path | 0.88% | **0.89%** | 已退出优化优先级 |

这进一步确认 P5k 没有“优化错对象”：它删除的是 output drain 之后的额外 padded
copy，因此 direct FP16 drain 自身应保持不变。下一步要先审计这 72 个 HMX result 的
downstream consumer：若 consumer 的 elementwise/bias contract 能在 VTCM tile drain
时直接满足，则融合形成最终表示并删除中间 DDR round trip；若中间 result 语义上必须
materialize，则该 6.78% residual drain 才进入 double-buffered asynchronous evacuation，
尝试与下一 HMX tile 的 compute 重叠。

LWP 产物已直接移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_p5k_hexkl_phase_lwp_dinov2_20260827
```

### 11.40 论文叙事保护线：prefetch 必须有独立可验证贡献（2026-08-27）

当前完整 DINO 的强收益主要来自 consumer-driven direct formation 与 continuity：
P5h 删除物理 copy，P5i 让 patch producer 直接形成可向量化的连续布局，P5j 删除
FP32 widen/narrow round trip，P5k 删除 HMX padded-output copy。它们与“减少/提前数据
搬动”的总方向一致，但 **不能把这些收益统称为 prefetch 收益**；目前各强结果中的
runtime prefetch hints 和 issued bytes 均为 0。

因此论文应把统一机制准确表述为 **consumer-contract-driven data-movement
orchestration**，其决策顺序是：

1. 能通过 direct/in-situ formation 消除的 transfer 不执行；
2. 不能消除、但可跨 consumer/layer 保留的 representation 做 persistence；
3. 仍必须执行且有真实 overlap distance 的 critical transfer 才用 V-DAE +
   VTCM/DMA asynchronous supply；
4. PMU/traffic control 负责 admission、节流和拒绝有害 prefetch。

这种叙事允许 ablation 中各项贡献不等大：一个有效系统本来就应让 elimination 优先于
speculative movement。但为了避免 headline 与证据不一致，异步 prefetch/supply 必须设置
硬 gate：

- 至少在多个完整模型（最好跨 domain）上有稳定、重复的正增量，而非单个样本噪声；
- 同时降低 exposed transfer/stall cycles，或在 matched sysMon 中降低/隐藏 AXI traffic；
- correctness、映射、频率和 thermal state 固定；
- 无 overlap distance 或 PMU 判定无收益时，能展示 admission 拒绝带来的“避免负收益”。

若最终只能展示“拒绝错误 prefetch”，而 active asynchronous supply 在大多数模型上仍
为 0 或不稳定，则 prefetch 不应继续作为标题级主贡献：应降为 runtime policy/negative
result，把论文核心收敛到 consumer-driven representation formation。反之，只要 residual
critical transfer 上有数个百分点但稳定、可归因的额外收益，prefetch 仍有说服力——它
不需要超过 P5i，但必须证明是在 direct formation 无法继续后有效隐藏了剩余搬运。

当前最合适的 prefetch 验证点不是已经降至 0.89% 的 patch path，而是 P5k 后仍占
6.78% 的 HMX FP16 output drain。先做 downstream-consumer audit：可融合则继续消除；
不可融合部分再做 ping-pong VTCM + asynchronous evacuation。这样无论结果为正或被
admission 拒绝，都与统一故事线和审稿所需的因果证据一致。

### 11.41 P5l：HMX F16 bias-drain formation 的实现、失败与停止线（2026-08-28）

P5k 后的 downstream audit 找到 72 个 F16 result consumer，其中 36 个是严格的
rank-2 broadcast bias：`C[m,n] + bias[n] -> final[m,n]`，覆盖 14,211,072 B result。
P5l 因此增加了独立、默认关闭的 tensor contract、bufferization interface、HMX micro
drain、LLVM lowering 和 v73 HVX runtime。runtime 在完整 32-column tile 上把两行
F16 result 与重复 bias 合成一个 128 B HVX add，边界 tile 使用有界 fallback。所有
新增路径均由 `--alps-p5l` / `ALPS_ENABLE_HMX_F16_BIAS_EPILOGUE_FORMATION` 单独控制。

实现过程暴露了一个不能绕过的 buffer lifetime 问题。allocation-liveness 会让同一个
memref 承载多个顺序 value epoch；全局 SSA user list 因此不能证明当前 matmul 与
epilogue 的物理 ownership。第一版严格 user matcher 留下 illegal epilogue。改成“第一个
lexical writer/reader”后 36 个 contract 都被识别，但 final destination 在若干 site
中位于 matmul 之后，直接融合产生 dominance error。把整段 HMX compute 移到 epilogue
虽消除了 verifier error，却跨越了 mutable-buffer lifetime，设备结果错误：

| 轮次 | 编译/设备状态 | Latency | Correctness |
|---|---|---:|---|
| compute 延迟到 epilogue | 编译、v73 执行成功 | 3,142.19 ms | FAIL；max diff 2.1235 |
| 同上 + subview descriptor offset | 编译、v73 执行成功 | 3,073.68 ms | FAIL；max diff 2.1235 |

第二版把 consumer final `tensor.empty` 建在 producer matmul 之前，并禁止 decomposition
移动 HMX compute。bufferization 后 36 个 site 中只有 23 个满足 producer-point
dominance，另外 13 个被 admission 正确拒绝。为拒绝项加入普通 elementwise fallback
后，23 fused + 13 fallback 的完整模型能够执行，但仍未通过数值门槛：

| 指标 | P5k（有效 control） | P5l admitted/fallback |
|---|---:|---:|
| Latency | **3,278.35 ms** | 3,648.61 ms（无效，不得用于 speedup） |
| Correctness | **PASS；max diff 0.0046** | **FAIL；max diff 2.2158，top-1 mismatch** |
| 可计入论文结果 | 是 | **否** |

这说明当前 `matmul -> standalone epilogue` contract 在 tensor 层仍不足以表达一个原子的
producer value epoch；bufferization 后再依据相邻 memref 操作融合，无法对 alias、stride、
destination lifetime 和 producer input lifetime给出完整证明。真正要继续该方向，应新增
单一 `matmul_with_bias` tensor op，使 lhs/rhs/bias/final destination 从一开始就在同一
DestinationStyle contract 中，再统一 bufferize/decompose；这属于后续工程项，不能为了
追逐一次无效 latency 继续给当前 P5l 打补丁。

因此执行停止线：P5l 代码保留为默认关闭的实验原型，但不进入 ALPS 有效累计方案；当前
最佳正确版本仍是 P5k。下一步回到 11.40 的论文保护线，只研究 P5k 后**无法直接消除**的
residual HMX drain/transfer：先用 matched LWP/PMU 证明它暴露在 critical path，再让异步
VTCM/DMA supply 与 HMX compute 重叠。若 active prefetch 没有稳定独立收益，则按既定
规则把 prefetch 降为 admission/runtime policy，而不夸大为主要加速来源。

完整产物与失败证据：

```text
nano:/home/huzq85/2-working/working_set/alps_p5l_dinov2_dominance_fixed_20260828
nano:/home/huzq85/2-working/working_set/alps_p5l_dinov2_offset_fixed_20260828
nano:/home/huzq85/2-working/working_set/alps_p5l_dinov2_tensor_order_fixed_20260828
nano:/home/huzq85/2-working/working_set/alps_p5l_dinov2_admitted_23_20260828
```

### 11.42 P5m：先证明 residual HMX drain 具备异步准入条件（2026-08-28）

P5m 是严格 analysis-only 阶段，执行路径仍与正确的 P5k 完全相同。它对每个静态 HMX
result drain 枚举 32×32 F16 descriptor，并分别统计完整 tile、边界 tile、可在后续 HMX
计算后隐藏的 descriptor、目标 stride 和 UserDMA 2D 合法性。完整 DINOv2-small 得到：

| 指标 | P5m 静态结果 |
|---|---:|
| HMX sites / admitted sites | 72 / 72 |
| 总 drain bytes | 21,316,608 B |
| 总 descriptors | 11,664 |
| 完整 / 边界 descriptors | 10,368 / 1,296 |
| DMA-admitted bytes | 21,233,664 B（99.61%） |
| 可与后续 HMX 重叠 bytes | 21,086,208 B（98.92%） |
| 边界同步 bytes | 82,944 B |
| 2D DMA legality | 通过；最大 destination stride 3,072 B |

P5m latency 为 3,268.25 ms，正确性与 P5k 相同（max abs diff 0.0046、allclose、top-1
match），说明 analysis 没有改变执行。P5n 需要额外保留 8 KiB VTCM：4 KiB 中容纳两个
2 KiB ping-pong result tile，另一个 4 KiB 保证随 slab 末端定位的 HexKL config block
不会与新增 slot 重叠。

完整产物：

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_dinov2_20260828
```

### 11.43 P5n：VTCM ping-pong + UserDMA 异步 HMX result evacuation（2026-08-28）

P5n 只在 P5m admitted 的完整行 tile 上改变 P5k drain：HMX accumulator 先在独立 VTCM
slot 中完成 AH→RM；编译器在覆盖该 slot 前等待旧 token，然后发出 VTCM→DDR 2D
UserDMA 并立即继续下一个 HMX tile。两个 slot 交替使用，短 M 边界仍走有界同步 copy，
result 逃逸和 VTCM dealloc 前统一 flush。该路径由
`ALPS_ENABLE_HMX_ASYNC_DRAIN` / `--alps-p5n` 独立控制，默认关闭；P5m/P5n 以及此前
P0–P5k 的开关仍可分别消融。

定向测试曾捕获一个实现错误：异步分支最初落在 weight-prepack 调度内，而准入条件又排除
weight-prepack，因此只生成 flush。修复后异步逻辑位于实际使用的 M-outer 调度，并由
FileCheck 同时验证 `wait -> AH-to-RM -> start`、边界同步 copy 和最终 flush；方言
round-trip、HexKL-to-LLVM lowering 与增量构建均通过。

同日、相同完整模型与设备配置的 matched 结果为：

| 指标 | P5k matched control | P5n async drain |
|---|---:|---:|
| Latency | 3,234.81 ms | **2,993.49 ms** |
| P5n incremental speedup | 1.00x | **1.0806x** |
| Latency reduction | 0% | **7.46%** |
| Correctness | PASS；max diff 0.0046 | **PASS；max diff 0.0046** |

独立 HexKL-phase LWP/P5n telemetry run 为 3,048.39 ms，并给出实际运行时证据：

| 证据 | P5k | P5n |
|---|---:|---:|
| 同步 `copy_f16_to_submatrix` root share | 6.78% / 7,200 次 | **0.05% / 1,296 次** |
| async start exposed share | NA | 0.07% |
| async wait exposed share | NA | 0.03% |
| async flush exposed share | NA | <0.01% |
| UserDMA issued / completed | NA | **10,368 / 10,368** |
| UserDMA issued bytes | NA | **21,233,664 B** |
| synchronous DMA fallback | NA | **0** |

LWP 只用于热点排名，实际 descriptor 次数以 runtime telemetry 为准。运行时 issued 数与
P5m admitted descriptor 数完全一致，completed=issued 且 fallback=0；因此这次收益不是
L2 hint（L2 issued 仍为 0），而是把 P5k 后语义上必须执行、且已被 LWP 证明暴露在关键
路径上的 result movement 提前发出，并与下一 HMX tile 计算重叠。同步 drain 从 6.78%
降到仅剩 1,296 个边界 descriptor，也与 7.46% formal latency reduction 在量级上吻合。

这为论文中的 prefetch 部分提供了首个完整模型、正确性通过、matched control、实际 DMA
计数和关键路径变化相互闭合的正证据。它应准确称为 **consumer-contract admitted
asynchronous evacuation/supply**，不能泛化成任意 data prefetch 都有效。当前硬门控结论
是：P5n 在 DINOv2-small 上通过；但进入论文最终累计方案前，仍需在其他完整模型/至少
另一个 domain 上验证可迁移性。若没有 residual HMX drain 或缺乏 overlap distance，
admission 应拒绝，而不是强制启用。

完整产物：

```text
nano:/home/huzq85/2-working/working_set/alps_p5k_matched_control_dinov2_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_dinov2_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hexkl_phase_lwp_dinov2_20260828
```

### 11.44 跨 domain 验证与正式 ablation study 计划（2026-08-28）

P5n 已在完整 DINOv2-small 上形成“静态准入—实际 UserDMA—关键路径变化—formal
latency—正确性”闭环，但单一 Vision 模型不足以支持可迁移性结论。下一阶段不把所有
模型盲目送入 P5n，而是按 P5m 先筛选 residual HMX drain 和真实 overlap distance：

1. **Speech：完整 Whisper-tiny。** 先运行 P5m analysis-only；只有 admitted bytes、
   overlap bytes、边界比例和 VTCM 需求通过，才运行同配置 P5k/P5n。
2. **Language：完整 Qwen2.5-0.5B。** 保持现有分层编译、分层设备执行和 Perf 求和
   语义；同样先 P5m，禁止仅因旧 item7 有收益就推断 P5n 必然有效。
3. **Vision replication：ViT-Base 或 BEiT-Base。** DINO 已作为 Vision 正例；第三个
   新模型优先选择无需额外模型适配且 P5m HMX 覆盖更高者，用于同 domain 复现，而非
   替代跨 domain 证据。

初筛只执行一次完整模型设备测量，且严格串行。停止线如下：

- P5m 没有合法 descriptor、静态 overlap 很低或有效数据量太小：记为 admission
  negative case，不运行 P5n；
- P5n correctness 失败、UserDMA issued/completed 不相等或出现 fallback：立即排除；
- P5n 相对 matched P5k 明确回退：停止该模型，不做长时间局部调参；
- 初次增益低于约 3%：记为不确定，不进入正式重复实验；
- 只有增益明确且可由 LWP/PMU/DMA telemetry 解释的模型，才复用同一编译产物做至少
  3 次串行正式测量，报告 median、离散度和设备状态。

#### 11.44.1 最终 ablation 矩阵

消融必须区分“删除搬运”和“提前搬运”，不能把 layout/direct formation 的收益归入
prefetch。建议冻结以下嵌套配置：

| ID | 配置 | 隔离的因果问题 |
|---|---|---|
| A0 | HMLIR HVX，HexKL Off | 纯 HVX framework baseline |
| A1 | HMLIR HVX，HexKL On | HMX/HexKL mapping baseline |
| A2 | A1 + consumer-driven layout/continuity/direct formation | consumer contract 能删除多少 representation movement |
| A3 | A2 + HMX F16 epilogue/direct-output formation（P5j/P5k） | HMX 边界的 widen/narrow 与 padded-output copy 收益 |
| A4 | A3 + admitted asynchronous drain（P5n） | 无法删除的 residual movement 被异步隐藏的独立收益 |
| A5 | A4 + PMU/traffic admission | runtime monitoring、节流和拒绝策略的增量 |

此外增加两个机制消融：

- **forced asynchronous movement vs P5m admission**：证明“选择性 prefetch”优于
  无条件发出；无 overlap distance 的模型应由 admission 拒绝；
- **synchronous drain vs ping-pong async drain**：P5k 与 P5n 的一对一比较，是论文中
  prefetch/提前搬运贡献的主要归因实验。

实现开关必须继续默认关闭且可独立控制。正式表格至少报告 latency、speedup、正确性、
HMX rewrite/site 数、P5m admitted/overlap bytes、UserDMA issued/completed/bytes/fallback、
LWP exposed transfer share，以及可获得时的 PMU/sysMon 指标。编译产物和设备结果仍直接
移动到 nano 的 `working_set`，本地只保留小型 CSV/Markdown 摘要。

### 11.45 P5n 跨 domain 初筛：Whisper、Qwen 与 BEiT（2026-08-28）

本轮配置没有跳过 P5h/P5i。`--alps-p5m` 是累计配置，包含 P0、P2e、P2g/P2gc、
P5fa、P5gd/P5gf/P5gg、**P5h、P5i、P5j、P5k** 和 P5m；`--alps-p5n` 在此基础上
仅增加异步 drain。P5i 的 patch-conv contract 对非视觉模型通常自然为零，P5h 则仍按
合法性匹配。最终 ablation 仍须将 P5h/P5i 单独拆分，不能用累计配置推断单项贡献。

完整 Whisper-tiny 的 P5m 静态筛选显示很高的理论覆盖：65 个 site 中 53 个准入，
22,405 个 descriptor 中 21,973 个准入；总 drain 为 45,774,400 B，准入
45,000,256 B，可重叠 44,892,160 B。P5m 的执行路径与 P5k 相同，但设备进程以
exit 13 / `0x8000040d` 异常退出。按停止线没有重复运行，也没有继续 P5n；因此该模型
当前只能记为“静态潜力高、累计 control 在设备上不受支持”，不能提供 latency 结论。

完整 Qwen2.5-0.5B（24 层分层执行，seq_len=32）的 P5m/P5k 等价 control 正确通过：
latency 10,582.80 ms，最终 top-5 匹配。静态聚合为 169 个 admitted site、14,252 个
descriptor、29,188,096 B admitted bytes，以及 14,083 个可重叠 descriptor、
28,841,984 B overlap bytes。由于 P5m 是 analysis-only 且执行与 P5k 完全相同，该值
作为本轮 matched synchronous control，避免再花约 22 分钟重复编译同一路径。

Qwen P5n 的 24 个 transformer layer 均正确通过；每层 UserDMA issued/completed 均为
396/396、811,008 B、fallback=0。但 head 的 4,748 个 DMA 也显示 issued=completed、
9,723,904 B、fallback=0 时，输出仍退化为全零量级并触发 correctness 失败。因此
10,025.77 ms 只是无效运行时间，**不得计算 speedup**。后续源码核查找到了确定根因：
UserDMA 2D descriptor 的 source/destination stride 都只有 16 bit，而 head stride 为
303,872 B；当前 runtime 未做范围校验，setter 直接 mask 成 41,728 B，P5m 又错误地把
它报告为 `dma2d_legal=1`。所以 completed 只表示截断后的错误 descriptor 完成，并不
表示目标 tensor 正确。按预定停止线不重复运行、不做该模型局部调参。

完整证据：

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_whisper_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_qwen_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_qwen_20260828
```

完整 BEiT-Base 提供了同 domain 的独立复现。P5m/P5k 等价 control 正确通过，latency
8,864.83 ms；72 个 site 中 60 个准入，28,311,552 B admitted、28,188,672 B 可与后续
HMX 重叠。P5n latency 为 8,292.04 ms，相对 matched control 为 **1.0691x**，延迟下降
**6.46%**；运行时 issued/completed=13,824/13,824、issued bytes=28,311,552 B、
fallback=0，且 max abs diff 仍为 0.0056、top-1 匹配。该结果复现了 DINO 上“无法删除的
result movement 通过 VTCM ping-pong + UserDMA 提前搬运”的独立收益。

需要严格区分归因：BEiT 的 P5m/P5n 都包含 P5h 与 P5i，因而这一对 matched comparison
只隔离 **P5n**，不能证明 P5h 或 P5i 单独贡献。P5h/P5i 的收益与交互必须留到 A2/A3
拆分消融中测量。当前筛选集合为：DINO、BEiT 是 correctness-closed 正例；Whisper 是
device-unsupported negative；Qwen 是 hardware-descriptor-range negative。跨 domain
结果说明 P5n 不能无条件泛化，admission 必须验证硬件 descriptor 的 address、width、
height 和 stride 字段范围，而不只是 bytes 与 overlap distance。

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_beit_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_beit_20260828
```

随后对层级窗口注意力的完整 Swin Transformer 进行同一筛选。P5m/P5k 等价 control
latency 为 41,401.57 ms，35/35 site 准入，14,770,176 B admitted、14,698,496 B
可重叠；P5n 正确通过，latency 40,898.92 ms，issued/completed=7,212/7,212、
issued bytes=14,770,176 B、fallback=0，top-1 匹配。相对 control 仅 **1.0123x**，
延迟下降 **1.21%**，低于预设 3% 继续门槛，因此记为“机制正确执行但收益不确定”，
不进行重复测量或局部调参。这也说明 admitted/overlap bytes 是必要而非充分条件；窗口
注意力中残余 drain 占总关键路径的比例明显低于平坦 ViT 类模型。

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_swin_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_swin_20260828
```

Speech/Language 的下一轮筛选进一步收紧了 admission 边界。完整 HuBERT-base 在 P5m
编译阶段由 P5g-e/f 主动拒绝。进一步源码与日志对照表明，并不是抽象的 alias 不安全：
P5g-f 的 head-major VTCM type builder 明确只接受 static rank-3 root，但 admission 把
HuBERT 的 rank-2 `512x64`/`768x64` roots 标成 rewrite-ready，rewrite 入口随即失败，
最后才被统一包装成“failed to clone alias chain”。因此没有生成设备 module；按停止线
不绕过证明、不运行 P5n。完整 GPT-2 的 P5m control
正确通过，latency 3,583.37 ms，但 12 个 block 的 48 个 HMX site 全部被拒绝，唯一静态
准入项是最终 vocabulary head（3,216,448 B admitted，3,215,360 B overlap）。其
100,514 B destination stride 同样超过16位上限，若执行会被截断成34,978 B；因此基于
Qwen 的确定根因直接拒绝 P5n，而不是重复一次已知高风险执行。

这两个 negative case 不属于“测试没跑完”：HuBERT 是 compiler rank-contract negative，
GPT-2 是 hardware-descriptor-range negative。正式 P5m admission 和 runtime fallback 都
必须固化 16-bit width/height/stride 检查；P5g-f admission 则必须要求 rank-3，或安全回退
到非 head-major P5g-e，而不能让 rewrite 阶段才失败。

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_hubert_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_gpt2_20260828
```

完整 Whisper-tiny 的功能二分确认了第三个独立根因边界。P2g analysis-only control 正确
通过，latency 76,799.45 ms，max diff 0.0039、last-token top-1 匹配；仅增加 P2g-c 后，
编译成功但设备复现 exit 13。P2g-c 报告 21 个 register-tile vectorization candidate，
13 个成功、8 个失败。继续向后的 P5g-g 仍 exit 13，且其 `rewritten_epochs=0`，所以
Whisper 的异常不是 P5g/P5h/P5j/P5k 或 P5n 引入，而是首次由 P2g-c register-tile direct
formation 引入。下一层定位应在这13个成功 site 上做 compiler-side per-site admission/
bisection；sysMon 只能记录系统 PMU/traffic，无法把 DSP exception 映射回某个 IR site，
不适合作为此功能错误的首要工具。

```text
nano:/home/huzq85/2-working/working_set/alps_p2g_whisper_rootcause_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_rootcause_20260828
nano:/home/huzq85/2-working/working_set/alps_p5gg_whisper_rootcause_20260828
```

### 11.46 Whisper P2g-c exit 13 的逐站点根因与修复（2026-08-28）

P2g-c 增加了 compiler-side demand window 和逐站点 telemetry。完整 Whisper 共找到
13 个 register-tile direct site；只启用 demand 0 即可稳定复现 exit 13，因此不再对其余
site 做线性盲跑。该 site 把 `tensor<1x384x1500xf16>` 直接形成
`tensor<1x1500x384xf16>`，permutation 为 `[0,2,1]`。失败 object 的 V73 反汇编没有
VTCM `vgather`/`vscatter`，实际生成的是成组 `vmemu`、寄存器内 `vdeal` 和 `vmemu`
store；所以这条路径是 HVX VRF register-tile formation，而不是 VTCM gather。

原准入只证明 affine map 可形成 128 B register tile，却没有证明源端整宽 load 的尾部
安全。V73 的 128 B HVX 向量对 FP16 是 64 elements，而源物理最内层 extent 1500 不能
被 64 整除；最后一个整宽 `vmemu` 会跨过逻辑 row 边界。对照 object 还排除了栈大小
假说：失败与正确 control 的 frame 只差约 256 B，control 本身已有约 174 KiB frame。

修复后的 P2g-c admission 要求 source physical innermost row 是整数个 native 128 B
vector；否则保留原生 transpose，直到未来具备 masked/padded tail lowering。该规则不是
性能启发式，而是 codegen safety proof。单元测试同时覆盖 64xf16 安全正例和 96xf16
非整向量拒绝例。完整 Whisper 修复后结果为：

| 指标 | 修复后 P2g-c |
|---|---:|
| Latency | 77,912.83 ms |
| Correctness | PASS；max diff 0.0039；last-token top-1 match |
| demands / producer-direct / register-tile-direct | 114 / 48 / 12 |
| rejected unsafe-tail sites | 1（原 demand 0，source inner extent 1500） |
| eliminated materialization bytes（静态） | 28,237,824 B |

相比同日 P2g analysis-only control 的 76,799.45 ms，修复后的 P2g-c 为约 **0.9857x**
（慢 1.45%）。因此本结果首先关闭了 Speech 路径的功能正确性缺口，但没有证明 P2g-c
本身有性能收益。12 个安全站点虽然静态替代了 transpose materialization，其
`vmemu + vdeal` producer-side formation 仍可能留在关键路径；后续不能把静态 eliminated
bytes 直接等价为物理 traffic 或 latency reduction。

```text
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_site_diag_all_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_site_0_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_tailfix_20260828
```

Whisper 修复后也重新关闭了 P5m/P5n 的 matched comparison。P5m/P5k synchronous
control 为 71,240.91 ms；52/65 site 通过新的 descriptor-range admission，20,352 个
descriptor、41,680,896 B admitted，其中 41,574,400 B 具有静态 overlap。其余 13 个
site 因最大 destination stride 达 103,730 B 而拒绝，证明 aggregate
`dma2d_legal=0` 不等于所有合法子 site 都必须放弃。

P5n 为 70,414.99 ms，相对 control **1.0117x**、延迟下降 **1.16%**；正确性保持
max diff 0.0039、last-token top-1 match。运行时 issued/completed 为
20,352/20,352，issued bytes 41,680,896 B，fallback=0。它说明异步 HMX result drain
在 Speech 模型上功能正确且真正执行，但收益低于预设 3% continuation threshold，故
停止重复测量和局部调参。Whisper 应作为 correctness-qualified weak-performance case，
不能进入显著收益集合；高 admitted/overlap bytes 仍不是关键路径占比的充分条件。

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_whisper_tailfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_whisper_tailfix_20260828
```

descriptor-range 修复后，完整 Qwen2.5-0.5B P5n 也从此前的错误全零 head 恢复为正确：
24 个 transformer layer 各自发出并完成 396 个 descriptor、811,008 B；聚合为
9,504/9,504、19,464,192 B、fallback=0。最终 vocabulary head 的 destination stride
为 303,872 B，新的 P5m admission 正确报告 `dma2d_legal=0`、admitted=0，并保留同步
drain，因此 head 不再发生 silent 16-bit truncation。最终 finite=True、top-5 match，
max abs 为 0.56640625（沿用该分层完整模型既有 top-5 correctness gate）。

修复后 P5n latency 为 10,364.01 ms，相对同配置 P5m/P5k matched control
10,582.80 ms 为 **1.0211x**、延迟下降 **2.07%**。它把 Qwen 从“无效运行”升级为
Language domain 的 correctness-qualified weak-performance case，但仍低于 3% continuation
threshold，不做正式重复测量。该结果同时验证了新的 admission policy：合法的 layer
descriptor 继续异步执行，非法的大 stride head 局部同步回退，而不是整模型禁用 P5n。

```text
nano:/home/huzq85/2-working/working_set/alps_p5n_qwen_stridefix_20260828
```

完整 HuBERT-base 在 rank-contract 修复后的 P5m control 为 180,829.05 ms，P5n 为
176,891.35 ms，即 **1.0223x**、延迟下降 **2.18%**。74/74 site 合法，运行时
issued/completed=5,234/5,234、10,719,232 B、fallback=0；正确性 finite、max diff
0.0083、last-frame top-1 match。与 Whisper 一致，HuBERT 证明 Speech encoder 上机制
可以正确执行，但未跨过 3% continuation threshold，因此停止重复测量。

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_hubert_rankfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hubert_rankfix_20260828
```

完整 GPT-2 在 descriptor-range 修复后重新运行 P5m，latency 3,583.47 ms，12 层
finite、last-token top-1 match。12 个 block 的 48 个 HMX site 全部没有 P5k residual
direct-output descriptor，最终 vocabulary head 的 100,514 B destination stride 被正确
报告为 `dma2d_legal=0`；因此完整模型 admitted descriptor/bytes 为 0。P5n 不再运行：
它不会发出异步 DMA，与 control 相同，属于明确的 static admission negative，而不是测试
缺失。

```text
nano:/home/huzq85/2-working/working_set/alps_p5m_gpt2_stridefix_20260828
```

本轮修复后的跨 domain 结论是：Vision 的 DINO/BEiT 仍是 P5n 明确正例；Swin、Qwen、
Whisper、HuBERT 是 correctness-qualified weak cases；GPT-2 是零合法工作量的 admission
negative。sysMon/PMU 不适合定位本轮三个功能错误，因为它不能把 DSP exception 反向
映射到 IR demand 或证明 descriptor field 未截断。现在功能正确性已经关闭，后续若要
解释弱模型的性能差异，再对代表性 weak case 做 LWP/PMU critical-path profiling；不能
用系统 traffic 计数替代 compiler-side safety proof。

### 11.47 关闭电源混淆并恢复 item7 组合后的 Qwen 归因（2026-08-28）

此前 Qwen P5m/P5n 的 10.58/10.36 s 不能与历史 item7 约 5.86 s 直接比较：前者是
P5 累计配置但没有组合 item7，后者启用了 item7。手机当时还处于
`low_power=1`、sticky battery saver 和 Doze，进一步形成了设备状态混淆。新增
`scripts/script_release/internal/prepare_phone_benchmark.sh` 在每次正式运行前关闭 low-power 与 adaptive
power saver、解除并禁用 light/deep device idle、唤醒设备；thermal protection 保持
启用，不以关闭安全保护换取数字。vendor 禁止 shell 修改 stay-on/app-standby，因此
脚本只报告这些只读状态，不伪造“已固定”。本轮三次完整模型均确认
`low_power=0`、sticky=0、device-idle disabled/ACTIVE。

维护脚本 `run_full_hvx_five_way.sh` 现支持 `--item7-only`，以及把
`--with-item7` 与 `--alps-p5m`/`--alps-p5n` 组合。完整 Qwen2.5-0.5B、24 层、FP16、
seq_len=32 的 matched 结果如下；三项均 finite、top-5 match，max abs 为 0.58984375：

| 配置 | Latency | 相对前项 | 相对 item7-only | 运行时机制 |
|---|---:|---:|---:|---|
| item7-only | 5,911.34 ms | 1.0000x | 1.0000x | K/V pair=1；L2 issued=0 |
| item7 + P5m/P5k | 5,639.33 ms | **1.0482x** | **1.0482x** | 累计 layout/direct formation；同步 residual drain |
| item7 + P5n | **5,328.06 ms** | **1.0584x** | **1.1095x** | 9,504/9,504 async DMA；19,464,192 B；fallback=0 |

item7-only 与历史 5,859.60 ms 只差 0.88%，因此此前“item7 收益消失”的主要原因是
实验配置错配，不是新 compiler 修改破坏了该路径。item7 仍然没有发出硬件 K/V L2
hint；它的收益应归因于 attention propagation/tiling/slicing topology，而不能写成
runtime data-prefetch 收益。P5m 相对 item7 的 4.82% 是累计 representation/layout
formation 的增量；只有 P5n 相对 matched P5m 的 5.52% 才隔离异步提前搬运。

逐阶段 Perf 排除了单层偶然值。24 层全部在 P5n 下快于 P5m；layer 总和从
4,467.11 ms 降至 4,150.30 ms，即 **1.0763x**。逐层 P5m/P5n speedup 范围为
1.0276x--1.1056x。head 因 303,872 B stride 不合法而同步回退，从 1,171.92 ms
轻微变为 1,177.45 ms；所以完整模型的 P5n 收益确实来自24个合法 transformer layer，
不是 vocabulary head 或 embedding。

为了在不重编译完整模型的情况下做硬件归因，新增
`scripts/script_legacy/profile_archived_hexagon_stage.sh`。它复用远端归档的同一个完整 layer-0
产物，保留 wrapper 编译时嵌入的原设备目录名，并在同一个 sysMon 窗口内串行执行20次。
首次复用曾因临时目录改名导致 wrapper 找不到硬编码 input、以 `0x8000040d` 退出；确认
根因后才修复脚本，没有把 exit 13 当性能运行重复。单次窗口太短时 SDK parser 不生成
PMU CSV，因此20次只用于稳定硬件计数，不替代上面的全模型 latency。

| 代表层 matched sysMon 指标 | item7 + P5m | item7 + P5n | P5n 变化 |
|---|---:|---:|---:|
| 最后一轮 wrapper Perf | 187.26 ms | **179.68 ms** | **-4.05% / 1.0422x** |
| 20次 host window | 9.026 s | **8.902 s** | **-1.38%** |
| processor cycles | 8,792,980,362 | **8,571,065,337** | **-2.52%** |
| AXI cached read | 2,305,764,096 B | 2,306,258,432 B | +0.02% |
| AXI cached write | 791,842,432 B | 971,270,400 B | +22.66% |
| AXI total | 3,097,606,528 B | 3,277,528,832 B | +5.81% |
| HVX+HMX-active window AXI | 791,542,400 B | **739,557,376 B** | **-6.57%** |
| HVX packet / HMX active event | 20,006,622 / 14,724,042 | 20,111,989 / 14,727,075 | +0.53% / +0.02% |

两次 PMU 前后均为 USB powered、100% battery、27.4--27.6 C、low-power off、idle
disabled，因而没有 thermal/battery-saver 解释。P5n 没有减少总物理流量；异步 drain
增加了可见写事务，但 processor cycles 和 HVX+HMX active-window AXI 同时下降。当前
证据支持的机制是：P5n 把无法消除的 residual HMX result movement 与后续计算重叠，
缩短关键路径，而不是通过减少 DDR 字节加速。`neither` 活动窗口还包含每次独立 runner
加载约30 MB constants 等系统域开销，所以不应把20次 system-domain AXI 总量外推为
一次完整推理的 tensor traffic。

完整证据已移动到：

```text
nano:/home/huzq85/2-working/working_set/alps_qwen_matched_item7_power_20260828
nano:/home/huzq85/2-working/working_set/alps_qwen_matched_item7_p5m_20260828
nano:/home/huzq85/2-working/working_set/alps_qwen_matched_item7_p5n_20260828
nano:/home/huzq85/2-working/working_set/alps_qwen_item7_p5n_sysmon_20260828
```

因此跨 domain 筛选结论应分配置表述：不组合 item7 的原 P5n 矩阵中，显著正例仍是
DINO/BEiT；在论文目标的 item7 + representation formation + async supply 统一配置下，
Qwen 已成为首个超过3%门槛且有 PMU/DMA 闭环的 Language 正例。正式 median/离散度不在
本轮继续重复，按既定约定与所有模型的最终 ablation 一起串行执行，避免筛选阶段演变为
无休止测试。

## 12. 完整15模型 LWP + sysMon 瓶颈语料库（2026-08-29 起）

### 12.1 为什么需要全模型 profiling，以及如何防止偏离论文故事

对15个完整模型全部建立 LWP + sysMon 语料库是必要的。此前按单模型静态 bytes 或
latency 猜测热点，已经多次出现“admitted bytes 很高但收益很弱”和“显式 copy 减半但
真实关键路径只下降数个百分点”。统一 profile 的目的不是为每个模型分别发明一个优化，
而是先得到可比较的事实，再把热点归并成少数可由 ALPS 统一处理的类别。

固定设备执行配置为完整 FP16 模型、HVX vector、HexKL on、item7 synchronous control。
已经能承受 P5m/P5k 编译内存的分层模型额外使用该配置观察尚未被 async drain 隐藏的
residual movement；单体大图不得为了 profiling 强制打开整条 P1--P5 编译期分析链。
编译期 movement ledger 是离线辅助证据，不是 sysMon 测量语义的一部分。每个模型分两次
运行：

1. **LWP插桩运行**：分层模型先用loop depth 1；单体大图先用函数/顶层region，仍超过
   内存预算时只插桩由Perf/sysMon锁定的热点函数。启用HexKL phase markers，用exclusive
   cycles定位operator/stage critical path；该延迟只用于排名，不作为正式latency。
2. **非插桩sysMon运行**：使用与正式item7运行相同的device code，优先重放已验证的归档
   产物；
   所有input/constant/shared object在PMU启动前预置，避免把编译和ADB上传混入窗口。
3. 两次均在 `low_power=0`、device-idle disabled、thermal protection enabled下串行运行；
   不设置timeout，不自动重试exit 13；每个case完成即移动到nano并删除本地大产物。

LWP与sysMon回答不同问题：LWP把cycles映射回stage/operator，sysMon给出真实HVX/HMX、
AXI read/write和burst行为。sysMon是CDSP system-domain统计，不能单独把某些bytes归给某
个MLIR op；分层模型的whole-model replay还包含每个runner process的启动成本，因此正式
模型Perf仍取wrapper内部各stage之和。

实现入口：

```text
scripts/script_legacy/run_full_bottleneck_corpus.sh
scripts/script_release/internal/profile_archived_hexagon_model.sh
scripts/script_release/internal/summarize_alps_lwp.py
scripts/script_release/internal/summarize_model_sysmon.py
```

论文适配采用四类标签：

| 标签 | 热点类型 | ALPS中的处理 | 是否进入论文核心 |
|---|---|---|---|
| E | 可消除的layout/materialization/representation round trip | consumer-driven direct/in-situ formation、persistent representation | 是，第2贡献 |
| P | 无法消除但暴露在关键路径且有overlap distance的transfer | VTCM ping-pong、UserDMA async supply/drain | 是，第1/2贡献交界 |
| R | 带宽/traffic burst、资源冲突或prefetch负收益风险 | PMU admission、traffic control、拒绝/节流 | 是，第3贡献 |
| C | 纯算术、低效scalar codegen、通用算子实现 | 修复baseline或使用既有HVX/HMX优化 | 工程项；不包装成ALPS创新 |

只有满足以下条件的优化才进入统一方案：至少跨两个模型复现，最好跨domain；覆盖显著
exclusive cycles或真实traffic；能由E/P/R之一表达；具有独立开关和matched ablation。
只服务一个模型的普通vectorization、特殊算子或精度技巧保留为baseline engineering，
不能为了提高数字破坏“先消除、再提前搬运、最后runtime admission”的故事线。

### 12.2 Corpus状态与逐模型优化路径

下表的 `measured` 只在LWP和sysMon均完成后填写；其余行是等待profile验证的假说，不能
当作测量结论。DINO/Qwen此前数据保留为先验证据，但主corpus仍以本节固定配置为准。

| Domain | 完整模型 | LWP/sysMon状态 | 当前最大瓶颈 | 候选优化路径 | 论文标签 |
|---|---|---|---|---|---|
| Language | GPT-2 | **measured** | vocabulary head占约60%；blocks约40%；HMX outer chain与未归因等待占主导 | vocabulary projection分块direct formation；合法stride的chunked drain；细分head root等待 | E/P |
| Language | SD/CLIP | **LWP + sysMon measured** | 12 layers占99.65%；HMX outer 66.09%，output copy 8.26%，GELU 7.75% | HMX direct consumer formation + async drain；GELU归C | E/P/C |
| Language | Qwen2.5-0.5B | **LWP + sysMon measured** | head21.69%；HMX outer63.50%；extf链16.22%；output copy6.18% | head分块direct formation + async drain；保留item7 topology | E/P/R |
| Language | TinyLlama-1.1B | **LWP + sysMon measured** | layers96.69%；HMX outer79.48%；extf链11.08%；output copy2.55% | layer projection persistent representation、合法分块drain | E/P/R |
| Language | SmolLM2-1.7B | **LWP + sysMon measured** | layers94.42%；HMX outer84.60%；head5.58%；output copy2.12% | layer projection persistent representation、合法分块drain | E/P/R |
| Vision | Swin Transformer | **full-graph LWP + sysMon measured** | extf/mulf/addf累加链94.00%；HMX仅2.22%；51.10 s | 首先修复window/MLP基础HVX codegen C；随后对window consumer layout做E | C/E |
| Vision | SegFormer MiT-B0 | **full-graph LWP + sysMon measured** | extf/mulf/addf累加链85.68%；HMX仅2.63%；AXI 0.32 GB | 首先修复多尺度conv/MLP基础HVX codegen C；随后做consumer layout E | C/E |
| Vision | DeiT-small | **full-graph LWP + sysMon measured** | extf/mulf/addf累加链57.64%；HMX链26.21%；softmax/reduction 4.23%；5.09 s正式运行 | 先修FP32累加/扩展链C，再对HMX表示做E/P；不再假设transpose是第一热点 | C/E/P |
| Vision | BEiT-base | **full-graph LWP + sysMon measured；P5n正例** | HMX chain 47.57%；extf/mulf/addf 34.90%；softmax/reduction 11.34%；AXI 3.54 GB | HMX direct/persistent representation E + residual async drain P；累加/softmax归C | E/P/C |
| Vision | DINOv2-small | **LWP + sysMon measured** | patch热点已由P5i消除；剩余HMX outer、attention arithmetic、residual drain；item7 replay 6.07 s | persistent HMX layout；P5n；纯attention算术归C | E/P/C |
| Speech | Whisper-tiny | **full-shape local LWP + sysMon measured** | encoder frontend 88.37%；一个encoder layer 8.38%；head 2.30%；conv累加链92.43% | 先做frontend HVX conv C，再将conv结果直接形成consumer layout E；P仅处理剩余tile supply | C/E/P |
| Speech | HuBERT-base | **local LWP + sysMon measured** | frontend90.74%、position9.17%；conv codegen dominated | shared HVX conv C，再做conv→token E | C/E |
| Speech | Wav2Vec2-base | **local LWP + sysMon measured** | frontend90.96%、position8.95%；conv codegen dominated | shared HVX conv C，再做conv→token E | C/E |
| Speech | UniSpeech-base | **local LWP + sysMon measured** | frontend90.73%、position9.17%；共享conv瓶颈 | 共享speech C/E contract，不做模型特例 | C/E |
| Speech | UniSpeech-SAT-base | **local LWP + sysMon measured** | frontend90.03%、position9.88%；共享conv瓶颈 | 共享speech C/E contract，不做模型特例 | C/E |

### 12.3 首个统一样本：完整 GPT-2

完整12层FP16 GPT-2的LWP和非插桩sysMon均通过，正确性为finite、last-token top-1
match。LWP运行3,625.62 ms、非插桩正式运行3,652.26 ms；二者接近只是设备稳定性旁证，
不能用插桩值计算speedup。LWP stage aggregate为：vocabulary head **60.26%**，12个
transformer block合计39.66%，embedding 0.08%。每个block约3.27--3.43%，说明不是单个
异常layer。

operation aggregate中，root/unattributed为49.80%，完整HMX outer chains约31.50%，
block内FP16扩展/标量算术链11.43%；独立可见的HMX FP16→FP32 output copy仍占3.66%，
RM→WH占1.42%。`micro_hmx_mm`本身的marker值很小不能解释为HMX计算免费：MM异步完成
等待被计入outer chain/root，后续需要在head增加更细completion marker。

whole-model replay的14个stage Perf之和为3,594.78 ms，其中head 2,163.52 ms
（60.19%），12个block约1,428.32 ms（39.73%），再次独立复现LWP分布。sysMon窗口为
8.293 s（包含14次runner process启动但不含文件上传），processor cycles 8.405B，
AXI read/write为1.417/0.549 GB，总量1.966 GB；平均170.84/66.23 MB/s。AXI并非持续
峰值饱和，且1.512 GB出现在HVX/HMX均未记录活跃的system-domain窗口，主要受分层runner
启动/constant mapping影响。因此GPT-2的第一优化对象不是全图L2 prefetch，而是占60%的
vocabulary head：按consumer需要对large-N projection分块，避免大stride/padded result，
能直接形成就消除；无法消除且满足16-bit descriptor限制的chunk才进入异步drain。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/gpt2
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/sysmon/gpt2
```

### 12.4 DeiT暴露出的profiling可扩展性问题与修正协议

DeiT-small的完整单体图分别尝试了LWP depth 1、LWP depth 0，以及把P1--P5m编译期分析
链绑入非LWP sysMon构建。三条路径均在手机运行前被Linux global OOM终止，python匿名
RSS约14.67 GB；LWP没有生成`lwp.json`。单线程和depth 0仍失败，证明根因不是手机、
FP16、编译并行或sysMon，而是完整单体IR与全图分析/插桩的结构性内存峰值。继续重复同一
路径既不能增加证据，也违反“失败不盲目重跑”的实验约定。

因此用2026-08-15已经通过正确性检查的完整item7产物进行非插桩sysMon replay。wrapper
Perf为 **5,086.94 ms**；PMU窗口5.531 s，processor cycles 7.754B，AXI read/write为
620.73/248.43 MB，总869.15 MB；HVX/HMX event为104.06M/7.53M。按1 ms活动窗口：
HVX+HMX共同活跃1408个sample并产生404.75 MB AXI，HVX-only 644个sample/172.40 MB，
neither 3478个sample/291.90 MB。它说明DeiT并非“没有vector/HMX工作”，而是规则ViT
pipeline中计算与外存活动明显重叠；仅凭总AXI仍不能判定某个transpose或drain在关键
路径上。

DeiT下一步不是再次全图插桩，而是：先以已有compiler ledger列出的96个produced
representation和72个HMX drain site建立静态候选；再仅对patch embedding、单个
attention block和HMX completion/drain插入局部LWP marker。若局部exclusive cycles与
高AXI burst重合，则走E（直接形成/persistent layout）或P（VTCM async drain）；若计算
marker占主导且搬运不在critical path，则归C，不为追求统一数字强行增加prefetch。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/sysmon/deit-small/item7-archived-replay
```

这几次失败更新了15模型的执行策略：**所有模型都要有sysMon；LWP要覆盖每个模型的关键
路径，但不要求每个模型都用相同的全图插桩粒度。** 分层Language模型按stage完整画像；
单体Vision/Speech先做非插桩sysMon和wrapper Perf，再对排名靠前的函数/region做局部LWP。
两种工具最终仍在统一的E/P/R/C分类上汇合，既保证profiling可扩展，也避免让工具开销
改变论文故事。

### 12.5 远端直供回放验证：DINOv2-small

为满足本地磁盘约束，新增`profile_remote_archived_hexagon_model.sh`：输入、主shared
object和constant shared objects从nano经SSH直接流到手机，编译产物不落地`/tmp`；PMU
启动前完成全部传输，故网络传输不进入sysMon窗口。DINOv2-small的完整item7归档通过
该路径复测，wrapper Perf为 **6,071.57 ms**，与此前同配置约5.95--6.1 s量级一致。

sysMon窗口6.893 s，processor cycles 9.188B，AXI read/write为842.61/338.96 MB，总
1.182 GB；HVX/HMX event为173.44M/9.69M。HVX+HMX共同活跃窗口产生488.25 MB AXI，
HVX-only产生316.43 MB，说明该模型同时存在显著vector、matrix和memory activity。结合
此前LWP消融，优化判断仍然成立：原patch convolution的39.09%热点可以通过P5i direct
formation退出关键路径；其后应对residual HMX representation/drain采用E/P，而纯
attention arithmetic归C。不能仅依据1.182 GB总AXI继续扩大无差别data prefetch。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/sysmon/dinov2-small/item7-remote-replay
```

### 12.6 首个Speech样本：Wav2Vec2-base

完整FP16 Wav2Vec2-base远端item7归档通过sysMon replay，wrapper Perf为
**184,932.30 ms**，与原归档185,248.95 ms一致。PMU窗口185.752 s，processor cycles
273.84B、committed packets 89.69B；但HVX/HMX event只有37.78M/8.47M。185,752个1 ms
sample中，**183,760个**既无HVX也无HMX event，HVX+HMX共同活跃仅1,799个。AXI总量虽有
5.492 GB，平均read/write bandwidth仅27.28/2.29 MB/s，不是持续DRAM带宽饱和。

编译日志还显示原始98个`batch_matmul`中只flatten了74个，item7传播了24个K/V pair，
但最终runtime prefetch hint为0。结合PMU，当前最大问题不是“没有发足够prefetch”，而是
大部分计算没有落到HVX/HMX：可能来自conv frontend、normalization、activation、
reduction或未满足vector/HMX contract的逐元素loop。下一步要把完整wrapper只作为正式
latency/sysMon载体，另建保持原shape/weight的frontend、单个encoder block、CTC head局部
LWP载体；先按exclusive cycles排序，再把layout造成的失败归E，把普通vectorization/
算子实现归C。只有E路径和随后暴露出的不可消除transfer才能进入ALPS，不把通用Speech
codegen修复伪装成prefetch贡献。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/archived_sysmon/wav2vec2-base
```

BEiT-base的item7 replay为 **13,824.45 ms**，PMU窗口18.630 s、processor cycles
24.64B，AXI read/write为2.981/0.555 GB。HVX+HMX共同活跃6575 ms并产生2.340 GB AXI，
明显不同于Wav2Vec2的scalar-dominated形态；结合此前P5n正收益，它是E/P统一路径的强候选。
但原始72个batch matmul没有被Python文本rewrite计数命中，不能把所有HMX activity简单
归因于该rewrite；仍需用局部LWP把attention、MLP和drain completion拆开。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/archived_sysmon/beit-base
```

### 12.7 SD/CLIP：最清晰的跨层重复HMX模式

完整FP16 SD/CLIP的非插桩14-stage Perf合计 **3,438.01 ms**；12个encoder layer占
99.65%，embedding和final norm合计仅12.14 ms。item7 LWP运行3,489.10 ms并通过完整
12层正确性；12层的exclusive share分别约8.23--8.58%，没有异常单层。

跨stage LWP operation aggregate显示：完整HMX outer chains占 **66.09%**，独立
`copy_f16_to_f32_submatrix`占 **8.26%**，GELU标量链7.75%，reduction 4.11%，
RM→WH 2.00%。这给出了比“总traffic很大”更直接的critical-path证据：每层都重复相同
HMX计算、accumulator读取、layout/output copy；优先路径是让下游consumer直接接受HMX
形成的表示（E），不能消除的f16→f32/chunked drain再异步化（P）。GELU和reduction若
不能由layout contract改善则归C，不写成prefetch贡献。

sysMon窗口11.649 s（包含14次runner启动），processor cycles 7.638B，AXI read/write
1.700/0.468 GB；HVX+HMX共同活跃2695 ms并产生1.053 GB AXI。它与LWP共同说明CLIP不是
Wav2Vec2式的纯scalar failure，而是一个适合ALPS E→P路径的规则Transformer正例。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/sd-clip
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/archived_sysmon/sd-clip
```

### 12.8 15模型sysMon全覆盖后的瓶颈分群

截至2026-08-29，15个完整模型的非插桩item7 sysMon均已完成。下表中的`active`是至少
记录到HVX或HMX event的1 ms sample占比；Language模型是多runner分层回放，`neither`
包含每层进程启动/constant mapping，不能直接解释为scalar比例。Vision/Speech单体模型
的PMU窗口与wrapper Perf接近，因此其active比例可以更直接用于执行路径分类。

| Domain | Model | Perf | AXI total | HVX/HMX active samples | 测得的瓶颈类别 | 优化决策 |
|---|---|---:|---:|---:|---|---|
| Language | GPT-2 | 3.67 s | 2.01 GB | 12.27%* | head 60.26%，large-N projection | E分块direct formation；合法chunk才P |
| Language | SD/CLIP | 3.44 s | 2.17 GB | 29.12%* | HMX outer/output copy跨12层重复 | E direct consumer layout + P async drain |
| Language | Qwen2.5-0.5B | 5.88 s | 5.92 GB | 19.41%* | HMX outer 63.50%，head 21.69% | E/P；R约束大流量burst |
| Language | TinyLlama-1.1B | 17.77 s | 11.61 GB | 43.49%* | layers 96.69%，HMX outer 79.48% | 跨层E/P；head不是首要目标 |
| Language | SmolLM2-1.7B | 29.73 s | 18.99 GB | 51.06%* | layers 94.42%，HMX outer 84.60% | 跨层E/P；R admission价值最高 |
| Vision | DINOv2-small | 6.07 s | 1.18 GB | 42.36% | patch已优化，剩余HMX/attention | E/P；纯attention算术归C |
| Vision | DeiT-small | 5.09 s | 0.87 GB | 37.12% | 规则ViT compute+movement混合 | hotspot-local LWP后选择E/P |
| Vision | BEiT-base | 13.82 s | 3.54 GB | 46.69% | HMX+HVX活跃窗口2.34 GB AXI | E/P强候选，复用DINO路径 |
| Vision | Swin Transformer | 51.10 s | 1.10 GB | **3.00%** | window路径绝大部分未用HVX/HMX | 先C修baseline；window layout可归E |
| Vision | SegFormer MiT-B0 | 8.93 s | 0.32 GB | **3.64%** | 低traffic且绝大部分未用HVX/HMX；宽泛FP16 accumulator实验反而慢21.03% | C应保留f16 operands + vector f32 accumulate并消除scalar helper；多尺度layout只在LWP命中时归E |
| Speech | Whisper-tiny | 70.07 s | 6.86 GB | 17.20% | HVX event 1.93B，HMX较少 | 定位HVX conv/attention/reduction；E优先于P |
| Speech | HuBERT-base | 172.77 s | 5.46 GB | **1.27%** | scalar/codegen dominated | 共享Speech局部LWP；先C再E |
| Speech | Wav2Vec2-base | 184.93 s | 5.49 GB | **1.07%** | scalar/codegen dominated | 共享Speech局部LWP；先C再E |
| Speech | UniSpeech-base | 184.30 s | 5.48 GB | **1.08%** | 与Wav2Vec2相同signature | 不做模型特例，共享C/E修复 |
| Speech | UniSpeech-SAT-base | 179.39 s | 5.48 GB | **1.11%** | 与Wav2Vec2相同signature | 不做模型特例，共享C/E修复 |

`*`：分层runner的active百分比不能与单体模型横向解释为scalar占比。

这组数据把后续工作明确分成两条，而不是15条模型特例：

1. **ALPS主线（E→P→R）**：CLIP、Qwen、TinyLlama、SmolLM、DINO、BEiT及
   GPT head均已显示HMX representation/output或consumer layout问题。先消除/直接形成，
   对无法消除且位于critical path的chunk异步搬运，最后按PMU traffic admission。
   DeiT是C+E/P混合型，先处理57.64%的累加链，同时保留26.21%的HMX表示路径。
2. **baseline enablement（C→E）**：Swin、SegFormer、Whisper、Wav2Vec2、HuBERT、
   UniSpeech家族
   当前主要问题是没有进入HVX/HMX；此时prefetch搬运得再好也无法带来1.8x。必须先修通
   shared operator/vectorization contract，再检查新暴露的layout/materialization是否
   能由ALPS E处理。C类收益必须在论文ablation中与ALPS收益分栏报告。

Whisper的局部LWP已经把此前的“两者之间”假说收敛为C→E：frontend占88.37%，卷积累加
链占92.43%，attention不是第一热点。该分群支持论文统一故事，而不是削弱prefetch：
prefetch被严格
定义为只服务“不能消除、可重叠、runtime允许”的residual supply；没有满足条件时正确
行为就是拒绝prefetch。

当前LWP状态为：GPT-2、SD/CLIP、Qwen、TinyLlama、SmolLM、DINO、DeiT、BEiT、Swin和
SegFormer已有全图或分层operator级结果；Whisper及四个共享Speech encoder已有完整shape
局部LWP。15个模型的LWP覆盖已经闭环。不能把“带全套重分析pass的LWP编译因内存失败”
写成模型瓶颈，也不能用
sysMon system-domain bytes虚构operator归因。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/archived_sysmon
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp
```

### 12.9 共享Speech encoder的完整shape局部LWP闭环

Wav2Vec2、HuBERT、UniSpeech和UniSpeech-SAT使用相同的完整20560-sample输入、64 frames、
768 hidden和12层结构，分别把frontend、position formation、一个代表性encoder layer和
CTC head编译为独立LWP stage。它们不是Debug/缩小模型；正式latency仍取完整单体wrapper，
局部载体只为避免12层IR复制导致host OOM并做operator attribution。

| Model | Frontend | Position | One encoder layer | CTC head | 用局部Perf估算完整路径 | 正式完整Perf |
|---|---:|---:|---:|---:|---:|---:|
| Wav2Vec2 | 162.48 s / 90.96% | 15.98 s / 8.95% | 164.89 ms / 0.09% | 1.51 ms | 180.44 s | 184.93 s |
| HuBERT | 158.16 s / 90.74% | 15.98 s / 9.17% | 165.41 ms / 0.09% | 1.48 ms | 176.12 s | 172.77 s |
| UniSpeech | 158.09 s / 90.73% | 15.98 s / 9.17% | 164.53 ms / 0.09% | 1.46 ms | 176.05 s | 184.30 s |
| UniSpeech-SAT | 160.83 s / 90.03% | 17.65 s / 9.88% | 166.30 ms / 0.09% | 1.53 ms | 180.48 s | 179.39 s |

四个模型一致复现：约99.9%的局部exclusive cycles在7-layer convolution frontend和
position convolution；最大的operation class都是`extf, extf, mulf, addf`卷积累加链。
单个Transformer layer已有6/8 batch-matmul rewrite且只约165 ms，attention并不是当前
百秒级瓶颈。局部估算与完整Perf相差约2--5%，足以形成归因闭环，而不是把局部stage
延迟冒充完整模型测量。

因此共享Speech优化顺序已经确定：

1. C：给长序列conv1d/feature extractor建立真正的HVX convolution lowering；这是
   baseline enablement，必须与ALPS贡献分开报告。
2. E：在conv kernel形成结果时直接产生`(batch, frames, hidden)` consumer layout，合并
   现有transpose/position input formation；这与consumer-driven in-situ formation主线
   无缝衔接。
3. P：只有在C/E之后仍存在不可消除、LWP证明位于critical path的conv tile supply，才
   使用VTCM async prefetch；当前阶段对attention K/V追加prefetch不会触及主瓶颈。
4. R：sysMon证明当前平均AXI只有约30 MB/s，runtime应拒绝无差别traffic prefetch；待
   HVX conv提高带宽需求后再由PMU admission重新决策。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/hotspot_lwp_audio
```

### 12.10 Whisper-tiny的完整shape局部LWP闭环

Whisper使用完整80x3000 mel输入、4层encoder、4层decoder、384 hidden和32-token target，
分别编译encoder frontend、一个代表性encoder layer、一个decoder layer和vocabulary head。
局部载体保持完整发布shape和真实算子，不是Debug模型；正式延迟仍取非插桩完整wrapper的
70,067.02 ms。

| Stage | 局部LWP Perf | Exclusive-cycle share |
|---|---:|---:|
| Encoder frontend | 45,146.27 ms | **88.37%** |
| One encoder layer | 4,346.28 ms | 8.38% |
| One decoder layer | 494.13 ms | 0.95% |
| Vocabulary head | 1,194.63 ms | 2.30% |

跨stage operation aggregate中，`extf, extf, mulf, addf`卷积/FP32累加链占
**92.43%**，HMX链仅2.06%。按4个encoder和4个decoder外推约65.7 s，与完整模型
70.07 s处于同一量级。因此此前“HVX event较高，可能是attention/layout feeding”的假说
被测量结果否定：当前第一瓶颈仍是frontend convolution的基础codegen，而不是K/V搬运。
优化顺序与共享Speech encoder一致：C先建立HVX conv lowering；E让conv直接产生后续token
layout；只有无法消除且位于critical path的tile supply进入P，最后由R控制流量。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/hotspot_lwp_whisper
```

### 12.11 DeiT轻量全图LWP恢复与瓶颈更正

旧的DeiT全图LWP把P1--P5m重分析链、movement ledger和插桩同时启用，host RSS达到约
14.67 GiB后OOM。改用与正式item7语义匹配的轻量协议——只启用item7、关闭movement
ledger、LWP loop depth 0、单线程编译——完整单体图成功编译和运行。编译耗时
1,264.58 s，插桩Perf为5,006.24 ms，输出finite、max abs diff 0.0035且top-1 match。
因此旧失败是profiling配置的组合内存峰值，不是DeiT本身不能做全图LWP。

全图exclusive-cycle aggregate为：FP32扩展/乘加链 **57.64%**，HMX chain
**26.21%**，root/unattributed 8.43%，softmax/reduction 4.23%，其余3.48%。最大单region
就是`extf, extf, mulf, addf`，占47.76%。这直接更正了“先找transpose/drain”的先验：
DeiT当前首要路径是C（消除不必要的FP16↔FP32扩展并形成真正HVX累加kernel）；随后才对
26.21%的HMX输出/表示应用E与P。sysMon总869.15 MB仍说明存在物理流量，但不能跨过
57.64%的计算/codegen热点直接声称prefetch是第一优化项。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/deit-small/item7-only
```

### 12.12 BEiT-base轻量全图LWP闭环

BEiT-base采用与DeiT相同的item7-only、movement-ledger off、LWP depth 0、单线程协议，
完整单体模型成功编译并在手机运行。插桩Perf为13,910.97 ms，输出finite、max abs diff
0.0049且top-1 match；与非插桩sysMon正式值13,824.45 ms接近，但插桩值仍只用于热点排名。

exclusive-cycle aggregate为：HMX chain **47.57%**，FP32扩展/乘加链 **34.90%**，
softmax/reduction **11.34%**，其余/未归因6.19%。最大单region是
`extf, extf, mulf, addf`，占34.79%。结合sysMon 3.54 GB AXI和此前P5n的6.46%收益，
BEiT是明确的混合路径：E/P可以处理接近一半的HMX表示与drain，C处理35%的累加链和11%的
softmax/reduction。它支持统一的E→P→R故事，但也说明仅prefetch不可能覆盖全部critical path。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/beit-base/item7-only
```

### 12.13 Swin Transformer全图LWP闭环

Swin完整模型轻量全图LWP成功，编译949.95 s，插桩Perf 50,973.58 ms，max logit diff
0.0015且top-1 match。exclusive-cycle aggregate非常集中：`extf, extf, mulf, addf`
累加链占 **94.00%**，HMX chain 2.22%，softmax/reduction 1.87%，其余1.91%。前十个
region均为同一累加链，每个约2.63--2.74%，说明问题遍布window blocks，而不是单个异常
transpose。

它与sysMon中约50秒无HVX/HMX activity形成闭环：Swin当前首要瓶颈是C类基础codegen，
不是外存带宽，也不是可以直接用prefetch隐藏的layout copy。正确顺序是先把这些规则
window/MLP累加链lower到真正HVX kernel，再观察随之暴露的window partition/merge物理布局；
后者才进入consumer-driven E，只有仍不可消除的tile supply进入P。当前对Swin增加无差别
prefetch预计只会增加traffic，与此前P5n仅1.21%的弱收益一致。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/swin-transformer/item7-only
```

### 12.14 SegFormer与15模型统一决策闭环

SegFormer MiT-B0轻量全图LWP成功，编译94.94 s，插桩Perf 8,954.01 ms，输出finite、
max abs diff 0.0013且top-1 match。exclusive-cycle aggregate为：FP32扩展/乘加链
**85.68%**，其他elementwise 5.92%，softmax/reduction 4.26%，HMX chain仅2.63%，
未归因1.49%。结合sysMon总AXI仅0.32 GB、绝大多数时间无HVX/HMX活动，当前瓶颈明确是
C类多尺度conv/MLP codegen；layout formation E应在C修通后处理新暴露的多尺度边界，
不能把当前低traffic路径包装成prefetch机会。

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/segformer-mit-b0/item7-only
```

15个完整模型的LWP + sysMon现已形成统一决策，而不是15条模型特例：

| 路径 | 模型 | 测量依据 | 进入论文主线的方式 |
|---|---|---|---|
| E→P→R（representation/movement first） | GPT-2、SD/CLIP、Qwen、TinyLlama、SmolLM、DINO、BEiT | vocabulary/HMX outer/output、较高AXI或已验证P5n critical overlap | 主论文核心：consumer-driven direct/persistent formation；残余搬运异步化；PMU admission |
| C+E/P（mixed） | DeiT | 累加链57.64%，HMX 26.21%，AXI 0.87 GB | C收益单列；E/P仍可做第二大路径，不把总收益都归prefetch |
| C→E（baseline enablement first） | Swin、SegFormer、Whisper、HuBERT、Wav2Vec2、UniSpeech、UniSpeech-SAT | 累加/conv链85.68--99%或frontend 88--91%，HMX很低 | 先修共享HVX conv/accumulation contract；随后让producer直接形成consumer layout |

论文故事仍保持三项贡献不变：V-DAE负责把consumer representation contract传播到producer；
prefetch + in-situ transformation只处理“不能直接消除、LWP证明位于critical path、可与
计算重叠”的residual supply；PMU runtime负责traffic admission和退避。C类codegen修复是
让模型真正进入目标HVX/HMX路径的baseline工程，不应伪装成ALPS prefetch贡献。最终
ablation必须分别报告C、E、P、R：这样即使P在某些模型上正确拒绝工作，也体现机制的
选择性，而不是把无效prefetch当成普适贡献。

### 12.15 15个完整模型的推理精度审计（2026-08-29）

必须区分以下三种精度，避免把LWP中的`arith.extf`误读成“模型使用FP32推理”：

1. **模型存储/接口精度**：浮点权重、主激活和浮点输入的dtype；
2. **kernel输入/输出精度**：HVX/HMX实际消费和产生的数据类型；
3. **累加精度**：matmul、convolution、LayerNorm、softmax等为避免溢出而临时使用的
   accumulator dtype。

本次corpus的15个模型在**模型层面**全部使用FP16权重、浮点输入和主激活，也没有运行时
quantize/dequantize。`.half()`与`torch_dtype=torch.float16`在模型导出前完成，编译产物
中的权重常量已经是FP16；`enableConversionToFp16=0`表示不在Hexagon pipeline中插入额外
全图转换pass，而不是关闭FP16。

但模型dtype不能替代kernel dtype审计。当前HMX matmul使用FP16 operands；一部分HVX
convolution/elementwise lowering却生成`extf(f16→f32), extf, mulf(f32), addf(f32)`。
这里不仅是FP32 accumulator，乘法本身也是FP32 arithmetic。因而最准确的描述是
**FP16 model/storage with mixed kernel arithmetic**，不能笼统声称15个模型都是纯FP16
compute。这是当前Hexagon-MLIR codegen限制，不是模型额外量化开销。

| Domain | 完整模型 | 浮点模型/输入证据 | 累加策略 | 当前corpus是否已是FP16 |
|---|---|---|---|---|
| Language | GPT-2 | 统一driver显式`--dtype fp16`；checkpoint以`torch.float16`加载 | HMX matmul为f16 operands；stable norm/softmax允许FP32 | **是** |
| Language | SD/CLIP | 统一driver显式`--dtype fp16`；text model `.to(float16)` | HMX为f16 operands；GELU/norm存在FP32 arithmetic | **是** |
| Language | Qwen2.5-0.5B | checkpoint `torch_dtype=float16`，mask/RoPE cache为FP16 | HMX f16 matmul；部分elementwise扩展至FP32 | **是** |
| Language | TinyLlama-1.1B | 与Qwen layered runner相同的FP16加载和输入协议 | HMX f16 matmul；部分elementwise扩展至FP32 | **是** |
| Language | SmolLM2-1.7B | 与Qwen layered runner相同的FP16加载和输入协议 | HMX f16 matmul；部分elementwise扩展至FP32 | **是** |
| Vision | DINOv2-small | model `.half()`，pixels与固定position embedding为FP16 | HMX为f16；patch/部分HVX算术扩展至FP32 | **是** |
| Vision | DeiT-small | model `.half()`，pixels与position embedding为FP16 | FP32 arithmetic链57.64%，HMX f16链26.21% | **是** |
| Vision | BEiT-base | model `.half()`，pixels与lifted bias为FP16 | FP32 arithmetic链34.90%，HMX f16链47.57% | **是** |
| Vision | Swin Transformer | model `.half()`，pixels为FP16 | FP32 arithmetic链94.00%；HMX仅2.22% | **是** |
| Vision | SegFormer MiT-B0 | model `.half()`，pixels为FP16 | FP32 arithmetic链85.68%；HMX仅2.63% | **是** |
| Speech | Whisper-tiny | model `.half()`，mel features为FP16；token ID为整数 | frontend conv主要为FP32 arithmetic链 | **是** |
| Speech | HuBERT-base | model `.half()`，audio samples为FP16 | GroupNorm及conv当前扩展到FP32 arithmetic再写回FP16 | **是** |
| Speech | Wav2Vec2-base | 同一full-audio FP16 runner | conv当前扩展到FP32 arithmetic再写回FP16 | **是** |
| Speech | UniSpeech-base | 同一full-audio FP16 runner | conv当前扩展到FP32 arithmetic再写回FP16 | **是** |
| Speech | UniSpeech-SAT-base | 同一full-audio FP16 runner | conv当前扩展到FP32 arithmetic再写回FP16 | **是** |

需要特别澄清两个容易造成误解的历史入口：

- `benchmark_models/run_gpt2lmheadmodel.py`的monolithic兼容路径仍保留FP32模型选项；本次
  15模型corpus没有使用它，而是使用`probe_gpt2_layered_export.py --dtype fp16`。
- 旧Stable Diffusion/UNet runner中可能存在FP32 timestep；本次列表中的`SD/CLIP`只指
  FP16 CLIP text encoder，不是完整UNet。

#### 是否需要重新跑15个模型的LWP与sysMon

**当前不需要因模型dtype重新跑。** 现有LWP和sysMon已经是在同一FP16模型/输入协议下
产生；仅重复执行不会把FP32 HVX arithmetic自动变成FP16，只会重复数十小时编译与设备
运行。日志字段从`uniform_fp16=1`改成更严格的
`fp16_model_storage=1 kernel_precision=mixed_f16_hmx_f32_hvx`，该改名不改变任何编译
选项或二进制，operator级判断仍以LWP IR为准。

只有以下情况才需要选择性重跑，而不是全量重跑：

1. 某个runner从`.half()`/`torch_dtype=float16`改为不同dtype；
2. 修改HVX lowering，形成helper-free的FP16 operand/vector路径（允许V73高效FP32
   vector accumulator），从而改变实际object；
3. 引入INT8/INT4或显式quantize/dequantize；
4. 修改会改变实际kernel dtype的HexKL/HVX lowering。

当前论文配置应表述为 **FP16 model/storage with mixed FP16-HMX and FP32-HVX kernel
arithmetic**。在实现helper-free的FP16 HVX conv/elementwise lowering之后，必须重新运行
受影响的Vision和Speech模型的LWP与sysMon，并对baseline与ALPS使用完全相同的kernel precision。
Language模型只有在其实际lowering发生变化时选择性重跑。最终正式论文表格则应在代码
冻结后对15模型统一重跑一次，避免把精度/codegen变化误算成ALPS加速。

### 12.16 V73 FP16-HVX lowering实验与12.8分群更新（2026-08-29）

为验证12.15中的精度假说，新增了默认关闭、可独立消融的
`enableAlpsFP16HVXArithmetic`。它不改变模型dtype、item7、layout、prefetch或HexKL配置，
只复用上游`conversion-to-fp16`，把FP16模型中满足现有规则的FP32 convolution/elementwise
island在HVX tiling/vectorization之前改为FP16。统一runner新增
`--alps-fp16-hvx`，固定比较`hmlir-hvx-hexkl-on`与
`alps-fp16-hvx-arithmetic`；两组均为完整模型、HVX vector、HexKL on。该开关保留为负向
消融，**默认关闭，不能并入最终ALPS组合**。

SegFormer入口IR验证显示，旧pass可以把`f32`类型引用从1,105处降到118处，并将首个
`conv(f16,f16)->f32 -> truncf`改为`conv(f16,f16)->f16`。这证明转换真实发生，而不是
开关空转。然而完整模型matched结果否定了“更少FP32 IR自然更快”的假说：

| 测量 | HexKL-on control | FP16-HVX arithmetic | 变化 |
|---|---:|---:|---:|
| 非LWP wrapper latency | **9,262.15 ms** | 11,209.29 ms | **慢21.03%** |
| LWP wrapper latency | **9,301.48 ms** | 11,300.98 ms | 慢21.50% |
| LWP root exclusive cycles | **13.724 B** | 16.668 B | 增加21.45% |
| main object instruction count | **107,523** | 150,896 | 增加40.34% |
| HVX-like instruction占比 | **4.54%** | 2.52% | 下降44.42% |
| `__extendhfsf2` relocation sites | **7,636** | 13,294 | 增加74.09% |
| `__truncsfhf2` relocation sites | **5,111** | 10,530 | 增加106.03% |
| sysMon processor cycles | **13.928 B** | 16.809 B | 增加20.68% |
| sysMon AXI total | 319.24 MB | **292.24 MB** | 减少8.46% |
| correctness | finite；max diff 0.0016；top-1 match | finite；max diff 0.0017；top-1 match | 均通过 |

这里最重要的结论是：**减少storage/AXI bytes不等于缩短critical path。** FP16组确实少
约27 MB AXI，却由于scalar half helper和更差的vector coverage增加约2.88 B processor
cycles。LWP中原control的主要热区是`extf, extf, mulf, addf`，但V73反汇编表明其中已有
有效`vmpy/vadd/vmem`；它表示FP16 input/weight在HVX中做widening multiply并使用FP32
accumulator，并不等同于整个卷积退化为FP32模型。强行把accumulator和中间elementwise
都变为FP16，反而使LLVM/Hexagon backend在未完整vectorize的loop/tail上反复调用
half-to-float/float-to-half helper。

因此“真正FP16 HVX lowering”在V73上的工程定义必须修正为：

1. convolution保留FP16 input/weight和连续vector load，使用V73高效的widening multiply
   与FP32 vector accumulator，只在最终边界trunc一次；不能把“FP16 compute”等同于
   “FP16 accumulator”；
2. elementwise只有在证明完整vector coverage、无scalar tail/helper且目标指令确实为
   HVX half-vector op时才允许FP16算术；否则保持当前mixed arithmetic；
3. 编译期必须加入object审计门：若`__extendhfsf2/__truncsfhf2`增加、HVX-like比例下降，
   自动拒绝该precision policy；
4. FP16 storage/layout仍可作为E/P减少搬动的基础，但precision lowering本身归C，不能
   计入ALPS prefetch贡献。

这也更新了12.8的分群，但没有改变统一故事线：SegFormer以及共享Speech frontend仍属于
**C→E→P→R**。C的任务不再是“盲目消除所有extf”，而是建立helper-free的FP16-load/
FP32-vector-accumulate kernel；随后E让该kernel直接形成consumer layout，P只覆盖不能
消除且位于critical path的tile supply，R依据PMU拒绝额外traffic。对其余14个模型不做
无意义的宽泛FP16重跑；只有新的helper-free selective lowering改变了实际object后，才
按12.8分群选择受影响的Vision/Speech模型重新跑LWP和sysMon。代码冻结后仍统一重跑15个
模型。

`scripts/script_release/internal/audit_hexagon_codegen.sh`现已把`extendhfsf2_relocations`、
`truncsfhf2_relocations`和二者之和写入每个case的`codegen.csv`。这使后续selective
lowering不再只凭tensor/vector IR获准：即使正确性通过，只要object层half helper回归，
该case就必须保留为C类负向实验，不能进入ALPS组合。

原始证据：

```text
nano:/home/huzq85/2-working/working_set/alps_fp16_hvx_segformer_lwp_20260829
nano:/home/huzq85/2-working/working_set/alps_fp16_hvx_segformer_sysmon_20260829
```

### 12.17 C阶段：native-width HVX widening convolution（2026-08-29）

按照12.16修正后的定义，实现了默认关闭、可独立消融的
`enableAlpsHVXWideningConv`。该pass不把accumulator降成FP16，而是在
post-bufferization阶段只匹配`f16 input/filter -> f32 output`的静态
`linalg.conv_2d_nchw_fchw`和`linalg.conv_1d_ncw_fcw`：沿独立输出位置形成64-lane
FP16 operand vector，保持每个lane原有reduction顺序，在FP32 vector pair中累加并
masked-store。64个FP16恰好占V73的一条128 B HVX vector；LLVM因而能够选择
`Vdd.sf = vmpy(Vu.hf,Vv.hf)`一类native widening指令，而不是把partial half vector
反复落到scalar conversion helper。该策略只消除被重写卷积内部的动态helper路径；
object中其余norm、softmax和未匹配算子的helper仍然存在，因此更准确的名称是
**selective helper-free widening convolution**，不能声称整个模型helper-free。

完整SegFormer的matched结果如下；两组均为FP16模型、HVX vector、HexKL on，且item7、
layout、prefetch、DMA和PMU admission全部关闭：

| 测量 | HexKL-on control | ALPS C widening conv | 变化 |
|---|---:|---:|---:|
| 正式latency | 9,313.73 ms | **6,352.36 ms** | **1.4662x；下降31.80%** |
| LWP Perf | 9,310.15 ms | **6,482.57 ms** | **1.4362x** |
| LWP root exclusive cycles | 13.723 B | **9.540 B** | 下降30.48% |
| sysMon processor cycles | 13,917,700,995 | **9,642,931,398** | 下降30.71% |
| sysMon committed packets | 4,970,712,872 | **3,190,233,004** | 下降35.82% |
| sysMon HVX packet event | 5,569,789 | **234,548,436** | 增加42.11x |
| sysMon AXI total | 318,152,704 B | 317,474,688 B | 基本不变（-0.21%） |
| main object instructions | 104,983 | 118,014 | 增加12.41% |
| HVX-like instructions | 4,881 (4.65%) | 7,108 (6.02%) | 增加45.62% |
| vector load/store mentions | 5,926 | 9,357 | 增加57.89% |
| half-helper relocations | 12,745 | 12,651 | 减少94个静态site |
| correctness | finite；max diff 0.0016；top-1 match | finite；max diff 0.0015；top-1 match | 均通过 |

LWP进一步给出直接因果证据：control的纯`extf, extf, mulf, addf`类占root
47.05%，而被重写的10个卷积在treatment中成为显式
`maskedload/gather/extf/mulf/addf/maskedstore`类并只占12.37%；root周期与正式latency
同方向下降。AXI几乎不变，排除了“少读DRAM”作为本轮主要解释；收益来自把原关键路径
上的scalar/mixed卷积循环变为native-width HVX widening kernel。虽然尚未达到1.8x，
但C阶段已经通过跨层级因果门，不再对SegFormer反复调参。

同一contract随后扩展到完整Whisper的两个`conv_1d_ncw_fcw`。修复Whisper runner遗漏的
CLI参数传播后，真正启用的matched结果为113,415.80 -> **108,246.36 ms**，即
**1.0478x（下降4.56%）**；finite、max diff 0.0044、last-token top-1 match。main object
的HVX-like count由20,398增至20,678，vector load/store由24,261增至24,835，half-helper
relocation由1,306降至1,300。收益小于SegFormer并不否定C：Whisper总时间还包含大量
decoder/attention计算，而本pass只改变两个frontend conv；它证明相同lowering能跨
Vision/Speech正确生效。第一次Whisper候选因runner漏传开关而显示
`alps_hvx_widening_conv=0`、object与control完全相同，该空转数据已明确作废，不能用于
论文。

由此，既定顺序继续为：**C已完成阶段验证 -> E consumer-driven direct layout -> P仅对
无法消除且位于critical path的residual movement做async DMA/VTCM -> R用PMU admission
控制traffic**。C是baseline enablement，不冒充prefetch贡献；E/P/R仍承担ALPS论文的
representation-aware prefetch主线。

原始证据：

```text
nano:/home/huzq85/2-working/working_set/alps_c64_segformer_full_20260829
nano:/home/huzq85/2-working/working_set/alps_c64_segformer_lwp_20260829
/tmp/alps_c64_segformer_sysmon_control_20260829
/tmp/alps_c64_segformer_sysmon_candidate_20260829
nano:/home/huzq85/2-working/working_set/alps_c_whisper_full_20260829
```

### 12.18 E阶段：C之上的consumer-driven direct layout（2026-08-29）

为避免把C的codegen收益误算成E，统一runner新增`--alps-c-e`，严格比较
`ALPS C widening conv`与`ALPS C + consumer-driven layout`。两组都开启相同的FP16
model、HVX vector、HexKL和C lowering；treatment只额外开启现有P2e
consumer-driven direct formation。item7、prefetch、DMA、VTCM staging、continuity扩展
和PMU admission均关闭。

| 模型 | C control | C + E | E独立增量 | 相对原HexKL-on累计 | P2e direct / demands | 静态消除materialization |
|---|---:|---:|---:|---:|---:|---:|
| SegFormer MiT-B0 | 6,352.36 ms | **5,489.21 ms** | **1.1572x** | **1.6967x** | 24 / 116 | 1,655,808 B |
| Whisper-tiny | 108,246.36 ms | **71,512.90 ms** | **1.5137x** | **1.5859x** | 36 / 114 | 18,923,520 B |

两模型的输出正确性都保持不变：SegFormer finite、max diff 0.0015、top-1 match；
Whisper finite、max diff 0.0044、last-token top-1 match。BackendConfig同时验证C和P2e均为1，
排除了开关空转。

结果支持统一因果链而非简单叠加：C先把producer的卷积关键路径变成native-width HVX
widening kernel；E再从terminal consumer的layout contract倒推producer/elementwise链的
最终表示，删除中间transpose/materialization。Whisper的E增量远大于C，说明Speech此前
的主要限制并非只有conv arithmetic，还包括frontend输出到后续sequence/attention表示的
大规模layout round trip。SegFormer的E增量较小但仍稳定，使C+E相对原HexKL-on达到
1.70x，已经逼近1.8x目标。

这也是ALPS论文第2贡献的直接正例：**prefetch不是无条件提前读canonical buffer，而是
先由consumer contract决定最终representation；可消除的搬动由in-situ/direct formation
删除，只有无法消除且处于critical path的residual tile supply才进入P。** 因此下一阶段
必须做严格的`C+E control`对`C+E+P treatment`，不能复用会同时打开P2g/P5h/P5i等多个
机制的旧累计开关来声称P收益。

原始证据：

```text
nano:/home/huzq85/2-working/working_set/alps_ce_segformer_full_20260829
nano:/home/huzq85/2-working/working_set/alps_ce_whisper_full_20260829
```

### 12.19 P阶段：C+E之后的residual HMX async drain（2026-08-29）

统一runner新增默认关闭的`--alps-c-e-p`。为隔离P，control与treatment都开启相同的
HVX vector、HexKL、C、P2e consumer-driven layout、P5j HMX F16 direct epilogue和P5k
direct output；两组都运行P5m静态准入，只有treatment开启P5n双VTCM slot + UserDMA
异步result drain。旧的P1--P5累计组合没有进入本实验。DINO的`14x14/stride-14`小宽度
非重叠patchify同时从C中拒绝：它只有16个输出列，不属于64-lane sliding convolution，
应留给E的patch/direct formation；该门不影响SegFormer的7/4、3/2重叠卷积或Speech
1D卷积。

完整DINOv2-small结果：

| 测量 | C+E direct-output control | C+E+P async drain | P增量 |
|---|---:|---:|---:|
| latency | **5,768.58 ms** | 5,828.72 ms | **0.9897x；慢1.04%** |
| P5m admitted sites | 72 / 72 | 72 / 72 | 相同 |
| P5m predicted overlap | 21,086,208 B | 21,086,208 B | 相同 |
| runtime DMA issued/completed | 0 / 0 | 10,368 / 10,368 | 真实执行 |
| runtime DMA bytes/fallback | 0 B / 0 | 21,233,664 B / 0 | 真实执行 |
| correctness | finite；max diff 0.0049；allclose/top-1 | finite；max diff 0.0051；allclose/top-1 | 均通过 |

因此P的机制成立，但仅凭静态descriptor合法、tile数量和producer内的后续HMX调用距离
仍会误准入：这些drain没有足够关键路径占比，DMA launch/wait与新增AXI traffic略高于被
隐藏的同步copy。本结果不是“prefetch没有执行”，而是一个有实际DMA证据的负向消融。
它直接要求R把PMU observation纳入admission：只有观测到同步drain stall/packet占比和
可用memory slack同时满足阈值的site class才启用P5n；否则保持C+E synchronous direct
output。R必须以这一组作为reject regression test，而不能通过重复测量或调distance把
负值包装成正收益。

第一轮配置因DINO runner漏传C且未启用P5j，得到`C=0`、`admitted=0`、DMA=0；该轮明确
作废，不计入性能数据。有效原始证据：

```text
nano:/home/huzq85/2-working/working_set/alps_cep_dinov2_valid2_20260829
```

### 12.20 R阶段：把PMU/poll traffic admission接到P5n（2026-08-29）

此前P4A虽然会尝试读取V73 HAP user PMU，但只控制旧P3b weight DMA，对P5n HMX drain
完全空转。本阶段把R接入P5n的真实wait-slot和start路径，并增加严格
`--alps-c-e-p-r`：control和treatment均开启同一C+E+P，只有treatment开启R。

R每64个完成descriptor形成一个固定window，读取UDMA active、DMPoll、coherent-read
stall和VTCM-write stall；若Unsigned PD拒绝HAP user PMU，则明确报告unavailable，并使用
同一wait-slot的真实软件poll次数作为保守fallback。零poll表示DMA已在demand前完成，必须
hold；只有平均DMPoll/poll达到每descriptor 4次才throttle。throttle后不改变layout或
legality，而是调用HexKL原生`hexkl_micro_hmx_copy_f16_to_submatrix`恢复同步direct drain，
不能用另一套通用copy冒充matched fallback。

最终完整DINOv2-small结果：

| 测量 | C+E+P | C+E+P+R |
|---|---:|---:|
| latency（单次formal sample） | 5,825.25 ms | **5,487.22 ms（1.0616x）** |
| DMA issued/completed | 10,368 / 10,368 | 10,368 / 10,368 |
| issued bytes | 21,233,664 B | 21,233,664 B |
| R windows / hold / throttle | NA | 162 / 162 / 0 |
| suppressed / poll retries | NA | 0 / 0 |
| HAP PMU status / reads | NA | unavailable(0) / 0 |
| correctness | finite；max diff 0.0049；allclose/top-1 | 相同 |

因为本次所有window都hold，R没有改变执行的DMA集合，5.825→5.487 s的单样本差值不能
归因成admission收益，只能作为正确性/无回归观测。可以确认的因果事实是：R不再错误地
把零等待的理想overlap关闭，并能在未来出现持续late-arrival时恢复到完全相同的HexKL
同步drain。当前手机Unsigned PD不允许进程内HAP PMU，因此“PMU counter真正触发
throttle并改善latency”仍是最终ablation的待补证据：优先使用可授权PMU的PD；否则将
sysMon的模型/site-class先验固化为下一次invocation的admission，同时把进程内软件poll
作为安全反馈。不能把`pmu_status=0`描述成已完成PMU性能验证。

原始证据：

```text
nano:/home/huzq85/2-working/working_set/alps_cepr_dinov2_final_20260829
```

### 12.21 当前阶段结论、12.8完成度与冻结后的完整模型计划（2026-08-29）

截至commit `3b90cd4`，必须把“12.8的profiling/路线分类闭环”和“15个模型上的实现/
性能闭环”分开表述。前者已经完成，后者尚未完成，不能因为C、E、P、R都有代码和代表性
模型结果，就声称12.8中的每条路线已经在全部模型上兑现。

#### 12.21.1 当前可以得出的因果结论

1. **C（baseline enablement）已经跨Vision/Speech完成阶段验证，但不是ALPS prefetch
   收益。** SegFormer的HexKL-on control到C为9,313.73 -> 6,352.36 ms（1.4662x）；
   Whisper为113,415.80 -> 108,246.36 ms（1.0478x）。SegFormer的AXI基本不变，而HVX
   packet显著增加，收益主要来自把scalar/mixed卷积关键路径改成native-width HVX
   widening kernel。
2. **E（consumer-driven direct layout）是目前最清楚的representation/movement收益。**
   SegFormer从C到C+E为6,352.36 -> 5,489.21 ms（E独立1.1572x，相对原HexKL-on
   累计1.6967x）；Whisper为108,246.36 -> 71,512.90 ms（E独立1.5137x，累计
   1.5859x）。这支持“先由consumer contract决定最终表示并直接形成，再对残余供给做
   prefetch”的论文故事。
3. **P（residual async DMA/VTCM）机制真实执行，但DINO上的当前静态准入没有性能
   收益。** 72/72 site获准、10,368个DMA完成且搬运21,233,664 B，但5,768.58 ->
   5,828.72 ms（0.9897x）。这说明descriptor合法和存在可重叠距离不足以证明搬运位于
   critical path。
4. **R已经接入P5n真实路径，但真正的PMU自适应性能闭环尚未完成。** 当前Unsigned PD
   不允许进程内HAP user PMU；DINO的162个window全部hold，没有改变DMA集合。因此
   5,825.25 -> 5,487.22 ms的单样本差异不能归因于R，只能作为正确性/无回归观测。
5. 当前统一故事仍成立：**C修复硬件执行前提；E删除可消除的representation movement；
   P只提前供应无法消除且位于critical path的残余tile；R拒绝无收益或造成traffic压力的
   P请求。** 不能把C混写成prefetch，也不能把静态“saved bytes”直接写成物理DDR下降。

#### 12.21.2 12.8路线完成度

| 工作 | 状态 | 边界 |
|---|---|---|
| 15个完整模型非插桩sysMon | 已完成 | 已覆盖Language/Vision/Speech |
| 15模型LWP热点定位 | 已完成 | 单体或分层/局部完整shape方式覆盖 |
| 12.8瓶颈分群 | 已完成 | 收敛为E→P→R与C→E→P→R两类共享路线 |
| C/E/P/R独立机制和开关 | 已完成 | 可独立启停和审计 |
| C代表性完整模型验证 | 已完成 | SegFormer、Whisper；尚未覆盖全部C类模型 |
| E代表性完整模型验证 | 已完成 | SegFormer、Whisper，并有DINO的HMX direct路径 |
| P真实DMA执行验证 | 已完成 | DINO有真实DMA证据，但当前为负向性能结果 |
| R接入P5n实际执行路径 | 已完成 | hold/throttle/suppression可审计 |
| R由真实PMU改变决策并改善latency | **未完成** | 当前进程内PMU不可用；DINO未发生throttle |
| 当前冻结代码下15模型最终latency | **未完成** | 需要统一重跑完整非Debug模型 |
| 5个预注册正例模型的完整消融 | **已完成** | 15个A1--A3 case全部PASS；A0/A4复用冻结主表 |

因此，对“12.8中的路线是否已经都完成”的准确回答是：**profiling、分群和共享机制原型
已经完成；全部15模型上的路线部署、最终性能确认和R的真实反馈闭环尚未完成。**

#### 12.21.3 冻结后的完整模型主实验

下一阶段不再围绕单模型反复调参，而是冻结代码、精度、设备设置和统计方法，串行运行15个
完整非Debug模型。每个模型依次记录：

1. Prefetch-Kernel-HX（PK HVX）；
2. APT-GET-HX（APT HVX）；
3. Hexagon-MLIR HVX（HexKL Off）；
4. Hexagon-MLIR HVX（HexKL On，ALPS matched control）；
5. ALPS C+E+P+R最终组合。

所有case使用相同的FP16 model/storage与mixed FP16-HMX/FP32-HVX kernel precision、
相同输入shape、模型层数、单HVX执行线程、设备performance设置和统计窗口。ALPS开关可以
统一打开，但每个机制必须依据合法性/admission选择性生效；未匹配时记录`not admitted`，
不能把零rewrite/零DMA写成该机制的有效实验。模型严格串行；不设置超时；遇到exit 13
立即调查而不盲目重跑。每个模型完成后把编译产物和原始日志**移动**到
`nano:/home/huzq85/2-working/working_set`，确认远端完整后删除本地`/tmp`大文件。

正式表格至少记录latency、相对ALPS的加速比、correctness、实际C rewrite、E formation、
P DMA issued/completed/bytes、R window/hold/throttle/suppressed、进程内PMU状态和sysMon摘要。
主实验先回答“冻结后的最终系统在多少完整模型上有效”，随后才进行大规模消融。

本轮必须在`experimental_data.md`中创建一个**新的冻结版本表格**，不得覆盖、合并或静默
改写此前的历史表。runner同时生成独立的`frozen_full_matrix.csv`、
`frozen_full_matrix_long.csv`和`frozen_full_matrix.md`：wide表用于论文主表，long表保留每个
模型/配置的原始指标，Markdown表在全部15模型完成后原样追加到实验文档。

#### 12.21.4 每个模型的数据物化量对比

可以且应当记录“数据物化量减少”，但需要同时报告三个不同层级，避免把预测量冒充硬件
流量：

| 层级 | 指标 | 计算方式 | 含义与限制 |
|---|---|---|---|
| Compiler logical | `static_materialization_bytes` | 在相同post-bufferization观察点分别运行P1 ledger；`baseline - ALPS` | 静态shape下显式copy/transpose/physical transform的读写物化估计 |
| Transformation causal | `eliminated_materialization_bytes` | 汇总P2e/P5h/P5i等真正rewrite的eliminated/residual字段 | 可归因到具体ALPS rewrite；不能重复累计同一value/version |
| Runtime added movement | DMA issued/suppressed bytes | P5n runtime ledger | P为提前供给额外发出的真实DMA量；它可能抵消E节省的流量 |
| Hardware external traffic | sysMon AXI read/write/total | matched kernel window硬件PMU | L2 miss导致的DDR/AXI请求；既包含demand也包含prefetch，不等于逻辑物化量 |

每个模型新增以下派生量：

```text
logical_materialization_reduction_bytes =
    baseline_static_materialization_bytes - alps_static_materialization_bytes

logical_materialization_reduction_percent =
    reduction / baseline_static_materialization_bytes * 100

external_traffic_reduction_bytes =
    baseline_sysmon_axi_total_bytes - alps_sysmon_axi_total_bytes
```

分层Language runner必须按每个stage的真实调用次数加权后再求和；不能只把一个layer的
静态字节当成完整模型，也不能在多处ledger对同一materialization重复计数。动态shape无法
静态确定的site记录`NA`并保留site count，后续如有必要加入runtime byte counter。论文中
应并排报告逻辑物化减少和物理AXI变化：前者回答“编译器删除了什么”，后者回答“设备最终
少访问了多少外部内存”。若两者不一致，差值本身就是cache/reuse/critical-path分析证据。

#### 12.21.5 进程内PMU不可用时的sysMon方案

sysMon可以采集与PMU相同或直接相关的量，因为SDK文档明确说明sysMon profiler service
本身采样硬件PMU。当前限制不是“手机没有PMU”，而是Unsigned PD中的ALPS进程不能直接
编程/读取HAP user PMU；sysMon通过系统侧profiler服务仍能取得计数器。当前Default mode
每1 ms固定采集八个事件，并已能得到：

- processor cycles、committed packets与cycles/packet；
- HVX packet和HMX active事件；
- L2 miss引起的AXI cached read/write bytes与带宽；
- L2fetch miss/traffic；
- 每1 ms AXI burst的p50/p90/p99/max，以及HVX-only、HMX-only、两者同时活跃窗口；
- raw CSV中的core/bus clock vote、DCVS、thermal throttle、BLC transaction/latency、
  BWMON bytes和packet count。

因此主实验可以用sysMon回答：是否compute-bound、是否有持续或突发的DDR压力、prefetch
是否增加L2/AXI traffic、HVX/HMX覆盖是否改变，以及设备频率/热状态是否匹配。后续应在
现有`summarize_sysmon_profile.py`中统一派生pCPP、effective utilization、BLC latency和
频率/thermal摘要，写入每个模型结果。

sysMon还支持User mode、`/data/pmu_events.txt`中的自定义4/8个PMU event以及STID/marker
过滤，理论上可选择DMPoll、coherent-read stall、VTCM-write stall等V73事件。但User
mode会占用PMU counter：四counter模式可能改变DCVS决策，八counter模式会关闭DCVS。
所以它只适合单独的diagnostic/profile run，不能与Default-mode formal latency直接混比；
具体event ID也必须由V73 event定义和设备实际输出双重验证，不能仅凭名字推断。

sysMon不能完全替代同一次invocation内的R：它是进程外、约1 ms粒度、结束后解析的采样，
无法在某个DMA descriptor deadline之前把决策返回P5n。当前可行的R闭环应分为两层：

1. **同一次invocation内**：保留P5n软件poll/late-arrival反馈，负责安全fallback和粗粒度
   throttle；
2. **跨invocation profile-guided admission**：先用sysMon profile run生成模型/site-class
   policy，再在下一次formal run中关闭高pCPP、高AXI burst、低overlap收益的P请求；formal
   latency本身不同时运行sysMon，或严格分开报告profiling overhead。

论文中应将第二层准确称为`sysMon/PMU-guided cross-invocation traffic admission`，而不是
声称descriptor级在线PMU控制。若后续获得可授权PMU的PD，再补充真正的within-invocation
R消融；在此之前，sysMon提供的是可信的硬件反馈和下一次执行策略输入，而非即时控制面。

#### 12.21.6 主实验后的消融顺序

15模型主表完成且不存在零工作误标、correctness failure或配置漂移后，再运行：

```text
A0  Hexagon-MLIR HVX + HexKL On
A1  A0 + C
A2  A1 + E
A3  A2 + P
A4  A3 + R
```

历史item7中的semantic/topology/slicing/runtime-prefetch仍单独拆分，不能把topology收益
自动归因于data prefetch。对于某模型未获准的阶段，表中写`not admitted`而不是把0%变化
解释成机制无效。这样主表回答最终效果，消融回答效果来自C、删除物化、异步供给还是反馈
准入，二者不会互相污染。

#### 12.21.7 UniSpeech-SAT重复模型审计与15模型修正（2026-08-30）

冻结矩阵执行到UniSpeech-SAT时做了结构审计。`UniSpeechConfig()`与
`UniSpeechSatConfig()`虽然属于不同Transformers类，但当前两个ForCTC runner在主实验实际
执行的图是等价重复：都是12层、hidden 768、12 heads、FFN 3072、相同七层卷积前端和
`1x20560` FP16输入；总参数同为94,396,320，213个参数tensor的名称（归一化root前缀后）
与shape逐项相同，模块类型只是`UniSpeech*`/`UniSpeechSat*`类名前缀不同。两者导出IR均有
98个`linalg.batch_matmul`，HexKL均rewrite 74个。

设备侧证据也一致：PK两者均产生387个hint、1,925,676次runtime issue、401,762,008 B
issued bytes和24,625,664 B静态materialization；latency分别为172,840.50和174,607.95 ms，
差异仅1.02%。因此随机seed不同不能把它们算作两个独立模型。UniSpeech-SAT PK保留为
重复性验证，其余四组在APT编译早期停止，不进入论文主表或模型数量统计。冻结主表改为
15个**结构独立**的完整模型，并用已支持的完整ViT-Base替换该位置。历史UniSpeech-SAT
实验不删除，但解释为共享speech graph的回归/negative-control证据，而不是额外模型证据。

#### 12.21.8 正式消融缩减为5个预注册正例模型（2026-08-30）

15模型冻结主表已经回答ALPS的总体有效性、跨模型覆盖和负例；再对全部15模型运行完整
A0--A4不会成比例增加因果证据。正式消融改为采用主表结果确定、且不依赖人工挑选的准入
规则：**相对冻结HexKL-On control达到至少1.50x的全部模型**进入消融。满足规则的恰好是
DINOv2-small（1.80x）、Swin Transformer（1.55x）、SegFormer MiT-B0（1.72x）、
DeiT-Small（1.65x）和Whisper-Tiny（1.65x）。该集合覆盖Vision和Speech，并包含不同的
attention、window attention、multi-scale vision以及audio frontend路径。

正式表仍使用统一嵌套配置：

```text
A0  Hexagon-MLIR HVX + HexKL On
A1  A0 + C
A2  A1 + E
A3  A2 + P
A4  A3 + R
```

冻结主实验中的A0和A4直接复用，因此只需补5个模型的A1--A3，共15个配置，而不是原计划
的45个配置。2026-08-29已有的SegFormer/Whisper C与C+E、DINO C+E/P/R等阶段数据保留为
机制证据和结果sanity check；由于它们不是同一冻结批次，不能与新表跨代计算incremental
speedup。只有配置、代码fingerprint、输入、设备状态和测量窗口完全匹配的数据才能进入
正式消融格子。

正式消融单独成表，同时报告每一级latency、相邻阶段增量`A(i-1)/Ai`和相对A0累计加速。
若某阶段没有合法site，写`not admitted`；若R只有hold而没有改变P决策，则写
`monitoring-only / no causal latency claim`，不能把自然波动归因于traffic control。

执行结果：5个模型的15个A1--A3配置全部PASS。E是统一主导项，增量为1.30x--1.69x；
P在DINOv2/DeiT为约1.06x，在SegFormer/Whisper约1.02x，Swin约1.00x。C只在
SegFormer（1.31x）和Whisper（1.04x）明确有益，Swin先回退约2.8%但被E恢复。A1到A3的
sysMon AXI总量在五个模型分别下降18.28%、9.53%、41.67%、17.68%和12.86%，支持收益
来自consumer-driven representation formation减少物理流量，并由选择性异步供给隐藏一部分
残余搬运。R仍只有hold，A3/A4的微小差异不作因果性能声明。正式数值和raw artifact路径见
`experimental_data.md`的独立Ablation Study表。

#### 12.21.9 冻结E职责修正与full-E复测（2026-08-31）

对历史`2,993.49 ms`与窄E冻结`5,499.60 ms`的配置审计确认，两者模型、输入、FP16、
V73、HVX vector、HexKL和`-O3`均一致；差异来自冻结E错误地只保留基础P2e和HMX
output formation，而遗漏了旧累计链中的P2g continuity/register-tile、P5h attention
destination formation和P5i patch/token producer formation。旧`--alps-p5n`名称实际代表
累积策略，不是单独的异步drain。

机制上P5h和P5i都属于E而不是新的论文顶层组件：P5h令attention consumer直接在最终
destination上形成结果并删除temporary/writeback；P5i把token consumer的连续layout
contract传播回patch producer，直接形成`[N,tokens,channels]`，并由连续OC内层维解锁HVX
codegen。P2g及P5f/P5g中必要的continuity/head-major分析是这些realizer的内部基础设施。
因此冻结runner已将E统一为：

```text
consumer contract discovery/propagation
  -> P2e generic producer-direct formation
  -> P2g continuity/register-tile realization
  -> P5h attention destination realization
  -> P5i patch/token producer realization
  -> P5j/P5k HMX output realization
```

各历史gate仍可独立开关，供E内部消融和legality回归使用；论文顶层消融仍保持
`A0 baseline -> A1 +C -> A2 +full E -> A3 +P -> A4 +R`。

五个预注册完整模型的同轮复测全部PASS。full E相对A1的独立增量为：DINOv2
`2.99x`、DeiT `2.87x`、Whisper `1.61x`、Swin `1.53x`、SegFormer `1.37x`；加入P后
分别再获得`1.08x`、`1.12x`、`1.01x`、`1.00x`和`1.01x`。DINO最终为
`2,976.63 ms`，复现历史约3秒量级；DeiT最终由窄E的`5,016.40 ms`降至
`2,594.67 ms`。这证明性能恢复来自完整consumer-driven formation，而不是设备频率或
测量偶然性。

P5h在DINO/Swin/SegFormer/DeiT/Whisper分别改写`11/3/0/11/6`条链，P5i分别形成
`1/0/2/1/0`个producer；因此它们是同一E抽象下按topology合法性选择的realizer，不是
DINO专用的无条件开关。A1到A3的sysMon AXI总量分别下降25.50%、10.34%、42.06%、
22.75%和23.61%。R仍只有hold窗口，没有改变P决策，A3/A4差异不作R的因果收益声明。

完整数值与权威产物路径见`experimental_data.md`的
“Re-frozen full-E selected-model matrix and ablation”章节。

#### 12.21.10 Consumer-contract topology admission与全量复测（2026-09-01）

恢复item7后，DeiT完整ALPS编译在约16 GB WSL内存上稳定触发OOM（exit 137）。审计排除
movement ledger后，根因是item7兼容umbrella在P5h之前保护全函数attention topology、关闭
原生slicing并传播K/V tiling；而后续P5h已经重写11条attention destination链并删除约
20.07 MB copy。最终runtime没有发出任何K/V L2 hint（eager K/V均为本调用内producer），
但编译器已经为同一个consumer representation contract重复构造了大规模IR。

最终修复没有按模型名关闭item7，而是增加模型无关的contract subsumption规则。早期P2e若
证明`demands > 0 && producer_direct == demands && native == 0`，函数记录
`alps.kv_topology_admission = "covered_by_consumer_formation"`，后续K/V semantic/fusion/tiling
metadata不再生成；存在任何未形成需求时则记录`admit_residual_topology`。同时Hexagon
slicing不再因为item7而全模块禁用，而是只跳过实际携带、且通过admission的K/V op。由此
建立统一顺序：**先消除/原位形成目标表示，再仅为无法消除的residual transfer保留
topology和prefetch**。

这项修复强化而非削弱论文故事：P5h/full-E与item7是同一个consumer contract的可选
realizer，不应无条件叠加；P仍独立处理无法消除的HMX/VTCM drain。DeiT修复后完整编译
成功，ALPS为2,586.91 ms、matched HexKL-On为8,282.03 ms（3.20x），并继续发出7,776个
async drain descriptor/15,925,248 B，因此不是通过删除prefetch伪造收益。结构规则还在ViT
和DINO上拒绝冗余topology；Swin、Qwen和分层LLM只有部分consumer formation，仍保留
item7 residual topology及原有收益路径。

随后对15个完整FP16模型从头运行30个matched case，全部PASS。全体几何平均1.67x，
Vision为2.43x、Language/text为1.41x、Speech为1.18x；9个模型达到1.50x、7个达到1.80x，
DeiT和DINO超过3x。Qwen为2.04x，TinyLlama为1.65x，说明admission没有误伤item7的分层
LLM正例；三个结构近似speech encoder仍约1.01--1.02x，继续作为“合法异步搬运不等于
关键路径收益”的negative evidence。完整表和raw路径见`experimental_data.md`的
“Consumer-contract-admitted full rerun”章节。
