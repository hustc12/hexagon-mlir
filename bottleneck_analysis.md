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
- `scripts/layered_hvx_options.py`
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
- `scripts/layered_hvx_options.py`
- `qcom_hexagon_backend/lib/Transforms/LowerTmTensor.cpp`
- `qcom_hexagon_backend/lib/Transforms/PrefetchInsertPass.cpp`
- `qcom_hexagon_backend/lib/Transforms/LayoutOpsEliminationPass.cpp`
- `qcom_hexagon_backend/lib/Dialect/OmniFetch/IR/LayoutAwareMapping.cpp`
- `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/FusionPass.cpp`
- `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`
- `qcom_hexagon_backend/include/hexagon/Dialect/OmniFetch/IR/OmniFetchOps.td`
- `docs_engineering/engineering_work.md`
- `docs_engineering/omnifetch_history.md`
- `docs_engineering/omnifetch-prefetch-insitu-innovation.md`

### V73 手册

- `docs_engineering/Hexagon_V73_Programmers_Reference_Manual.pdf`，Memory / Cache prefetch / software `l2fetch`；
- `docs_engineering/Hexagon_V73_HVX_Programmers_Reference_Manual.pdf`，HVX local memory、VMEM/L2、VTCM、scatter/gather、memory performance 与 PMU events。

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
bash scripts/build_hexagon_mlir_incremental.sh
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
2. 对每组保存最终 IR、编译日志、正确性、latency，并用 `scripts/audit_hexagon_codegen.sh` 统计 object instruction/HVX/VMEM/HexKL 差异；
3. 补充 fusion group、slicing site、K/V marked/rejected/runtime site、alloc/copy/layout bytes 汇总；
4. 只有明确定位历史 item 7 的正收益和负回退来自哪个独立策略后，才宣布 P0 通过；
5. P0 通过后进入 P1 analysis-only representation/movement ledger，不提前扩展 HRA/HPA 的全局 action search。

这个顺序保持论文的三点主线不变：P1/P2 确定并减少必要 movement，P3 用 V-DAE overlap 剩余 movement，P4A 用 PMU/traffic feedback 调节已选路径；Hierarchical Representation Admission 仍作为未来论文方向保留。

统一脚本已增加 `--alps-p0` 模式，不为每个模型创建新脚本。三模型 gate 的执行形式为：

```bash
OUTPUT_DIR=/path/to/local/results \
REMOTE_RESULTS_DIR=/home/huzq85/2-working/working_set/alps_p0_YYYYMMDD \
scripts/run_full_hvx_five_way.sh --alps-p0 \
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
scripts/run_full_hvx_five_way.sh --alps-p0b \
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
scripts/run_full_hvx_five_way.sh --alps-p2a --compile-threads 4 \
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
scripts/run_full_hvx_five_way.sh --alps-p2c --compile-threads 4 \
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
