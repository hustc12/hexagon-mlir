# Alps 研究与工程历史梳理

> 状态快照：2026-08-07，`baseline_5` 分支。
>
> 本文不是新的功能清单，而是对 Alps 到目前为止的设计、实现、实验和方向修正做一次因果审计。主要依据是
> [创新设计文档](docs/alps-prefetch-insitu-innovation.md)、
> [工程排障记录](Engineering_work.md)、
> [两个独立预取 baseline 的记录](docs/hexagon-prefetch-baselines-plan.md)、
> 仓库提交历史，以及仓库根目录中的两份 Hexagon V73 手册。

## 1. 先给结论：我们不是没有方向，而是曾经使用了不一致的基线

Alps 的主线始终是连贯的：

```text
减少不必要的数据搬动
        -> 对无法消除的搬动做 in-situ 布局形成
        -> 让一次搬动服务尽可能多的消费者
        -> 只对仍然不可避免且有足够提前量的数据做 prefetch / DMA
        -> 在 VRF、VTCM、L2、DDR 之间保持数据驻留
```

真正造成“兜兜转转”的原因，不是这条故事线错误，而是历史实验一度混合了四种不同效应：

1. 名为 `HVX` 的旧配置实际关闭了 vectorization、VTCM tiling 和部分 HexagonMem lowering，主要执行 generic scalar Hexagon 代码；
2. `HexKL` 在很多模型上没有任何 HMX rewrite，因而不能代表 HMX baseline；
3. item 7 不只插入 K/V `l2fetch`，还保留了 attention fusion boundary、改变了布局和融合拓扑；
4. cumulative runner 的 warm/persistent-cache 计时与单次 HexKL 计时曾经不完全对称。

修复真实 HVX 后，DINOv2 Debug 从旧 scalar 的 174.832 ms 降到 21.258 ms，单是 baseline 修复就达到 8.224x。此前 cumulative item 7 的 60.927 ms 虽然比旧 scalar 快 2.870x，却比真实 HVX 慢 2.866x。这说明早期很多大倍率主要反映“旧 baseline/codegen 被组合策略绕开或改变”，不能直接解释为 prefetch 本身的收益。

目前最可信的结论是：

- Alps 已经形成了完整的编译器机制框架，并证明能够显著减少预取命令数、VTCM 静态峰值和某些布局变换；
- 早期 Debug 结果证明“融合边界 + 布局 + 预取”的组合策略有潜力，但没有完成对真实 HVX/HMX 的独立归因；
- 在匹配的真实 HVX 上，当前 K/V L2 prefetch 的收益只有噪声级，DINOv2 Debug 为 1.03x，Falcon Debug 为 0.997x；
- 全量 HuBERT 上，cumulative items 1–7 相对 HVX 和 HexKL 都严重回退；单独修复后的 K/V L2-only 也比 HVX 慢 2.39%；
- 所以不能继续把“多插一些 prefetch”当成主方向。正确下一步是以全模型的真实 HVX/HMX 瓶颈为输入，做 region-level movement elimination、reuse 和 residency。

## 2. 目标必须如何定义

最终愿景仍然是让模型列表中的模型在 Hexagon NPU 上普遍加速，期望至少 1.8x，部分达到 3x。但这应被区分为研究愿景和单个机制的准入条件。

如果某模型有比例 `f` 的时间属于可被当前数据搬动方案优化的部分，而这部分被加速 `r` 倍，则 Amdahl 上限为：

```text
Speedup = 1 / ((1 - f) + f / r)
```

即使把目标部分加速到无限快：

- 达到 1.8x 也要求至少 44.4% 的端到端时间可被消除；
- 达到 3x 要求至少 66.7% 的端到端时间可被消除。

因此，“所有模型都由同一个 prefetch 开关得到 1.8x”不是可信的科学假设。若所有模型都能达到这个倍率，通常意味着当前 baseline 存在系统性 codegen 缺陷；修复这种缺陷是必要的工程工作，但不能全部算作 Alps 的论文贡献。

合理的验收层次应为：

1. 每个模型都使用正确、匹配的真实 HVX/HMX baseline；
2. 每个模型都由同一套 movement analysis 自动分类，但可选择不同策略；
3. 对 memory/layout-bound 子集争取稳定 `>=1.8x`，个别达到 `>=3x`；
4. 对 compute-bound 或没有可消除搬动的模型要求 no-regression，而不是伪造统一倍率；
5. 汇报全模型 geometric mean、P50/P90、正确性、最终目标码和 PMU/traffic 证据；
6. QNN HTP 作为闭源 vendor upper bound，而不是把 LiteRT-QNN重复算成独立 NPU backend。

## 3. 研究过程的时间线

### 3.1 初始 Alps：从提前搬动到编译器可见的数据供应

初始实现于提交 `018c588` 附近形成，核心是预取、V-DAE/DMA 和 layout-aware reshape。随后 `94872a7` 与 `c794cf8` 增加消融并将 V-DAE、prefetch、layout reshape 解耦。

最初已经意识到：`reshape` 不能仅是一个语法 view；编译器必须跟踪物理 layout、buffer identity、lifetime、placement、readiness 和 reuse。这个认识后来发展成 item 1–7。

### 3.2 item 1–3：先建立分析和消除机制

| 项目 | 内容 | 已确认结果 | 正确解读 |
|---|---|---|---|
| item 1 | layout-value liveness | Falcon Debug 约 0.16% 回退，后续样本约 0.7% 正向 | 分析基础设施，不是独立性能贡献 |
| item 2 | transform cost model | Falcon Debug 约 0.34% 正向 | 噪声级；价值在于选择/拒绝机制 |
| item 3 | layout-carry fusion | 消除 2 个 collapse view，5 个 activation site；正确性通过 | 证明能消除 IR 变换，但当时没有干净独立性能归因 |

这一阶段的重要经验是：编译器“识别了机会”不等于物理内存流量真的下降。必须检查最终 object 的 load/store、spill 和实际延迟。

### 3.3 item 4–6：跨调用复用、二维流水和 VTCM 生命周期

| 项目 | 主要机制 | Falcon Debug 累计结果 | 已证实的机制价值 | 尚未证实的性能价值 |
|---|---|---:|---|---|
| item 4 | generation-safe persistent WH cache | 1612.315 ms vs HexKL 1628.410 ms，1.010x | 99.31% warm hit；8 MiB cache；失效语义正确 | 约 1% 小于可靠论文信号 |
| item 5 | DDR load / reshape / compute 二维 ping-pong | 1586.580 ms vs 1622.670 ms，1.023x | 真正 bootstrap + `t+1` 异步加载；与 item 4 组合正确 | 约 2.2%，仍需交错重复 |
| item 6 | VTCM lifetime coloring | 1598.756 ms vs 1619.855 ms，1.013x | 静态峰值 45056 -> 16384 B，减少 63.64% | 与 item 5 跨 run 差异不可归因 |

这些项目不是失败。它们证明了 lifetime、ownership、persistent identity 和流水正确性。但它们也说明：减少静态 VTCM 峰值或获得高 cache hit，并不自动带来 1.8x 端到端收益。

### 3.4 item 7 与 `alps-2x-improvement`：第一次大幅收益，也是最大归因陷阱

item 7 包含四件事：

1. 给 QK 的 K、AV 的 V 标记 compiler-visible semantic role；
2. 让 K 以 transpose-aware indexing 直接被消费，避免 cache-wide transpose；
3. 在 bufferization 之前保留 K/V attention boundary，避免融合丢失身份；
4. 按 stream/page 合并并发出 L2 prefetch hint。

Falcon Debug、seq=128 的历史结果为：

| 配置 | 延迟 |
|---|---:|
| 旧 `HVX` | 11742.816 ms |
| HexKL | 1614.896 ms |
| HexKL + items 1–7 | 599.969 ms |

即 `2.692x` over HexKL。提交 `b6f5548` 因此被标记为 tag `alps-2x-improvement`。

这个 tag 很重要，应该永久保留为“组合策略产生显著历史信号”的可复现实验快照；但它不应被描述成“item 7 的八条 l2fetch 单独带来 2.692x”。原文当时已经正确注明，大幅变化应归因于：

```text
KV-aware fusion boundary + direct K layout + page prefetch
```

而且 Falcon runner 使用 `use_cache=False`，测的是 prefill 中刚刚产生的 K/V，不是真正 DDR-resident 的 autoregressive past-K/V decode。

### 3.5 将组合方案扩展到模型列表：Debug 15 模型筛选

`c233a78` 将 items 1–7 接入更多 runner，随后多个提交完成了三领域候选筛选。历史 Debug 筛选最终得到一个 5/5/5 的 `>=1.8x` relaxed 集合：

| Domain | 历史筛选模型 |
|---|---|
| Language/Text | Falcon、GPT-2、Qwen2.5-0.5B、TinyLlama、SD/CLIP text encoder |
| Computer Vision | Swin、SegFormer、DeiT-Small、BEiT、DINOv2 |
| Speech/Audio | Whisper、HuBERT、Wav2Vec2、UniSpeech、UniSpeech-SAT |

其中有代表性的历史结果包括：Swin 3.239x、Whisper 2.985x、SegFormer 2.314x、DINOv2 proxy 2.898x、Falcon 2.741x。这个阶段证明组合 pass 的适用面不局限于 Falcon。

但该“15 模型”只能被称为历史架构筛选集合，不能继续称作真实 HVX 上的 15 个合格模型，原因如下：

- 旧 `HVX` helper 实际关闭了 vectorization；
- GPT-2 为 FP32 graph，HexKL 动态做大规模 f32->f16 转换，109539.127 ms 的 HexKL baseline 是病态值；其组合虽比 HexKL 快 3.43x，却比旧 HVX 还慢 1.42x；
- Qwen 的 warm 结果为 1.936x，但 cold 只有 1.423x；
- SD text encoder 是亚毫秒结果，统计稳定性不足；
- DeiT、BEiT 输出 NaN，只是用户当时允许的 accuracy-waived screening；
- 一些音频 proxy 减少到一层，不是全量模型；
- item 7 改变 fusion boundary，因而这些数据主要是“组合 codegen topology”的信号。

所以这 15 个模型没有消失，但其状态应从“论文已合格”降级为“需要在修复后的真实 HVX/HMX 上重新验证的候选集”。

### 3.6 全量模型：可执行性收益与性能收益开始分离

`507603b` 建立了串行全模型矩阵。早期结果包括：

- Swin-Tiny：只有 cumulative row 完成，206326.114 ms；两个 baseline exit 13，不能计算加速比；
- SegFormer：只有 cumulative row 完成，19785.205 ms；属于可执行性改善，不是速度对比；
- DeiT-Small：三个配置都有 timing，组合表面为 3.428x，但三个输出都有 NaN；
- DINOv2-small：当时三个配置正确，组合 300484.161 ms vs HexKL 1284233.665 ms，表面 4.274x；但 HexKL 有 0 次 HMX rewrite，且后来证明当时的 `HVX` 不是正确 vector baseline；
- Whisper-tiny：组合正确完成 1093076.414 ms，两个 baseline 失败，因此只能算 feasibility；
- BEiT-base：HVX/HexKL exit 13，组合 host codegen exit 137。

这一阶段的价值是发现 capacity、ABI、host peak memory、device exit 13 和 codegen 复杂度问题。它没有建立“全量模型上稳定 1.8x”的结论。

### 3.7 QNN / LiteRT 对比：暴露的是整个编译栈差距

直接 QNN HTP 和 LiteRT-QNN 的 DINOv2 Debug bring-up 证明：LiteRT 的 Qualcomm NPU 后端最终仍通过 QNN HTP，因此不能把 LiteRT-QNN 与 direct QNN 当成两个独立 NPU compiler baseline。

QNN 的数量级优势也不能简单归因于 HMX。QNN HTP 是完整图编译器和运行时，包含 vendor kernel selection、fusion、layout propagation、constant preparation、VTCM planning、buffer reuse 和调度。相比之下，当时 Hexagon-MLIR graph 有 84 个 generic op、0 个 HexKL rewrite，并关闭了向量化。这个对比促使项目先修 baseline，而不是继续在弱 baseline 上累计倍率。

### 3.8 Engineering 修复：终于建立真实 HVX baseline

工程审计发现 benchmark helper 强制设置：

```text
enableVectorization = False
enableVTCMTiling = False
部分路径 enableConvertToHexagonmem = False
```

修复配置、ABI、late `ub.poison`、LLVM `PS_aligna` prologue ordering 后，DINOv2 Debug 得到：

| 配置 | 延迟 | 解释 |
|---|---:|---|
| legacy scalar | 174.832 ms | 无 HVX-like 指令 |
| true HVX vector | 21.258 ms | 8.224x baseline 修复 |
| HVX + VTCM | 21.931 ms | 与 HVX object 等价 |
| HexKL + vector + VTCM | 22.232 ms | 没有 HMX/HexKL 目标码证据 |
| cumulative item 7 scalar | 60.927 ms | 比真实 HVX 慢 2.866x |

LWP 进一步显示 DINOv2 Debug 的动态瓶颈是：patch embedding convolution 46.90%、attention AV 15.43%、QK 9.35%、output projection 9.16%。这证明“只处理 K/V stream”无法覆盖该模型的主要时间。

### 3.9 手册驱动的 M1–M10 与 MAR/N1–N10：从 feature list 转向 movement abstraction

两份 V73 手册促成了两轮设计收敛：

- M1–M10：page-safe/single-flight L2 scheduler、L2-vs-DMA、layout equivalence、VRF forwarding、alignment/page placement、VTCM coloring、nontemporal last-use、store/load hazard、tiered residency、PMU feedback；
- MAR/N1–N10：stationary scheduling、activation multicast、online reduction、residual rendezvous、recompute-vs-reload、VRF pressure、overwrite suppression、page supertile、HMX/HVX boundary residency、multi-layer circular arena。

统一抽象是 Movement-Amortization Region（MAR）：

1. 这个 movement 是否必要；
2. 必要的字节应何时、通过什么路径移动；
3. 一次 movement 能服务多少消费者。

这与原故事线一致，并不是为了加速而另起炉灶。不过前两个原型也暴露了成本模型问题：

- N1 explicit transpose-stationary 在 Falcon true HVX 上从 509.019 ms 回退到 661.809 ms，慢 1.300x；ledger 错把逻辑 weight references 当成物理 DDR reads，而真实新增 transpose 是不可忽略的流量；
- N2 activation multicast 的编译器机制已完成，但 Falcon 导出 fused QKV，因此候选为 0；其他适合模型当时没有合法 true-HVX baseline，尚未性能准入。

因此，MAR 是正确抽象，但不能靠静态“逻辑字节数”预测收益，必须结合最终 VMEM、cache-line、spill、PMU 和端到端 timing。

### 3.10 两个独立 prefetch baseline：证明命令效率，不证明大幅延迟优势

在 `baseline_5` 上实现了 Prefetch-Kernel-HX 和 APT-GET-HX，并与 Alps 在 DINOv2/ViT Debug 上串行对比。Alps 的重要优势是请求数显著更少：

- DINOv2：Alps 36 次 runtime request，对方 122843 次；
- ViT：Alps 72 次，对方 36274 次。

延迟改善并不显著，因此这支持的论文点是“reuse-aware coalescing 避免 prefetch command storm”，不是“当前实现已经普遍得到大幅 speedup”。

### 3.11 返回全模型与最新 HuBERT 结果：当前方向必须收敛

最新全量 HuBERT-base 的 matched true-vector 结果为：

| 配置 | 延迟 | 相对关系 |
|---|---:|---|
| HVX vector | 212074.277 ms | baseline |
| HexKL + HVX vector | 268255.667 ms | 比 HVX 慢 1.265x |
| HexKL + items 1–7 + HVX vector | 620050.837 ms | 比 HexKL 慢 2.312x，比 HVX 慢 2.924x |
| HVX + repaired K/V L2-only | 217146.200 ms | 比 HVX 慢 2.39% |

最终 object 中 plain HVX 的 HVX instruction line 占 47.41%，K/V 版本为 45.04%。所以它确实在使用 HVX，但静态指令仍有一半以上为 scalar，且当前 `hvx-vector` 配置没有启用 VTCM tiling。

修复后的 item 7 找到 24 个 K/V sites（12 K + 12 V）和 288 个 hints，但没有带来收益。这与 DINOv2/Falcon true-vector 结果一致：prefill 中 K/V 是本层刚产生并立即消费的数据，通常已经在 L2；为它保留边界和再次 hint 不会隐藏 DDR miss，反而可能损伤 fusion。

## 4. 哪些结果现在可以相信

### 4.1 强证据

- 修复后的 DINOv2 Debug true HVX 比 legacy scalar 快 8.224x；
- item 6 的 VTCM 静态峰值确实减少 63.64%，且经过 lifetime bug 修复后正确；
- persistent WH cache 的 generation/site identity 和 99% 级 warm hit 机制正确；
- 二维 pipeline、K/V semantic tracking、page accounting、prefetch counters 和两个独立 baseline 均有编译器/runtime 实现与测试；
- Alps 在两个 Debug 模型上将 prefetch 请求数量降低数个数量级；
- N1 explicit transpose-stationary、whole-stream VTCM staging、full HuBERT cumulative 是明确负结果；
- true HVX 上 fresh-prefill K/V hint 仅噪声级或轻微回退。

### 4.2 有价值但必须重新验证的信号

- tag `alps-2x-improvement` 的 Falcon 2.692x；
- 历史 15-model Debug 筛选；
- 旧 full DINOv2 的 4.274x；
- Swin/Whisper/SegFormer 等组合方案的大倍率或 feasibility。

这些结果说明改变 fusion/layout/data-supply topology 可能非常有效，但必须在 true HVX、真实 HMX coverage、对称计时和全模型正确性下重新建立。

### 4.3 不能再使用的表述

- “item 7 的 K/V prefetch 单独带来约 2.7x”；
- “当前已经有 15 个真实 HVX 模型达到 1.8x”；
- “HexKL 行一定使用了 HMX”；
- “只要请求了 VTCM 就是在 VTCM 上高效执行”；
- “prefetch 次数越多越好”；
- “组合 row 能跑、baseline exit 13，所以组合 row 有无限加速”；
- “LiteRT 和 QNN 是两个独立 Qualcomm NPU backend”。

## 5. V73 memory hierarchy 对后续方向的硬约束

这次重新核对两份手册后，以下约束必须写进实现的 admission gate，而不只是写在设计文档里。

### 5.1 VRF 优先于任何 memory optimization

HVX 是 load-store 架构。手册明确建议减少 VMEM，因为 VRF access 比任何 memory access 都便宜。能通过 producer-consumer fusion、online reduction 或 multicast 留在 vector register 的值，不应该先存入 L2/VTCM 再读回。

### 5.2 HVX VMEM 直接连接 L2，不经过 scalar L1

因此 scalar `dcfetch` 不是 HVX tensor path 的核心优化。对一次性 DDR stream，在有足够提前量时使用 L2FETCH；对高复用 tile，才考虑显式 DMA 到 VTCM。

### 5.3 `l2fetch` 是稀缺的 single-flight、best-effort 资源

手册说明：

- 建议请求小于 8 KiB；
- 应在首次使用前数百 cycles 发出；
- 太早会被逐出，太晚无法覆盖 latency；
- 新 `l2fetch` 会停止尚未完成的旧 prefetch；
- 跨 page 生成的地址会被丢弃；
- PFA 与 PMU 可以观测 active/completion/access。

这解释了独立 baseline 的数万/十万请求为何不合理，也要求全图只有一个 compiler-visible scheduler，而不是每个 op 自己发 hint。

### 5.4 VTCM 不是更快的 cache，而是软件管理的显式驻留空间

VTCM non-evictable、低延迟、适合 intermediate vector data，并且是 scatter/gather 必需目标；但 DDR->VTCM copy、同步和 bank pressure 都有成本。只有满足以下至少一个条件才应选择 VTCM：

- tile 被多个算子、head 或 position 重用；
- DMA 能与足够长的计算窗口重叠；
- producer 能直接形成 consumer/HMX layout；
- scatter/gather 或 deterministic residency 的价值超过搬运成本。

短序列 whole-stream staging 已经被 DINOv2 Debug 的 0.53x/0.51x 结果否定。

### 5.5 对齐、page/TLB、store-to-load 与 final-use 必须联合处理

- 优先 128-byte aligned VMEM，避免昂贵的 VMEMU；
- page-working-set 和 micro-TLB 应进入 tile 选择；
- 同地址 VMEM store 后紧接 load 有较大 penalty，约需 15 个 packets 的间隔；
- ping-pong 不只是并行结构，也可提供 hazard distance；
- final one-pass data 可使用 `:nt`，但必须通过 last-use/alias proof；
- external DMA 与 HVX thread noncoherent，必须有明确 completion、barrier 和 ownership。

## 6. 是否需要再来一轮头脑风暴

需要，但不是第四份“十个独立点子”的清单。现有 item 1–7、M1–M10、N1–N10 已经覆盖了绝大多数名词空间。下一轮应称为 **bottleneck-driven convergence**，只围绕一个统一机制做实现：

> **Fusion-preserving Movement Region（融合保持的数据搬动区域）**：在 producer、多个 consumer、HVX/HMX 边界之间，为一个 tile 建立从产生到最后使用的 memory contract；优先不落地，其次一次搬动多次使用，最后才调度 page-safe L2FETCH 或 DMA/VTCM。

它是原 Alps、in-situ reshape、M1–M10 和 MAR 的收敛版本，不是新故事。

### 6.1 一个 region 只能选择以下路径之一

```text
P0: VRF forward / fusion                 # 零物理搬动
P1: producer-side direct layout store    # 消除 reshape/transpose/pack
P2: VTCM resident tile                   # 一次 DMA，多个消费者
P3: page-safe L2 lease                    # 一次性 cold DDR stream
P4: ordinary demand VMEM                  # 预取无收益时的安全回退
```

选择器必须依据 producer distance、reuse count、physical bytes、page count、alignment、VRF spill、VTCM bank/footprint、HMX/HVX consumer 和 final use，而不能只看 op 类型。

### 6.2 优先实现的三个可证伪方向

#### A. Fresh-tensor path：在线 attention / reduction + producer-side layout

对象是 prefill、ViT、speech encoder 中刚产生的 Q/K/V。不要预取刚写出的 K/V，而是：

1. projection producer 直接产生 head/tile consumer layout；
2. QK 使用 online softmax max/sum；
3. 当前 score tile 直接消费 V，避免完整 score/probability matrix 落地；
4. 用 VRF/VTCM 保存 bounded accumulator；
5. 只有下一块真正冷的 source tile 才提前 prefetch。

这同时减少 reshape/transpose、score tensor store/load，并创造真正的预取 overlap window。它比继续调整 K/V lookahead 更有机会在 attention 占比较高的 LLM、vision 和 audio transformer 上得到大幅收益。

#### B. Cross-engine path：HMX output -> HVX epilogue -> next HMX 的 VTCM residency

对象是已经合法进入 HMX 的 matmul region：

```text
HMX output tile in public VTCM
  -> HVX bias / activation / norm / residual in situ
  -> next HMX AH layout in the same or colored slot
```

目标不是扩大 HMX shape coverage；后者仍是文档中标记的 **Next Paper Idea**。当前论文只优化“已经能进入 HMX 的数据供应和边界搬动”，这样故事线仍然完整。

#### C. Domain hotspot path：让同一 movement contract 覆盖非 attention 模型

若目标包括模型列表中的所有模型，就不能只做 attention：

- Vision：优先处理 LWP 已证实占 DINOv2 Debug 46.90% 的 patch embedding convolution，采用 patch/output-channel tile、producer-side layout 和 VTCM ping-pong；
- Audio：profile positional convolution、feature extractor、LayerNorm/attention 的真实占比，再在最高占比 region 应用相同 contract；
- LLM：prefill 优先 weight/projection/MLP tile reuse；真实 decode 才对 persistent past-K/V page 使用 L2 lease 或 DMA/VTCM；
- convolution-only negative controls：若 profile 证明没有可消除的跨算子 traffic，应明确归类为 no-regression，而不是强行插 hint。

### 6.3 真正的 item 7 应拆成两种语义

| 场景 | K/V 来源 | 正确策略 |
|---|---|---|
| Prefill / encoder | 本层刚产生 | fusion、direct layout、online consume；通常不做 L2 prefetch |
| Autoregressive decode | 跨 token 的历史 DDR state | page/head-aware prefetch，必要时 ping-pong VTCM，并与 query/softmax 计算重叠 |

当前 `use_cache=False` 的 Falcon/GPT-2/Qwen/TinyLlama runner 不能验证第二行。若要把 attention K/V prefetch 作为核心贡献，必须实现固定 shape decode-step ABI、`past_key_values` 输入输出、cache position/page table、连续多 token device invocation，以及 GQA/MQA/sliding-window 覆盖。

## 7. 下一阶段按顺序执行的计划

### Phase 0：冻结可比较基线，禁止继续累计混杂结果

1. 为每个 runner 输出 `scalar / true-HVX / VTCM / HexKL-HMX / Alps` 的实际 option manifest；
2. object audit 必须报告 HVX instruction、HMX call/rewrite、DMA、VMEM、spill；
3. HexKL 行若 HMX rewrite 为 0，标成 `HexKL enabled, HMX inactive`，不能算 HMX baseline；
4. 计时统一 cold/warm、iteration、heap、power mode、input hash，设备执行严格串行；
5. exit 13 立即保存 DSP log、perf/output 文件和 object，不重复盲跑。

### Phase 1：建立全模型 movement/bottleneck census

对 15 个候选和全部 `run_*.py` 先做一次低成本编译/profile 分类，而不是立刻跑三个超长全量配置。每个模型至少记录：

- top-10 LWP regions 和占比；
- logical tensor bytes 与最终 VMEM/load-store；
- L2/DDR/TLB/VTCM stall 相关 PMU；
- materialized transpose/reshape/pack/unpack bytes；
- attention fresh-K/V 与 persistent-K/V 字节；
- HMX eligible/rejected shape 及原因；
- region reuse、first-use distance、VRF spill、VTCM peak。

只有 profile 预测可优化比例 `f >= 44.4%` 的模型，才把 1.8x 作为该轮硬目标。其余模型先以 no-regression 和最大热点优化为目标。

### Phase 2：先做 A——online attention / producer-layout region

选择三个 true-HVX 可运行的代表：一个 LLM、一个 vision、一个 audio。准入条件：

- 最终 object/PMU 证明 score/K/V 或 layout materialization bytes 下降；
- 没有新增等量 spill/transpose；
- 正确性通过；
- 相同 true-HVX baseline 下串行重复稳定改善；
- 不满足即停止扩展，不把它默认接入所有 runner。

### Phase 3：再做 B——合法 HMX region 的 HMX/HVX boundary residency

先找一个真实 HMX rewrite 非零且 baseline 正确的完整模型。若当前没有，就先修模型/shape 到已有 HexKL public ABI，而不是在本论文中顺手实现通用非对齐 HMX lowering。

### Phase 4：实现真实 decode K/V pipeline

用 Falcon 或 Qwen 的固定 shape decode step 验证 historical K/V，而不是继续在 prefill 上调 page token。至少比较：

1. true HVX decode；
2. HexKL/HMX decode；
3. past-K/V page-safe L2 lease；
4. reused K/V tile 的 DMA/VTCM ping-pong；
5. in-situ K/V consumer layout。

### Phase 5：按 domain hotspot 扩展到完整模型

先完成一个机制的三领域证据，再扩展到全部模型。正式结果表同时报告：

- HVX、HexKL/HMX、HexKL/HMX+Alps；
- speedup、geometric mean、P50/P90；
- correctness；
- HMX active 与否；
- eliminated/added bytes、VMEM、PMU stalls；
- prefetch requested/issued/suppressed/page-clipped；
- DMA overlap、VTCM peak/bank stalls；
- compile time 与 host/device capacity failure。

## 8. 论文故事线的最终建议

论文不应写成“我们实现了很多 prefetch 技巧”，而应写成：

> 现有 Hexagon 编译路径把数据供应当作各算子的局部问题，导致相邻 operator 之间反复 materialize layout、重复从 L2/DDR 读取、产生互相覆盖的 prefetch，并在 HMX/HVX 边界丢失 residency。Alps 建立跨 producer-consumer 的 movement contract：先消除可避免的 movement，再在产生时形成消费布局，让一次必要 movement 服务多个消费者，最后依据 V73 的 single-flight L2、DMA coherency、VTCM 和 VRF 约束隐藏剩余 latency。

三项论文核心贡献可以收敛为：

1. **Movement Region IR/analysis**：统一 identity、layout、lifetime、reuse、first use、ownership 和 physical-byte ledger；
2. **Eliminate and amortize**：producer-side in-situ layout、online/fused consumer、multi-consumer residency；
3. **Hierarchy-aware supply**：在 demand VMEM、page-safe L2 lease、DMA/VTCM 和 HMX/HVX residency 之间选择，并由 PMU/最终目标码闭环验证。

这样 prefetch 仍然是核心，但不再被错误地当作万能答案；in-situ reshape、减少数据搬动、提前搬动、VTCM、DMA、HVX 与 HMX 都自然地属于同一故事。

## 9. 当前决策

1. 保留 `alps-2x-improvement` tag，作为组合策略第一次出现大幅收益的历史锚点；不把它当作当前 true-HVX 论文结果。
2. 历史 15-model Debug 集合保留为候选集，但全部需要在修复后的 baseline 下重新资格审查。
3. 停止把 fresh prefill K/V 的 broad L2 prefetch 默认接入 cumulative policy。
4. 不再新增与 M1–M10、N1–N10 重复命名的独立 ideas。
5. 下一项实现优先选择 **online attention / producer-side direct layout 的 Fusion-preserving Movement Region**，同时建立全模型 movement census。
6. 真正 past-K/V prefetch 单独进入 decode 实验线。
7. 非对齐/批量 MatMul 到 HMX 的通用 lowering 继续作为 **Next Paper Idea**；当前工作只优化已有 HMX region 的数据供应。

最重要的是：接下来每一次声称加速，都必须回答三个问题——**究竟少搬了多少物理字节、究竟隐藏了哪一段不可避免的 latency、这一次 movement 究竟服务了几个消费者**。回答不了这三个问题的倍率，不再进入 Alps 的正式结论。
