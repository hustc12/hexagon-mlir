# ALPS 工程交接说明

本文档记录截至 2026-09-05 的 ALPS 设计边界、代码状态、有效与无效的
实验结论，以及从干净 shell 复现实验所需的步骤。它描述的是
`alps_vdae_runtime_control` 分支，不应把历史目录中的 OmniFetch 名称或早期
Debug/GEMM 实验当作当前实现。

## 1. 当前版本

- Repository：`git@github.com:hustc12/hexagon-mlir.git`
- Branch：`alps_vdae_runtime_control`
- 本文档之前的功能提交：`1d870d1` (`Panelize residual V-DAE weight access`)
- 上游：`git@github.com:qualcomm/hexagon-mlir.git`
- 真机：OnePlus CPH2449 / Snapdragon 8 Gen 2 (`kalama`)，Hexagon v73 CDSP
- 主要精度：完整模型 FP16；不把混合精度或量化作为 ALPS 贡献
- 执行原则：模型之间严格串行；单个模型的编译可以并行；无 timeout；失败
  后不自动重试，尤其是 exit 13 必须先检查生成物、传输和设备日志。

当前 release 主实验仍使用冻结的 `alps-final` endpoint。最新加入的
panelized exact-weight V-DAE 是独立实验路径，通过
`alps-vdae-full-e`/`alps-vdae-full-e-traffic-control` 开启；它尚未进入
`alps-final`，原因见第 5 节。

### 1.1 项目目标

ALPS 的最终目标是在不依赖闭源 QNN graph optimizer、也不改变模型语义的前提
下，加速模型列表中完整、非 Debug 模型在 Hexagon NPU 上的端到端推理。优化
对象不是某一个算子或 GEMM microbenchmark，而是模型执行过程中由表示不匹配
造成的数据物化、layout conversion、跨 memory tier 搬运和不可隐藏的访存等待。

预期的统一方法是：由 consumer 反推并直接形成其所需的物理表示；使用 V-DAE
将确实无法消除的搬运与 HVX/HMX 计算解耦，并通过 VTCM queue 提前供给；runtime
再根据真实 traffic/stall evidence 拒绝不合算的异步搬运。换言之，ALPS 的目标
不是增加 prefetch 数量，而是减少需要搬的数据，并只提前搬运剩余且可被隐藏的
数据。

工程和论文的成功标准是：

- 以完整 FP16 模型而非 Debug 层或 GEMM 为评价对象；
- 主要比较 Prefetch-Kernel-HX、APT-GET-HX、原生 Hexagon-MLIR 和 ALPS，硬件
  归因时区分 scalar、HVX 和 HMX/HexKL lowering；
- 在真实 v73 手机上获得稳定、correctness-qualified 的端到端结果，并用
  materialization ledger、UserDMA telemetry、LWP 与 sysMon traffic 解释来源；
- 目标是模型列表中的每个完整模型至少达到 1.8x，代表模型达到 3x 以上，同时
  不能通过删除弱模型或隐藏负结果达成；
- 设计应能平滑移植到 v75/v79 等相近 Hexagon revision，不侵入式修改上游
  Hexagon-MLIR，且所有组件可独立开关以支持 matched ablation。

截至当前版本，这一目标只部分达成：15 模型整体 geometric mean 为 1.67x，
7 个模型达到至少 1.80x，DINOv2 和 DeiT 超过 3x；三个完整 speech encoder 和
部分 language model 尚未达到目标。主收益已被定位到 consumer-driven
formation，而 V-DAE residual supply 和 runtime admission 仍需要证明独立的正
收益。这一差距正是后续工作的范围，不能在交接或论文中写成“所有模型均已达到
1.8x”。

## 2. 对齐后的三层设计

三层共同解决一个问题：避免先生成错误表示、再搬运和转换；不可消除的搬运
才由解耦 Access 流提前完成；当提前搬运会争用资源时，runtime 应拒绝它。
三层不是三个互不相关的优化开关。

### 2.1 Vectorized Decoupled Access--Execute（V-DAE）骨架

V-DAE 将地址生成、descriptor 管理、DMA 发起和 layout formation 放入
scalar/scout Access stream，将实际计算留在 HVX/HMX Execute stream。两者通过
有界 VTCM tile/panel queue 以及精确的 ready/free token 解耦：

```text
producer/source representation
        |
        v
scalar/scout Access: address + DMA + formation
        |
        v
bounded VTCM queue: FREE -> LOAD_PENDING -> LAYOUT_PENDING -> READY
        |
        v
HVX/HMX Execute: consume -> release -> FREE
```

正确性约束是 `(invocation, generation, value version, tile/panel, layout,
source tier, destination tier)` 全部匹配后才允许消费。Access 可以领先 Execute，
但不得用 process-global FIFO 猜测 tile ownership，也不得覆盖未释放的 VTCM
slot。

主要代码：

- `qcom_hexagon_backend/include/hexagon/Dialect/Alps/IR/AlpsOps.td`：descriptor、
  invocation、exact-weight kick/consume/release 等 ALPS op；
- `qcom_hexagon_backend/lib/Conversion/AlpsToLLVM/AlpsToLLVMPass.cpp`：runtime ABI
  lowering；
- `qcom_hexagon_backend/bin/runtime/src/AlpsRuntime.c`：descriptor state machine、
  scout、UserDMA、VTCM queue、计数器和 traffic actuator；
- `qcom_hexagon_backend/lib/Transforms/PrefetchInsertPass.cpp`：Access/Execute
  schedule、lookahead、panel boundary 和静态 amortization gate；
- `qcom_hexagon_backend/lib/Transforms/DecomposeHexKLMatmulPass.cpp`：HMX tile、
  VTCM arena/lifetime coloring 和 panel slot 映射。

最新 exact-weight 实现把一个 2 KiB useful-data micro-tile descriptor 合并成
4/8-tile panel。一个 descriptor 发起一次 2-D DMA、形成整个 WH panel，并只
发布一个 token。编译器为它保留
`(lookahead + 1) * panel_tiles` 个物理 VTCM tile，并拒绝无法摊销 descriptor
开销的短循环。`ALPS_VDAE_PANEL_TILES` 可取 1--8；1 保留旧路径，正式 panel
实验使用 8。

### 2.2 Prefetching in-situ transformation

这一层把 consumer-driven representation formation 和 residual asynchronous
movement 视为同一个 supply 问题，而不是先做 transpose、再对 transpose 结果
做普通 prefetch。

编译器首先从 consumer 反推其 layout contract。若 producer 可以直接形成该
表示，就在 producer/consumer 边界吸收 transpose、reshape、patch/token
reorder、attention destination formation 或 HMX output formation。只有不能
消除但又处在关键路径上的 residual movement，才交给 V-DAE 提前搬入 VTCM。
因此优先级固定为：

1. 消除搬运并直接形成 consumer-ready representation；
2. 对不可消除的 residual representation 做异步 supply；
3. 若没有足够 Execute slack 或会增加流量，则保持 native demand path。

当前 full-E 包含：generic P2e consumer formation、P2g continuity/register-tile
支持、P5h attention destination formation、P5i patch/token formation，以及
P5j/P5k HMX output formation。它们有 legality/topology gate，不按模型名称
启停。P5n 是 residual HMX result evacuation 的 VTCM ping-pong/UserDMA 路径。

历史脚本仍使用 C/E/P/R 缩写。论文和对外说明不应只写字母：

- C 是 native-width HVX widening-convolution 的工程支持，不是论文独立贡献；
- E 是上述 consumer-driven in-situ representation formation；
- P 是不可消除 residual movement 的异步 supply；
- R 是下一小节的 runtime traffic admission。

### 2.3 Runtime traffic control/admission

Admission 的作用不是证明“prefetch 越多越好”，而是在 runtime evidence 表明
Access 跟不上 Execute 或造成资源争用时，减少/关闭异步请求并回到 native
path。

当前实现分为两个时间尺度：

- invocation 内部，P4A 使用 DMA completion、poll retry、late-completion window
  和周期性 probe 做真实闭环，actuator 会 suppress 或重新开放 DMA；
- invocation 之间，`derive_alps_traffic_policy.py` 可以把 matched latency、LWP
  和 sysMon 摘要生成 versioned policy，下一次 compilation/invocation 按
  `residual_vdae_admitted` 决定是否生成 descriptor machinery。

必须明确：当前设备上进程内可用 PMU read 为 0；sysMon 是设备外的系统级采样
工具，不能在同一次 kernel 内直接回调 ALPS runtime。因而当前 runtime 快速环
依赖 DMA/poll telemetry，sysMon 用于跨 invocation 的 policy。不能在论文中将
它描述成“sysMon 每个 tile 实时控制 prefetch”。116-event sysMon 数据是多轮
multiplexed/rotating 采样，不是 116 个事件在一次运行中同时读取。

## 3. 当前代码路径和开关状态

### 3.1 对外冻结路径

`scripts/script_release/internal/run_full_hvx_five_way.sh` 中的 `alps-final` 是
当前 15 模型、ablation、movement audit 的冻结 endpoint。它包含经过全模型
验证的 consumer formation 和现有 residual drain/traffic machinery，但不启用
最新 exact-weight V-DAE。

这一区分很重要：主表中 ALPS 的 1.67x overall geometric mean 不能归因于最新
exact-weight panel V-DAE。现有 ablation 表明主要收益来自 full consumer
formation；P 只有部分模型有小收益，R 在五模型正式 ablation 中没有触发有效
的 throttle/suppression 决策。

### 3.2 exact-weight V-DAE 实验路径

以下两个 scheme 只用于研究 V-DAE residual contract：

- `alps-vdae-full-e`：formation + exact-weight V-DAE；
- `alps-vdae-full-e-traffic-control`：上述路径 + P4A runtime admission。

相关非公开实验环境变量：

```bash
ALPS_VDAE_PANEL_TILES=8
ALPS_VDAE_TREATMENT_ONLY=1
ALPS_VDAE_UNREGULATED_ONLY=1   # 只跑未调控 V-DAE
ALPS_VDAE_REGULATED_ONLY=1     # 只跑 V-DAE + traffic admission
```

`TREATMENT_ONLY`、`UNREGULATED_ONLY`、`REGULATED_ONLY` 只减少实验矩阵，不改变
compiler mechanism。panel size 通过 Python backend option、MLIR pass option、
ALPS op 和 runtime ABI 全链路传递。

### 3.3 尚未成为正收益组件的部分

- Exact-weight V-DAE：机制正确且 panelization 大幅消除了开销，但 DINO/BEiT
  上仍未超过 formation-only；当前应由 admission 拒绝。
- Runtime R：actuator 有实际 suppress/probe 行为，但主 ablation 中没有独立
  latency 收益；不得把自然波动写成 R 的贡献。
- Speech HuBERT/Wav2Vec2/UniSpeech：完整模型只提升约 1.01--1.02x，说明其
  主要 critical path 没有被现有 formation contract 覆盖。
- V75/V79：只有 hexagon-sim relative counter 和完整图 compile/link 证据，
  不能声称为真实设备 latency。

## 4. 已完成的正式实验

### 4.1 15 个完整模型 end-to-end

冻结 15 模型包含 6 个 Vision、4 个 Speech 和 5 个 Language/text 模型。最新
consumer-contract-admitted 两列重跑全部 correctness PASS：

- 全 15 模型 geometric mean：1.67x（Hexagon-MLIR HexKL pipeline / ALPS）；
- Vision：2.43x；Language/text：1.41x；Speech：1.18x；
- 9 个模型达到至少 1.50x，7 个至少 1.80x；
- DeiT 3.20x，DINOv2 3.25x；
- HuBERT、Wav2Vec2、UniSpeech 是保留的弱/负证据，不应删除。

完整五列汇总在 `docs/alps/experimental_data.md` 的
“Consolidated complete-model end-to-end table (2026-09-02)”以及
`docs/alps/data/alps_end2end_data.csv`。该五列表跨实验 generation 合并；PK/APT、
HexKL-Off 与 2026-09-01 两列并非同一次 replay，引用时必须保留此说明。

### 4.2 五模型 ablation

模型为 DINOv2-small、Swin Transformer、SegFormer MiT-B0、DeiT-Small 和
Whisper-Tiny。rename 后的完整 A0--A4 重跑与 2026-09-01 matched control 接近，
25 个 case 全部 correctness PASS，最终 endpoint 差异为 -0.11% 到 +2.12%。

主要因果结论：

- full consumer formation 是主收益：DINO 2.99x、DeiT 2.87x、Whisper 1.61x、
  Swin 1.53x、SegFormer 1.37x（相对上一阶段）；
- residual async P 在 DINO/DeiT 为 1.08x/1.12x，在其余模型约 1.00--1.01x；
- R 未产生 causal performance claim。

### 4.3 movement、materialization 和 sysMon audit

五模型 matched audit 中，A1 到 A2 的静态 materialization reduction 为：
DINO 32,378,880 B、Swin 2,033,664 B、SegFormer 0 B、DeiT 18,302,976 B、
Whisper 315,592,704 B。P2e/P5h/P5i 的 realizer estimate 会互相重叠，绝不能
与 ledger delta 相加。

对应的 A1 到 A3 sysMon AXI reduction 为 DINO 25.50%、Swin 10.34%、SegFormer
42.06%、DeiT 22.75%、Whisper 23.61%。全 15 模型 audit 也保存在
`docs/alps/experimental_data.md`；其中 Qwen、TinyLlama 和 SmolLM2 的 AXI
traffic 曾上升，这正是 topology/runtime admission 的必要性证据。

### 4.4 V75/V79 portability

Hexagon Tools 19.0.02 的 `v75na_1` 和 `v79na_1` functional simulation 上，
DINO proxy 的 HexKL-Off/ALPS relative ratio 都为 4.25x，Swin proxy 都为
4.02x；所有 proxy correctness PASS。完整 DINO 12-block 和完整 Swin
`[2,2,6,2]` 图在两个 target 上均完成 lowering/codegen/link。

`Kernel PerfP50` 只是 simulator counter，frequency/bus penalty 未按真机校准。
这些结果只支持跨 ISA revision 的适用性和趋势，不支持 V75/V79 latency 声明。

## 5. 最新 panelized V-DAE 结果与判断

### 5.1 DINOv2-small

| 配置 | Latency | Descriptor/kick | Ready before demand | Execute wait cycles |
|---|---:|---:|---:|---:|
| Formation only | 3,391.19 ms | 0 | -- | -- |
| 原始 2 KiB V-DAE | 4,921.67 ms | 163,296 | 约 12% | 约 3.15B |
| 8-tile panel V-DAE | 3,582.52 ms | 5,184 | 49.42% | 649,159,789 |

Panel 将 descriptor 减少 96.83%，相对旧 V-DAE 降低 latency 37.38%，但仍比
formation-only 慢 5.64%。DINO 的 dominant movement 已被 formation 消除，
剩余 exact-weight stream 不足以偿还 Access 固定成本。

### 5.2 BEiT-Base

| 配置 | Latency | Scheduled descriptors | 关键结果 |
|---|---:|---:|---|
| Formation only | 9,218.58 ms | 0 | matched control |
| 原始 2 KiB V-DAE | 21,270.46 ms | 544,320 | ready 2.44%，wait 23.71B cycles |
| panel8 + 256-entry cache | 11,681.02 ms | 36,288 | cache working set 不足 |
| panel8 + 512-entry cache | 9,766.29 ms | 36,288 | cache hit 66.06% |
| panel8 + 512 entries + admission | 9,377.41 ms | 36,288 | suppress 30,119；wait 1.16B cycles |

最终路径比旧 V-DAE 快 2.27x，也比未调控的 panel path 快 3.98%，但仍比
formation-only 慢 1.72%。因此不能宣称 exact-weight V-DAE 已转正。正确处理是
让下一 invocation/compile 的 contract admission 拒绝它，而不是继续调阈值把
自然波动包装成收益。

512-entry formed-panel cache 位于 DDR，大小上限为 16 MiB，不是 VTCM，也不是
免费资源。若以后保留，论文必须计算这部分 memory overhead。

权威工程记录在 `docs/alps/vectorized_dae_engineering_plan.md` 第 9--11 节。
有效的新实验归档为：

```text
nano:/home/huzq85/2-working/working_set/alps_vdae_panel8_dino_20260904
nano:/home/huzq85/2-working/working_set/alps_vdae_panel8_beit_20260904
nano:/home/huzq85/2-working/working_set/alps_vdae_panel8_cache512_beit_20260904
nano:/home/huzq85/2-working/working_set/alps_vdae_panel8_cache512_p4a_beit_20260904
```

名称含 `panel4_dino` 或 `panel4_dino_fixed` 的早期目录没有把 panel option
完整传入 compiler，实际仍是旧 per-tile 路径，不得作为 panel 结果引用。

## 6. 环境准备

项目不再要求修改 `~/.bashrc`。从干净 shell 执行：

```bash
cd /home/huzq85/2-working/hexagon_npu/hexagon-mlir
source scripts/script_release/setup/set_local_env.sh
```

脚本会检查并设置 repository-local Triton/plugin、`../LLVM_DIR`、
`../HEXAGON_SDK`、`../HEXAGON_TOOLS`、`../HEXKL_DIR`、`../HOST_TOOLCHAIN` 和
`../mlir-env`。默认 target 是 v73。

开始真机实验前至少确认：

```bash
adb devices
ssh nano true
scripts/script_release/internal/prepare_phone_benchmark.sh apply
```

手机应关闭 battery saver，保持供电和散热条件稳定。不要用一次 QNN warm-up
来改变 ALPS 的定义；若使用它只能作为明确记录的设备状态准备步骤，并对所有
比较列一致执行。

## 7. 编译和测试

### 7.1 只编译一次 toolchain/runtime

```bash
./run_alps.sh --build-only
```

也可以直接使用增量构建脚本：

```bash
bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh --arch 73
```

`ALPS_BUILD_JOBS=N` 或 `--jobs N` 只控制单模型/单 toolchain 的编译并行度；
绝不能并行运行多个模型。资源紧张时建议从 4--8 jobs 开始。

### 7.2 一键正式实验

```bash
./run_alps.sh --end-to-end   # 15 个完整模型、五配置
./run_alps.sh --ablation     # 五个完整模型 A0--A4
./run_alps.sh --movement     # 复用 end-to-end 数据生成 movement/traffic audit
./run_alps.sh --portability  # V75/V79 proxy + 完整图 compile/link
```

组合执行：

```bash
./run_alps.sh -e -a
./run_alps.sh --all
```

无参数 `./run_alps.sh` 会先警告完整流程可能运行很多小时，并等待 Yes；
`--all` 不询问。所有模型和配置串行、无 timeout、失败不自动重试。

默认本地恢复目录为 `/tmp/alps_reproduce_<git-sha>`，归档目录为：

```text
nano:/home/huzq85/2-working/working_set/alps_reproduce_<git-sha>
```

同一 commit 使用相同 `ALPS_RUN_ID` 可恢复 PASS case。一个模型完成并同步到
nano 后应及时释放本地大型生成物；compact CSV/log 可以保留。不要把编译生成
物提交到 Git。

### 7.3 targeted compiler regression

```bash
source scripts/script_release/setup/set_local_env.sh --quiet
FC="$LLVM_PROJECT_BUILD_DIR/bin/FileCheck"

linalg-hexagon-opt \
  qcom_hexagon_backend/test/Transforms/alps-exact-overlap.mlir \
  -pass-pipeline='builtin.module(func.func(alps-minimal-static-admission{min-dma-bytes=2048 min-overlap-ops=2 enable-p3-exact-readiness=true},prefetch-insert{lookahead=2 enable-layout-aware=true enable-two-dim-pipeline=true enable-alps-exact-overlap=true exact-weight-panel-tiles=4},alps-exact-readiness))' \
  2>&1 | "$FC" \
  qcom_hexagon_backend/test/Transforms/alps-exact-overlap.mlir \
  --check-prefix=PANEL

linalg-hexagon-opt \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/alps-exact-readiness.mlir \
  -pass-pipeline='builtin.module(alps-to-llvm)' | "$FC" \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/alps-exact-readiness.mlir

python3 -m py_compile \
  benchmark_models/hexkl_utils.py \
  scripts/script_release/internal/layered_hvx_options.py
```

完整 lit 测试可通过下面命令运行；在受限环境中 ccache 必须使用可写目录，
release build script 已将其放到 `/tmp`：

```bash
bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh --tests
```

### 7.4 复现最新 V-DAE treatment

DINO panel8：

```bash
ALPS_VDAE_PANEL_TILES=8 \
ALPS_VDAE_TREATMENT_ONLY=1 \
ALPS_VDAE_UNREGULATED_ONLY=1 \
bash scripts/script_release/internal/run_full_hvx_five_way.sh \
  --alps-vdae-full-e --device-iterations 1 \
  --output-dir /tmp/alps_vdae_panel8_dino_recheck \
  --remote-dir /home/huzq85/2-working/working_set/alps_vdae_panel8_dino_recheck \
  dinov2-small
```

BEiT panel8 + runtime admission：

```bash
ALPS_VDAE_PANEL_TILES=8 \
ALPS_VDAE_TREATMENT_ONLY=1 \
ALPS_VDAE_REGULATED_ONLY=1 \
bash scripts/script_release/internal/run_full_hvx_five_way.sh \
  --alps-vdae-full-e --device-iterations 1 \
  --output-dir /tmp/alps_vdae_panel8_p4a_beit_recheck \
  --remote-dir /home/huzq85/2-working/working_set/alps_vdae_panel8_p4a_beit_recheck \
  beit-base
```

`results.csv` 是 aggregate latency 权威来源。layered model 的 `run.log` 会包含
多个 `Perf:`，不能拿其中一个 stage 代替完整模型 latency。正确性至少检查
finite、误差门槛和 top-1/model-specific gate。

## 8. 结果解释纪律

1. `HMLIR HVX (HexKL On)` 是启用 HexKL lowering pipeline 的 matched control，
   不等同于证明每个 matmul 都在 HMX；必须结合 rewrite/codegen 计数判断。
2. 对外论文正文统一称 `Hexagon-MLIR` baseline，不使用 HexKL On/Off 作为主要
   算法名称；On/Off 只在工程归因中保留。
3. `runtime_issued_bytes` 是 ALPS/UserDMA 计数；sysMon AXI bytes 是系统级物理
   traffic；static materialization ledger 是编译期估计，三者不可混用。
4. sysMon kernel-host window 包含 launcher/sampling overhead，不等同于
   `Perf:` latency。
5. simulator PerfP50 不是毫秒，也不能跨未校准 core 解释绝对速度。
6. 相邻 ablation speedup 使用 `A(i-1)/Ai`，end-to-end speedup 使用
   `baseline/ALPS`；ALPS 固定为 1.00x，lower latency is better。
7. 负结果必须保留。尤其不能把 exact-weight V-DAE、R 或三个 speech encoder
   写成已有显著正收益。

## 9. 建议的下一步

不要继续对 DINO/BEiT exact-weight panel size 或 P4A threshold 做无界调参。
这两个模型已给出明确 break-even 结论。下一步应按以下顺序：

1. 将 exact-weight residual contract 的静态/跨 invocation rejection 固化，
   确保被拒绝时不生成 descriptor，性能回到 matched formation native path；
2. 以此前曾产生约 6.46% 收益的 BEiT residual HMX output drain 为候选，把它
   明确表达为 V-DAE 的 producer/consumer、VTCM queue 和 ready/free contract；
3. 只在该 residual movement 的 native critical-path cost 大于 Access 固定开销
   且存在 independent Execute slack 时 admission；
4. 用 formation-only、formation + residual V-DAE、再加 runtime admission 的
   三列 matched complete-model 实验判断是否真正转正；
5. 若仍不能超过 formation-only，就把 V-DAE 保留为被 admission 管理的框架，
   不在当前论文中声称它有独立 latency contribution。

这一方向没有偏离论文：它仍是 consumer-ready formation、不可消除 residual
movement 的 asynchronous prefetch/supply，以及 runtime traffic admission。
不应通过混合精度、模型名特判或无关算术优化人为制造 V-DAE 收益。

## 10. 主要文档索引

- `docs/alps/bottleneck_analysis.md`：P0--P6、15 模型 LWP/sysMon、formation、
  HMX drain 和历史 gate 的完整工程记录；
- `docs/alps/vectorized_dae_engineering_plan.md`：V-DAE exact readiness、runtime
  control、panel redesign 和最新 break-even 结论；
- `docs/alps/experimental_data.md`：所有正式 end-to-end、ablation、movement、
  sysMon 和 portability 数据；
- `docs/alps/data/alps_end2end_data.csv`：15 模型 end-to-end 机器可读表；
- `scripts/script_release/README.md`：对外一键复现入口；
- `docs/user-guide.md`：Hexagon-MLIR/SDK 基础构建要求；
- `archive/` 和 `scripts/script_legacy/`：历史开发资料，只用于追溯，不是 release
  入口。
