# OmniFetch 改进项交接文档 (Handoff)

> 目的：记录"OmniFetch 可改进之处"四项改进(#1/#2/#3/#4)的**当前实现状态、精确代码位置、
> 待做项设计、构建绕过方法、设备验证步骤**，供任何人直接接手。
> 生成时间上下文：device adb `49d1c7b2`，venv `/home/huzq85/2-working/hexagon_npu/mlir-env`。
> 相关既有文档：`benchmark_models/OMNIFETCH_ANALYSIS_AND_ROADMAP.md`、`plan_todo.md`。

---

## 0. TL;DR 当前进度

| 项 | 内容 | 状态 | 位置 |
|----|------|------|------|
| #2 | adaptive_control 真实化 | **已实现(未提交)** | `bin/runtime/src/OmniFetchRuntime.c` |
| #3 | 权重 WH 离线预打包 | **已实现(未提交)** | `lib/Transforms/DecomposeHexKLMatmulPass.cpp` |
| #4 | VTCM 持久 arena | **未做** | `DecomposeHexKLMatmulPass.cpp:189-193/381` |
| #1 | 真双线程 DAE | **未做** | runtime + UserDMA/multithreading |

所有 backend 改动均为**未提交(git M 状态)**，没有丢失。待验证 = 构建 + 设备跑通(无 exit13 /
Bad VA / NaN / 结果损坏 / 无性能回归)。

---

## 1. 关键约束(实现任何一项前必读)

- runtime(`bin/runtime/src/*.c/.cpp`)每个源被 hexagon-clang `-emit-llvm` 编成 `.bc`，再由
  `bitcode2array.py` 转 C++ 字节数组链接进模型 `.so`（见 `bin/runtime/CMakeLists.txt`）。
- **runtime C 代码禁用**：`qurt.h` / `stdatomic.h` / `hexagon_protos.h` / `assert`
  （会引入 `_Assert` 依赖导致链接失败）。跨源全局用 file-static + extern-C 函数暴露。
- 信号量：`volatile int omni_sem_pool[16]`（单 HW 线程，volatile 足够）。
- HexKL 权重 WH 字节格式是**闭源**的；`libsdkl` 只有 ARM/DSP，无 x86。任何 WH 预打包
  **必须在设备上跑真实 `hexkl_micro_hmx_rm_to_wh_f16`**，禁止在 x86 复刻格式。
- 组件门控：`enablePrefetch` 是 #2/#3(部分)/VDAE 的基座；`OmniFetchToLLVM` 在
  `enablePrefetch || enableOmniFetchWeightPrepack` 时运行(`LinalgToLLVMPass.cpp:491-492`)。

---

## 2. #2 adaptive_control 真实化 — 已实现

**思路**：不改 IR 迭代变量，闭环状态放 runtime 全局；`AdaptiveControlOp` 调用点每轮触发真实
更新。用**软件测得的自旋等待数**作为"预取及时性"信号（无 PMU 依赖）。

**已落地代码**（`bin/runtime/src/OmniFetchRuntime.c`）：
- 参数：`MIN_LOOKAHEAD 1` / `MAX_LOOKAHEAD 8` / `STALL_THRESHOLD 8000`（line 45-50）。
- 全局：`omni_stall_accum`(自旋累计)、`omni_stall_events`(wait 次数)、
  `omni_eff_lookahead`(当前有效预取距离，初值 MAX)（line 59-61）。
- `__omni_fetch_wait`：局部 `spins` 累加进 `omni_stall_accum` / `omni_stall_events++`（line 411-412）。
- `__omni_fetch_update_distance`：读并清零累计，算均摊 `avg=accum/events`；
  `avg>STALL_THRESHOLD` → 距离+1（上限 MAX）；`avg*4<STALL_THRESHOLD` → -1（下限 MIN）；
  写回 `omni_eff_lookahead` 并返回（line 711-739）。
- async 预取深度消费 `omni_eff_lookahead`：预热接下来 (omni_eff_lookahead-1) 个 K-tile
  （line 303-305）。

**接线点(未改动，已存在)**：`VDAEDecouplePass.cpp:197-209` 生成 `AdaptiveControlOp`；
`OmniFetchToLLVMPass.cpp` LowerAdaptiveControl → `__omni_fetch_update_distance(i32)->i32`。

**待验证**：`enableOmniFetchAdaptive=True` 编 GPT-2，设备结果与关闭 adaptive 一致（自适应只影响
时序不影响数值），延迟不劣化。

**可选加强(更高风险，未做)**：把距离作为 `scf.for` iter_arg 真正回灌 IR 而非仅 runtime 全局。

---

## 3. #3 权重 WH 离线预打包 — 已实现

**思路**：`rm_to_wh` 只依赖 `(kt,colTile)`，原本嵌在 M×N×K 三重循环里被重算 `ceil(M/32)` 次。
把它提到**每 matmul 一次的 prologue**，预打包所有 tile 到临时 **DDR WH buffer**；内层 K-loop 只
把预打包好的 2048B tile 复制进 VTCM ping-pong 槽。收益随 `numMTiles=ceil(M/32)` 放大（GPT-2
HexKL 强制 seq=32→1 个 M-tile，需在 seq=128/256 才见收益，seq=32 作对照）。

**已落地代码**（`lib/Transforms/DecomposeHexKLMatmulPass.cpp`，`enableWeightPrepack` 门控）：
- prologue 双层 `scf.for(colTile,kt)`：`MicroHMXRmToWhF16Op(whW, byteOff, rhsWork, kt, colTile, N)`
  产出每个 WH tile 到 DDR（line 219-253）。
- 内层 K-loop：用 `omni_fetch.prefetch_in_situ(whTile→vtcm+curW, LayoutTransform::None,
  lookahead=0)` 把预打包 tile 拷进 VTCM 槽（line 310-338），保留 ping-pong `curW` 让 OmniFetch
  仍可重叠。
- `whW` 在 outer M-loop 后 dealloc（line 382-383）。option 关闭时走原路径不变。

**接线**：`hexagon_options.py enableOmniFetchWeightPrepack=False`(默认关)；
`Passes.td` 的 `DecomposeHexKLMatmul` option `enable-weight-prepack`；
`LinalgToLLVMPass.cpp:400 decomposeOptions.enableWeightPrepack=enableOmniFetchWeightPrepack`；
`MLLVMIRTranslation.cpp` 解析转发。

**正确性论证**：同一 HEXKL `rm_to_wh` 产出同样 WH 字节，只是每 tile 只算一次缓存进 DDR 再原样
拷入同一 VTCM 槽 → VTCM 结果字节与逐次路径完全一致。

**待验证**：seq=32(对照,约持平) / seq=128 / seq=256(应随 seq 变快)；同 seq 下 on/off 输出字节一致。

---

## 4. #4 VTCM 持久 arena — 未做(设计)

**现状**：runtime `VtcmPool`(`bin/runtime/src/VTCMPool.cpp:14-56`)已一次性持有全部 VTCM 并用
free-list arena 管理；但 IR 层 `DecomposeHexKLMatmulPass.cpp` 每个 matmul 都
`hexagonmem.alloc`(line 189-193) / `dealloc`(line 381)，产生反复申请/释放 churn。

**低风险方案**：把 VTCM 缓冲 hoist 到 **func 入口**，按 pass 内跨 matmul 的**最大**字节数只分配
一次，func 出口再 dealloc，各 matmul 复用同一 buffer。
- 改点：`DecomposeHexKLMatmulPass.cpp:189-193`(alloc 移到 func 首个 matmul 前，尺寸取 max)、
  `:381`(dealloc 移到 func 末)。
- 若跨 matmul 静态求 max 困难 → 退化"每 func 一个、遇更大再重分配"或先只做"单 matmul 内不重复
  alloc"的保守版。

**待验证**：GPT-2 + `micro_bench` matmul 结果不变，VTCM alloc 次数下降、延迟不劣化。
**注意**：`VtcmPool` 收益可能已被运行时 arena 吸收，预期收益偏低 —— 先量化再决定是否深做。

---

## 5. #1 真双线程 DAE — 未做(设计，最高风险，最后做)

**现状**：async DMA 在 `__omni_fetch_signal → omni_async_complete`(OmniFetchRuntime.c:120-126,
191-196)**同步排空**，非真正双线程；deferred WH 在 signal() 里完成（在 wait() 完成会损坏结果）。

**目标**：DMA scout 线程独立于 compute 线程，用现有 UserDMA(DM0) 引擎 + 硬件信号量真正并行
access/execute。
- 复用 `bin/runtime/UserDMA/`、`bin/runtime/multithreading/` 线程池、`hexagon_runtime_dma_*`。
- 仅在 `enableOmniFetchVDAE=True` 且循环有 async prefetch(lookahead>0) 时启用。
- 设备反复验证无 exit13 / Bad VA / 结果损坏；不稳定即回退到 signal()-drain 语义(flag 默认关)。
先出最小可跑版本，逐步开启，全程 device gate。

---

## 6. 构建（重要：绕过 requirements pin）

`scripts/script_legacy/build_hexagon_mlir_working.sh` 会 `source scripts/script_release/setup/build_triton.sh`，其中
`pip install -r ci/requirements.txt` 死 pin 了 `torch-mlir==20260325.762`（已从 dev-wheels 索引
下架，只剩 20260401+），导致构建在编译后端前中止。**venv 里 torch-mlir/torch/triton 已装好**，
可直接跑后端增量编译，绕过该重装：

```bash
BASE_DIR=/home/huzq85/2-working/hexagon_npu
export HOST_TOOLCHAIN=$BASE_DIR/HOST_TOOLCHAIN
export PATH="$HOST_TOOLCHAIN/bin:$PATH"
export CC="$HOST_TOOLCHAIN/bin/clang" CXX="$HOST_TOOLCHAIN/bin/clang++"
export HEXAGON_SDK_ROOT=$BASE_DIR/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/
export HEXAGON_TOOLS=$BASE_DIR/HEXAGON_TOOLS/Tools
export HEXKL_ROOT=$BASE_DIR/HEXKL_DIR/hexkl_addon
export LLVM_PROJECT_BUILD_DIR=$BASE_DIR/LLVM_DIR/llvm-project/build
source $BASE_DIR/mlir-env/bin/activate
export HEXAGON_MLIR_ROOT=$BASE_DIR/hexagon-mlir
export HEXAGON_ARCH_VERSION=75
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
cd "$TRITON_ROOT"
TRITON_BUILD_WITH_CLANG_LLD=1 TRITON_BUILD_WITH_CCACHE=true \
LLVM_INCLUDE_DIRS="$LLVM_PROJECT_BUILD_DIR/include" \
LLVM_LIBRARY_DIR="$LLVM_PROJECT_BUILD_DIR/lib" \
LLVM_SYSPATH="$LLVM_PROJECT_BUILD_DIR" \
pip install -e . --no-build-isolation --verbose
```

ninja 会因 `OmniFetchRuntime.c` 变更重新生成 `.bc` → 字节数组（#2/#3 runtime 改动被吸收），
C++ pass 走主 ninja。
**永久修复(可选)**：把 `ci/torch-requirements.txt` 里的 `torch-mlir==20260325.762` 放宽为
`>=20260401` 或最新可用版本。

---

## 7. 设备验证与门槛

1. 编 + 跑 GPT-2：`benchmark_models/run_gpt2lmheadmodel.py`（device 49d1c7b2, venv mlir-env）。
2. 门槛：Pass 成功 + prefetch inserts>0 + 设备无 exit13/Bad VA/NaN + 数值/字节与基线一致 +
   延迟不回归。`micro_bench` matmul 作 HexKL 数值对照。
3. 基线：A=HVX(`enableHexKL=False`)；B=Hexagon NN 库(仓库外，另跑，CSV 列)。
4. 已知：OmniFetch 相对 HexKL 实测仅 ~0–4%；HexKL 才是主要收益(2–24×)。见 `plan_todo.md` Results。

---

## 8. 相关文件索引

- 选项：`qcom_hexagon_backend/backend/hexagon_options.py`
- 流水线接线：`qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`
- 预取插入：`qcom_hexagon_backend/lib/Transforms/PrefetchInsertPass.cpp`
  （注意 line ~863 `doLayoutAware=false` 强制 HVX 线性预取；~1010 只处理最内层循环）
- V-DAE：`qcom_hexagon_backend/lib/Transforms/VDAEDecouplePass.cpp`
- HexKL 分解：`qcom_hexagon_backend/lib/Transforms/DecomposeHexKLMatmulPass.cpp`
- OmniFetch 降级：`qcom_hexagon_backend/lib/Conversion/OmniFetchToLLVM/OmniFetchToLLVMPass.cpp`
- 外部函数名：`qcom_hexagon_backend/include/hexagon/Conversion/OmniFetchToLLVM/OmniFetchExternalFnNames.h`
- Op 定义：`qcom_hexagon_backend/include/hexagon/Dialect/OmniFetch/IR/OmniFetchOps.td`
- runtime：`qcom_hexagon_backend/bin/runtime/src/OmniFetchRuntime.c`、`VTCMPool.cpp`、`UserDMA/`
- Pass options td：`include/hexagon/Transforms/Passes.td`、
  `include/hexagon/Conversion/LinalgToLLVM/Passes.td`
