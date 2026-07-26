# OmniFetch Analysis & Improvement Roadmap

Status: analysis snapshot (branch `alps_2`). This document captures a full read of the
OmniFetch subsystem (dialect/IR, transform passes, LLVM conversion, device runtime),
the relevant Hexagon SDK / HEXKL primitives, and the benchmark models, plus a
prioritized improvement roadmap. It is the working reference for the OmniFetch effort.

---

## 1. Current design blueprint

OmniFetch is a three-component (Plan-A, independently toggled) prefetch subsystem
inserted **after bufferization and before LLVM lowering**, built around HEXKL's HMX
matmul micro-kernels.

### IR layer (`include/hexagon/Dialect/OmniFetch/IR/OmniFetchOps.td`)
- `omni_fetch.prefetch_in_situ` — core op. Args: `src`, `dest` (memrefs),
  `layout_transform` ∈ {None, HMXWeight, HMXActivation, Custom, L2Hint},
  `lookahead` (default 2), optional `index_map`, variadic `tile_params`.
  - Weight tile_params (3 or 4): `[tile_row, tile_col, src_cols, (weight_off)]`
  - Activation tile_params (6): `[tile_row, tile_col, src_cols, act_off, scr_off, src_rows]`
- `omni_fetch.create_sem` / `signal` / `wait` — V-DAE semaphores.
- `omni_fetch.adaptive_control` — PMU-driven lookahead tuning (see limitation below).

### Pass pipeline (in `lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp` order)
1. `MatmulToHexKL` — `linalg.matmul` → `hexkl.matmul`, **only** static rank-2 with
   M/K/N all %32==0 and **non-attention** (`K==M || N==M` skipped: device exit 13).
2. `DecomposeHexKLMatmul` — triple-nested `scf.for` of MicroHMX ops; allocates VTCM
   with **dual ping-pong weight slots** (`w0`/`w1`) for lookahead.
3. `PrefetchInsert` (Component 1) — recognizes MicroHMX loops; layout-aware phase
   replaces `RmToWh`/`Copy+RmToAh` with `prefetch_in_situ`; `lookahead>=1` emits sync
   current-slot + async next-slot prefetch (software pipeline). Innermost loops only.
4. `LayoutOpsElimination` (Component 2) — deletes layout ops absorbed by in-situ prefetch.
5. `OmniFetchVDAEInsert` (Component 3) — wraps async loops with `create_sem/wait/…/signal`
   (+ optional `adaptive_control`); skips L2Hint and sync-only prefetches.
6. `HexKLToLLVM` + `OmniFetchToLLVM` — lower to extern-C runtime calls.

### Device runtime (`bin/runtime/src/OmniFetchRuntime.c`)
- Staging tiers: DDR → `l2fetch` → L2 → UserDMA 2D → VTCM.
- HMX-native layout via `hexkl_micro_hmx_rm_to_wh/ah_f16`.
- Constraint: must NOT use `qurt.h`, `stdatomic.h`, `hexagon_protos.h`, `assert()`
  (Unsigned PD / DSP symbol limits).

### Pipeline / architecture context
- Two front-ends (Triton via triton-shared; torch-mlir `.mlirbc`) share
  `translate_linalg_to_obj` → `LinalgToLLVMPass` (~50 passes) → LLVM IR (O3) →
  link runtime bitcode → `hexagon-clang++` + `libhexkl_micro.a` → `.so` → `adb` to device
  (`run_main_on_hexagon`) or `hexagon-sim`.
- HEXKL = optional matmul accel (dialect + passes + runtime bridge), gated by `enableHexKL`.
- **Hexagon NN is NOT integrated** — only an external baseline column in plotting/CSV.

---

## 2. Improvement opportunities (by impact)

Key finding: several headline features are currently stubs or degraded.

1. **V-DAE is not true decoupled access-execute.** Runtime is software-pipelined on a
   single HW thread; the semaphore is a `volatile int` spin counter
   (`OmniFetchRuntime.c:86,381`). SDK offers `worker_pool` (up to 12 workers), an
   independent UserDMA engine (DM0 runs parallel to compute), and
   `HAP_compute_res_hmx_lock2/3`. Real fix: a dedicated DMA **scout thread** + compute
   thread synchronized via `memw_locked` HW semaphores.
2. **`adaptive_control` is a no-op** — `__omni_fetch_update_distance` returns its input
   unchanged (`:691-696`), PMU never read. Any "PMU-adaptive lookahead" claim is empty.
   Fix: read AXI-stall via QuRT PMU (`qurt_pmu_get`) and actually adjust distance.
3. **Weights are static but re-laid-out every inference.** Biggest free win: pre-pack
   weights to WH layout **offline** (`sdkl_cpu_rm_to_wh_f16`); runtime does pure DMA only.
4. **VTCM via `memref.alloc`**, not `HAP_compute_res_acquire_cached` (persistent address /
   page layout across calls) nor single-page alloc for DMA stride. A persistent VTCM
   arena removes per-layer alloc/free overhead.
5. **Prefetch only covers the HMX matmul path.** Layout-aware is force-disabled in the
   non-HexKL HVX path (`PrefetchInsertPass.cpp:860-863`); `enableDmaToVtcm` forced off.
   Attention (QK^T/AV), softmax, LayerNorm, conv have no prefetch.
6. **Innermost loops only** (`hasNestedFor` skip) — cannot prefetch the next layer's large
   weights while computing the current layer (the ideal case for LLM inference).
7. **Data-credibility gap.** `plotting/ALPS_Prefetcher_Data.csv` shows ALPS 2–4×, but
   `PHASE4_STATUS_AND_OMNIFETCH.md` measures OmniFetch at **0–4% over HexKL**, and the CSV
   baseline columns exclude HexKL. The CSV appears to be projected/target values and
   contradicts measurement — must reconcile before any experiments.

---

## 3. Non-prefetch NPU acceleration methods (available but unused)

1. **Quantized execution (largest headroom).** All models are fp16; the "quantized"
   script is fake-quant→dequant back to fp16 (no integer exec). HEXKL provides
   `hexkl_micro_hmx_mm_u8i8` (64×32×32, ~2× fp16 throughput) and `u8i4` (int4 weights,
   512B/tile, half the bandwidth).
2. **Multi-threading** (`worker_pool`, up to 12 threads / `num_hvx128_contexts`): fan matmul
   M-row tiles across HW threads (HexKL Micro path is single-thread today).
3. **Operator fusion**: LayerNorm/RMSNorm+matmul; attention QK^T→softmax→AV (attention is
   not even on HMX now) to cut VTCM↔DDR round trips.
4. **KV cache / incremental decode**: all models are `use_cache=False` prefill-only; no
   autoregressive path — mismatched with real on-phone LLM inference.
5. **Weight-compression DMA**: UserDMA supports DLBC compression (`srccomp/dstcomp`) and
   `srcbypass` to skip L2 for large weight streams.
6. **DCVS / HMX time-sharing** (`hmx_lock2/lock4`).
7. **CPU↔NPU pipelining** via `dspqueue` async packet queue (avoid per-layer FastRPC).

---

## 4. Novel points to combine WITH prefetching (paper contributions)

1. **Layout-aware prefetch + offline weight pre-packing** — prefetch becomes pure
   bandwidth movement; lookahead becomes analytically modelable.
2. **Quantization-aware prefetch** — int4 tile = 512B, so VTCM holds 4× more tiles →
   deeper lookahead; jointly optimize prefetch distance with bit-width.
3. **True dual-thread DAE + adaptive scout** — DMA scout thread tunes lookahead from PMU
   AXI-stall feedback (makes §2.2 real; matches the "DAE Scout" naming in the CSV).
4. **Inter-layer prefetch** — exploit Transformer's layer-by-layer predictability to
   prefetch the next layer's weights during the current layer (breaks the innermost-only
   limit); the best-fit prefetch scenario for LLMs.
5. **Elastic VTCM allocation** — per-layer dynamic ping-pong / multi-buffer depth atop a
   persistent `acquire_cached` arena.
6. **Attention-specific in-flight reshape** — pad/reshape HMX-excluded attention shapes
   during prefetch so they become HMX-eligible again.

---

## 5. Baselines for all future experiments

- **Baseline A = Hexagon-MLIR (vanilla HVX):** this repo with HexKL/OmniFetch off
  (`enableHexKL=False`). Runnable in-repo.
- **Baseline B = Hexagon NN library:** NOT integrated in this repo — only an external
  numbers column in `plotting.py`/CSV. Real comparison requires running the same models on
  the Qualcomm Hexagon NN SDK separately; do NOT trust the (apparently projected) CSV values.

Ablation ladder: Hexagon NN | HVX | HVX+HexKL | +Prefetch | +LayoutAware | +V-DAE(real
dual-thread) | +Quantization. **Always list HexKL separately** — measurements show HexKL is
the dominant win (GPT-2 ~2×, tiny LLM 10–20×) while OmniFetch is a small increment.

---

## 6. Model structural-correctness issues

| Issue | Impact |
|---|---|
| **GPT-2 12L device logits all NaN** (all 3 configs) | Only full-depth LLM with complete on-device A/B; results invalid. Likely fp16 saturation (~after 4 layers). Blocks OmniFetch numeric validation. |
| **Qwen/TinyLlama RoPE baked into fixed-position constants** (`_ConstRope`) | Exported model only valid for the exported seq/positions; not general-position-correct RoPE. |
| **Most vision/BERT/CLIP/UNet/VAE use `from_config` random weights** | Validates compiler numerics only, not real model quality. |
| **GELU approximated** (`gelu_new`/`gelu_fast`/tanh; no `math.erf`) | Small numeric divergence vs published models. |
| **All `use_cache=False` (prefill-only)** | No KV cache / autoregressive decode — mismatched with real phone LLM inference. |
| Mamba random weights (vocab 50280 vs 50277); quantized GPT-2 hardcoded 2L fake-quant | Shrunk topology / non-real weights. |
| VAE rewrites GroupNorm and Conv as matmul (claimed equivalent) | Structural substitution; needs numeric verification. |

Structurally faithful: GPT-2 (real weights, but NaN), Falcon, Real-ESRGAN.
Most urgent: GPT-2 NaN (blocking bug); Qwen/TinyLlama fixed-position RoPE (correctness hazard).

---

## Immediate roadmap (agreed order)

1. Implement improvements from §2 and novel points from §4.
2. Benchmark against baselines (§5) to measure any speedup.
3. Then, as warranted, pursue §3 (non-prefetch acceleration).
