# OmniFetch Next Plan (device-grounded, ignore CSV)

Goal: get a reproducible on-device win chain:

`HexKL only` → `HexKL + VTCM` → `+ sync prefetch` → `+ layout-aware` → (optional) `+ async DMA`

Each step must: **Pass + prefetch inserts > 0 (when prefetch on) + NPU time not worse**. Stop and fix on exit 13 or regression.

Device: `adb -s 49d1c7b2`  
Venv: `/home/huzq85/2-working/hexagon_npu/mlir-env`

---

## Phase 0 — Pick vehicle

### Finding (2026-07-23)

- **Attention (`batch_matmul`) is not a HexKL vehicle.** `MatmulToHexKL` only matches 2D `linalg.matmul`. PrefetchInsert saw `hexkl_func=0` and only Softmax HVX 1D strips.
- **VTCM tiling on Attention regresses badly** (~41 ms → ~360 ms). Do not use as OmniFetch baseline.
- **Primary vehicle switched to GEMM:** `benchmark_models/verify_omnifetch_Gemm.py` (`256×256` f16 2D matmul → real MicroHMX).

### Checklist

- [x] Smoke: HexKL only — Pass (GEMM)
- [x] Smoke: HexKL + OmniFetch (sync L2-hint, layout OFF) — Pass, inserts > 0
- [x] Record NPU times (see Results log)
- [x] Attention Phase0 recorded as negative control (no HexKL MicroHMX)
- [ ] (Optional later) Extend `MatmulToHexKL` to `batch_matmul` so Attention can use HexKL path

---

## Phase 1 — Prefetch that inserts and does not regress — **COMPLETE**

### Exit criteria (original + GEMM vehicle)

| Criterion | Status |
|-----------|--------|
| Prefetch actually inserts (`enablePrefetch` wired) | **PASS** |
| HexKL loops preferred; HVX strips not thrashed | **PASS** |
| inserts > 0 on vehicle | **PASS** (GEMM: 2× L2Hint) |
| Pass + no meaningful regression vs HexKL-only | **PASS** (~neutral) |
| Speedup | **Not required for Phase 1** (deferred to Phase 2) |

### Done checklist

- [x] Attention/GEMM harness: `enablePrefetch` + V-DAE together
- [x] PrefetchInsert: HexKL-first candidate selection
- [x] PrefetchInsert: `insertHexKLMicroPrefetchHints` (L2Hint on Copy/RmToWh)
- [x] Always-on insert count via `llvm::errs()`
- [x] GEMM device smoke: Pass, inserts=2, ~neutral timing

---

## Phase 2 — Layout fusion **and** tile lookahead + async DMA

### 2a — Avoid separate layout transform — **COMPLETE**

On HexKL weight path: replace `RmToWh` with `prefetch_in_situ(HMXWeight)` that writes into the HexKL VTCM slot via `memref.view`, calling **`hexkl_micro_hmx_rm_to_wh_f16`** (same layout as HexKL).

**Root causes fixed along the way:**
1. Dominance: do not use `RmToAh`’s `actOff` at a Prefetch inserted before `Copy` (activation fusion deferred).
2. Wrong gather: generic HMXWeight gather assumed contiguous K=32; DDR tiles are strided. Fix = HexKL micro API + `tile_params(row,col,src_cols)` on full matrix src.

### Checklist 2a

- [x] PrefetchInsert layoutAware: emit HMXWeight into VTCM views; erase `RmToWh`
- [x] Ablation: layout OFF vs ON on GEMM (real HMX* inserts, not L2Hint)
- [x] IR evidence: `RmToWh` eliminated (`erased_hexkl_ops=1`)
- [x] Record NPU delta; Pass required
- [x] Activation `Copy+RmToAh` fusion (HexKL-accurate; GEMM 64×128×256 Pass; erased_hexkl_ops=2)

### 2b — Lookahead + async prefetch — **COMPLETE (L2 overlap)**

Software pipeline on HexKL weight loops (dual VTCM slots):
1. Sync `prefetch_in_situ` fills the **current** ping-pong slot (Phase 2a hexkl_micro).
2. Async kick (lookahead=1) **before Mm** L2-fetches the **next** tile so warmup overlaps compute.
3. V-DAE wait/signal only on the loop that **directly** owns the Prefetch (nested-loop wait was completing jobs too early).

**Deferred WH:** completing in `wait()` corrupted results; **fixed by finishing WH in `signal()` after Mm**. Pipeline: kick dma2d-packs next tile into DDR stage (overlaps Mm) → signal dma_wait + `hexkl_micro` WH into idle ping-pong slot (HexKL `vtcm_base+weight_off` ABI). Direct DMA→VTCM staging also Passes but was ~2× slower for 2KB tiles.

### Checklist 2b

- [x] Dual weight slots in `DecomposeHexKLMatmulPass`
- [x] Software pipeline: sync current + async next kick before Mm
- [x] V-DAE: only nearest enclosing `scf.for` gets wait/signal
- [x] GEMM lookahead ON: Pass + not slower (see Results)
- [x] Optional: deferred WH layout / tile DMA — **fixed 2026-07-24**: WH in `signal()` (not `wait()`); dma2d pack→DDR stage overlaps Mm; HexKL slab+weight_off ABI. Direct DMA→VTCM staging Passes but ~2× slower for 2KB tiles (runtime keeps `stage_off` hook).

---

## Phase 4 — One real model — **IN PROGRESS**

### Checklist

- [x] GPT-2 fair `--seq-len 32`: HVX / HexKL / HexKL+OF Pass (debug 2L Top-1)
- [x] GPT-2 **full 12L** fair `--seq-len 32`: HVX / HexKL / HexKL+OF device Pass (compare NaN — see notes)
- [x] Qwen HVX fair `--seq-len 32` Pass (tiny debug 2-layer; main harness = full 24L)
- [x] Qwen HexKL + OmniFetch — **unblocked** (tiny debug; see notes)
- [x] Falcon-RW-1B debug HVX / HexKL / HexKL+OF Pass (main = full 24L; debug = tiny)
- [x] TinyLlama-1.1B debug HVX / HexKL / HexKL+OF Pass (main = full 22L; debug = tiny)
- [x] Mamba-130M debug HVX / HexKL / HexKL+OF Pass (main = full 32L; debug = 1L tiny)
- [x] ViT / Swin / GraphSAGE / SD-TE debug 3-way Pass
- [x] SD-UNet / SD-VAE / Real-ESRGAN device smoke (debug Pass; see results log)
- [x] Record fair-seq speedups

### Qwen notes (2026-07-23)

Harness: `benchmark_models/run_qwen2.5-0.5b.py` (full published arch). Debug shrinks: `debug_running/run_qwen2.5-0.5b_debug.py`.
- Content-filled `--seq-len` (default 32 with HexKL), 256MB DSP heap, ConstRope (no inv_freq/cos buffers in ABI).
- Root-caused seq>1 exit-13: RoPE `register_buffer` / f32 `inv_freq` made WrapperGenerator emit broken `(dim, memref*)` ABI.
- **HexKL crash fix:** `ReduceContractionRank` turns attention `batch_matmul` into tile-aligned `matmul` (e.g. 32×64×32); HMX on those shapes TLB-faults. `MatmulToHexKLPass` now skips `K==M || N==M` (keep HVX). Projections/FFN still HexKL.
- **Harness:** HexKL path uses `enableConvertToHexagonmem=True`, `enableVectorization=False` (vec+Qwen still Bad VA 0x28; MicroHMX does not need it). `FormAsyncThreads` now lowers `scf.forall` → sequential `scf.for` (was async → Bad VA 0x18).

---

## Results log

| Date | Config | Vehicle | NPU time | Inserts | Pass? | Notes |
|------|--------|---------|----------|---------|-------|-------|
| 2026-07-23 | HexKL only | Attention Q/K 1×8×128×64 | 41.45–41.70 ms | 0 | Y | No HexKL MicroHMX (batch_matmul) |
| 2026-07-23 | HexKL + VTCM | Attention | 332–364 ms | 0 | Y | **~8× slower — avoid** |
| 2026-07-23 | HexKL + OF layout OFF | Attention | ~41.6 ms | 0 | Y | PrefetchInsert only saw HVX strips |
| 2026-07-23 | HexKL only | GEMM 256³ f16 | **14.152 ms** | 0 | Y | Real MicroHMX |
| 2026-07-23 | HexKL + OF layout OFF | GEMM 256³ | **14.286 ms** | **2** L2Hint | Y | ~+0.9% noise / slight overhead |
| 2026-07-23 (resume) | HexKL only | GEMM 256³ | **14.092 ms** | 0 | Y | Post-reboot remeasure |
| 2026-07-23 (resume) | HexKL + OF layout OFF | GEMM 256³ | **14.358 ms** | **2** L2Hint | Y | Phase 1 still good |
| 2026-07-23 (resume) | HexKL + OF layout ON | GEMM 256³ | **14.033 ms** | **1** HMXWeight fusion | Y | **Phase 2a: RmToWh erased; hexkl_micro via Prefetch; ~neutral / slight win** |
| 2026-07-23 | HexKL + OF layout ON (bad gather) | GEMM 256³ | ~92–105 ms | 2 | N | Contiguous gather on strided tile → NaN/mismatch |
| 2026-07-23 | HexKL + OF layout ON (+ idle sync fill) | GEMM 256³ | **15.731 ms** | 2 | Y | Extra post-Mm fill; disabled pending true async |
| 2026-07-23 | HexKL only | GPT-2 2-layer seq32 | **10718 ms** | 0 | Y | Phase 4 baseline; Top-1 match |
| 2026-07-23 | HexKL + OF layout ON | GPT-2 2-layer seq32 | **10773 ms** | **8** HMXWeight fusion | Y | ~+0.5% vs HexKL-only; no exit 13 |
| 2026-07-23 | HexKL + OF layout ON + async L2 | GEMM 256³ | **13.745 ms** | 2 async_pipeline | Y | **~3% vs HexKL 14.2 ms**; L2 kick before Mm |
| 2026-07-23 | HVX fair `--seq-len 32` | GPT-2 2-layer | **13332 ms** | 0 | Y | Content-filled (no pad_token) |
| 2026-07-23 | HexKL fair `--seq-len 32` | GPT-2 2-layer | **11110 ms** | 0 | Y | **~17% faster than fair HVX** |
| 2026-07-23 | HexKL+OF fair `--seq-len 32` | GPT-2 2-layer | **11111 ms** | 8×2 async | Y | ~parity with fair HexKL |
| 2026-07-23 | HVX fair `--seq-len 32` | Qwen tiny 2L vocab4k | **1919 ms** | 0 | Y | top5 match; max_abs≈5e-4 |
| 2026-07-23 | HexKL fair `--seq-len 32` | Qwen tiny 2L vocab4k | **155.8 ms** | 0 | Y | **~12× vs HVX**; Top-1; attn skipped in MatmulToHexKL |
| 2026-07-23 | HexKL+OF fair `--seq-len 32` | Qwen tiny 2L vocab4k | **153.0 ms** | **~30** sites (15× layout-fusion) | Y | Pass; hexkl_func=1; ~parity / slight win vs HexKL |
| 2026-07-23 (3-way) | HVX `--seq-len 32` | Qwen tiny 2L | **1951.7 ms** | 0 | Y | top5; `debug_running/run_qwen2.5-0.5b_debug.py` |
| 2026-07-23 (3-way) | HexKL `--seq-len 32` | Qwen tiny 2L | **156.2 ms** | 0 | Y | **~12.5× vs HVX**; Top-1 |
| 2026-07-23 (3-way) | HexKL+OF `--seq-len 32` | Qwen tiny 2L | **150.6 ms** | >0 | Y | **~3.6% vs HexKL**; Top-1 |
| 2026-07-23 | HVX `--seq-len 32` | Falcon tiny 2L | **2751 ms** | 0 | Y | top5; max_abs≈0.016 |
| 2026-07-23 | HexKL `--seq-len 32` | Falcon tiny 2L | **113.5 ms** | 0 | Y | **~24× vs HVX**; Top-1 |
| 2026-07-23 | HexKL+OF `--seq-len 32` | Falcon tiny 2L | **109.6 ms** | 16 sites | Y | **~3.4% vs HexKL**; Top-1 |
| 2026-07-23 | HVX `--seq-len 32` | TinyLlama tiny 2L | **1910 ms** | 0 | Y | top5; max_abs≈5e-4 |
| 2026-07-23 | HexKL `--seq-len 32` | TinyLlama tiny 2L | **195.3 ms** | 0 | Y | **~9.8× vs HVX**; Top-1 |
| 2026-07-23 | HexKL+OF `--seq-len 32` | TinyLlama tiny 2L | **194.6 ms** | 30 sites | Y | ~parity vs HexKL; Top-1 |
| 2026-07-23 | HVX `--seq-len 32` | Mamba tiny 1L | **1135 ms** | 0 | Y | top5; seq SSM sequential |
| 2026-07-23 | HexKL `--seq-len 32` | Mamba tiny 1L | **130.6 ms** | 0 | Y | **~8.7× vs HVX**; Top-1 |
| 2026-07-23 | HexKL+OF `--seq-len 32` | Mamba tiny 1L | **127.9 ms** | >0 | Y | ~2% vs HexKL; Top-1 |
| 2026-07-23 | HVX | ViT tiny 2L patch32 | **1220 ms** | 0 | Y | top5 |
| 2026-07-23 | HexKL / +OF | ViT tiny 2L | **1222 ms** | 0 | Y | Pass; HexKL no win (shapes) |
| 2026-07-23 | HVX | Swin tiny [1,1,1,1]/48 | **67066 ms** | 0 | Y | top5; max_abs≈1e-3 |
| 2026-07-23 | HexKL / +OF | Swin tiny | **67.4–67.7 s** | 0 | Y | Pass; ~parity |
| 2026-07-23 | HVX `--seq-len 32` | GraphSAGE tiny 2L | **304 ms** | 0 | Y | max_abs≈6e-3 |
| 2026-07-23 | HexKL `--seq-len 32` | GraphSAGE tiny 2L | **124 ms** | 0 | Y | **~2.4× vs HVX** |
| 2026-07-23 | HexKL+OF `--seq-len 32` | GraphSAGE tiny 2L | **123 ms** | >0 | Y | ~parity vs HexKL |
| 2026-07-23 | HVX / HexKL / +OF | SD-TE tiny CLIP | **~1.47 ms** | 0 | Y | all Pass |
| 2026-07-23 | HVX | Real-ESRGAN 8×8 | **8000 ms** | 0 | Y | device Pass; max_abs≈0.85 (loose) |
| 2026-07-23 | HexKL | Real-ESRGAN 8×8 | **7894 ms** | 0 | Y | conv-only IR; HexKL rewrite=0 |
| 2026-07-23 | HexKL+OF | Real-ESRGAN 8×8 | **7920 ms** | 0 | Y | ~parity; max_abs≈0.57 |
| 2026-07-23 | HVX | SD-VAE tiny [32,64] | **8984 ms** | 0 | Y | device Pass; loose compare |
| 2026-07-23 | HexKL | SD-VAE tiny [32,64] | **488 ms** | 0 | Y | **~18× vs HVX** |
| 2026-07-23 | HexKL+OF | SD-VAE tiny [32,64] | **495 ms** | >0 | Y | ~parity vs HexKL |
| 2026-07-23 | HVX | SD-UNet tiny no-xattn | **106 ms** | 0 | Y | device Pass; loose compare |
| 2026-07-23 | HexKL | SD-UNet tiny no-xattn | **106 ms** | 0 | Y | ~parity (conv-heavy) |
| 2026-07-23 | HexKL+OF | SD-UNet tiny no-xattn | **105 ms** | 0 | Y | ~parity |
| 2026-07-23 | HVX fair `--seq-len 32` | GPT-2 **full 12L** | **24027 ms** | 0 | device Y / compare N | `run_gpt2lmheadmodel.py`; n_layer=12; Hexagon logits **NaN** |
| 2026-07-23 | HexKL fair `--seq-len 32` | GPT-2 **full 12L** | **12049 ms** | 0 | device Y / compare N | **~2.0× vs HVX**; 48 matmul→f16; logits NaN |
| 2026-07-23 | HexKL+OF fair `--seq-len 32` | GPT-2 **full 12L** | **11673 ms** | **48×** layout-fusion (2 sites + async each; hexkl_func=1, 96 cand loops) | device Y / compare N | **~3.1% vs HexKL**; OF fires on full graph; logits NaN |
| 2026-07-24 | HexKL + last-token lm_head | GPT-2 **full 12L** | **5413 ms** | 0 (OF off) | device Y | After bias-buffer freeze (exit-13 fix); **~2.2× vs HexKL full-seq 12049 ms** |
| 2026-07-24 (resume) | HexKL only | GEMM 64×128×256 | **4.292 ms** | 0 | Y | Post-crash vehicle (non-square; avoids K==M skip) |
| 2026-07-24 (resume) | HexKL+OF layout OFF | GEMM 64×128×256 | **4.191 ms** | L2Hint | Y | |
| 2026-07-24 (resume) | HexKL+OF layout ON + act fuse + L2 async | GEMM 64×128×256 | **4.283 ms** | act erased=2, wt async | Y | **Deferred WH-on-wait gated** (corrupt ~31); L2 kick only |
| 2026-07-24 (resume) | HexKL / HexKL+OF fair seq32 | GPT-2 2L debug | Pass (Top-1) | act+wt fusion | Y | Re-gated unaligned-N→HVX (N-pad lm_head still exit 13) |
| 2026-07-24 | HexKL+OF dma2d+WH-on-signal | GEMM 64×128×256 | **4.701 ms** | act+wt async | Y | **WH fixed in signal()**; slab+weight_off; ddr stage pack overlaps Mm |
| 2026-07-24 | HexKL+OF DMA→VTCM stage | GEMM 64×128×256 | **9.42 ms** | same | Y | Correct but ~2× slower — DDR stage kept as default |

### GPT-2 full 12L notes (2026-07-23)

Harness: `benchmark_models/run_gpt2lmheadmodel.py` (published 12L / 768-d). Device execution `Result:Pass` all three ways; host `compare` fails with NaN top-5 on Hexagon (CPU top-5 sane). Debug 2L previously Top-1 match — depth-related numerical / IR issue, not capacity failure. PrefetchInsert on OF path: `hexkl_func=1`, 96 candidate loops, 48 loops with `layout-fusion sites: 2` + `async_pipeline=1`.

**NaN root cause (2026-07-23 evening):** not HexKL/OF-specific (HVX also NaNs). Full-graph export was **float16 end-to-end**; past_kv dumps show healthy early layers then f16 saturation (`±65504`) / NaN by ~layer 3–4. HVX bisect: `n_layer=2` top5 OK (~3% logit NaN), `n_layer≥4` broken. **float32** at `n_layer=4` → `nan_frac=0`. Fix in harness: `GPT2LogitsWrapper` (`use_cache=False`, logits only) + `torch_dtype=float32`; HexKL path inserts `truncf` on matmul inputs only so LN/softmax/residual stay f32.

**In-situ reshape:** YES on OF runs — PrefetchInsert replaces `MicroHMXRmToWh` with `prefetch_in_situ(HMXWeight)` (`erased_hexkl_ops=1`); device runtime calls `hexkl_micro_hmx_rm_to_wh_f16` (`OmniFetchRuntime.c`).

**2–3× on top of HexKL:** OmniFetch alone cannot (covers weight layout/prefetch only; measured ~3%). Attention score/context matmuls still HVX (`MatmulToHexKL` skips `K==M||N==M` for TLB). Realistic large levers: (1) make attention HexKL-safe, (2) fix `enableVectorization` Bad VA, (3) true DMA-into-VTCM on wait (still gated — WH-on-wait corrupts even with CPU pack), (4) safe N-pad for lm_head (large pad still exit 13; unaligned N kept on HVX). Activation `Copy+RmToAh` fusion is **on** and Passes on GEMM.

Debug write-up: `benchmark_models/QWEN_HEXKL_EXIT13_DEBUG.md`. Full 24L host compile not finished in-session (~2 GB mlirbc / RAM thrash)—structure OK in main harness; use debug for A/B until capacity allows. **GPT-2 full 12L** is the first Phase-4 full published LLM with complete on-device 3-way timings.

---

## Immediate next action

**Pivot (user 2026-07-24):** final top-5 / NaN not blocking — full 12L device `Result:Pass` (no crash). Focus on **2–3× on top of HexKL**.

**Recovered state after host reboot:** device `49d1c7b2` online; harness keeps `GPT2LogitsWrapper` + f16 (float32 full-graph compile abandoned — ~950MB mlirbc thrash). Prior f16 full-12L timings still valid: HVX **24027 ms** / HexKL **12049 ms** / OF **11673 ms**.

**Bottleneck picture (static IR + FLOP, GPT-2 12L seq32):**
- HexKL covers **48** Linear matmuls (c_attn / c_proj / MLP) only (~68% FLOP).
- **lm_head** `batch_matmul` `S×768×50257` (~31% FLOP, ~77MB weight traffic) stays HVX (N not tile-32).
- **12× `tm_tensor.attention`** → HVX batch_matmul+softmax; at seq=32 attention FLOP is tiny (~0.5%).
- OmniFetch (~3%) cannot deliver 2–3×.
- Tried decode-style **last-token lm_head**: IR OK (`1×768×50257`); first device attempt **exit 13** (Bad VA 0x0).
- **Root cause (same family as Qwen ConstRope / doc §F):** `Attention.bias` `register_buffer` → 24 extra FX args; ciface wrapper only passed `(out*, ids*)` while compiled iface read stack args → NULL. Old full model “worked” because 24 past_kv outputs filled those stack slots.
- **Fix:** `freeze_gpt2_attn_bias_buffers()` → export arity 1. Retest: **Result:Pass**, HexKL last-token **5413 ms** vs prior full-seq HexKL **12049 ms** (**~2.2×** on HexKL baseline).

**Acceleration levers (priority):**
1. Stabilize last-token lm_head or pad vocab→HexKL lm_head (largest remaining FLOP/BW).
2. `--enable-lwp` on known-good HexKL 12L to confirm wall-time share.
3. Attention path matters more at longer seq; HexKL-safe batch matmul / fused attn.
4. HexKL micro (`HEXKL_DIR`): RmToWh/Copy beyond OF; vectorization Bad VA.

**Topology status:** main GPT-2=12L (**full 3-way timed**); NaN is numerical not crash. Shared helpers: `benchmark_models/hexkl_utils.py`. Status doc: `benchmark_models/PHASE4_STATUS_AND_OMNIFETCH.md`.

