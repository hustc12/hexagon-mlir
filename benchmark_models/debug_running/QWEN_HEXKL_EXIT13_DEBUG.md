# Qwen2.5-0.5B Hexagon exit-13 / HexKL debug notes

Living reference for “DSP crashed / no `perf.txt` / adb exit 13” on the Qwen harness.
Device used in this work: `adb -s 49d1c7b2`. Venv: `mlir-env`.

## Harnesses (do not shrink the main one)

| Script | Role | Layers / topology |
|--------|------|-------------------|
| `run_qwen2.5-0.5b.py` | **Full** published Qwen2.5-0.5B | **24L** / 896-d / vocab≈152k / `from_pretrained` |
| `run_qwen2.5-0.5b_debug.py` | Fast DSP iteration only | **2L** / hidden=64 / vocab=4096 / `from_config` |
| `run_gpt2lmheadmodel.py` | **Full** GPT-2 | **12L** / `n_embd=768` / `from_pretrained` |
| `run_gpt2lmheadmodel_debug.py` | Fast DSP iteration only | **`n_layer=2`** |
| `run_falcon_rw_1b.py` | **Full** Falcon-RW-1B | **24L** / 2048-d / vocab=50304 / `from_pretrained` |
| `run_falcon_rw_1b_debug.py` | Fast DSP iteration only | **2L** / hidden=64 / vocab=4096 / `from_config` |

**Important (status as of 2026-07-23):**

- Main harnesses keep **full published structure** (including layer count). Hooks `customize_*_config` are identity unless a `*_debug.py` overrides them.
- Phase-4 / three-way **measured** numbers (HVX / HexKL / HexKL+OF) so far used the **debug** topologies (GPT-2 2L, Qwen tiny 2L), same practical reason: full Qwen host lowering ~2 GB mlirbc / RAM thrash; full GPT-2 12L is heavy on DSP time/VA.
- Do **not** treat those fair tables as “full-depth model” numbers. Re-measure on main scripts when capacity allows.

Fair Phase-4 numbers that claim “Qwen” / “GPT-2” must say **debug vs full**.

Three-way ablation (same protocol as GPT-2 fair `seq=32`; default helper uses **debug**):

```bash
bash benchmark_models/python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py
# or: python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 [--enable-hexkl] [--enable-omnifetch-vdae]
```

**Full 24L capacity:** main harness is unshrunk; host lowering ~2 GB `.mlirbc` / RAM thrash blocked on-device finish in this session. Prefer `*_debug.py` for the HVX / HexKL / OF matrix until capacity allows.

---

## Symptom → how to triage

1. **Compile OK, run dies in <1s, no `perf.txt`** → almost always DSP fault or **dlopen** failure, not a Python exception.
2. Always pull logcat immediately after a fail:

```bash
adb -s <serial> logcat -d | grep -iE 'undefined symbol|plt object|TLBMISS|Bad VA|Fault PC|CRASHED'
```

3. Map `Fault PC` into the kernel `.so` when load address is printed:

```text
Fault PC : 0xFDF1083C
load address : 0xFDF00000
→ offset 0x1083C   # hexagon-nm / objdump -d --start-address=...
```

4. Check wrapper ABI vs exported symbol:

```bash
hexagon-nm _mlir_ciface_QwenWrapper.o | grep mlir_ciface
# Good HVX/HexKL:  T _mlir_ciface_QwenWrapper
rg 'extern "C" void' *_wrapper.cpp
# Good: (MemRef*, MemRef*, MemRef*, MemRef*)  — out + 3 inputs
# Bad:  (int64_t, MemRef*, int64_t, MemRef*, ...)  — broken (dim, memref*) ABI
```

5. Confirm HexKL actually linked:

```bash
hexagon-nm -u _mlir_ciface_QwenWrapper.o | grep hexkl_micro
```

---

## Root causes found (2026-07-23)

### A. Broken host/DSP ABI — RoPE / extra buffers (seq > 1)

**Symptom:** exit 13; wrapper emits `(dim, memref*)` pairs.

**Cause:** `register_buffer` cos/sin or a naked f32 `inv_freq` memref in the torch-mlir signature. WrapperGenerator mis-parses ranks and generates a bad starter.

**Fix (harness):** precompute RoPE as **closure constants** (`_ConstRope`), not buffers; keep only `(input_ids, attention_mask, position_ids)` as runtime args. Constants land in `consts.so`.

### B. Attention shapes on HexKL / HMX — TLB miss

**Symptom:** dlopen OK; `TLBMISS` / precise exception; Bad VA often **`0xD1F854`** (corrupt/unmapped memref store), Fault PC inside `QwenWrapper` epilogue / descriptor writes.

**Cause chain:**

1. Qwen emits `linalg.batch_matmul` (batch=1).
2. Compiler `ReduceContractionRank` collapses many of them to 2D `linalg.matmul`.
3. At `seq=32`, attention score/context mats are **HMX-tile-aligned** (e.g. QKᵀ `32×64×32`, AV `32×32×64`).
4. `MatmulToHexKL` accepted them → MicroHMX path → device fault.
5. Text rewrite in the harness that skipped `K==M || N==M` was **not enough alone**: rank reduction still fed attention matmuls into HexKL.

**Fix (compiler):** in `MatmulToHexKLPass.cpp`, after the `% 32 == 0` check, also skip:

```text
if (K == M || N == M)  // attention-like; keep HVX
```

Projections / FFN / lm_head (e.g. `32×64×64`, `32×64×128`) still convert.

**Harness still:** optional `rewrite_batch_matmul_to_matmul` with the same skip (defense in depth). HexKL options:

- `enableHexKL=True`
- `enableConvertToHexagonmem=True` (VTCM)
- **`enableVectorization=False`** (see C)

### C. Vectorization + `scf.forall` → async — Bad VA `0x18` / `0x28`

**Symptom:** `enableVectorization=True` alone (or with HexKL) → TLBMISS Bad VA **`0x18`** or **`0x28`**.

**Cause:** Hexagon tiling can emit `scf.forall`. `FormAsyncThreadsPass` used to lower each iteration to `async.execute` → `new AsyncToken()` on the DSP User PD heap → NULL/`0x18` destructor crash. Same class of bug documented in `run_swin_transformer.py`.

**Fix (compiler):** `FormAsyncThreadsPass` now lowers `scf.forall` → **sequential `scf.for`** (no async tokens).

**Harness:** Qwen HexKL path still keeps **vectorization off**. MicroHMX does not need HVX vectorization; with Qwen IR, vec+HexKL still showed Bad VA `0x28` in one retest after the sequential forall change—leave vec off until that is fully cleared.

### D. Misleading “undefined symbol `_mlir_ciface_QwenWrapper`”

**Symptom:** logcat `undefined symbol PLT #N _mlir_ciface_QwenWrapper` / `plt object relocation failure`.

**Cause (bisect pitfall):** Python wrapper class named `W` while launcher `func_name='QwenWrapper'` → object exports `_mlir_ciface_W`, wrapper calls `QwenWrapper`.

**Rule:** class name / exported MLIR func / `func_name` must match (`QwenWrapper`).

### E. False leads

| Lead | Why it looked real | Outcome |
|------|--------------------|---------|
| Expand all `ub.poison` → 0 | Vec IR has poison; translation can fail | Rebuild did not fix DSP crash |
| “hexagonmem alone crashes” | `hex_execution` clears convert when HexKL off | Need to force-convert carefully; real HexKL crash was attention HMX |
| Async symbols in `.so` via `strings` | Always link `-lhexagon_mlir_async_runtime` | Check **undefined** refs on the **`.o`**, not strings on `.so` |

---

### F. GPT-2 logits-only / last-token — Bad VA `0x0` at `_mlir_ciface_*+0x14`

**Symptom (2026-07-24):** `GPT2LogitsWrapper` + HexKL compiles; dlopen OK; crash in \<1s; logcat:

```text
TLBMISS RW
Bad VA     : 0x0
Fault PC   : ... _mlir_ciface_GPT2LogitsWrapper+0x14
```

**Cause (same family as A — extra buffers in FX signature):**

1. GPT-2 `Attention` keeps `bias` / `masked_bias` as `register_buffer`.
2. torch-mlir FX promotes them to **runtime args** (12× `1x1x1024x1024xi1` + 12× `f16` scalars + `input_ids` = 25 args).
3. WrapperGenerator emits a starter with only `(logits*, input_ids*)`.
4. Compiled `_mlir_ciface_*` still loads further args from the Hexagon **stack** (`memw(r29+#0x544)` → NULL).

**Why full `GPT2LMHeadModel` “worked” before:** it returned logits **plus 24 past_kv** outputs → many output `MemRef*` pointers were passed on the stack and accidentally satisfied those loads. Logits-only / last-token (1 output) removes that luck.

**Fix (harness):** `freeze_gpt2_attn_bias_buffers()` in `run_gpt2lmheadmodel.py` — pop buffers, reattach as plain tensor attributes (optionally sliced to `seq_len`) so FX folds them as constants. Export arity becomes **1** (`tensor<1xSxi64>`).

**Triage tip:** after export, `func.func` must have a single `%arg0: tensor<…xi64>`; if you still see `1024x1024xi1` args, the freeze did not take effect.

---

## Working flag matrix (tiny debug, seq=32, after fixes)

| enableHexKL | enableConvertToHexagonmem | enableVectorization | Result (post-fix) |
|-------------|---------------------------|---------------------|---------------------|
| F | F | F | HVX Pass (~1.9 s) |
| T | T | F | HexKL Pass (~156 ms) |
| T + OmniFetch | T | F | Pass (~153 ms, prefetch sites > 0) |
| * | * | T | Avoid on Qwen until Bad VA 0x28 cleared |

---

## Recommended debug workflow next time

1. Reproduce on **`run_qwen2.5-0.5b_debug.py`** only; never shrink the main harness.
2. Confirm HVX Pass + correct wrapper ABI.
3. HexKL with **vec off**; `nm -u` for `hexkl_micro_*`.
4. If TLB fault: check whether attention `M/K/N` with `K==M` or `N==M` are still becoming `hexkl.matmul` (dump after `MatmulToHexKL` or count micros vs expected projection/FFN count).
5. If Bad VA `0x18`: grep kernel for async / forall lowering.
6. Only then enable OmniFetch (`--enable-omnifetch-vdae`); require PrefetchInsert `hexkl_func≥1` and inserts > 0.

---

## Code touchpoints

- `qcom_hexagon_backend/lib/Transforms/MatmulToHexKLPass.cpp` — tile + attention-shape skip
- `qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/FormAsyncThreadsPass.cpp` — forall → sequential for
- `benchmark_models/run_qwen2.5-0.5b.py` — ConstRope, heap 256MB, HexKL option gating, batch_matmul rewrite
- Rebuild: from `triton/build/cmake.linux-x86_64-cpython-3.11`, `ninja libtriton.so`

---

## Related results log

See `plan_todo.md` Phase 4 / Results table (HVX vs HexKL vs HexKL+OF for Qwen tiny and GPT-2).

### Fresh 3-way (2026-07-23, `python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py`, seq=32, tiny debug)

| Config | NPU time | Correctness |
|--------|----------|-------------|
| HVX | **1951.7 ms** | top-5 match |
| HexKL | **156.2 ms** (~12.5× vs HVX) | Top-1 |
| HexKL + OmniFetch | **150.6 ms** (~3.6% vs HexKL) | Top-1 |

Logs: `/tmp/omnifetch_qwen/ablation_3way/`.
