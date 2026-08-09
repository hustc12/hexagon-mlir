# Phase-4 Status, Harness Topology & OmniFetch Assessment

Date: 2026-07-26 (updated: build § aligned with docs/user-guide.md)

---

## 1. Ablation shell scripts

`run_qwen_ablation_3way.sh` and `run_tinyllama_ablation_3way.sh` are **not required**.
They only wrapped three sequential debug runs (HVX → HexKL → HexKL+OmniFetch).
They were removed; use the debug Python scripts directly, e.g.:

```bash
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 --enable-hexkl
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 --enable-hexkl --enable-omnifetch-vdae
```

---

## 2. Debug script layout

All `*_debug.py` runners live under:

`benchmark_models/debug_running/`

They override parent harness `customize_*` / `load_*` hooks for DSP smoke.
**Fair / full-structure numbers must use** `benchmark_models/run_*.py` (parent).

Shared helper module: `benchmark_models/hexkl_utils.py`
(renamed from `hexkl_phase4_utils.py`).

---

## 3. Debug vs full: what actually ran on device?

| Mode | Topology | Device 3-way (HVX / HexKL / HexKL+OF) |
|------|----------|--------------------------------------|
| **Debug** (`debug_running/*`) | Reduced layers / width / vocab / spatial size | **Pass** on the Phase-4 matrix so far |
| **Full GPT-2** (`run_gpt2lmheadmodel.py`) | Published **12L / 768-d** | **Device Pass** all three; timings below; host compare **NaN** |
| **Full other** (`run_*.py`) | Published structure kept in harness | Mostly **not** finished on-device (host lowering RAM, DSP VA / consts.so, compile time) |

So: debug paths are green; **GPT-2 full 12L** is the first fair published LLM with complete on-device A/B. Larger LLMs (Qwen/Falcon/…) remain capacity-blocked.

---

## 4. Full (main) harness topology checklist

Main scripts keep **published architecture** (layer counts / depths / channel lists).
`customize_*_config` is identity unless a debug script overrides it.

| Main harness | Published structure (main target) |
|--------------|-----------------------------------|
| `run_gpt2lmheadmodel.py` | 12L / 768-d |
| `run_qwen2.5-0.5b.py` | 24L / 896-d / vocab=151936 |
| `run_falcon_rw_1b.py` | 24L / 2048-d |
| `run_tinyllama.py` | 22L / 2048-d |
| `run_mamba-130m.py` | 32L / 768-d |
| `run_vit.py` | 12L / patch16 / 224 |
| `run_swin_transformer.py` | depths=[2,2,6,2] / embed_dim=96 |
| `run_graphsage.py` | 12L / 768-d |
| `run_sd_text_encoder.py` | CLIP 12L / 768-d; default seq_len=**77** |
| `run_sd_unet.py` | channels=[320,640,1280,1280] + CrossAttn blocks |
| `run_sd_vae_decoder.py` | channels=[128,256,512,512]; latents 64×64 |
| `run_real-esrgan.py` | Full RRDBNet; default input **64×64** (spatial, not layer count) |

### Non-topology compromises still present in main harnesses

These do **not** shrink layer counts, but they are intentional backend / ABI / capacity knobs:

| Item | Where | Why |
|------|--------|-----|
| `gelu_new` / `gelu_fast` / GELU→tanh | ViT, Swin, GraphSAGE, Falcon, SD-TE | Avoid Hexagon `math.erf` |
| `from_config` (random weights) | Mamba, ViT, Swin, GraphSAGE, some SD | Checkpoint / size / vocab padding; **structure unchanged** |
| ConstRope (closure cos/sin) | Qwen, TinyLlama, … | Avoid RoPE buffer / `inv_freq` ABI break |
| HexKL: `enableVectorization=False` | HexKL paths | Avoid Bad VA from async/forall |
| Attention matmul skip (`K==M \|\| N==M`) | `MatmulToHexKLPass` + harness rewrite | Avoid HMX TLB fault on attn shapes |
| `run_gpt2lmheadmodel_quantized.py` | Separate script | Still hardcodes `n_layer=2` (quantized vehicle, not main GPT-2) |
| `run_stable_diffusion.py` | Orchestrator | `from_config`; not in Phase-4 main matrix |

---

## 5. OmniFetch: prefetch + in-situ reshape — did it show up?

**Short answer: it is not invisible, but end-to-end wins are small and inconsistent on tiny debug models.**

### What “prefetch + in-situ reshape” means here

Layout-aware OmniFetch on the HexKL weight path:

1. **PrefetchInsert** finds HexKL MicroHMX loops (`hexkl_func≥1`).
2. Fuses / replaces layout (`RmToWh`) with **`prefetch_in_situ(HMXWeight)`** writing into the HexKL VTCM slot (`hexkl_micro_rm_to_wh` / layout-fusion).
3. Optional **lookahead / async** kick so the next tile’s L2 fetch overlaps Mm.

Compiler evidence that the path **fired** (not just flags): PrefetchInsert logs with `HexKL layout-fusion sites`, `erased_hexkl_ops`, `async_pipeline`, and non-zero insert counts in `plan_todo.md` Results log.

### Where it showed a measurable (small) win vs HexKL-only

| Vehicle | HexKL | HexKL+OF | Δ vs HexKL | Inserts / notes |
|---------|-------|----------|------------|-----------------|
| GEMM 256³ + async L2 | ~14.2 ms | **13.75 ms** | **~3%** | 2 async_pipeline — clearest microbench win |
| **GPT-2 full 12L** | **12049 ms** | **11673 ms** | **~3.1%** | hexkl_func=1; 48× layout-fusion (2 sites+async); also **~2.0× vs HVX 24027 ms** |
| Qwen tiny 2L (3-way) | 156.2 ms | **150.6 ms** | **~3.6%** | layout-fusion sites > 0 |
| Falcon tiny 2L | 113.5 ms | **109.6 ms** | **~3.4%** | 16 sites |
| Mamba tiny 1L | 130.6 ms | **127.9 ms** | **~2%** | sites > 0 |
| Qwen earlier OF run | 155.8 ms | 153.0 ms | ~1–2% | ~30 sites (15× layout-fusion) |

**Caveat (full GPT-2):** device `Result:Pass` and Perf timings are usable for OF vs HexKL ranking, but Hexagon logits are **NaN** on all three configs (HVX included); debug 2L was Top-1 OK. Treat as perf/ablation evidence pending NaN root-cause.

### Where inserts happened but time was ~parity (noise / no clear win)

| Vehicle | Notes |
|---------|--------|
| GPT-2 2L fair | OF ~11111 ms vs HexKL 11110 ms; 8×2 async inserts |
| TinyLlama tiny | 30 sites; ~parity |
| GraphSAGE tiny | sites > 0; ~parity |
| SD-VAE tiny | sites > 0; ~parity vs strong HexKL vs HVX |

### Where the method barely / did not engage

| Vehicle | Why little/no OF benefit |
|---------|---------------------------|
| ViT / Swin tiny | PrefetchInsert often **0 sites** (shapes / no tile-aligned HexKL HMX path) |
| Real-ESRGAN | **conv-only** IR; HexKL rewrite=0 |
| SD-UNet tiny (no cross-attn) | conv-heavy; HexKL≈OF≈HVX ~106 ms |
| SD-TE tiny | Too small / little HexKL work (~1.5 ms) |

### Interpretation

1. **Mechanism works:** PrefetchInsert + in-situ HMXWeight fusion runs on HexKL-heavy LLM graphs (debug and **full GPT-2 12L**: `hexkl_func=1`, many layout-fusion sites).
2. **Dominant win is still HexKL vs HVX** (tiny Qwen/Falcon ~10–20×; **full GPT-2 ~2×** — attention/HVX still large share at full depth).
3. **OF vs HexKL** stays in the **~0–4%** band even on full GPT-2 (~**3.1%**) — larger depth/width **did not** unlock a double-digit OF story; still proves path is live at published topology.
4. Likely reasons OF stays muted:
   - Prefetch / layout only covers HexKL weight Mm; attention remains HVX.
   - Async overlap is lookahead=1–2 software pipeline, not full DMA concurrency.
5. **Best OF demos:** GEMM 256³ (~3%), Qwen/Falcon tiny (~3–4%), **GPT-2 full 12L (~3.1%)** — consistent small win class.

### What would make OF more visible next

- Fix full GPT-2 NaN so Top-1/compare validates the 3.1% claim.
- Vehicles where HexKL Mm dominates wall time even more (wider FFN / less HVX attention share), or extend HexKL to more matmul shapes.
- Keep fair `--seq-len 32` content-filled protocol; avoid pure-conv / non-tile-aligned graphs.

---

## 6. Build (canonical: `docs/user-guide.md`)

Official source of truth: [`docs/user-guide.md`](../docs/user-guide.md)
(“Building Hexagon-MLIR Compiler”) and scripts under `scripts/`.
One-shot automation: `bash ./scripts/build_hexagon_mlir.sh`.

There are **two** CMake layers. Do not conflate them:

| Layer | What you configure | Where `Hexagon` appears |
|-------|--------------------|-------------------------|
| **1. LLVM for Triton** | `llvm-project/build` | **Must** pass `-DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;Hexagon"` so LLVM can emit Hexagon IR/asm |
| **2. Triton + Hexagon plugin** | `triton` via `scripts/build_triton.sh` | **Not** via `LLVM_TARGETS_TO_BUILD`. Hexagon backend comes from `TRITON_PLUGIN_DIRS=…/qcom_hexagon_backend` + linking against the LLVM built in layer 1 |

Paths below assume layout under `/home/huzq85/2-working/hexagon_npu/` (`BASE_DIR`). Adjust if yours differs.

### 6.1 Environment (manual, matches user-guide + `scripts/set_local_env.sh`)

```bash
cd /home/huzq85/2-working/hexagon_npu
source mlir-env/bin/activate          # or: export CONDA_ENV=...; source $CONDA_ENV/bin/activate
unalias python3 2>/dev/null || true   # only if python3 is a broken alias

cd hexagon-mlir
export HEXAGON_MLIR_ROOT=$PWD
export BASE_DIR=/home/huzq85/2-working/hexagon_npu

# SDK / Tools / HexKL (user-guide §Required Environment Variables)
export HEXAGON_SDK_ROOT=$BASE_DIR/HEXAGON_SDK/Hexagon_SDK/6.4.0.2
export HEXAGON_TOOLS=$BASE_DIR/HEXAGON_TOOLS/Tools
export HEXKL_ROOT=$BASE_DIR/HEXKL_DIR/hexkl_addon
export HEXAGON_ARCH_VERSION=75        # match device (v73/v75/v79)

# Host clang used to compile LLVM + Triton host bits
export HOST_TOOLCHAIN=$BASE_DIR/HOST_TOOLCHAIN
export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"

# LLVM built in §6.2 (user-guide uses LLVM_PROJECT_BUILD_DIR)
export LLVM_PROJECT_BUILD_DIR=$BASE_DIR/LLVM_DIR/llvm-project/build
# Triton setup expects headers/libs under this tree (install/ or build/)
export LLVM_SYSPATH=$LLVM_PROJECT_BUILD_DIR/install
export LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include
export LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib

# Triton + Hexagon plugin (user-guide “Building Triton Locally”)
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:${PYTHONPATH:-}

# Shorthand: after cd hexagon-mlir, also OK to:
#   source scripts/set_local_env.sh
# (sets SDK/Tools/HexKL/TRITON_* relative to parent of HEXAGON_MLIR_ROOT)

export ANDROID_SERIAL=49d1c7b2   # device used for GPT-2 2L ablations
```

### 6.2 Layer 1 — build LLVM **with Hexagon target** (required once / on LLVM upgrade)

Pin commit to `triton/cmake/llvm-hash.txt` (or the hash used by `scripts/build_hexagon_mlir.sh`).
**This** is where `-DLLVM_TARGETS_TO_BUILD=…;Hexagon` belongs — without it, Hexagon codegen libraries are missing and the NPU backend cannot lower to Hexagon object code.

From `docs/user-guide.md` / `scripts/build_hexagon_mlir.sh`:

```bash
# Example (hash must match Triton’s expected LLVM):
cd $BASE_DIR/LLVM_DIR/llvm-project
# git checkout <hash-from-triton/cmake/llvm-hash.txt>

mkdir -p build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_ASM_COMPILER="${CC}" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;Hexagon" \
    -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_BUILD_EXAMPLES:BOOL=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu \
    -DCMAKE_INSTALL_PREFIX="${LLVM_PROJECT_BUILD_DIR}/install"

cmake --build . -j
cmake --install .
export LLVM_PROJECT_BUILD_DIR=$BASE_DIR/LLVM_DIR/llvm-project/build
```

Sanity: `ls $LLVM_PROJECT_BUILD_DIR/install/lib | grep -i Hexagon` should show Hexagon target libs.

### 6.3 Layer 2 — build Triton + Hexagon-MLIR plugin (canonical)

**Do not** hand-roll a bare `cmake -S triton …` as the primary flow.
User guide: set env (§6.1), then:

```bash
cd $HEXAGON_MLIR_ROOT
./scripts/build_triton.sh
```

What that script does (see `scripts/build_triton.sh`):

* `pip install -r ci/requirements.txt`
* Sets `TRITON_PLUGIN_DIRS` (includes `qcom_hexagon_backend`)
* Builds with local LLVM via:

```bash
cd $HEXAGON_MLIR_ROOT/triton
TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=true \
LLVM_INCLUDE_DIRS="$LLVM_PROJECT_BUILD_DIR/include" \
LLVM_LIBRARY_DIR="$LLVM_PROJECT_BUILD_DIR/lib" \
LLVM_SYSPATH="$LLVM_PROJECT_BUILD_DIR" \
pip install -e . --no-build-isolation --verbose
```

(`build_triton.sh` uses `LLVM_PROJECT_BUILD_DIR` as `LLVM_SYSPATH`; if you installed under `…/build/install`, point `LLVM_*` at the install tree that actually contains headers/libs.)

Verify (user-guide “Verify the Setup”):

```bash
find . -name linalg-hexagon-opt
# → …/triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt

lit triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/test/
```

### 6.4 Day-to-day incremental rebuild (after a successful §6.3)

Only when the existing Ninja build dir is healthy. Build dir:

`triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}`

```bash
B=triton/build/cmake.linux-x86_64-cpython-3.11   # adjust PYTHON_VERSION

# OmniFetchRuntime.c / DMA / device bitcode → must rebuild hexagon_runtime
ninja -C $B hexagon_runtime

# Dual-thread scout enqueue → hexagon_mlir_async_runtime
ninja -C $B hexagon_mlir_async_runtime

# Host passes / options → libtriton
ninja -C $B libtriton.so

# If pip editable install did not refresh the .so Python loads:
cp -f $B/libtriton.so triton/python/triton/_C/libtriton.so

python3 -c "from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions; print(HexagonOptions())"
```

If the CMake cache is corrupted, prefer **re-running `./scripts/build_triton.sh`** (or the `pip install -e .` block in §6.3) over inventing a custom Triton `cmake` line. Avoid deleting the whole tree unless necessary.

### 6.5 What to rebuild after which edits

| Change | Rebuild |
|--------|---------|
| `OmniFetchRuntime.c` / DMA / scout | `hexagon_runtime` (+ `hexagon_mlir_async_runtime` if scout enqueue) |
| Prefetch / Decompose / MatmulToHexKL / options | `libtriton.so` (or full `./scripts/build_triton.sh`) |
| Both | both, then re-run device harness (fresh DSP `.so`) |
| LLVM hash / need Hexagon codegen libs | rebuild LLVM (§6.2) then Triton (§6.3) |

---

## 7. Run comparison experiments (GPT-2 2L)

Fair debug vehicle: `benchmark_models/debug_running/run_gpt2lmheadmodel_debug.py` (`n_layer=2`).
Full 12L: `benchmark_models/run_gpt2lmheadmodel.py` (host NaN known; device Pass usable for Perf).

HexKL requires `--seq-len` multiple of 32.

### 7.1 Flags (defaults off unless noted)

| CLI | `HexagonOptions` | Meaning |
|-----|------------------|---------|
| `--enable-hexkl` | `enableHexKL` | HexKL / HMX matmul path |
| `--enable-omnifetch-vdae` | `enablePrefetch` + `enableOmniFetchVDAE` | OmniFetch + V-DAE |
| `--enable-omnifetch-weight-prepack` | `enableOmniFetchWeightPrepack` | VTCM-resident WH prepack (column-outer) |
| `--enable-omnifetch-dual-thread-dae` | `enableOmniFetchDualThreadDae` | Scout thread for deferred WH (needs VDAE) |
| `--enable-omnifetch-inter-layer-prefetch` | `enableOmniFetchInterLayerPrefetch` | Outer HexKL loop prefetch |
| `--enable-omnifetch-attention-hmx` | `enableOmniFetchAttentionHmx` | Pad K==M/N==M attn matmuls → HexKL |
| `--enable-omnifetch-dma-to-vtcm` | `enableOmniFetchDmaToVtcm` | DMA stage in VTCM (default **off**; DDR stage) |
| `--enable-hexkl-persistent-vtcm` | `enableHexKLPersistentVtcm` | Shared VTCM arena (default off) |
| `--omnifetch-lookahead N` | `omniFetchLookahead` | Lookahead (default 2) |
| `--disable-layout-aware` | layout-aware off | Linear prefetch only |
| `--disable-omnifetch-adaptive` | adaptive off | Stall adaptive off |

### 7.2 Phase-1 style A/B (HexKL / OF / prepack)

```bash
cd /home/huzq85/2-working/hexagon_npu/hexagon-mlir
source /home/huzq85/2-working/hexagon_npu/mlir-env/bin/activate
unalias python3 2>/dev/null || true
export ANDROID_SERIAL=49d1c7b2 HEXAGON_MLIR_ROOT=$PWD

DBG=benchmark_models/debug_running/run_gpt2lmheadmodel_debug.py

# seq=32
python $DBG --enable-hexkl --seq-len 32
python $DBG --enable-hexkl --seq-len 32 --enable-omnifetch-vdae
python $DBG --enable-hexkl --seq-len 32 --enable-omnifetch-weight-prepack

# seq=128 (prepack win scales with ceil(M/32))
python $DBG --enable-hexkl --seq-len 128
python $DBG --enable-hexkl --seq-len 128 --enable-omnifetch-vdae
python $DBG --enable-hexkl --seq-len 128 --enable-omnifetch-weight-prepack
```

### 7.3 Phase 2–4 smokes (optional flags)

```bash
# Dual-thread DAE scout (default off)
python $DBG --enable-hexkl --seq-len 32 \
  --enable-omnifetch-vdae --enable-omnifetch-dual-thread-dae

# Inter-layer prefetch
python $DBG --enable-hexkl --seq-len 32 \
  --enable-omnifetch-vdae --enable-omnifetch-inter-layer-prefetch

# Attention in-flight pad → HexKL
python $DBG --enable-hexkl --seq-len 32 --enable-omnifetch-attention-hmx
```

### 7.4 Micro GEMM smoke

```bash
python benchmark_models/verify_omnifetch_Gemm.py --enable-hexkl --m 64 --k 128 --n 256
python benchmark_models/verify_omnifetch_Gemm.py --enable-hexkl --enable-omnifetch --enable-layout-aware
python benchmark_models/verify_omnifetch_Gemm.py --enable-hexkl --enable-omnifetch-weight-prepack --m 64 --k 128 --n 256
```

### 7.5 Full GPT-2 12L (perf only; host compare may NaN)

```bash
python benchmark_models/run_gpt2lmheadmodel.py --seq-len 32 --enable-hexkl
python benchmark_models/run_gpt2lmheadmodel.py --seq-len 32 --enable-hexkl --enable-omnifetch-vdae
```

### 7.6 How to read results

- Device: `Test_Info` → `Result:Pass` / `Perf:` (units usually **us**; divide by 1e6 for seconds).
- Correctness (2L): `Top-1 token matched (HexKL numerical tolerance)`.
- Compiler: `[PrefetchInsert] Found N candidate loops`, `Total prefetch sites`.

### 7.7 Measured snapshot (2026-07-25, GPT-2 **2L**, device `49d1c7b2`)

| Config | seq32 | seq128 |
|--------|-------|--------|
| HexKL | 307.5 s | 1335.9 s |
| HexKL+OF | 345.7 s | 1337.2 s |
| HexKL+prepack | 334.5 s | **1278.2 s (~4% vs HexKL)** |
| OF+dual-thread | 316.4 s | — |
| OF+inter-layer | 322.6 s | — |
| attention-HMX | 314.2 s | — |

All listed rows: device Pass + Top-1 (2L).

---

## 8. Related living docs

- **Build / install (authoritative):** `docs/user-guide.md`, `scripts/build_hexagon_mlir.sh`, `scripts/build_triton.sh`, `scripts/set_local_env.sh`
- Living checklist / results table: `plan_todo.md`
- OmniFetch analysis / roadmap: `benchmark_models/OMNIFETCH_ANALYSIS_AND_ROADMAP.md`
- OmniFetch handoff: `benchmark_models/OMNIFETCH_IMPROVEMENTS_HANDOFF.md`
- Qwen HexKL exit-13 write-up: `benchmark_models/debug_running/QWEN_HEXKL_EXIT13_DEBUG.md`
- Debug runners README: `benchmark_models/debug_running/README.md`
