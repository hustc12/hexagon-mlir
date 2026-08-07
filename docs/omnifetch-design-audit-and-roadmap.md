# OmniFetch Design Audit, Model Validation, and Roadmap

Date: 2026-07-26

## Scope

This document records a source-level audit of the OmniFetch implementation in
Hexagon-MLIR, its relationship with HexKL and the Hexagon SDK, the current
on-device model status, and the implementation roadmap. It is the source of
truth for the work items below; benchmark claims must still be reproduced on a
device before publication.

The two required external comparison baselines are:

1. unmodified Hexagon-MLIR, with HexKL and OmniFetch disabled; and
2. Qualcomm Hexagon NN Library.

Hexagon NN Library is not integrated in this repository or in the inspected SDK
tree. Values in `benchmark_models/plotting/ALPS_Prefetcher_Data.csv` therefore
must not be presented as measured baseline results until a real runner, library
version, model conversion path, and device logs are checked in.

## Executive summary

OmniFetch is a cross-layer compiler/runtime prototype, not a single prefetch
pass. Its implemented path is:

```text
linalg.matmul
  -> MatmulToHexKL
  -> hexkl.matmul
  -> DecomposeHexKLMatmul
  -> HexKL Micro HMX loops
  -> PrefetchInsert
  -> omni_fetch.prefetch_in_situ
  -> VDAE wait/signal/adaptive_control
  -> OmniFetchToLLVM
  -> UserDMA, L2 fetch, and HexKL Micro runtime calls
```

Its sound central ideas are:

- move DDR-to-staging/VTCM traffic ahead of HMX consumption;
- combine transport with AH/WH layout preparation;
- ping-pong HMX weight slots;
- overlap access and execute work; and
- adapt scheduling from observed stalls.

The implementation is currently best described as a HexKL tile software
pipeline with partial UserDMA overlap. It is not yet a generally reentrant or
concurrency-safe runtime. Existing measurements show that HexKL/HMX is the
dominant speedup. OmniFetch adds approximately 0--4% over HexKL-only on the
working vehicles; the recorded full GPT-2 result is about 3.1%.

## Current architecture

### Dialect

`OmniFetchOps.td` defines:

- `prefetch_in_situ`;
- `create_sem`;
- `wait` and `signal`; and
- `adaptive_control`.

The layout kinds are `None`, `HMXWeight`, `HMXActivation`, `Custom`, and
`L2Hint`. HexKL tile metadata is carried as variadic `tile_params`.

### HexKL integration

The FP16 HMX tile is 32x32. HexKL requires 2048-byte activation alignment,
128-byte weight alignment, and 256-byte HMX configuration alignment.

For weights, PrefetchInsert replaces `MicroHMXRmToWhF16Op` with
`prefetch_in_situ(HMXWeight)`. For activations it can replace
`CopySubmatrixToF16 + RmToAhF16` with
`prefetch_in_situ(HMXActivation)`.

The async weight path currently:

1. synchronously prepares the current weight slot;
2. starts a DMA2D pack of the next strided 32x32 tile;
3. overlaps that transfer with the current HMX MM;
4. waits for DMA and performs WH conversion after MM; and
5. writes the converted tile into the idle ping-pong slot.

Direct DMA-to-VTCM is supported by an option but was about two times slower for
the tested 2 KiB tiles. DDR staging remains the default.

### V-DAE and adaptive control

VDAE creates a semaphore before an eligible loop, waits at the loop-body entry,
and signals at the loop tail. L2 hints and synchronous-only prefetches are not
synchronized.

The default semaphore implementation is a runtime counter with spin waiting,
not the hardware semaphore described by the dialect documentation. The
adaptive signal is the number of software wait spins, not a PMU AXI-stall
counter. The adaptive result changes runtime-global state but is not carried as
an `scf.for` iter_arg.

## Correctness and robustness findings

### Runtime state

The runtime currently has process-global async-job, staging-ring, semaphore,
adaptive-controller, and scout state. This prevents safe nested invocations,
parallel model execution, and robust true dual-thread DAE. The state needs an
explicit invocation context and a descriptor ring.

### Synchronization

Volatile counters do not provide sufficient multi-thread atomicity or
acquire/release ordering. Semaphore slots are reused from a fixed pool of 16
without generation IDs. A wait timeout currently allows execution to continue
with potentially incomplete data. These are correctness defects, not merely
performance limitations.

### Pipeline boundary

The generated next tile is clamped to the final tile. The last iteration can
therefore prefetch the last tile again instead of conditionally omitting the
request. This wastes traffic and complicates signal/slot accounting.

### Single outstanding job

Starting a new operation can drain the previous global job, serializing the
pipeline and allowing sibling loops to interfere. A real lookahead larger than
one requires a stateful descriptor ring:

```text
FREE -> DMA_PENDING -> LAYOUT_PENDING -> READY -> CONSUMING -> FREE
```

### Candidate analysis

The non-HexKL fallback uses coarse shape and size heuristics. Layout-aware
mapping is forced off outside the proven HexKL path. A production pass needs
reuse-distance, alias, lifetime, stride, VTCM-capacity, transfer-startup, and
critical-path analysis.

## Ordered implementation roadmap

### P0: correctness first

1. Introduce an invocation/context abstraction for async descriptors,
   semaphores, staging buffers, and adaptive state.
2. Use atomic acquire/release operations and generation-aware synchronization.
3. Treat timeout, descriptor exhaustion, and DMA failure as explicit errors or
   safe synchronous fallbacks.
4. Replace final-tile clamping with a guarded prefetch.
5. Add host runtime tests and MLIR transformation tests for zero/one/multiple
   iterations, nested loops, and repeated invocations.

### P1: real overlap and transfer policy

1. Replace the single async job with a descriptor ring of
   `lookahead + 1` entries.
2. Batch adjacent 2 KiB tiles into larger DMA transactions when profitable.
3. Select L2 hint, CPU/HVX copy, DDR staging DMA, or VTCM DMA from tile size,
   stride, reuse, and measured setup cost.
4. Feed DMA latency, compute cycles, wait time, VTCM occupancy, and available
   PMU data to the adaptive controller.
5. Jointly select lookahead, buffer count, tile size, destination, and cache
   policy.

### P1: increase HMX coverage

1. Add safe edge/remainder tiles instead of requiring all dimensions to be
   divisible by 32.
2. Add batch-matmul and attention-specific HMX paths.
3. Avoid full-size padding allocations for attention and vocabulary
   projection.
4. Add chunked and last-token-only LM-head lowering.
5. Validate every new shape on device before enabling it by default.

### P2: profitability model

Estimate for each candidate:

```text
benefit =
  hidden DMA latency
  + eliminated layout cycles
  + eliminated DDR traffic
  - DMA setup
  - synchronization
  - VTCM pressure
  - cache pollution
```

Only insert prefetches when the estimate is positive and all alias, dominance,
capacity, and lifetime constraints are proven.

## Optimizations beyond prefetch

The highest-value complementary work is:

- broader HMX coverage;
- last-token-only projection, KV cache, paged/sliding-window KV, GQA/MQA, and
  speculative decoding;
- FP16, W8A16, W8A8, and W4A16 mixed-precision kernels;
- norm/projection, bias/activation, RoPE/QK layout, attention, residual, and
  dequant/matmul fusion;
- online/Flash attention;
- model-level persistent VTCM allocation and native-layout lifetime extension;
- stable DCVS, DDR bandwidth, HMX, thermal, and warm-up control.

## Proposed combined research ideas

### Layout-Liveness Prefetch

Track the layout required by downstream consumers. Keep a tile in AH/WH form
and in VTCM across compatible operators, avoiding AH/WH-to-row-major-to-AH/WH
round trips.

### Critical-Path-Aware OmniFetch

Prefetch only tiles whose predicted hidden latency exceeds DMA, synchronization,
VTCM-pressure, and cache-pollution costs.

### Prefetch plus quantization co-scheduling

Have the access/scout side perform DMA plus fused dequantization and WH reorder,
while HMX consumes the previous tile.

### Multi-level prefetch

Schedule DDR-to-L2, L2-to-staging/VTCM, and staging-to-AH/WH as separate stages,
selecting the path from tile size and reuse.

### Layer- and token-aware scheduling

Use deep tile pipelines during prefill. During decode, prefetch the next layer's
weights while the current layer runs and retain frequently reused packed
weights across tokens.

## Required experiment protocol

Use this ablation matrix:

| ID | Configuration |
|---|---|
| B0 | Hexagon NN Library |
| B1 | unmodified Hexagon-MLIR, HexKL and OmniFetch off |
| B2 | Hexagon-MLIR plus HexKL |
| E1 | B2 plus plain prefetch |
| E2 | E1 plus layout fusion |
| E3 | E2 plus V-DAE |
| E4 | E3 plus adaptive control |
| E5 | E4 plus the proposed optimization |

All rows must use the same checkpoint, tokenizer/input, batch, sequence length,
precision, output scope, power/thermal state, warm-up, and timing boundary.
Report compile, load, cold inference, warm p50/p90/p99, energy, and peak memory.
Validate numerical output; device process success is not output correctness.

## Model audit

The main runners generally retain published topology:

- GPT-2: 12 layers, width 768;
- Qwen2.5-0.5B: 24 layers, width 896;
- Falcon: 24 layers, width 2048;
- TinyLlama: 22 layers, width 2048;
- Mamba-130M: 32 layers, width 768;
- ViT: 12 layers;
- Swin-T: depths `[2,2,6,2]`;
- CLIP text encoder: 12 layers, width 768; and
- Stable Diffusion UNet/VAE: main channel topology retained.

Only full GPT-2 has recorded complete three-way device timings. Debug runners
shrink depth, width, heads, vocabulary, channels, attention structure, or image
size, and often use random weights. They are compiler/device smoke tests, not
valid published-model performance results. In particular, the debug UNet
removes cross-attention.

The full GPT-2 FP16 graph produces NaN logits on HVX, HexKL, and OmniFetch.
Recorded bisection shows FP16 saturation around layers 3--4. This is not an
OmniFetch-only error, but it means the full-model timing is performance evidence
only. Norm, softmax, and residual accumulation should remain FP32 while HMX
matmuls use FP16, followed by layer-by-layer differential validation.

The small `verify_omnifetch_model.py` Transformer is not a full language-model
block: it omits normalisation, residuals, MLP, causal masking, cache semantics,
and other production details. It validates data movement and matmul mechanics
only.

## Build and validation

The canonical build description is `docs/user-guide.md`. For an already
provisioned checkout, use `scripts/build_hexagon_mlir_incremental.sh`; it
validates the SDK, Tools, HexKL, LLVM, Python environment, and plugin paths
without downloading or reinstalling dependencies on each incremental build.

Typical commands are:

```bash
# Rebuild the optimizer, DSP runtime, and Python compiler extension.
bash scripts/build_hexagon_mlir_incremental.sh --jobs 12

# Also run the targeted test suite.
bash scripts/build_hexagon_mlir_incremental.sh --tests --jobs 12

# Re-run the editable Python package build when packaging/configuration changed.
bash scripts/build_hexagon_mlir_incremental.sh --full --jobs 12
```

The Python extension target is intentional: the device benchmark loads passes
from `triton/_C/libtriton.so`. Rebuilding only `linalg-hexagon-opt` leaves the
benchmark on stale pass code.

## Implementation status on 2026-07-26

This iteration completed the following safe parts of the ordered roadmap:

- the final K tile is no longer issued twice; the next-tile operation is guarded
  by `nextKt < upperBound`;
- async layout-prefetch is disabled for statically short K pipelines (fewer
  than eight 32-element tiles), while synchronous layout fusion remains
  available;
- semaphore state uses acquire/release atomics;
- semaphore handles carry a generation, preventing an old scout callback from
  satisfying a newly reused slot;
- the single global async descriptor was replaced by a four-entry FIFO tied to
  staging slots, with explicit full-queue backpressure;
- timeout and descriptor-full conditions are recorded in runtime error flags;
  and
- adaptive stall counters and effective lookahead are updated atomically.

These changes improve correctness and remove unnecessary work, but they do not
yet provide full invocation isolation. The descriptor FIFO remains global.
Nested or concurrent model invocations therefore require a later explicit
per-invocation context design. The HMX remainder path, attention-like shapes,
dynamic cost model, and true dual-thread stress matrix also remain open.

Validation completed in this iteration:

| Check | Result |
|---|---|
| shell syntax and `git diff --check` | pass |
| Hexagon runtime bitcode and runtime archive | build pass |
| `linalg-hexagon-opt` and `triton/_C/libtriton.so` | build pass |
| targeted static-prefetch LIT test | 1/1 pass |
| phone GEMM 64x128x256, layout-aware | correct, 3.057 ms |
| phone GEMM 64x256x512, HexKL | correct, 8.169 ms |
| phone GEMM 64x256x512, plain OmniFetch | correct, 8.227 ms |
| phone GEMM 64x256x512, async layout-aware | correct, 8.442 ms |

The 64x128x256 result confirms that the short-loop gate suppresses async
synchronization and removes the earlier large regression (the result is about
0.4% slower than its 3.044 ms HexKL run). The 64x256x512 test really enters the
async path and validates the new runtime queue, but it is about 3.3% slower than
HexKL. Thus eight tiles is a correctness-safe initial threshold, not yet a
profitable cost model. Future enabling should use measured transform cost,
expected overlap, synchronization cost, reuse, and VTCM/cache pressure.

The 256x256x256 smoke result must not be used as async evidence. Its compiler
log reports `hexkl_func=0`: the current attention-like guard (`K == M` or
`N == M`) prevents HexKL conversion when attention HMX support is disabled.

## Model-level continuation

Further performance validation no longer uses GEMM microbenchmarks. The detailed
Prefetch plus in-situ reshape innovation plan and first Falcon model result are
recorded in `docs/omnifetch-prefetch-insitu-innovation.md`.

The reusable model ablation entry point is:

```bash
bash scripts/run_omnifetch_model_ablation.sh \
  --model falcon-debug --seq-len 128 --timeout 240
```

## Drift-controlled percentile measurement (2026-07-30)

The experiment protocol above requires warm p50/p90/p99, which the historical
single-mean timing could not provide. Two additions close this gap:

1. **Device-side per-iteration sampling.** `hexagon_benchmark.h` now provides
   `benchmark_samples_us` and `report_percentiles`, which record each
   iteration's wall-clock time and append `PerfP50/PerfP90/PerfP99/PerfMin/
   PerfSamples` lines to stdout and `perf.txt`. The legacy `Perf:` mean line is
   byte-identical, so existing awk parsers (anchored on `^\s*Perf:`) are
   unaffected. Wired into both the default benchmark codegen
   (`hexagon_launcher_base.py`) and the OmniFetch WH-cache warm phase
   (`torch_mlir_hexagon_launcher.py`).

2. **Interleaved compile-once round-robin.** `run_torch_mlir_interleaved`
   compiles each backend profile once, then executes profiles round-robin for
   N rounds so thermal/DVFS drift is shared across configs rather than
   penalizing whichever ran last. Host-side aggregation emits
   `InterleaveResult profile=… p50_us/p90_us/p99_us/min_us`. Exposed via
   `--interleave-profiles` (labels: `legacy-scalar`, `hvx-vector`,
   `hvx-vector-vtcm`, `hexkl`, `hexkl-items17`) and `--rounds`.

**Fake-HMX guard.** HexKL profiles report their batch_matmul/f16 rewrite count;
a model that yields 0 rewrites (shapes not 32-aligned, e.g. the DINOv2-debug
proxy or DINOv2-full's 257 tokens) is labeled "no HMX coverage" rather than
presented as an HMX baseline. First B0–B3 percentile rows are in
`plan_todo.md` (Interleaved percentile baselines).

**PS_aligna frame fix status.** The Hexagon LLVM aligned-frame-base ordering fix
(`patches/llvm/0001-hexagon-order-aligned-frame-base-setup.patch`) is validated
on device: the previously-faulting vector item-7 Falcon path and DINOv2-debug
HVX-vector both pass with correct output. DINOv2-*full* still faults, but in
output-tensor serialization (`tensor.h` `dump_to_file` fwrite / `remote_munmap64`
ENOMEM), a separate capacity issue distinct from the frame bug.
