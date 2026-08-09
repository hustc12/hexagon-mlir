# OmniFetch / Hexagon-MLIR Engineering Work

## Purpose

This document records the engineering diagnosis and remediation plan for the
unexpectedly long DINOv2-small latency observed with the current
Hexagon-MLIR-based HVX, HexKL, and HexKL + OmniFetch items 1–7 paths.

The immediate objective is to separate three effects that were previously
mixed together:

1. benchmark configuration errors that disable backend optimizations;
2. limitations or bugs in the current Hexagon-MLIR fork;
3. the real incremental benefit of OmniFetch.

No latency result is considered valid unless the run completes, produces a
device-side `Perf` result, and passes the numerical correctness gate.

## 1. Current evidence

### 1.1 DINOv2 workload identity

The current full DINOv2-small workload has:

- input: `1x3x224x224`, FP16;
- 257 tokens: one class token plus 256 image-patch tokens;
- 12 transformer layers;
- hidden size 384;
- six attention heads;
- FFN intermediate size 1536;
- 22,825,192 parameters;
- deterministic random full-structure weights, seed 142.

These weights are not a pretrained accuracy checkpoint, but the same values
and input are shared by all backends in the controlled performance test.

### 1.2 Invalid rerun and ABI defect

After the DINOv2 model factory was shared by the Hexagon and QNN runners, the
fixed position-embedding tensor was registered as a non-persistent buffer.
Torch-MLIR lifted this buffer to the first public function argument:

```text
Dinov2SmallWrapper(position_embeddings, pixel_values)
```

The Hexagon runner originally passed only `pixel_values`. The resulting null
argument caused a precise cDSP exception at the function entry:

```text
Bad VA: 0x9
_mlir_ciface_Dinov2SmallWrapper+0x24
```

The runner now explicitly supplies both device ABI arguments while retaining
the one-argument Python model interface for the reference calculation.

After this ABI repair, the HVX path entered the long-running device
calculation, but still exited with status 13 after roughly 18 minutes and did
not produce a valid `Perf` result. This is a failure record, not a latency
measurement.

## 2. Current benchmark configuration is not a valid HVX baseline

The backend defaults are:

```python
enableVectorization = True
enableVTCMTiling = True
enableConvertToHexagonmem = True
```

The shared OmniFetch benchmark helper currently overrides them:

```python
options["enableVectorization"] = False
options["enableVTCMTiling"] = False
if not options.get("enableHexKL"):
    options["enableConvertToHexagonmem"] = False
```

Consequently, the configuration named `hvx` is primarily generic scalar
Hexagon lowering rather than a representative vectorized HVX baseline.

Disassembly of the generated full DINOv2 main object supports this diagnosis:

- approximately 65,670 instructions;
- only approximately 65 clearly recognizable HVX `v*` instructions;
- approximately 0.099% HVX-like instruction share.

The exact instruction classification is only a static heuristic, but the
result is sufficiently extreme to show that the main graph is not being
systematically vectorized.

This issue must be repaired before attributing the measured gap to
Hexagon-MLIR versus QNN.

## 3. DINOv2 matmul and HMX coverage

The exported DINOv2 graph contains 96 `linalg.batch_matmul` operations:

| Operation family | Count | Representative shape |
|---|---:|---|
| Q/K/V/output projections | 48 | `1x257x384` by `1x384x384` |
| FFN up projection | 12 | `1x257x384` by `1x384x1536` |
| FFN down projection | 12 | `1x257x1536` by `1x1536x384` |
| attention QK | 12 | `6x257x64` by `6x64x257` |
| attention AV | 12 | `6x257x257` by `6x257x64` |

The current backend includes `ReduceContractionRankPass`, which can collapse
batch-one contractions to 2-D matmul. However, the local
`MatmulToHexKLPass` then requires M, K, and N all to be multiples of 32.
DINOv2's token dimension is 257, so all 72 projection/FFN contractions are
rejected. The 24 batch-six attention contractions are not accepted by the
2-D HexKL pattern.

The observed full-model HexKL result therefore does not exercise HMX for the
dominant transformer matrix multiplications.

## 4. Official upstream assessment

The official Qualcomm repository was inspected at:

```text
qualcomm/hexagon-mlir main
9b4b8fcea2b93c801b5de784ee750ca9350d504f
```

The current fork and official `main` share ancestor:

```text
dc4667fef6e6fe5d1c642bd41b3028a8c3f77adc
```

Official `main` has ten commits after that ancestor. Most are documentation
and v73/v75/v79/v81 compatibility changes, but commit `93ade94` is a large
backend update with approximately 20,828 inserted lines. It adds or expands:

- Crouton layouts;
- FP16 conversion;
- folding casts into matmul;
- pack/unpack frontier propagation;
- pack/unpack and transpose constant folding;
- convolution preprocessing and tiling;
- HexKL micro/macro lowering;
- VTCM and external scratch support;
- DMA and runtime support.

The official default HexKL micro mode does not impose the local pass's strict
32-alignment rejection at `MatmulToHexKLPass`. Together with
`ReduceContractionRankPass`, this makes the 72 batch-one DINOv2
projection/FFN operations plausible HMX candidates. Device correctness and
tail handling must still be proven on v73.

The remaining 24 batch-six attention contractions still require a batched
HMX lowering or a loop/packing strategy.

### Upgrade decision

An upstream experiment is required, but it must not be implemented as an
in-place merge into `alps_improve_3`:

- the fork has 68 divergent commits after the common ancestor;
- approximately 243 backend files differ from official `main`;
- the no-index comparison contains about 21,071 added and 9,043 removed
  lines.

The safe strategy is a separate branch/worktree rooted at official
`9b4b8fc`, followed by layered porting:

1. deterministic DINOv2 runner and correctness harness;
2. OmniFetch dialect and runtime registration;
3. core prefetch/in-situ/V-DAE passes;
4. items 1–7;
5. tests, scripts, and documentation.

The same pinned Triton and triton-shared commit hashes are used by the
official tree, although the official patch application flow and the
triton-shared repository URL have changed.

## 5. Profiling options on the current v73 device

### 5.1 Static IR and assembly audit

This is the first and cheapest profiling layer:

1. save the initial Torch-MLIR Linalg IR;
2. enable per-pass IR dumps;
3. count contraction, vector, layout, memory, and DMA operations after key
   passes;
4. dump or disassemble the final Hexagon object;
5. report the ratio of vector-like versus scalar instructions and the
   presence of HexKL/DMA/runtime calls.

The audit must at least count:

- `linalg.batch_matmul`;
- `linalg.matmul`;
- `hexkl.matmul`;
- `vector.*`;
- `hexagonmem.*`;
- DMA operations;
- pack/unpack/transpose/copy;
- SCF loops.

### 5.2 LWP

The current Torch-MLIR path supports Lightweight Profiling:

```python
options["enableLWP"] = True
options["disableLWPLoop"] = False
options["LWPloopDepth"] = 1
```

The output is:

- `/tmp/lwp.json`;
- `/tmp/lwp_infodump.txt`;
- `/tmp/lwp_output.csv`.

It can be processed with:

```bash
python test/python/process_lwp.py \
  /tmp/lwp.json \
  /tmp/lwp_infodump.txt \
  /tmp/initial-linalg.mlir
```

LWP reports function/loop pcycles, call counts, parent loops, MLIR source
locations, and a summary of contained operations. It adds overhead and uses
bounded recording buffers, so it should first be validated with DINOv2 Debug.
LWP results are for bottleneck attribution, not formal end-to-end latency.

### 5.3 ETM limitation

The current fork's `HexagonProfiler` raises `NotImplementedError`.

Official `main` contains an ETM implementation, but it currently:

- requires root/remount and CoreSight access;
- requires PyETM and matching CDSP firmware binaries;
- checks for a specific Lanai firmware;
- asserts Hexagon v75.

It is not directly applicable to the current v73 phone. ETM should therefore
not block the immediate investigation.

## 6. Repair and evaluation order

### Phase A: repair the current baseline

1. Stop unconditionally overriding vectorization and VTCM settings.
2. Add explicit, logged configuration profiles:
   - legacy scalar;
   - HVX vectorization only;
   - HVX vectorization plus VTCM;
   - HexKL micro;
   - HexKL plus OmniFetch items 1–7.
3. Keep multithreading disabled during the first isolation experiments.
4. Run DINOv2 Debug serially with correctness gates.
5. Perform static IR/object audits for every configuration.
6. Enable LWP only on configurations that already pass normally.
7. Run full DINOv2 only for the best correct configuration.

### Phase B: official upstream experiment

1. Create a separate worktree at official `9b4b8fc`.
2. Build with the official dependency/patch flow.
3. Port only the deterministic DINOv2 Debug runner first.
4. Compare official HVX and HexKL before porting OmniFetch.
5. Verify actual vector/HMX coverage from IR and assembly.
6. Port OmniFetch in layers, with unit tests after each layer.
7. Compare:
   - legacy current configuration;
   - repaired current configuration;
   - official upstream baseline;
   - official upstream plus OmniFetch.

## 7. Interpretation rule

QNN remains a production graph compiler/runtime with proprietary kernel
selection, graph fusion, layout planning, HMX/HVX mapping, memory planning,
and device performance control. Hexagon-MLIR is an open and actively
developing compiler stack rather than an equivalent implementation of QNN.

However, the previously observed difference cannot be assigned entirely to
that framework gap. The current experiment simultaneously:

- disabled systematic HVX vectorization;
- disabled VTCM tiling;
- disabled Hexagon memory conversion on the HVX path;
- routed none of DINOv2's dominant transformer matmuls to HMX.

Only after repairing these issues can the residual difference be described as
the Hexagon-MLIR versus QNN stack gap.

## 8. Repairs implemented on 2026-07-30

The current branch now contains the following engineering repairs:

1. `hex_execution` no longer silently forces
   `enableLinalgToHVX=false`, `enableLinalgToVTCM=false`, and
   `enableLinalgToHexagonMem=false`.  The requested backend configuration is
   now preserved.
2. The runners expose explicit scalar, HVX, VTCM, HexKL, LWP, and cumulative
   OmniFetch item 1--7 controls.
3. Both DINOv2 runners perform finite-output, maximum-difference, and top-1
   correctness checks.  The full runner also uses the correct device ABI:
   fixed position embeddings and pixels are passed to the compiled wrapper,
   while the PyTorch reference receives pixels only.
4. A late `ub.poison` rewrite and UB-to-LLVM conversion were added to the
   Linalg-to-LLVM pipeline.  This is necessary because OmniFetch item 7 can
   introduce poison values after the earlier cleanup point.
5. `scripts/run_dinov2_codegen_profiles.sh` runs configurations strictly
   serially and records status, latency, correctness, artifacts, and logs.
6. `scripts/audit_hexagon_codegen.sh` reports final-object instruction,
   HVX-like instruction, vector load/store, HexKL/HMX, and DMA evidence.

The incremental v73 build described in `docs/user-guide.md` succeeds:

```bash
bash scripts/build_hexagon_mlir_incremental.sh --arch 73 --jobs 8
```

The new late-poison regression test also passes.

## 9. DINOv2 Debug repair results

All cases below were run serially on device `49d1c7b2`.  A `PASS` requires
finite output, the configured maximum-difference tolerance, and matching
top-1 output.

| Configuration | Latency (ms) | Correct | Static observation |
|---|---:|---|---|
| legacy scalar | 174.832 | PASS | 54,183 instructions; no HVX-like instructions |
| HVX vector | 21.258 | PASS | 61,099 instructions; 2,201 HVX-like (3.602%) |
| HVX vector + VTCM | 21.931 | PASS | byte/codegen-equivalent to HVX vector |
| HexKL + vector + VTCM | 22.232 | PASS | no HMX/HexKL instruction evidence |
| HexKL + prefetch/VDAE | 21.727 | PASS | no material object-level change |
| cumulative item 4 | 20.321 | PASS | byte/codegen-equivalent to HVX vector |
| cumulative item 5 | 20.462 | PASS | byte/codegen-equivalent to HVX vector |
| cumulative item 6 | 20.176 | PASS | byte/codegen-equivalent to HVX vector |
| cumulative item 7, scalar | 60.927 | PASS | item-7 attention path is active |
| cumulative item 7, vector | device failure | no | 38 HMX/HexKL and 91 DMA mentions |

The repaired HVX baseline is **8.224x faster** than the legacy scalar
configuration.  This is the dominant confirmed repair.  The apparent
item-4--6 latency differences cannot be attributed to those schemes on this
Debug model: their final objects are equivalent to the HVX baseline, so the
differences are run-to-run noise.  Item 7 is meaningful: in scalar mode it is
2.870x faster than legacy scalar, but it is still 2.866x slower than the
repaired HVX baseline.

The Debug shape does not route its dominant transformer matrix operations to
HMX.  Therefore it is suitable for correctness, option, and profiling
isolation, but not for claiming HexKL/HMX performance.

## 10. LWP bottleneck attribution

LWP was enabled only after the normal repaired HVX run passed.  Its measured
end-to-end time was 21.937 ms; this instrumented number is not used as formal
latency.  The profile contains 25,684,820 pcycles:

| Rank | Operation region | Pcycles | Share |
|---:|---|---:|---:|
| 1 | patch-embedding `linalg.conv_2d_nchw_fchw` | 12,046,121 | 46.90% |
| 2 | attention AV batch matmul | 3,962,390 | 15.43% |
| 3 | attention QK batch matmul | 2,400,871 | 9.35% |
| 4 | attention output projection | 2,353,421 | 9.16% |
| 5 | FFN down/GELU/reduction region | 1,558,215 | 6.07% |
| 6 | FFN up/layer-normalization region | 861,023 | 3.35% |

After restoring HVX, patch embedding is the first optimization target.
Attention and its matrix engine eligibility are next.  A generic prefetch
change should not be expected to dominate unless it addresses these measured
regions and changes the final object.

## 11. Full DINOv2 validation and newly isolated LLVM defect

The full-model runner was verified to use:

- input `1x3x224x224`, FP16;
- 257 tokens;
- 12 transformer layers;
- hidden size 384;
- 6 attention heads;
- intermediate size 1536;
- 22,825,192 parameters;
- 96 batch matmuls and one ordinary matmul.

Compilation completed in 280.58 seconds.  The resulting object contains
265,787 instructions and 23,949 HVX-like instructions (9.0106%), including
29,547 vector loads/stores.  Thus the previous near-scalar code generation
has been fixed for the full model as well.

The device run currently fails before a valid timing result.  The cDSP fault
is a Bad VA at `Dinov2SmallWrapper+0x1014d4`.  Disassembly shows that the
large aligned stack-frame base register `r16` is used to save incoming
pointers before it has been initialized:

```text
memw(r16 + large-negative-offset) = r1
allocframe(...)
r16 = and(r30, #-128)
memw(r16 + another-offset) = r0   # bundled with the r16 definition
```

The epilogue later reloads the output descriptor through the invalid saved
pointer and faults at offset `0x28`.  The same malformed prologue occurs in
the item-7 vector Debug object, explaining why scalar item 7 passes while
vector item 7 fails.

This is not a DINOv2 structure or ABI mismatch.  It is an LLVM Hexagon
frame-lowering/packetization ordering defect exposed by a large,
vector-spilling, 128-byte-aligned stack frame.  `PS_aligna` defines the
aligned frame base but is declared as a side-effect-free pseudo and is
expanded after register allocation.  Its definition is not kept in a
separate packet before all aligned-frame uses.

A `no-realign-stack` workaround was tested and rejected.  It repaired the
prologue ordering but made vector spills insufficiently aligned and corrupted
pointers during deallocation.  It has been completely reverted.

The next bounded repair is therefore to preserve the `PS_aligna` dependency
and packet boundary in the pinned LLVM backend, track that dependency patch
reproducibly from this repository, rebuild, and rerun:

1. item-7 vector Debug;
2. repaired HVX Debug;
3. full DINOv2 HVX;
4. full DINOv2 HexKL and HexKL + items 1--7 only after correctness passes.

No full-model performance claim is valid until this backend defect is fixed.

### 11.1 Resolution (updated 2026-08-08): official LLVM fixes applied

The earlier local `PS_aligna` workaround has been superseded by the exact
official LLVM Hexagon fixes that resolve the complete failure chain:

- `patches/llvm/0001-hexagon-handle-truncating-subreg-copies.patch`
  (`689ecf880373bb4e0f01ed5e004f19a466e869dc`);
- `patches/llvm/0002-hexagon-fix-ap-prologue-use-before-def.patch`
  (`3ef59d80c5ce51738a055d9e8eb98aa3c8effb2f`); and
- `patches/llvm/0003-hexagon-add-ap-liveins.patch`
  (`2e10b62995915d35ba528872e70aacda7223bd18`).

`scripts/apply_llvm_hexagon_fixes.sh` applies these patches idempotently to the
pinned LLVM checkout.  Their ordering has been verified from a clean
`ac5dc54d509169d387fcfd495d71853d81c46484` worktree.  The rebuilt
`libLLVMHexagonCodeGen.a` is linked into the runtime-loaded `libtriton.so` and
`linalg-hexagon-opt`.

Device validation:

1. **item-7 vector Falcon debug — PASS.** `Result:Pass`, no Bad VA (the primary
   PS_aligna blocker).
2. **DINOv2-debug HVX-vector — PASS** with correct output (`top1_match=True`,
   `max_abs_diff≈3e-4`).
3. **Full DINOv2 HVX-vector — PASS after the remaining ABI and VTCM lifetime
   issues were repaired/configured as documented in section 17.** The native
   v73 result is 30,396.932 ms with finite output and top-1 match.

The official fixes are therefore validated on both Debug and complete-model
v73 executions. Fair-baseline measurement (§ percentile tooling; see
`plan_todo.md`) proceeds on the vehicles that run.

## 12. Item-7 acceptance criterion

The current DINOv2 Debug evidence distinguishes implementation activity from
performance:

- items 4--6 did not change the final HVX object on this shape;
- item 7 did change the object and introduced HexKL/DMA evidence;
- scalar item 7 proved that the path is executable and can outperform the
  legacy scalar configuration;
- the vector item-7 result is still unavailable because it exposed the LLVM
  aligned-frame defect described above.

Scalar speedup is not the target result.  Item 7 is accepted as a useful
OmniFetch optimization only if all of the following hold:

1. both pure HVX and HVX + item 7 use real vector code;
2. both pass identical correctness gates;
3. final-object auditing proves item 7 is active;
4. runs are serial and repeated sufficiently to distinguish speedup from
   noise;
5. HVX + item 7 has a stable, material latency improvement over pure HVX.

The same rule applies to HMX: a claim requires proof that eligible matrix
operations actually lower to HMX and that the added data-movement scheme
improves the HMX baseline.

If item 7 does not satisfy this criterion after the backend repair, it remains
an ablation result rather than the final optimization.  The next design loop
will focus on measured HVX/HMX bottlenecks:

- software-pipelined prefetch at producer/consumer tile boundaries;
- double-buffered DMA with computation overlap;
- keeping Q/K/V and FFN tiles resident across adjacent fused operations;
- in-situ layout formation during producer stores;
- eliminating materialized transpose/reshape/pack/unpack chains;
- reuse-aware cache/VTCM placement and eviction;
- reducing repeated weight, activation, and normalization-stat reads;
- patch-embedding convolution tiling, because LWP attributes 46.90% of Debug
  cycles to that region.

Ideas are promoted only when they change the relevant IR/object and improve a
real HVX or HMX baseline.

## 13. Reassessment of item 7 under the V73 memory hierarchy

### 13.1 The scalar 2.87x result is not an isolated item-7 speedup

The previously reported scalar comparison was:

| Configuration | Latency |
|---|---:|
| legacy scalar | 174.832 ms |
| cumulative items 1--7, scalar | 60.927 ms |

Those configurations were not a controlled item-7 ablation.  The legacy case
disabled HexagonMem, HexKL, and all OmniFetch components.  The cumulative case
enabled HexagonMem, HexKL, items 1--7, and the special OmniFetch runtime
measurement path.  More importantly, the old K/V metadata handling caused the
entire function to skip Hexagon fusion.

The scalar cumulative run did execute K/V hints:

- two attention sites;
- four hints per model invocation;
- 4,352 requested K/V bytes per invocation;
- 12 issued requests across the runtime's three internal invocations.

However, its cold time was 67.418 ms and its reported warm average was
60.927 ms.  Warming accounts for only about ten percent, not the difference
from 174.832 ms.  Four kilobytes of K/V traffic also cannot plausibly explain
an end-to-end 2.87x reduction by itself.  The dominant causes are therefore
most likely the cumulative backend configuration and the changed fusion/code
generation topology.  The scalar result proves that the cumulative path is
executable and correct, but it is **not evidence of an isolated 2.87x K/V
prefetch speedup**.

Per the current experimental priority, no further scalar ablation will be
performed.  All item-7 acceptance experiments use a real HVX-vector baseline.

### 13.2 L2 is still part of the HVX data path

The V73 HVX manual states that the vector processor is attached directly to
L2.  VMEM loads and stores move data to and from L2 and do not use the L1 data
cache.  It explicitly recommends `L2FETCH` for cacheable data that VMEM will
consume, preferably:

- in requests smaller than 8 KB;
- several hundred cycles before first use;
- at row or tile granularity;
- neither so late that latency is uncovered nor so early that the line is
  evicted before use.

The V73 scalar programmer's manual further defines `L2FETCH` as a nonblocking
2-D request with byte width, height, and stride.  Requests are best effort,
lower priority than demand traffic, page constrained, and subject to a small
finite pending-request capacity.

Thus L2 prefetch is valid for HVX when the source remains in cacheable
DDR-backed memory and enough independent computation separates the hint from
the VMEM demand.  The error in the original item 7 was not "using L2 with
HVX"; it was issuing a late, layout-oblivious hint for a tiny, freshly produced
K/V stream.

### 13.3 Why DINOv2 Debug is difficult for the old item 7

After normal HVX fusion, DINOv2 Debug no longer materializes attention K/V in
the old `[B,H,S,D]` form.  Projection computation is fused into the attention
contractions and the physical activation remains `[B,S,H,D]`.  This has three
consequences:

1. the K/V data has just been produced and is likely already resident in L2;
2. forcing a separate K/V materialization can undo fusion and add a
   store/read pair;
3. a hint inserted immediately before the contraction has insufficient lead
   time to cover DDR latency.

The Debug sequence length is only 17 and its two K/V sites request 4,352 bytes
in total.  It is useful for correctness and code-generation isolation, but it
is not the strongest workload for a long-context K/V-cache optimization.
Autoregressive decode with a large DDR-resident historical cache is a more
natural item-7 target.

### 13.4 HVX placement policy: VRF, VTCM, and L2 are complementary

The V73 HVX manual recommends minimizing VMEM operations because VRF access is
cheaper than any memory access.  It describes VTCM as faster and lower power
than L2, non-evictable, free from cache-associativity management, capable of
continuous packet-level reads/writes, and mandatory for HVX scatter/gather.
These properties lead to the following placement hierarchy:

1. **VRF/fusion:** keep an immediate producer-consumer value in registers and
   avoid materialization entirely.
2. **VTCM staging:** use VTCM for a tile reused by multiple vector operations,
   heads, or consumers, or for scatter/gather and HMX-formatted data.
3. **L2 prefetch:** warm a one-pass cacheable DDR stream when it is not worth
   paying for an explicit VTCM copy.

VTCM is not automatically beneficial.  DDR-to-VTCM movement and
synchronization must be amortized or overlapped.  The manual also warns that a
VMEM load immediately following a store to the same address has a large
store-to-load penalty; approximately 15 intervening packets are recommended.
Ping-pong buffers naturally provide this separation.

External DMA is noncoherent with HVX threads and therefore requires explicit
completion polling/release/barrier handling before the consumer accesses the
destination.

### 13.5 Revised vector/HMX item-7 design

The revised implementation and evaluation order is:

1. preserve ordinary HVX fusion and identify K/V on the final contraction;
2. for fused `[B,S,H,D]`, issue a page-safe strided 2-D L2 request per head
   rather than treating the stream as contiguous;
3. measure that implementation against repeated pure HVX-vector executions;
4. for reusable or genuinely DDR-resident K/V, stage only the active tile into
   ping-pong VTCM buffers;
5. overlap DMA of tile `t+1` with HVX/HMX compute on tile `t`;
6. form the consumer/HMX layout during the producer store or DMA, eliminating
   separate transpose, reshape, pack, and unpack operations;
7. keep the query and immediate intermediates in VRF where fusion permits;
8. use 128-byte-aligned vector accesses and avoid VMEMU where practical.

For HMX, this becomes a unified data-supply pipeline: DDR prefetch, asynchronous
DMA, VTCM residency, and in-situ AH/WH layout formation.  For HVX, VTCM is
selected only when reuse and overlap compensate for the explicit movement.

Only the following vector experiments are in scope:

- pure HVX vector;
- HVX plus fusion-preserving 2-D K/V L2 prefetch;
- HVX plus K/V VTCM staging without overlap;
- HVX plus ping-pong DMA/VTCM overlap;
- HVX plus DMA/VTCM and in-situ layout formation.

Each result must use identical inputs and repetition protocol, pass the same
correctness gate, demonstrate actual HVX code and active movement operations,
and report stable end-to-end latency.  Scalar latency is no longer an
acceptance metric.

### 13.6 Vector-only results and the prefill/decode distinction

The first controlled vector experiments were run serially on DINOv2 Debug.
The current repaired LLVM backend was used, and each successful result passed
the numerical and top-1 checks.

| DINOv2 Debug configuration | Latency | Observation |
|---|---:|---|
| pure HVX vector, existing matched run | 21.258 ms | reference |
| fusion-preserving 2-D K/V L2 hint | 20.663 ms | 1.03x; four hints, 8,704 requested bytes |
| synchronous whole-stream VTCM stage | 40.150 ms | 0.53x; copy cost is not amortized |
| asynchronous whole-stream DMA/VTCM stage | 41.552 ms | 0.51x; no useful producer-consumer overlap window |

The L2 runtime reported page clipping on every DINO request: only 9,600 of
26,112 requested bytes were issued over the runtime's three internal
invocations.  The result is directionally positive but too small to satisfy
the material-speedup acceptance criterion.  Whole-stream VTCM staging is a
clear negative result for this short sequence and must not be included in the
default cumulative configuration.

Falcon Debug at sequence length 128 exposed two additional experimental
problems:

1. its runner still forced `enableVectorization=False`, so historical rows
   named HVX were not evidence of true HVX-vector execution;
2. the standalone K/V option did not enable the enclosing prefetch pass.

Both runner issues were corrected with an explicit `--enable-hvx-vector`
switch and proper item-7 pass gating.  The repaired true-vector measurements
were:

| Falcon Debug, seq=128 | Latency | Correctness |
|---|---:|---|
| HVX vector, item 7 not active | 509.019 ms | max abs 0.0236; top-5 match |
| HVX vector + active item-7 2-D K/V hints | 510.438 ms | max abs 0.0236; top-5 match |

The active run identified four attention consumers and emitted eight hints
covering 65,536 logical bytes.  It was 0.28% slower, which is noise-sized and
not a speedup.  A separate attempt to combine HexKL, the cumulative schemes,
and HVX vectorization compiled successfully but exited with status 13 on the
device.  Consequently HVX and HMX must remain separately validated codegen
profiles until that mixed-path failure is fixed.

This negative Falcon result has an important semantic explanation.  The
current benchmark sets `use_cache=False` and performs prompt prefill.  Each
layer produces K and V locally and consumes them immediately; their cache
lines have just been written and are normally still hot.  Prefetching those
fresh values cannot hide a DDR miss and adds runtime-call and `L2FETCH`
overhead.  In contrast, attention K/V-stream prefetch is naturally applicable
to autoregressive decode, where past K/V is a function input or persistent
state residing in DDR across token invocations.

The revised policy is therefore:

- do not present prefill K/V hinting as the source of the old scalar speedup;
- use producer-distance analysis to distinguish fresh local K/V from
  externally resident past K/V;
- preserve/fuse fresh K/V in VRF or form its consumer layout in situ;
- apply L2 page/tile prefetch to past K/V only when there is enough lead time;
- use VTCM only for a reused tile with an actual overlap window, not as a
  whole-stream copy;
- optimize prefill through stationary weight/layout schedules, because its
  dominant traffic is not a cold historical K/V cache.

The old scalar cumulative improvement remains useful evidence for the
combined HexKL/layout strategy, but it is not attributable to item 7.  The
next prefill implementation should use a weight-stationary or
consumer-stationary vector schedule: retain a weight/vector tile in VRF or
VTCM across multiple sequence positions, prefetch the next tile while the
current tile computes, and fold transpose/reshape formation into producer
stores.  This stays within OmniFetch's unified objective of moving data
earlier or avoiding the movement entirely.

## 14. New design space after the vector item-7 result

### 14.1 Why another list of independent prefetch mechanisms is insufficient

The earlier M1--M10 roadmap already covers page-safe `L2FETCH`, L2-versus-DMA
selection, physical-layout equivalence, VRF forwarding, aligned/page-aware
placement, VTCM bank coloring, nontemporal last use, store-to-load distance,
tiered residency, and PMU feedback.  Renaming those mechanisms would not
create a new contribution.

The vector experiments instead reveal a missing compiler abstraction.  The
compiler currently optimizes one movement site or one consumer at a time.  It
does not first ask whether several consumers can share the same physical read,
resident tile, or producer result.  Consequently it can prefetch a value that
is already hot, copy a whole stream into VTCM for one immediate consumer, or
materialize an intermediate that could have served several operations while
still in the vector register file.

The next design should be based on a **movement-amortization region (MAR)**:

```text
source tile / producer result
          |
          | one admitted movement or direct producer handoff
          v
  VRF or VTCM resident tile
     /          |          \
 consumer A  consumer B  consumer C
     \          |          /
      final-use store or eviction
```

An MAR groups compatible consumers of the same source version.  It decides:

- which loop order maximizes reuse before the tile is evicted;
- whether the tile should enter through L2 prefetch, DMA, or direct production;
- which consumer layouts can be formed during that movement;
- whether one vector load can be multicast to multiple computations;
- whether an intermediate should be forwarded, recomputed, or materialized;
- whether VRF pressure permits fusion without spills; and
- where the last use permits an nontemporal store/load.

This creates one coherent story:

> OmniFetch schedules a tile before its first unavoidable use, forms its
> required physical layout during that movement, and amortizes the tile across
> all compatible consumers before releasing its hierarchy residency.

The story remains precision-preserving and does not require quantization.

### 14.2 Manual-derived constraints used by the new design

The two V73 manuals place hard constraints on an MAR:

1. HVX VMEM accesses L2 directly and bypasses L1.  Scalar `dcfetch` is therefore
   not an HVX tensor optimization.
2. VRF access is substantially cheaper than VMEM.  Avoiding a store/load is
   preferable to replacing DDR traffic with two VTCM accesses.
3. VTCM has deterministic, non-evictable service and continuous packet-level
   access, but an explicit copy must be amortized and external DMA is
   noncoherent until completion/ownership is established.
4. `L2FETCH` is best below 8 KiB and several hundred cycles ahead of use.  It is
   low priority, page constrained, and a competing command can truncate useful
   work.
5. V73 vectors are 128 bytes.  VMEMU can touch multiple cache lines, so base
   and row-stride alignment should be a scheduling property, not a late fix.
6. A load shortly after a store to the same address can stall until the store
   reaches L2; roughly 15 independent packets are recommended.
7. Contiguous access reduces bank conflicts, cache-set aliases, and micro-TLB
   misses.  Page working set and lower address bits are part of scheduling.
8. Scatter/gather is VTCM-only and page-contained; producer-side direct
   placement is generally preferable to storing row-major and gathering later.
9. The `:nt` attribute describes final use and can protect future prefetched or
   reused data from one-pass traffic.

These facts favor reuse regions and loop/dataflow transformation over issuing
more independent hints.

### 14.3 New mechanism N1: phase-specialized stationary scheduling

Use different stationary dataflows for prefill and decode:

- **prefill:** weights are reused across many sequence/image positions, so
  make a weight/output-channel tile stationary in VRF or VTCM and process a
  strip of positions before advancing the weight tile;
- **decode:** one token reuses a large historical K/V state, so make the
  active K/V page or head tile stationary while queries stream through it;
- **convolution/vision:** choose output-channel-, input-channel-, or
  patch-stationary scheduling from reuse and VTCM/VRF pressure.

This directly targets the Falcon prefill result: prefetching fresh K/V cannot
help a run dominated by projection and vocabulary weights.  Reordering
`[sequence, output-channel, reduction]` so a weight vector serves several
sequence positions can reduce repeated VMEM reads even when the complete
weight tensor already fits in L2.

Admission test:

```text
saved VMEM bytes =
  weight_tile_bytes * (old_position_reloads - new_position_reloads)

admit only if:
  saved VMEM bytes >
  added output traffic + padding + spill bytes
```

The first implementation target should be a single static FP16 projection in
Falcon Debug, followed by GPT-2/Qwen and the QKV/MLP projections of ViT/Swin.

### 14.4 New mechanism N2: activation multicast across sibling projections

Transformer blocks contain natural sibling consumers:

- normalized activation feeding Q, K, and V projections;
- MLP activation feeding gate and up projections;
- encoder output feeding multiple cross-attention projections; and
- a feature tile feeding several convolution branches.

Current operator-by-operator lowering may reload the same activation for every
sibling.  An activation-multicast MAR loads one 128-byte-aligned activation
vector once into VRF and applies all compatible projection tiles before the
value is discarded.  If HMX is used, the activation is staged once into AH and
referenced by multiple HMX calls when the public HexKL ABI permits it.

This is stronger than ordinary fusion: the principal metric is eliminated
source reads, not merely fewer dispatches.  It composes naturally with
prefetch because the shared activation receives one lease instead of three
competing leases.

### 14.5 New mechanism N3: online reduction and consumer fusion

Large reduction intermediates should not be written and reread when a bounded
tile state suffices:

- LayerNorm/RMSNorm: retain partial sum/square-sum in vector registers and
  feed normalized tiles directly to projection staging;
- softmax: maintain online max and normalization sum by score tile, then
  consume V tiles without materializing the full score/probability matrix;
- pooling/statistics: combine reduction epilogue with the following transform
  or projection.

For attention this is an OmniFetch-compatible, hierarchy-specific online
attention path:

```text
prefetch K/V tile t+1 (<8 KiB, page-contained)
  while
Q tile + K/V tile t -> online max/sum/output accumulator in VRF/VTCM
```

It reduces score-matrix writes, softmax rereads, and V rereads, while providing
enough compute between prefetch and consumption.  This is a better prefill
target than hinting newly produced whole K/V streams immediately before use.

The implementation should begin on HVX attention; HMX matmul substitution is
an independent extension and is not required to validate reduced movement.

### 14.6 New mechanism N4: residual rendezvous

A residual tensor is produced early and consumed after several expensive
operators.  Ordinary lowering can write it to DDR/L2 and read it again at the
add.  A residual-rendezvous MAR chooses among:

- retaining a small tile in VRF across a fused local region;
- retaining a longer-live tile in a colored VTCM slot;
- processing a sequence/image strip through the block so residual production
  and addition are closer; or
- assigning an L2 lease when VTCM pressure makes explicit retention worse.

The final producer can write directly into the residual-add layout, and the
add can be fused with the last consumer store.  This removes one materialized
boundary without assuming that an entire model activation fits in VTCM.

### 14.7 New mechanism N5: recompute-versus-reload

Some values are cheaper to regenerate in VRF than to load:

- causal/window masks;
- position indices and simple affine address-derived values;
- broadcast constants and scale factors;
- cheap elementwise epilogues; and
- view/transpose address expressions.

Introduce a costed choice:

```text
recompute_cost(vector packets)
    versus
reload_cost(VMEM bytes + expected L2/TLB stalls)
```

This complements movement-equivalence analysis.  A value need not be
prefetched or retained if deterministic vector recomputation is cheaper.
Recomputation must be bit-compatible with the baseline where exact numerical
equivalence is required.

### 14.8 New mechanism N6: VRF-pressure-aware fusion and tile sizing

Fusion is beneficial only while live vector values remain in VRF.  Excessive
fusion causes spills, which silently converts “zero-copy forwarding” into more
VMEM traffic.

Add a vector-register liveness estimator to each proposed MAR:

- live Q/K/V, accumulator, normalization state, masks, and address vectors;
- vector-pair requirements of individual HVX operations;
- expected spill bytes;
- packet distance required by store/load or scatter/gather hazards; and
- remaining independent work available for L2FETCH latency hiding.

Select tile shape, number of fused consumers, and pipeline depth jointly.
Object auditing must verify that the selected region reduces VMEM
instructions and does not merely move spills to a different stack slot.

### 14.9 New mechanism N7: overwrite-aware allocation and zero suppression

Buffers that are fully overwritten should not trigger an initial read or a
separate DDR zero-fill:

- initialize reductions in VRF/HMX accumulators;
- construct full output tiles in registers and perform one aligned store;
- use VTCM scratch for partial results rather than zeroing a DDR tensor;
- avoid copy-on-initialization for destination-style ops whose entire output
  is proven overwritten; and
- mark final one-pass stores nontemporal when legal.

The scalar manual's `dczeroa` demonstrates the architectural value of
allocating a write-only cache line without fetching old contents.  HVX tensor
code should realize the same principle through full-tile vector stores,
destination ownership analysis, VTCM scratch, and cache-allocation policy
rather than routing tensors through scalar L1.

### 14.10 New mechanism N8: page-working-set supertiles

M5 proposed page-aware placement.  The stronger scheduling transformation is
to execute all compatible work for one page-contained supertile before moving
to the next page:

1. issue one page-safe 2-D L2 lease or DMA;
2. consume the resident tile across sibling operators/positions;
3. perform in-situ layout placement and local reductions;
4. release it at the final use; and
5. advance to the next page/supertile.

This reduces active-page count and micro-TLB churn while amortizing one
prefetch command.  It also prevents a burst of per-row hints from overwriting
the single-flight L2 prefetch engine.

### 14.11 New mechanism N9: HMX/HVX boundary residency

HMX and HVX should not be treated as unrelated kernels joined through DDR:

```text
HMX output in VTCM
  -> HVX bias/activation/norm/residual in the same tile
  -> in-situ AH staging for the next HMX operation
```

The MAR owns the public VTCM tile across the boundary.  HVX epilogues consume
and update it in place when aliasing is legal; the final HVX store forms the
next AH layout.  This removes HMX-output DDR materialization and the next
activation reload without relying on private HMX registers.

This direction connects cleanly to the current paper story.  General
non-aligned/batched MatMul-to-HMX lowering remains a separate “Next Paper
Idea”; boundary residency improves data supply for HMX operations that are
already legal.

### 14.12 New mechanism N10: multi-layer circular activation arena

For activation-dominated regions, process a bounded sequence/image strip
through several adjacent operators or layers using a circular colored VTCM
arena.  The strip remains resident while weights stream past it.  For
weight-dominated regions, retain the existing layer-major schedule instead.

This is not unconditionally profitable: crossing layers can increase weight
reloads.  A dynamic-programming partitioner should select region boundaries
from:

- activation bytes saved;
- additional weight bytes;
- VTCM/VRF capacity;
- residual lifetime;
- layout compatibility; and
- available prefetch overlap.

It generalizes residual rendezvous and stationary scheduling to graph regions
without requiring whole-model VTCM residency.

### 14.13 Priority and experimental gates

The new mechanisms should not be implemented in arbitrary order:

| Priority | Mechanism | Why first | Initial complete-model gate |
|---:|---|---|---|
| 1 | N1 phase-specialized stationary scheduling | directly targets the measured Falcon prefill bottleneck and repeated weight reads | Falcon/GPT-2 HVX vector |
| 2 | N2 activation multicast | common to LLM, ViT/Swin, and encoder-decoder models | Falcon plus Swin/Whisper |
| 3 | N6 VRF-pressure model | prevents N1/N2 fusion from winning only in source IR while spilling in the object | all N1/N2 models |
| 4 | N3 online reduction/attention | largest materialization reduction, but more complex correctness | GPT-2/Falcon and one vision/audio Transformer |
| 5 | N4 residual rendezvous | broad model coverage and composes with N3 | Transformer and ResNet-like model |
| 6 | N9 HMX/HVX boundary residency | connects the same movement abstraction to HMX | one already-HMX-eligible model |
| 7 | N5/N7 read suppression | smaller independent wins and useful cleanup | three domains |
| 8 | N8/N10 supertile/region partitioning | global scheduling after local cost models are trustworthy | selected full models |

Before changing code, each candidate site should emit a movement ledger:

```text
baseline physical reads/writes
predicted eliminated bytes
new prefetch/DMA bytes
expected reuse count
VRF and VTCM footprint
active pages and alignment padding
predicted spills
first-use distance
```

A mechanism enters the cumulative row only when:

1. the final object shows fewer VMEM operations or fewer physical bytes;
2. no new spill, alignment, page, or ownership problem invalidates the model;
3. identical complete-model inputs pass the correctness gate;
4. HVX vector improves over HVX vector, or the HMX-enabled path improves over
   its matched HexKL baseline; and
5. repeated serial measurements show a stable improvement beyond noise.

This replaces blind feature trials with falsifiable, byte-level predictions.

## 15. Return to complete-model execution (2026-08-06)

### 15.1 What the independent prefetch baselines established

The two-model Debug comparison did establish a useful OmniFetch advantage, but
the claim must stay scoped to those workloads.  For DINOv2 Debug, OmniFetch
issued 36 runtime prefetch requests versus 122,843 for the independent
baseline.  For ViT Debug it issued 72 versus 36,274.  Latency improved only
slightly, but the several-orders-of-magnitude reduction in commands is evidence
that reuse-aware coalescing can avoid a prefetch-command storm.

It is not valid to generalize that count result to every complete model.  The
first true-vector full HuBERT run below exposed the opposite regime and is now
the reason for adding an explicit traffic gate before continuing the remaining
15-model matrix.

### 15.2 Correct complete-model execution order

The balanced target remains five language/text, five vision, and five
speech/audio models.  Execution is strictly serial:

1. make every `hvx` row explicitly select the repaired `hvx-vector` profile;
2. screen one complete model in all three configurations;
3. preserve the first device failure and diagnose it instead of retrying;
4. reject or cap a cumulative plan whose predicted/runtime prefetch traffic
   scales pathologically with layer count;
5. continue HuBERT-family audio models, then language, then missing vision
   rows; and
6. only promote passing screens to repeated median/p90 experiments.

The full-matrix script previously used configuration name `hvx` without
explicitly selecting vector lowering for every runner.  It now passes
`--backend-profile hvx-vector` to the shared vision/audio runners and
`--enable-hvx-vector` to custom language/Stable-Diffusion runners.  Therefore
older full rows without `profile=hvx-vector vectorization=1` must not be mixed
with the new matrix.

### 15.3 Exit 13 diagnosis and fail-fast policy

The first HuBERT HVX execution returned exit 13.  The old executor silently
retried it, discarding the stdout from each failed attempt.  The SDK definition
is:

```text
AEE_EBADSTATE = 0x8000040d
```

The preserved HexKL stdout subsequently reported
`-2147482611 = 0x8000040d`, followed by `Failed to close handle`.  In both the
HVX and HexKL cases, `main()` had already written a non-empty `perf.txt` and the
complete `1x64x32` FP16 output.  The recovered HVX output was compared on the
host against the identical seed, model, and input:

```text
finite=True
max_abs_diff=0.009094
mean_abs_diff=0.001851
last_frame_top1_match=True
```

This instance of exit 13 is consequently a FastRPC/ribbon teardown-state
failure after valid computation, not a model-kernel failure.  The executor now:

- prints the first failing command's stdout and stderr;
- never retries exit 13;
- accepts exit 13 only when this invocation produced every expected output and
  `perf.txt`; and
- still requires the model's numerical correctness gate to pass.

`run_full_model_matrix.sh` also sets device retries to one.  A genuine exit 13
without complete files fails immediately and preserves the original site.

### 15.4 Full HuBERT-base true-vector result

This is the complete 12-layer, hidden-768, 12-head, intermediate-3072
HuBERT-base structure with 94,396,192 parameters and a 20,560-sample input
(64 encoder frames).  All rows use a 512 MiB QuRT heap and one measured device
invocation.

| Configuration | Host codegen | Device latency | Correctness |
|---|---:|---:|---|
| HVX vector | 273.9298 s | 212,074.277 ms | finite; max error 0.0091; top-1 match |
| HexKL + HVX vector | 176.5813 s | 268,255.667 ms | finite; max error 0.0088; top-1 match |
| HexKL + items 1-7 + HVX vector | 174.9162 s | 620,050.837 ms | finite; max error 0.0088; top-1 match |

HexKL is 1.265x slower than HVX.  The cumulative row is 2.312x slower than
HexKL (`HexKL/combo = 0.4326x`) and 2.924x slower than HVX.  This is a negative
full-model result; it must not be replaced by the much better one-layer Debug
screen.

The persistent wrapper's outer `Perf` timer was inconsistent with its command
wall time: `Perf=921.902 s`, while cold=270.895 s, invalidated=619.855 s, and
the whole ADB command was 1511.396 s.  The one warm sample
`PerfP50=620.051 s` closes the timing identity to within wrapper overhead, so
the full-matrix parser now uses the persistent warm sample for this row.  The
raw contradictory values remain in the log for auditability.

### 15.5 Why the complete model regressed

The cumulative compiler/runtime counters were:

```text
prefetch sites            220
in-situ operations        220
sync choices               74
async choices              73
persistent choices         73
VTCM peak bytes saved  413,696
runtime L2 requests    386,294
page-clipped requests  386,294
requested bytes       791,130,112
issued bytes           50,142,592
WH cold hits/misses     39,421 / 127,235
WH total hits/misses    78,388 / 254,924
```

No attention-K/V sites were selected (`sites=0`).  The regression is therefore
not caused by K/V stream prefetch.  It comes from applying layout-fusion and
weight movement choices independently at nearly every repeated layer/site.
Every L2 request was page-clipped, and the persistent cache had substantially
more misses than hits.  The Debug policy's selectivity did not scale to the
12-layer graph.

Before running the structurally similar Wav2Vec2/UniSpeech models, the
cumulative planner needs a full-graph traffic budget and reuse gate:

```text
accept only if predicted_saved_reload_bytes
  > issued_prefetch_bytes + command_cost + cache_miss_cost
```

The gate must also cap commands per unique physical page, deduplicate repeated
layer-local requests, and reject persistence when predicted misses exceed
reuse hits.  The full HuBERT row is the first mandatory negative test for that
policy; the two Debug baseline models remain positive selectivity tests.

## 16. HVX execution audit and Attention K/V prefetch (2026-08-06)

### 16.1 Is the complete model really running on HVX?

Yes, but “HVX is enabled” must not be confused with “all important work is
efficiently vectorized.” The executed full-HuBERT HVX object was audited with
the V73 `hexagon-llvm-objdump`, rather than relying only on the Python option:

```text
plain HVX object:
  instruction lines       116,139
  HVX instruction lines    55,057  (47.41%)
  vmem lines                8,159
  vector-math lines        16,579

HVX + K/V-prefetch object:
  instruction lines       132,665
  HVX instruction lines    59,749  (45.04%)
  vmem lines                9,971
  vector-math lines        20,217
```

The objects contain real V73 instructions including `vmem`, `vmpy`, `vadd`,
`vlut16`, `vzxt`, and `vsplat`. Therefore the 212-second result is not a
scalar-only execution. Static instruction share is not dynamic time share,
but it proves that HVX code executes while also exposing incomplete coverage:
more than half of the static instruction stream is still scalar,
address/control, or helper code.

The matched profile was:

```text
profile=hvx-vector vectorization=1 vtcm_tiling=0 hexagonmem=1 hexkl=0
```

Consequently, the current row uses HVX but not the compiler's VTCM tiling
profile. The remaining latency is consistent with a 94.4M-parameter monolithic
graph, generic unfused attention/softmax/transpose paths, partial vector
coverage, and DDR/L2-resident intermediate traffic. It is not evidence that
the Scalar Processor alone ran the model.

### 16.2 Why item 7 previously selected zero HuBERT sites

HuBERT exports each attention contraction as `[12,64,64]`: batch is flattened
to the 12 heads, sequence length is 64, and head dimension is also 64. The old
inference deliberately required `sequence != head_dim` to avoid guessing that
an arbitrary square batch matmul was attention. It therefore emitted
`KVCachePrefetch sites=0`.

The repaired analysis recognizes square attention structurally:

- QK is proved by the explicit/folded last-two-dimension K transpose;
- AV is paired with QK and its softmax/head-layout dataflow;
- K/V identity is attached before generalization/fusion;
- semantic attributes are preserved across HVX scheduling and tiling; and
- surviving SCF tile loops carry identity through vectorization and
  bufferization, after which requests are deduplicated by physical source.

For the complete 12-layer model, the final compiler result is:

```text
semantic K/V sites       24  (12 K + 12 V)
tiling carriers          72
deduplicated sites       24
coalesced L2 hints      288
logical pages           576
ordinary prefetch sites   0
```

This is an isolated K/V experiment; it does not enable the ordinary HexKL/HVX
weight-prefetch loop or items 1-6.

### 16.3 Complete-model K/V L2-prefetch result

The model, seed, input, heap, vector profile, and one-invocation timing policy
match section 15.4.

| Configuration | Host codegen | Device latency | Relative to plain HVX | Correctness |
|---|---:|---:|---:|---|
| HVX vector | 273.9298 s | 212,074.277 ms | 1.0000x | finite; top-1 match |
| HVX vector + K/V L2 prefetch-only | 373.6175 s | 217,146.200 ms | 0.9766x | finite; max error 0.0088; top-1 match |

The L2-only K/V experiment regressed latency by 2.39%. This falsifies the
claim that attention K/V L2 hints are automatically beneficial on this full
HuBERT shape. With only 64 sequence tokens, each K/V stream is small and is
consumed shortly after production; 288 hint commands add cost while much of
the data is plausibly already cache-resident.

There is one attribution caveat: preserving semantic K/V identity currently
keeps the marked attention boundary intact and disables the slicing rewrite
that drops those attributes. Thus this row measures the deployable
`boundary preservation + K/V hints` implementation, not a hypothetical
zero-codegen-difference hint toggle. A metadata/boundary-only control is
needed before assigning the full 2.39% delta solely to the hardware hints.

This negative L2 result does **not** settle the DMA-to-VTCM design. The next
memory-hierarchy experiment should stage only a bounded K/V tile or page into
double-buffered VTCM, preserve its subview offsets, overlap DMA with the
preceding projection/softmax work, and reject the transform unless predicted
DDR-stall savings exceed DMA, synchronization, and VTCM-pressure costs. A
full-tensor synchronous copy is not an acceptable substitute.

Reproduction command:

```bash
scripts/run_full_model_matrix.sh \
  --config hvx_kv_prefetch \
  --dsp-heap-mb 512 \
  --timeout 1200 \
  --output-dir /tmp/omnifetch-full-hubert-kv-result-20260806 \
  hubert-base
```

## 17. Latest upstream v73 bring-up and first complete-model PASS (2026-08-08)

### 17.1 Revisions and build configuration

The clean native baseline is Qualcomm `hexagon-mlir` revision
`9b4b8fcea2b93c801b5de784ee750ca9350d504f`, with LLVM pinned by that checkout
at `ac5dc54d509169d387fcfd495d71853d81c46484`.  The phone is the SM8550
(`kalama`) CPH2449 and the runtime/toolchain artifacts are explicitly v73.
Only the Hexagon backend is built; AMDGPU, NVIDIA/PTX, Proton, examples, and
unrelated Triton tools are disabled.  The native qcom suite reports 148 PASS,
one upstream UNSUPPORTED test, and zero FAIL.

The host plugin is compiled with `-O2 -g`; this only affects the compiler
executable.  Generated model code uses LLVM optimization level 3 and is linked
with `hexagon-clang++ -O3 -mv73 -mhvx`, so the host-tool `-O2` does not reduce
device inference optimization.

### 17.2 Compatibility and v73 code-generation defects repaired

Three independent failures had to be separated before a valid device result
was possible:

1. Full-model memref returns were changed to caller-owned trailing out params.
   The conversion must use `modifyPublicFunctions=true`; otherwise the public
   C interface retains the old by-value return descriptor ABI while the host
   wrapper passes an output tensor.
2. LLVM Hexagon commit `689ecf880373bb4e0f01ed5e004f19a466e869dc`
   fixes backend crashes on truncating DoubleRegs-to-IntRegs COPY operations.
3. LLVM commits `3ef59d80c5ce51738a055d9e8eb98aa3c8effb2f` and
   `2e10b62995915d35ba528872e70aacda7223bd18` fix the aligned-stack pointer
   (`r16`) use-before-definition and its live-in tracking.  Before these fixes,
   the generated DINOv2 prologue accessed `r16` before assigning it, producing
   a Bad VA at function entry.

These are baseline correctness/compatibility repairs and must not be counted
as OmniFetch speedups.  They are maintained as reversible patches rather than
mixed into the OmniFetch optimization implementation.

### 17.3 Full-graph VTCM staging failure and baseline policy

With upstream `enableVTCMTiling=true`, full DINOv2 lowers to 822 VTCM
allocations totaling 189,891,400 bytes; the largest individual allocation is
only 792,588 bytes.  All 822 frees are placed near function exit.  The runtime
VTCM pool returns null after exhaustion, and the unchecked pointer eventually
reaches `HexagonBuffer::CopyFrom`/`memcpy`, causing exit 13.

Disabling `buffer-loop-hoisting` did not change these counts or lifetimes.  The
deeper limitation is that one-shot full-graph bufferization/deallocation keeps
the independently tiled staging buffers alive across the monolithic linear
graph.  The existing `scratch/MemoryOffsets` path is not yet a solution because
it explicitly flat-sums all allocations without liveness reuse and would also
require roughly 190 MB.

For the native full-model baseline, both HVX and HexKL therefore use the
existing upstream option `enableVTCMTiling=false`.  This does not disable HVX:
the audited LLVM IR still contains 1,723 Hexagon HVX intrinsic references,
including 128-byte `V6.vmpy`, `V6.vadd`, and `V6.vlut` operations.  HexKL's
own local VTCM allocations are introduced later and explicitly deallocated.
A bounded lifetime-aware VTCM coloring/planning implementation remains a
separate OmniFetch optimization candidate.

### 17.4 Complete DINOv2-small result on the latest native baseline

Both rows use the identical 12-layer, hidden-384, six-head, intermediate-1536,
22,825,192-parameter random full-structure model, input `[1,3,224,224]` in
FP16, output `[1,1000]`, one device iteration, a 512 MB DSP heap, and v73.

| Native configuration | Host compile | Device latency | Correctness |
|---|---:|---:|---|
| HVX vector | 255.9736 s | 30,396.932 ms | finite; max error 0.0054; top-1 match |
| HexKL | 255.5466 s | 9,873.013 ms | finite; max error 0.0049; top-1 match |

HexKL is 3.078x faster than native HVX for this model.  The strict HexKL row
does not apply the project's host-side batch-matmul rewrite, so this is an
upstream-native baseline rather than OmniFetch.  Logs and CSV output are under
`/tmp/hexagon-mlir-native-v73-smoke-dinov2`; generated files are deliberately
not tracked by Git.

Reproduction command:

```bash
ANDROID_SERIAL=49d1c7b2 HEXAGON_ARCH_VERSION=73 \
scripts/run_full_model_matrix.sh --native-only --no-timeout --force \
  --runtime-root /home/huzq85/2-working/hexagon_npu/hexagon-mlir-native \
  --output-dir /tmp/hexagon-mlir-native-v73-smoke-dinov2 \
  dinov2-small
```

### 17.5 Second complete-model validation: DeiT-small

DeiT-small validates that the compatibility policy is not DINOv2-specific.
The tested graph has 12 layers, hidden size 384, six attention heads,
intermediate size 1536, 198 tokens, and 21,975,400 parameters.  It uses the
same v73, FP16, one-iteration, 512 MB heap, strict-native, and correctness
policy as section 17.4.

| Native configuration | Device latency | Relative result | Correctness |
|---|---:|---:|---|
| HVX vector | 24,075.094 ms | 1.000x | PASS |
| HexKL | 8,342.079 ms | 2.886x faster | PASS |

The successful HVX row again uses `vectorization=1, vtcm_tiling=0`; this is a
vector baseline without the unsafe monolithic-graph VTCM staging policy, not a
scalar fallback.  Logs and CSV output are under
`/tmp/hexagon-mlir-native-v73-smoke-deit` and are not tracked by Git.

### 17.6 Cross-domain validation: Whisper-tiny

Whisper-tiny exercises a materially different graph: an 80x3000 log-mel input,
four encoder and four decoder layers, d_model 384, six heads, target length 32,
37,760,640 parameters, convolutional audio frontend, self-attention, and
cross-attention.  The exported graph contains 89 batch matmuls.

| Native configuration | Device latency | Relative result | Correctness |
|---|---:|---:|---|
| HVX vector | 160,182.939 ms | 1.000x | PASS |
| HexKL | 112,048.671 ms | 1.430x faster | PASS |

Both rows use the same full graph and strict-native policy, and neither uses
host-side batch-matmul rewriting.  The smaller HexKL gain than on DINOv2/DeiT
is consistent with limited automatic HexKL coverage of the 89 batch matmuls
plus substantial convolution, attention-normalization, layout, and data
movement work outside eligible matrix kernels.  This is an observed
correlation; operator-level profiling is required before assigning exact time
shares.  Logs and CSV output are under
`/tmp/hexagon-mlir-native-v73-smoke-whisper` and are not tracked by Git.
