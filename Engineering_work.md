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
