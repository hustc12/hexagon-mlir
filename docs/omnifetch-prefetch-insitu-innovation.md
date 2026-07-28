# OmniFetch: Prefetch and In-situ Reshape Innovation Plan

## Scope and evaluation rule

HexKL remains the compute baseline. OmniFetch should optimize how HexKL-ready
data is produced, moved, retained, and scheduled rather than duplicate the
closed-source HMX kernels.

Quantization and runtime dequantization are no longer part of the planned
OmniFetch path. The active scope is prefetch, DMA, in-situ layout production,
removal of redundant reshape/transpose/copy operations, and memory-hierarchy
placement that reduces or advances data movement without changing model
precision.

Performance claims must come from models in `benchmark_models`, not standalone
GEMM tests. Reduced-depth runners may screen compiler/runtime mechanisms, but
final claims require the published-topology runner, real weights where
available, and comparison against both Hexagon-MLIR and Hexagon NN Library.

## What in-situ reshape should mean

The current layout-aware path combines movement with RM-to-AH/WH conversion.
That removes an intermediate materialization, but it still treats the converted
tile as a short-lived implementation detail of one matmul.

The larger opportunity is to make a transformed tile a compiler-visible value
with:

- an identity: source allocation, byte range, source version, target layout;
- a lifetime: first and last compatible consumer;
- a placement: DDR, L2, VTCM bank/offset, AH, or WH;
- a readiness event; and
- a reuse count and eviction cost.

Prefetch then becomes production of a future layout value, not just early byte
movement. This is the foundation for the following innovations.

## Ranked innovation candidates

### P0: Layout-value liveness and reshape memoization

Create a `layout_value` abstraction for AH/WH tiles. Reuse a transformed tile
when subsequent HMX consumers request the same source version and layout.

For constant weights, the version is immutable. For activations, versioning can
be derived from MemorySSA-like writes. This permits safe elimination of repeated
RM-to-WH/AH conversion without relying on fragile local def-use patterns.

Expected value:

- weight WH layout can survive all M tiles, layers with tied weights, and decode
  tokens;
- activation AH can flow directly into a compatible projection; and
- legality is explicit instead of inferred from adjacent operations.

First implementation should remain function-local and constant-weight-only.

### P0: Transform-aware profitability model

Choose among:

1. ordinary HexKL reshape;
2. synchronous in-situ reshape;
3. asynchronous prefetch plus reshape;
4. persistent prepacked layout; and
5. no HexKL conversion.

Estimate:

```text
benefit =
  eliminated_copies
  + eliminated_reshapes
  + hidden_memory_latency
  + reuse_count * saved_transform_cost
  - synchronization
  - queueing
  - VTCM_pressure
  - cache_pollution
  - layout_eviction_cost
```

Static shape, tile count, loop nesting, constant/activation status, and reuse
distance provide compile-time features. Runtime stall counters and queue
occupancy provide feedback. A per-site decision is preferable to one global
lookahead.

### P1: Layout-carrying producer/consumer fusion

Propagate the consumer's desired layout backwards through reshape-only,
collapse/expand, transpose, slice, and copy chains. Where legal, make the
producer write directly into AH/WH-compatible staging.

Examples:

- norm output directly staged for the QKV projection;
- Q/K projection output directly staged for attention;
- attention output directly staged for the output projection; and
- MLP activation directly staged for the down projection.

This generalizes `LayoutOpsElimination`, whose current local walk can recognize
a redundant `collapse_shape` but may be unable to delete it because uses remain.
The new approach rewrites all compatible consumers around a shared layout value.

### P1: Cross-token persistent WH cache

Decode repeatedly uses the same weights. Convert constant weight tiles to WH
once, retain a bounded cache across token invocations, and prefetch only misses.

Required mechanisms:

- model/invocation context rather than process-global anonymous state;
- cache keys containing weight identity and layout parameters;
- VTCM residency policy for hot tiles and DDR-packed fallback for the rest;
- generation-safe invalidation when a model is unloaded; and
- separate prefill and decode policies.

This should have much larger leverage than per-inference reshape hoisting.

### P1: Two-dimensional pipeline

Pipeline both K tiles and future consumers:

```text
DMA/load tile t+2
  -> in-situ reshape tile t+1
    -> HMX consume tile t
```

Use distinct staging slots and readiness events for load and transform. The
compiler should select the number of slots from measured load/reshape/compute
latencies rather than hard-code double buffering.

### P1: VTCM lifetime coloring

Build an interference graph for activation, weight, accumulator, and prefetch
tiles. Assign VTCM offsets by lifetime coloring so buffers with disjoint
lifetimes share storage, while live prefetch and compute tiles cannot alias.

This can increase legal lookahead without increasing total VTCM and makes
persistent layout caching less likely to evict useful compute buffers.

### P1: KV-cache-aware prefetch and layout

For autoregressive models:

- retain K/V in the layout consumed by attention;
- prefetch only the pages touched by the next attention step;
- combine RoPE application with K staging where legal;
- specialize for GQA/MQA reuse; and
- select sliding-window or paged policies from the model configuration.

Unlike weight-only prefetch, this attacks a decode-critical memory stream.

### Retired: Prefetch plus dequantization/reshape

The W8A16 prototype fused:

```text
packed quantized load -> dequantize -> WH reshape -> HMX-ready tile
```

It passed its model-level correctness screen but increased Falcon debug latency
by 12.11%. More importantly, quantization changes the research scope away from
the intended data-movement contribution. It remains documented as a negative
experiment but must not be enabled in the cumulative OmniFetch configuration or
used as the basis of future work.

### P2: Critical-path prefetch across operators

Schedule a future operator's weight/layout production only when it lies on the
predicted critical path. Avoid speculative prefetch for both branches or for
operators separated by long HVX work that naturally hides memory latency.

Use model graph order, operator timing history, and queue/VTCM pressure. This is
more precise than enabling all outer-loop prefetch.

### P2: Thermal- and token-phase-aware control

Use distinct policies for cold start, prefill, first decode token, and steady
decode. Reduce concurrency under thermal throttling or memory saturation.
Report the policy state with every benchmark so adaptation remains reproducible.

## Recommended implementation order

1. Add compiler-visible layout identities and constant-weight WH liveness.
2. Add per-site static profitability reporting without changing codegen.
3. Use that report to select sync in-situ, async in-situ, or persistent layout.
4. Introduce an explicit model/invocation context and cross-token WH cache.
5. Add VTCM lifetime coloring and multi-stage load/reshape/compute scheduling.
6. Extend to activation chains and KV cache without changing model precision.

Each stage needs an off switch and model-level A/B testing.

## Ordered implementation tracker

The eight directions are implemented strictly in this order. A later item may
be designed in advance, but it must not be enabled in the measured cumulative
configuration until the earlier items have passed correctness and model-level
regression gates.

| Order | Direction | First deliverable | Status |
|---:|---|---|---|
| 1 | layout-value liveness and reshape memoization | stable layout-site identity, source classification, reuse/liveness statistics, then constant-weight reuse | analysis foundation complete; codegen reuse pending item-2 gate |
| 2 | transform-aware profitability model | per-site choice among no transform, sync, async, and persistent layout | initial static model implemented; repeated model-level gate pending |
| 3 | layout-carrying producer/consumer fusion | propagate requested AH/WH layout through view/reshape chains | conservative activation-view fusion implemented; repeated model-level gate pending |
| 4 | cross-token persistent WH cache | model-context cache with generation-safe invalidation | implemented end to end; Falcon model/device correctness, warm-hit, and invalidation gates passed |
| 5 | two-dimensional load/reshape/compute pipeline | independently scheduled load and transform readiness | implemented; compiler/runtime tests and Falcon model/device gate passed; repeated performance gate pending |
| 6 | VTCM lifetime coloring | interference-based VTCM offset assignment | implemented; compiler/runtime tests and Falcon model/device gate passed; repeated performance gate pending |
| 7 | KV-cache-aware prefetch and layout | page-aware K/V staging and attention-consumer layout | compiler path implemented; Falcon prefill/device gate passed; true cross-token decode and repeated performance gates pending |
| 8 | prefetch plus dequantization/reshape | fused compressed load, dequantization, and WH production | retired: correctness passed but latency regressed 12.11%; excluded from the project direction and cumulative configuration |

Thermal/token-phase control and critical-path scheduling are cross-cutting
policies. They will be inputs to item 2 and are not counted as extra enabled
features outside this order.

## Mandatory three-way timing protocol

Every model experiment has exactly these primary rows:

| Row | Meaning |
|---|---|
| HVX | Hexagon-MLIR with HexKL and all OmniFetch features disabled |
| HexKL | the same model/input with HexKL enabled and OmniFetch disabled |
| HexKL + cumulative OmniFetch | HexKL plus only the innovation items that have passed the ordered gate |

Hexagon NN Library remains an additional external baseline when its equivalent
model is available. It must not replace any of the three primary rows.

For each row, keep checkpoint/random seed, model topology, input tensors,
sequence/image size, precision, output scope, warm-up, device power/thermal
state, and timing boundary identical. Report device p50/p90/p99 over repeated
runs rather than treating a single run as a final performance claim.

An item enters the cumulative row only when:

1. model output passes the same numerical check as HexKL;
2. no model in the negative-control set regresses beyond the declared noise
   band;
3. the compiler report proves that the intended sites were exercised; and
4. the result is repeatable across interleaved run order.

## Item 1 implementation status

The first compiler-visible analysis stage is implemented in
`LayoutOpsEliminationPass.cpp`.

Each layout-producing `prefetch_in_situ` now records:

- `omni_fetch.layout_value_id`;
- `omni_fetch.layout_source_kind`;
- `omni_fetch.layout_site_occurrences`;
- `omni_fetch.layout_estimated_executions`; and
- `omni_fetch.layout_dest_users`.

The function records unique site, reusable site, and prefetch-instance totals.
Identity traces through `memref.subview`, `memref.cast`, and `tensor.cast` to a
root SSA value and includes the target layout and HexKL tile parameters.
Estimated executions multiply enclosing static `scf.for` trip counts; dynamic
loops are conservatively marked reusable/unknown.

This stage intentionally does not memoize or retain a tile yet. The previous
unconditional column-resident WH experiment regressed Falcon, so a site may
change codegen only after item 2 proves profitability.

Supporting infrastructure completed with item 1:

- OmniFetch enum attributes now have textual parse/print support;
- PrefetchInsert, V-DAE, and LayoutOpsElimination are registered in
  `linalg-hexagon-opt`; and
- `omnifetch_layout_value_analysis.mlir` verifies identity grouping and static
  loop execution counts.

Build and the targeted LIT test pass. On Falcon debug sequence length 128, the
analysis reports 10 layout-prefetch instances, 10 unique sites, and all 10 as
runtime-reusable because they execute in enclosing loops.

### First mandatory three-way snapshot

With deterministic Falcon debug weights and identical sequence length 128:

| Primary row | Device time | Numerical result |
|---|---:|---|
| HVX | 11397.344 ms | exact top-5 check passed, max abs 0.0237 |
| HexKL | 1686.627 ms | top-1 check passed, max abs 0.0239 |
| HexKL + cumulative OmniFetch | 1689.332 ms | top-1 check passed, max abs 0.0239 |

HexKL is about 6.76 times faster than HVX. The cumulative row is about 0.16%
slower than HexKL in this single ordered run. A subsequent cumulative-only run
measured 1676.665 ms, confirming that differences at this scale require
repeated interleaved p50/p90/p99 measurements. Item 1 itself is analysis-only,
so it is not expected to change device execution time.

## Item 2 implementation status

The first transform-aware profitability model is implemented in
`PrefetchInsertPass.cpp`. It makes a per-weight-site decision among:

- `native`: retain the HexKL `rm_to_wh` transform and avoid constructing a
  one-use synchronous OmniFetch wrapper;
- `sync`: fuse the transform into a synchronous in-situ prefetch;
- `async`: emit the existing lookahead in-situ pipeline; and
- `persistent candidate`: report cross-outer-loop reuse without yet enabling
  the previously regressing persistent-WH code generation.

The initial policy is deliberately conservative. A weight site selects async
only when lookahead is enabled, its static inner trip count and useful tile
count are both at least eight, and
`100 * useful_tiles - 800 >= 0`. Lookahead zero selects sync. Short or
dynamically unknown weight loops remain native. Static outer reuse of at least
four, or dynamically unknown outer reuse, marks a persistent candidate.
Activation `copy + rm_to_ah` sites retain synchronous in-situ fusion because
copy elimination is already part of their benefit.

The constants above are stable compiler cost units, not calibrated Hexagon
cycles. They establish deterministic selection and observability; repeated
model measurements must later fit the coefficients and thresholds.

Every decided site records `omni_fetch.transform_mode`,
`omni_fetch.transform_score`, `omni_fetch.transform_useful_tiles`, and
`omni_fetch.transform_outer_reuse`. Persistent candidates additionally record
`omni_fetch.persistent_candidate`. Function-level native/sync/async/persistent
counts and `[TransformCostModel]` diagnostics make it possible to prove which
paths a model actually exercised.

`omnifetch_transform_cost_model.mlir` covers a short native weight site, a
synchronous activation site, a long async weight site, and persistent-candidate
reporting. Both this test and the item-1 layout-value LIT test pass after an
incremental rebuild.

### Item 2 Falcon decision report

For Falcon debug at sequence length 128, the compiler selects:

| Decision | Sites |
|---|---:|
| native weight transform | 5 |
| synchronous activation in-situ transform | 5 |
| asynchronous weight transform | 0 |
| persistent-WH candidate (report only) | 5 |

Each weight loop has only two useful tiles, yielding score `-600`, so the model
correctly rejects asynchronous setup. Their estimated outer reuse is 8, 24, or
512, which identifies future persistent-cache opportunities without enabling
the known-regressing loop interchange. Because native weight sites no longer
create one-use OmniFetch operations, item-1 layout analysis now observes the
five emitted activation sites rather than ten mixed sites.

### Item 2 three-way snapshot

With deterministic Falcon debug weights and identical sequence length 128:

| Primary row | Device time | Numerical result |
|---|---:|---|
| HVX | 11327.209 ms | exact top-5 check passed, max abs 0.0237 |
| HexKL | 1692.119 ms | top-1 check passed, max abs 0.0239 |
| HexKL + cumulative items 1-2 | 1686.430 ms | top-1 check passed, max abs 0.0239 |

HexKL is about 6.69 times faster than HVX. The cumulative row is about 0.34%
faster than HexKL in this one ordered run. This is encouraging mechanism-screen
evidence, not a passed performance gate: the difference is close to expected
run-to-run noise and still needs interleaved repeated p50/p90/p99 data and the
remaining model matrix.

The next ordered item, layout-carrying producer/consumer fusion, should first
target activation view/reshape chains: these are the five Falcon sites that
remain on the emitted OmniFetch path. Weight layout propagation should remain
separate from persistent-WH enablement until the cost model can account for
VTCM footprint and cache lifetime.

## Item 3 implementation status

The first layout-carrying producer/consumer fusion is implemented in
`LayoutOpsEliminationPass.cpp`. For a HexKL HMX activation prefetch carrying
the full six tile parameters, it walks backward through:

- `memref.cast`; and
- `memref.collapse_shape` whose source and result have static shapes, equal
  element counts, and statically provable contiguous strides.

The prefetch source is rewired directly to the producer-side memref and dead
metadata views are erased. This is valid for the current HexKL activation
runtime contract because it consumes an aligned base pointer and explicit
`tile_row`, `tile_col`, `src_cols`, and `src_rows`; the source descriptor rank
does not participate in its addressing.

The implementation deliberately does not bypass dynamic/non-contiguous
collapses, subviews with an offset, transpose/permute operations, or arbitrary
producer computation. Those cases require an explicit affine-layout proof
rather than pointer equality.

Each fused prefetch records:

- `omni_fetch.layout_carried_from_producer`;
- `omni_fetch.layout_carried_view_depth`.

The function records `omni_fetch.layout_carried_sites` and
`omni_fetch.layout_carried_views`, and emits a `[LayoutCarryFusion]` summary.
`omnifetch_layout_carry_fusion.mlir` verifies both a fused static-contiguous
case and a rejected dynamic-shape case. The new test plus the item-1 and item-2
regression tests pass after the incremental rebuild.

### Item 3 Falcon report

Falcon debug sequence length 128 has five activation prefetch sites. Two sites
are fed by contiguous `memref.collapse_shape` views from
`memref<128x2x32xf16>` to `memref<128x64xf16>`. Both are carried into the
prefetch consumer and erased:

| Layout-carry statistic | Count |
|---|---:|
| activation prefetch sites | 5 |
| fused sites | 2 |
| bypassed/dead views | 2 |

The post-fix cumulative run measured 1665.442 ms and passed the same top-1
check with max absolute logit difference 0.0239. In the immediately preceding
ordered three-way run, HVX measured 11287.286 ms and HexKL measured
1719.584 ms; the pre-fix cumulative row measured 1674.583 ms but reported zero
item-3 hits. A diagnostic cumulative run between the two implementations
measured 1699.544 ms, further showing the device-level variance.

Therefore item 3 has passed its compiler-path and numerical gates, but not yet
its performance gate. The apparent cumulative advantage over that HexKL sample
is about 3.15%, while the approximately 0.55% difference from the pre-fix
cumulative sample is not a controlled attribution. Interleaved repeated runs
are required before enabling item 3 in a publication result.

## Item 4 implementation and validation

The cross-invocation WH cache is now implemented end to end. It is intentionally
distinct from the existing function-scoped persistent VTCM arena and
column-local weight prepack: both of those lose their contents or identity at
the end of one invocation.

The runtime cache contains 2048 runtime-owned, 128-byte-aligned, 4096-byte WH
tile slots in DDR (about 8 MiB of tile payload). Eight-probe hashing replaced
the original linear 64-slot prototype after model testing showed that Falcon's
1152 tile accesses per forward thrashed the smaller cache. A cache key contains:

- embedding-runtime model context ID;
- model generation;
- cache epoch;
- source base pointer;
- weight tile row and column; and
- source column count; and
- compiler-assigned transform-site ID.

The site ID is required even when the source pointer participates in the key:
temporary allocator addresses are reused across distinct constant weights in
the generated Falcon module. Omitting the site ID produced stale-tile aliasing
and a numerical failure; adding it restored the expected result.

`__omni_fetch_wh_cache_set_context(context, generation)` selects the active
generation. The embedding runtime must increment generation whenever weights
belonging to the same context may change.
`__omni_fetch_wh_cache_invalidate(context, generation)` explicitly invalidates
that generation, and `__omni_fetch_wh_cache_stats()` exposes packed hit/miss
counters. This prevents pointer reuse across model reloads from silently
returning stale WH data when the host follows the generation contract.

The HMX-weight runtime path reserves `lookahead == -1` for persistent-cache
mode. A miss executes the exact HexKL RM-to-WH transform and stores the
resulting tile in the DDR cache; a hit copies the already transformed WH tile
into the current invocation's VTCM slot. Ordinary lookahead values are
unchanged.

The compiler exposes `enable-persistent-wh-cache` through the Python backend,
Linalg-to-LLVM pipeline, and PrefetchInsert pass. The transform cost model
selects persistent mode only for weight sites with statically high or unknown
outer reuse. Falcon selects five persistent weight sites and five synchronous
activation sites. Persistent prefetches carry the site ID in their runtime
parameters and do not acquire V-DAE wait/signal synchronization.

The Falcon runner keeps one loaded DSP module alive for a cold invocation,
multiple warm invocations, and an explicit invalidation probe. It sets a stable
function-derived context and configurable generation, and appends cold/warm
latency plus hit/miss counters to `perf.txt`. The model ablation script now uses
three device iterations and enables item 4 in the Falcon cumulative row.

On the final sequence-length-128 Falcon debug run:

- cold: 1722.707 ms, 864 hits and 288 misses;
- three warm invocations: 1612.315 ms average, 3432 hits and 24 misses in
  aggregate, or 1144 hits and 8 misses per invocation (99.31% hit rate);
- after explicit invalidation: 1586.472 ms, with the counters increasing by
  864 hits and 288 misses, matching the cold cache population pattern; and
- numerical gate: top-1 match, maximum absolute logit difference 0.0239.

The invalidation probe demonstrates that a generation is not silently reused.
The 8 residual warm misses per invocation are bounded hash/probe conflicts, not
whole-working-set thrashing.

This runner repeats the same full-sequence forward while retaining the loaded
module. It validates the cross-invocation lifetime and is a useful proxy for
stable weights across decode steps, but it is not yet a true KV-cache
autoregressive token loop. Results must therefore be described as a
cross-invocation model experiment, not token-per-second decode performance.

The final same-day, three-iteration primary comparison is:

| Configuration | Device time | Numerical result |
|---|---:|---|
| HVX / unmodified Hexagon-MLIR | 11296.286 ms | top-5 matched, max abs 0.0237 |
| HexKL | 1628.410 ms | top-1 match, max abs 0.0239 |
| HexKL + OmniFetch items 1–4 | 1612.315 ms warm average | top-1 match, max abs 0.0239 |

The item-4 cumulative row is 0.99% faster than this adjacent HexKL sample and
85.73% faster than HVX. This is a valid mechanism result, but the approximately
1% HexKL delta remains too small for a publication claim without interleaved
repetitions, thermal control, and confidence intervals.

## Item 5 implementation and validation

Item 5 implements a real load/reshape/compute pipeline for profitable HexKL
weight sites. It is controlled by `enableOmniFetchTwoDimPipeline` /
`--enable-omnifetch-two-dim-pipeline` and remains off by default.

The previous async prototype synchronously transformed the current weight tile
on every K iteration and also launched the next tile. That was correct, but the
repeated current-tile transform serialized the path and prevented it from being
a true pipeline. The item-5 compiler transformation now:

1. synchronously bootstraps only the first K tile;
2. starts the two-dimensional DDR load for tile `t+1` while HMX consumes tile
   `t`;
3. completes WH reshape into the idle ping-pong slot after tile `t` compute;
4. publishes transform readiness through the existing V-DAE semaphore; and
5. lets the following iteration consume the ready slot without repeating the
   synchronous WH transform.

The runtime descriptor has explicit `LOAD_PENDING`, `LOAD_READY`, and
`TRANSFORM_READY` phases. The compiler passes six tile parameters for the
hybrid item-4/item-5 path: tile row, tile column, source width, destination
offset, staging offset, and stable layout-site ID.

Item 5 composes with the cross-invocation cache rather than replacing it. A
warm cache hit copies an already transformed WH tile into the idle slot. A miss
enters the asynchronous load/reshape pipeline and inserts the completed WH tile
into the same context- and generation-safe cache. The first tile also uses the
persistent cache. Thus `async=5, persistent=5` means five sites use both
mechanisms, rather than two disjoint sets of sites.

The new
`qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_two_dim_pipeline.mlir`
test verifies the first-tile guard, persistent bootstrap, next-tile lookahead,
six-parameter identity, and removal of the original per-iteration
`micro_hmx_rm_to_wh` operation. Together with the item-4 and cost-model tests,
three targeted compiler tests pass.

On the sequence-length-128 two-layer Falcon model runner, five weight sites
selected the item-5 pipeline and five activation sites remained synchronous.
The final fixed-order three-way run was:

| Configuration | Device time | Numerical result |
|---|---:|---|
| HVX / unmodified Hexagon-MLIR | 11197.351 ms | top-5 matched, max abs 0.0237 |
| HexKL | 1622.670 ms | top-1 match, max abs 0.0239 |
| HexKL + OmniFetch items 1–5 | 1586.580 ms warm average | top-1 match, max abs 0.0239 |

The cumulative row is 2.22% faster than the adjacent HexKL run and 85.83%
faster than HVX (1.023x and 7.058x respectively). Its cold time was 1715.577
ms; the three warm invocations produced 4296 total hits and 312 total misses,
and the invalidation probe took 1611.002 ms. This is a positive model-level
mechanism result, not yet a publication-quality speedup: the experiment still
needs interleaved repetitions, temperature/power logging, and confidence
intervals.

The debug runner has hidden size 64 and only two K tiles at several sites. The
explicit item-5 gate therefore permits loops with at least two tiles so that
repository model tests actually exercise the pipeline. With the flag off, the
original conservative profitability threshold remains unchanged.

## Item 6 implementation and validation

Item 6 adds opt-in VTCM lifetime coloring to the decomposed HexKL matmul path.
It is controlled by `enableOmniFetchVtcmColoring` /
`--enable-omnifetch-vtcm-coloring`, is disabled by default, and is enabled only
in the cumulative items-1-through-6 model row.

The implementation represents activation tiles, weight ping-pong tiles,
prefetch staging, flattened output, and accumulator storage as half-open live
intervals with a size and alignment. A deterministic first-fit interval
colorer assigns offsets to contiguous tile ranges. Regions may share an offset
only when their live intervals do not overlap; simultaneously live regions
retain distinct colors. Both the normal M-outer schedule and the weight-prepack
schedule have explicit lifetime models.

For the sequence-length-128 Falcon debug module, all five decomposed HexKL
sites were colored. The reported per-function maximum changed from 45056 bytes
in the legacy append-only layout to 16384 bytes in the colored layout, saving
28672 bytes, or 63.64%. The compiler records
`omni_fetch.vtcm_legacy_peak_bytes`,
`omni_fetch.vtcm_colored_peak_bytes`,
`omni_fetch.vtcm_saved_peak_bytes`, and
`omni_fetch.vtcm_colored_sites` attributes so that the allocation result is
visible in IR and benchmark logs.

Model-level validation caught an important error in the first lifetime model:
it treated an activation tile as dead after one N-column computation, although
the default M-outer schedule reuses that activation across all N columns. This
allowed output storage to overwrite a live activation and produced a maximum
absolute logit difference of 0.7319. The corrected model keeps activation
storage live across the N loop and reuses the dead WH ping-pong slots for
flattened output and accumulator storage. In the weight-prepack schedule,
prepacked WH remains live across M rows and output storage instead reuses dead
activation colors. After this correction, top-1 matching and the established
0.0239 maximum absolute difference were restored.

The focused compiler checks are:

```bash
source /home/huzq85/2-working/hexagon_npu/mlir-env/bin/activate
lit -sv \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Transforms/omnifetch-vtcm-lifetime-coloring.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Transforms/decompose-hexkl-matmul.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_two_dim_pipeline.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_persistent_wh_cache.mlir
```

All four tests pass. The new coloring test checks the 45056-to-16384-byte peak
reduction and verifies the intended offset sharing between non-overlapping
scratch/weight/output regions. The incremental library build, Python syntax
checks, shell syntax check, and `git diff --check` also pass.

The model-level three-way command is:

```bash
ANDROID_SERIAL=49d1c7b2 \
  bash scripts/run_omnifetch_model_ablation.sh \
    --model falcon-debug --seq-len 128 --timeout 240
```

The final fixed-order run produced:

| Configuration | Device time | Numerical result |
|---|---:|---|
| HVX / unmodified Hexagon-MLIR | 11321.344 ms | top-5 matched, max abs 0.0237 |
| HexKL | 1619.855 ms | top-1 match, max abs 0.0239 |
| HexKL + OmniFetch items 1–6 | 1598.756 ms warm average | top-1 match, max abs 0.0239 |

The cumulative row is 1.30% faster than the adjacent HexKL sample and 85.88%
faster than HVX (1.013x and 7.081x respectively). Its cold invocation took
1707.232 ms. Before the explicit invalidation probe, the cache counters reached
4296 hits and 312 misses; the post-invalidation invocation took 1590.593 ms and
increased the totals to 5160 hits and 600 misses.

This run proves that the colored allocation composes correctly with prefetch,
in-situ reshape, the persistent WH cache, and the two-dimensional pipeline
while materially reducing the statically reserved VTCM peak. It does not yet
show an isolated item-6 latency improvement: the items-1-through-5 result from
the preceding run was 1586.580 ms, and cross-run thermal/noise effects are
larger than this difference. Publication-quality performance evaluation still
requires interleaved items-1-through-5 versus items-1-through-6 repetitions,
device temperature/power logging, and confidence intervals.

## Item 7 implementation and validation

Item 7 adds compiler-visible K/V stream identity, page accounting, coalesced
L2 prefetch, and a KV-aware fusion boundary. It is controlled by
`enableOmniFetchKvCachePrefetch` /
`--enable-omnifetch-kv-cache-prefetch`; the logical page size is controlled by
`omniFetchKvCachePageTokens` /
`--omnifetch-kv-cache-page-tokens` and defaults to 32 tokens. Both are opt-in,
and item 7 is enabled only in the cumulative Falcon row.

The implementation has four stages:

1. `tm_tensor.attention` lowering marks the QK contraction's K operand and the
   AV contraction's V operand with semantic roles. Metadata emission itself is
   gated, so HVX and plain HexKL retain the original IR and fusion behavior.
2. The transpose-aware matmul scheduler and named-to-generic conversion
   preserve only these semantic attributes. The QK path consumes K directly
   in `[batch-or-head, sequence, head_dim]` layout through transpose-aware
   indexing instead of materializing a cache-wide transpose.
3. Tensor fusion preserves the marked attention boundary until after
   bufferization. This prevents a replacement generic op from silently losing
   K/V identity and prevents unrelated producer fusion from obscuring the
   cache-consumer layout.
4. `PrefetchInsert` splits a static K/V memref into contiguous leading
   streams, accounts for logical 32-token pages, coalesces adjacent pages into
   one hardware hint per stream, and emits `L2Hint` `prefetch_in_situ` ops.
   The runtime lowers these to the existing asynchronous Hexagon `l2fetch`
   implementation; no copy or VTCM allocation is performed.

The compiler reports:

- `omni_fetch.kv_prefetch_sites`;
- `omni_fetch.kv_prefetch_hints`;
- `omni_fetch.kv_prefetch_pages`;
- `omni_fetch.kv_prefetch_bytes`; and
- `omni_fetch.kv_direct_layout_sites`.

The Falcon debug graph contains two attention layers. At sequence length 128
and two heads, item 7 finds four consumers (K and V per layer), accounts for
32 logical pages, coalesces them into eight contiguous hardware hints, and
prefetches 65536 bytes per invocation.

Two implementation failures were caught before accepting the result. Copying
the complete attribute dictionary while creating a transpose-aware matmul
overwrote structural op properties and caused a 64-versus-192 shape inference
failure. Copying only the two OmniFetch semantic attributes fixed the baseline.
Next, the generic tensor fusion pass replaced marked contractions and the
model reported zero KV sites. Metadata is now emitted only under the item-7
gate, and the gated path preserves the attention boundary until page hints are
inserted. A direct full-pipeline check then reports:

```text
[HexagonFusion] function=FalconForCausalLM skipped=1 reason=preserve_kv_cache_boundary
[KVCachePrefetch] function=FalconForCausalLM sites=4 hints=8 pages=32 bytes=65536 page_tokens=32
```

The focused regression command is:

```bash
source /home/huzq85/2-working/hexagon_npu/mlir-env/bin/activate
lit -sv \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Conversion/TmTensorToLinalg/SDPA.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Transforms/omnifetch-kv-cache-prefetch.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Transforms/omnifetch-vtcm-lifetime-coloring.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_two_dim_pipeline.mlir \
  triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_persistent_wh_cache.mlir
```

All five tests pass. The new test checks two K/V consumers, four coalesced
hints, 16 logical pages, 32768 bytes, per-stream subviews, and L2Hint emission.
The user-guide-based incremental build, Python syntax checks, shell syntax
check, and `git diff --check` also pass.

The model command remains:

```bash
ANDROID_SERIAL=49d1c7b2 \
  bash scripts/run_omnifetch_model_ablation.sh \
    --model falcon-debug --seq-len 128 --timeout 240
```

The final fixed-order device run produced:

| Configuration | Device time | Numerical result |
|---|---:|---|
| HVX / unmodified Hexagon-MLIR | 11742.816 ms | top-5 matched, max abs 0.0237 |
| HexKL | 1614.896 ms | top-1 match, max abs 0.0239 |
| HexKL + OmniFetch items 1–7 | 599.969 ms warm average | top-1 match, max abs 0.0231 |

The cumulative row is 62.85% faster than the adjacent HexKL sample and 94.89%
faster than HVX (2.692x and 19.572x respectively). It is 62.47% faster than
the preceding item-6 run, but that cross-run comparison is not an isolated
item-7 measurement. The cold invocation took 701.875 ms; the explicit
invalidation probe took 599.153 ms. Before invalidation, the WH cache counters
reached 3775 hits and 833 misses.

This large delta must be attributed to the combined
`KV-aware fusion boundary + direct K layout + page prefetch` policy, not to
eight `l2fetch` instructions alone. Preserving the attention boundary changes
the compiler schedule and reduced runtime substantially in this debug graph;
it also increased compilation time to approximately 212 seconds in the
measured cumulative build. An isolated four-way ablation—items 1–6,
boundary-only, hints-only, and boundary+hints—is required before assigning
causality or making a performance claim.

Most importantly, the current Falcon, GPT-2, Qwen, and TinyLlama runners set
`use_cache=False`. This experiment exercises K/V streams in a full-sequence
prefill graph, not a persistent `past_key_values` decode loop. Item 7's
compiler mechanism and model/device gate are implemented, but a true
cross-token result still requires a fixed-shape decode-step ABI with K/V cache
inputs/outputs, cache-position or page-table inputs, multiple sequential
device invocations, GQA/MQA coverage, and sliding-window/page selection.

## Retired item 8 negative experiment

Item 8 implements an opt-in W8A16 compressed-weight path that combines
compressed weight retention, tile-local dequantization, and immediate WH
production. It is controlled by `enableOmniFetchDequantReshape` /
`--enable-omnifetch-dequant-reshape`. The flag is disabled by default and is
not part of the formal items-1-through-7 cumulative row.

The implementation is end to end:

1. The OmniFetch dialect has a distinct `HMXWeightDequantI8` layout kind.
   `PrefetchInsert` selects it at eligible HexKL weight sites and attaches a
   stable compiler site ID to every tile request.
2. The runtime maintains a generation-safe, direct-mapped compressed tile
   cache. Its key contains the model context, generation, compiler site ID,
   tile row/column, and source stride. It deliberately excludes the transient
   source pointer because function-local lowered materializations can move
   between invocations.
3. A 32-by-32 source tile is represented by 1024 signed int8 values and
   4-by-32 FP32 group scales. The group size is eight along K. This consumes
   1536 bytes rather than the 2048 bytes required by the row-major FP16 tile,
   a 25% reduction before WH conversion.
4. On a compressed-cache hit, the runtime dequantizes into one short-lived,
   aligned FP16 tile and immediately invokes the HexKL RM-to-WH transform. It
   never materializes a complete dequantized weight matrix.
5. The Falcon CPU reference applies the same symmetric, group-size-eight
   fake-quantization to rank-two projection weights before compilation and
   reference execution. The comparison is therefore W8A16-equivalent on both
   sides rather than an unfair W8 device result against the original FP16
   model.
6. The launcher reports cold and cumulative compressed-cache hits/misses in
   `perf.txt`. The model ablation script exposes item 8 only under
   `--include-experimental`.

Model validation caught a cache-identity error in the first implementation.
Different layers could reuse allocator addresses, so a key without a compiler
site ID returned a stale compressed tile. That version produced a top-1
mismatch and a maximum absolute logit error of approximately 0.6309. Adding
the stable site ID restored top-1 matching and reduced the maximum absolute
difference to 0.0233. Removing the unstable source address from the otherwise
generation-safe key then improved warm cache reuse without weakening layer
identity.

The focused compiler regression is:

```bash
triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt \
  qcom_hexagon_backend/test/Transforms/omnifetch-dequant-reshape.mlir \
  -pass-pipeline='builtin.module(func.func(prefetch-insert{lookahead=2 enable-layout-aware=true enable-dequant-reshape=true}))' \
| /home/huzq85/2-working/hexagon_npu/LLVM_DIR/llvm-project/build/install/bin/FileCheck \
  qcom_hexagon_backend/test/Transforms/omnifetch-dequant-reshape.mlir
```

It checks the new layout kind, synchronous production mode, stable site
parameter, function-level accounting attributes, and removal of the original
native RM-to-WH operation. The existing two-dimensional pipeline regression
also passes, which ensures the new opt-in path does not change item 5 while
disabled.

Following `docs/user-guide.md`, the complete plugin/runtime build used for the
device test is:

```bash
CCACHE_DIR=/tmp/omnifetch-ccache \
CCACHE_TEMPDIR=/tmp \
ninja -C triton/build/cmake.linux-x86_64-cpython-3.11
```

Rebuilding only the standalone DSP runtime target is insufficient after a
runtime ABI change: the model path loads the runtime embedded through the
rebuilt Hexagon plugin, so `libtriton.so` must also be relinked.

The final model commands are:

```bash
# Formal items-1-through-7 three-way comparison.
ANDROID_SERIAL=49d1c7b2 \
  bash scripts/run_omnifetch_model_ablation.sh \
    --model falcon-debug --seq-len 128 --timeout 240

# Add the experimental item-8 row.
ANDROID_SERIAL=49d1c7b2 \
  bash scripts/run_omnifetch_model_ablation.sh \
    --model falcon-debug --seq-len 128 --timeout 240 \
    --include-experimental
```

The final fixed-seed Falcon debug screen, with sequence length 128 and three
warm device iterations, produced:

| Configuration | Device time | Numerical result | Compressed-cache result |
|---|---:|---|---|
| HexKL + OmniFetch items 1–7 | 597.726 ms warm average | top-1 match, max abs 0.0231 | not enabled |
| HexKL + OmniFetch items 1–8 (experimental) | 670.129 ms warm average | top-1 match, max abs 0.0233 | cold 546 hits / 606 misses; cumulative 2682 hits / 1926 misses |

Item 8 is 72.403 ms, or 12.11%, slower than the adjacent items-1-through-7
sample. Its cold invocation took 785.284 ms and the post-invalidation
invocation took 686.312 ms. Thus the compiler path, runtime mechanism, matched
quantized reference, cache identity, invalidation behavior, and model-level
correctness gate are complete, but the performance gate fails.

The principal bottleneck is structural rather than a missing prefetch flag.
Every compressed-cache miss still scans an FP16 tile and performs scalar
groupwise quantization at inference time; every hit performs scalar
dequantization before the synchronous WH transform. Direct-mapped cache
collisions add further misses, and the current item-8 path intentionally does
not enter the older asynchronous FP16 DMA pipeline. The next performance
version should precompress weights offline or at model load, DMA compressed
bytes directly from DDR/L2, vectorize groupwise dequantization with HVX, and
double-buffer dequantization/WH production with item 5's compute pipeline.
These observations explain the negative result but are not follow-up tasks:
item 8 must not enter the formal `HexKL + cumulative OmniFetch` row.

The project has since retired this direction. The implementation and negative
result remain in this document for reproducibility, but quantization,
dequantization, compressed-weight caching, and the item-8 experimental row are
not part of the forward OmniFetch roadmap.

## V73 memory-hierarchy-driven roadmap after retiring quantization

### Manual scope and hardware facts

This roadmap is based on:

- [Hexagon V73 Programmer's Reference Manual](../Hexagon_V73_Programmers_Reference_Manual.pdf),
  especially Section 1.1.1, Sections 5.10–5.11 on cache and memory ordering,
  and the PMU events in Chapter 9; and
- [Hexagon V73 HVX Programmer's Reference Manual](../Hexagon_V73_HVX_Programmers_Reference_Manual.pdf),
  especially Sections 3.1–3.9 on alignment, VTCM, memory types, ordering, and
  vector-memory performance, plus the HVX PMU events in Chapter 4.

The following facts materially constrain OmniFetch design:

| V73 hardware fact | Consequence for OmniFetch |
|---|---|
| HVX VMEM connects directly to L2 and does not use L1 data cache. | Weight/activation prefetch for HVX or HexKL staging should target L2 or VTCM; L1 `dcfetch` is relevant only for a subsequent scalar consumer. |
| VTCM is faster and lower-power than L2, is not evicted, and reduces L2 pressure. | Short-lived transformed tiles and repeatedly reused layout values should be explicitly placed in VTCM when their live ranges fit. |
| VRF access is much cheaper than VMEM access. | The best reshape elimination keeps producer results in registers or writes the consumer layout directly; replacing one DDR copy with two VTCM accesses is not the final goal. |
| V73 `l2fetch` is nonblocking and lower priority than demand traffic, but a new command can halt a still-active command. Zero-valued fields cancel activity. | The compiler/runtime must treat the L2 prefetch engine as a scheduled single-flight resource rather than emit independent hints at every site. |
| The HVX manual recommends an `l2fetch` region smaller than 8 KB, issued several hundred cycles before use. Fetching too early permits eviction. | Prefetch distance must be expressed in estimated cycles/bytes-to-use, not only loop iterations; large requests must be tiled and timed. |
| An `l2fetch` address generated on a different page from its start address is dropped. | Every request needs page-boundary splitting or a page-contained allocation contract. |
| Aligned VMEM is preferred; VMEMU can touch multiple cache lines and costs bandwidth and power. V73 HVX vectors are 128 bytes. | Layout contracts must carry 128-byte base/stride alignment and permit padding plus predication. |
| Contiguous access reduces bank conflict, cache-set aliasing, and micro-TLB pressure. Conflict-free vector access depends on lower address bits; scatter/gather on V73 is especially sensitive to bits `[10:3]`. | VTCM coloring must include bank phase, DDR/L2 allocation should avoid set aliases between simultaneous streams, and tile schedules should minimize active pages. |
| The `:nt` attribute tells the cache that data is at its final use. | Streaming outputs and one-use inputs should be marked nontemporal so they do not evict prefetched weights or future activations. |
| A VMEM load soon after a store to the same address can stall until the store reaches L2; the manual recommends about 15 intervening packets. | Producer/consumer fusion should remove the round trip; otherwise rotate buffers or schedule independent work before reload. |
| HVX scatter/gather works only in VTCM, must remain within one translated page, and gather cannot read directly from DDR/L2. Scatter is generally preferable; bursts and consumption distance must be controlled. | In-situ transforms should DMA a page-contained region once, then prefer producer-side scatter/direct placement over a later gather when legal. |
| External AXI DMA is noncoherent with coprocessor threads and requires explicit release or descriptor polling. | DMA readiness must be a compiler-visible ownership token; broad barriers and unsynchronized cache reuse are both incorrect. |

### Immediate audit finding in the current runtime

The current runtime does not yet honor the most important `l2fetch`
constraints:

- `omni_l2fetch_weight_tile()` issues 32 separate 64-byte commands for one
  32-by-32 FP16 tile. On V73, each new command can halt the preceding active
  command, so most rows may never become resident.
- `omni_l2fetch()` uses 32 KB chunks. This is legal for the extended
  descriptor but is four times the HVX manual's recommended maximum working
  request of less than 8 KB.
- neither helper splits a descriptor at a virtual-page boundary;
- the KV path can issue several stream hints back-to-back without a global
  L2-prefetch-engine arbitration policy; and
- current adaptation uses software wait time, but does not observe
  `L2FETCH_COMMAND_OVERWRITE`, `L2FETCH_ACCESS_CREDIT_FAIL`, L2 conflicts, or
  whether prefetched lines were actually missing.

For a strided 32-by-32 FP16 weight tile, the first replacement should be one
2D request with `width=64`, `height=32`, and
`stride=source_columns*2`, split only where the generated addresses cross a
page. If the page split would create too many commands, direct 2D UserDMA to
VTCM is preferable to a storm of L2 hints.

### One paper story: movement planning over the V73 memory hierarchy

The ten mechanisms and the three proposed abstractions are not independent
feature ideas. They form one extension of OmniFetch from a prefetch insertion
pass into a **hierarchy-aware movement planner**:

```text
movement equivalence class
  decides which physical movement is semantically unnecessary
          |
          v
layout residency graph
  chooses the physical layout and hierarchy tier that should own the bytes
          |
          v
prefetch lease
  schedules the remaining transfer early enough without overwriting or
  evicting another useful transfer
```

This gives one logical objective:

> Minimize mandatory physical bytes and exposed movement latency while
> preserving the model's physical address semantics and memory ownership.

Existing OmniFetch supplies the starting mechanisms: layout-aware prefetch,
in-situ reshape, DMA/VTCM staging, persistent WH reuse, two-dimensional
pipelines, VTCM lifetime coloring, and K/V prefetch. The three abstractions
generalize those mechanisms instead of replacing them:

| Unified decision | Existing OmniFetch instance | V73 mechanisms that complete it |
|---|---|---|
| Is this movement physically necessary? | in-situ reshape and layout-aware mapping | M3 movement equivalence, M4 VRF/direct producer placement, M8 store-to-load forwarding |
| In which layout and tier should the value reside? | DMA-to-VTCM, persistent WH cache, VTCM lifetime coloring | M2 path selection, M5 page/alignment placement, M6 bank phase, M9 residency graph |
| When should an unavoidable transfer occur and what protects it? | static/adaptive lookahead, inter-layer and K/V prefetch | M1 prefetch leases, M7 last-use nontemporal protection, M10 PMU feedback |

The paper story therefore has three evaluation axes rather than ten unrelated
speedup claims:

1. **less movement:** physical bytes, materialized transforms, and
   store/reload boundaries removed;
2. **earlier useful movement:** demand-stall time, useful issued bytes, and
   prefetch overwrite/credit-failure rate; and
3. **safer residency:** VTCM/L2 conflict stalls, active pages, ownership
   violations, and numerical correctness.

M1--M10 are implemented in order because later decisions consume facts
produced by earlier ones. Each stage is admitted to the cumulative row only
after a complete GPT-2 or Falcon run compares the same input and checkpoint on
HVX, HexKL, and HexKL plus the admitted OmniFetch stages. A mechanism that
fails correctness or regresses latency stays documented but does not silently
enter the cumulative configuration.

### M1/P0: Page-safe single-flight L2 prefetch scheduler

Model the L2 prefetch engine as a compiler-visible resource with at most one
active V73 command. Construct a global queue across weights, activations, and
K/V streams rather than letting each loop emit commands independently.

Each request carries:

- first-use cycle estimate and last-profitable issue cycle;
- page-contained 2D region `(base, width, height, stride)`;
- byte footprint, reuse count, and critical-path priority;
- expected consumer (`HVX`, `DMA`, scalar, or HMX staging); and
- cancellation/overwrite risk.

The scheduler coalesces rows into one 2D command, caps the normal working
request below 8 KB, splits at pages, and refuses to issue a younger command
while an older useful command is still active. `USR.PFA` or PMU overwrite
events can provide a runtime completion signal without inserting a blocking
wait on the compute path.

The novel abstraction is a **prefetch lease**: a tile owns a bounded L2
residency interval from issue to last use. Competing leases are rejected or
redirected to DMA/VTCM when their footprints and reuse windows would evict one
another. This combines temporal scheduling with cache-capacity reasoning
instead of choosing a fixed lookahead.

#### M1 implementation and full-model gate (2026-07-26)

The first M1 implementation is present in `OmniFetchRuntime.c` and is deliberately
nonblocking:

- the 32-command row storm for a 32-by-32 FP16 weight tile is replaced by one
  two-dimensional request `(width=64, height=32, stride=source_columns*2)`;
- `USR.PFA` bit 3 is read before issue; a younger request is suppressed while
  an older `l2fetch` remains active, so it cannot silently overwrite the older
  command;
- the descriptor is capped below 8 KiB and clipped so every generated address
  remains in the 4 KiB page containing the start address;
- unsupported extended-descriptor fields are rejected; and
- issued, busy-suppressed, page-clipped, unsupported, requested-byte, and
  issued-byte counters are appended to model `perf.txt`.

This is the **single-page first slice** of a prefetch lease, not the complete
multi-page queue. Clipped remainder pages are intentionally visible in the
counter report. M2 must choose between scheduling those remainder leases and
redirecting the tile to direct DMA; silently claiming that a clipped request
covered the entire tensor would be incorrect.

The runtime, Python launcher, and full-model runner compiled successfully:

```bash
CCACHE_DIR=/tmp/omnifetch-ccache CCACHE_TEMPDIR=/tmp \
  ninja -C triton/build/cmake.linux-x86_64-cpython-3.11

triton/build/cmake.linux-x86_64-cpython-3.11/bin/llvm-lit -v \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch-kv-cache-prefetch.mlir \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_two_dim_pipeline.mlir \
  qcom_hexagon_backend/test/Conversion/LinalgToLLVM/omnifetch_persistent_wh_cache.mlir

python -m py_compile benchmark_models/run_gpt2lmheadmodel.py
bash -n scripts/run_omnifetch_model_ablation.sh
git diff --check
```

The build completed and all three targeted lit tests passed. Disassembly of the
DSP runtime also contains the `USR` read and a single `l2fetch` issue site.

The formal full-model command is:

```bash
ANDROID_SERIAL=49d1c7b2 \
  bash scripts/run_omnifetch_model_ablation.sh \
    --model gpt2-full --seq-len 32 --timeout 1800 \
    --output-dir \
      benchmark_models/results/omnifetch_m1_gpt2_full_f32_seq32
```

The complete published GPT-2 topology is retained: 12 layers, hidden size 768,
12 heads, and vocabulary 50,257. A fully FP16 graph previously produced NaNs,
so normalization, residual, softmax, and the CPU reference now remain FP32.
Fixed-length formal runs use a stated whole-vocabulary qualification gate:
all logits finite, identical top-1, and centered cosine at least 0.80. This
does not claim strict CPU equivalence; top-5 overlap, cosine, and mean absolute
error are always printed.

The current host cannot yet complete the required three rows:

| Attempt | Outcome | Evidence |
|---|---|---|
| GPT-2 full, seq=128, HVX | device completed in 127,401,804 us; finite logits, identical top-1, top-5 overlap 3/5, centered cosine 0.824828 | qualified baseline result, not strict equivalence |
| GPT-2 full, seq=128, HexKL | host OOM before device execution | Linux OOM log: peak anonymous RSS 15,356,952 KiB |
| GPT-2 full, seq=32, HVX | host compilation exceeded the 1800 s case timeout | no device timing; recorded as FAIL |

Falcon is not a valid local substitute because its cached directory contains
configuration/tokenizer files but no checkpoint weights. Consequently M1 is
**implemented and statically validated but not admitted to the cumulative
performance row**. No M1 speedup is claimed, and M2 must not begin until either
a memory-efficient mixed-precision export or a larger-memory build host lets
HVX, HexKL, and HexKL+M1 finish with the identical model and input.

### M2/P0: Per-tile L2-prefetch versus direct-DMA path selection

L2 prefetch and UserDMA are different movement paths and should not be stacked
blindly:

```text
reused/cached path:
  DDR -> l2fetch -> repeated HVX/CPU reads

one-use layout path:
  DDR -> 2D UserDMA -> VTCM stage -> direct final-layout production

producer-resident path:
  producer VRF/VTCM -> consumer layout, with no DDR or L2 round trip
```

The compiler should select one path per tile from reuse, stride, page count,
alignment, transform cost, and VTCM pressure. For a one-use strided weight
tile, prefetching it into L2 and then DMA-reading it may add cache pollution
without reducing the mandatory transfer. Direct DMA with the appropriate
source/destination cache-bypass policy can avoid redundant snoop/allocation
work. For shared activations or K/V blocks with multiple HVX consumers, L2
residency can be valuable.

The UserDMA `cacheAllocationPolicy` and source/destination bypass fields are
currently always zero in the OmniFetch weight path. They should become explicit
codegen decisions. A policy verifier must reject incoherent combinations,
especially a bypassed DMA write followed by a cached consumer without a
completion/ownership transition.

### M3/P0: Physical-layout equivalence and direct producer placement

Reshape-like operators must be divided into two classes:

1. metadata-only views whose affine address function is unchanged; and
2. physical transforms such as a real transpose, permutation, or repack.

Class 1 operations can be erased after composing their affine maps into the
consumer. Class 2 operations cannot simply be deleted: their producer must be
rewritten to emit the consumer's desired order, or the transform must be fused
with the one unavoidable movement into VTCM.

Introduce a **movement equivalence class** for values with the same physical
bytes, source version, and composed address map. `reshape`, `collapse_shape`,
`expand_shape`, `subview`, transpose, and copy chains are normalized into this
representation. The compiler then:

- forwards a zero-copy alias when address functions are equivalent;
- makes an HVX producer store directly in AH/WH-compatible or
  consumer-contiguous order when possible;
- combines DMA copy and layout placement when a transfer is unavoidable; and
- materializes a standalone transform only as the last legal fallback.

This is stricter and safer than pattern-based reshape deletion, and it gives a
measurable objective: bytes moved per original model operator and the number of
materialized layout boundaries removed.

### M4/P1: VRF-resident epilogues and inter-operator layout forwarding

Because VRF access is cheaper than VMEM, fuse short reshape/transpose/slice
epilogues into the producer while its result is still in vector registers.
Where the next operator is an HVX kernel, forward vector tiles directly through
the fused region. Where the next operator is HexKL/HMX, write the tile once
into its final VTCM staging layout.

Initial legal targets are:

- elementwise/bias/activation followed by a view or transpose;
- normalization output followed by projection staging;
- attention output projection input views; and
- residual add where both consumers accept the same physical layout.

Crossing a closed HexKL kernel boundary cannot assume that private HMX
registers remain live. The optimization therefore stops at the public
VTCM/AH/WH ABI unless HexKL exposes a compatible fused entry point.

### M5/P1: Alignment-, page-, and micro-TLB-aware tensor placement

Extend layout values with:

- minimum 128-byte base alignment;
- row-stride alignment and permitted padding;
- virtual page span and page-contained subregions;
- last-use information for nontemporal marking; and
- a maximum live-page budget.

Pad rows or tile extents to vector boundaries and mask inactive lanes instead
of repeatedly using VMEMU. Allocate simultaneously consumed weights,
activations, output, and K/V pages so their hot regions do not unnecessarily
span many pages. Page-contained 2D tiles improve both `l2fetch` reliability and
scatter/gather legality.

Padding is profitable only when the extra fetched bytes cost less than
unaligned multi-line accesses and transform code. The cost model must report
both logical bytes and physical transferred bytes.

### M6/P1: Bank-phase-aware VTCM coloring and direct scatter placement

Item 6 colors VTCM by lifetime. Extend the interference graph with an address
phase:

- distribute concurrent accesses across relevant lower address bits;
- avoid giving the active DMA destination, HMX operand, and HVX transform
  scratch the same bank phase;
- keep scatter/gather regions within one page;
- limit V73 scatter/gather bursts to four per thread; and
- leave at least 12 packets before consuming conflict-free gather/scatter
  results, or approximately 24 for poorly correlated addresses.

When an HVX producer already holds values in vectors, prefer a producer-side
scatter into final VTCM layout over storing row-major and later gathering.
This converts:

```text
VRF -> row-major VTCM -> gather -> final VTCM layout
```

into:

```text
VRF -> final VTCM layout
```

The pass should use `HVX_VTCM_OUTSTANDING`, `HVX_SCATGATH_FULL`,
`HVX_SCATGATH_IN_FULL`, `HVXST_VTCM_FULL`, and
`VTCM_FIFO_FULL_CYCLES` to validate that reduced byte movement does not create
a worse bank/network bottleneck.

### M7/P1: Last-use nontemporal streaming and cache-residency protection

Use liveness to mark final-use HVX loads/stores with `:nt`. Typical candidates
are final output tiles, one-use staging inputs, and streamed residuals after
their last consumer. Do not mark reusable weights, K/V pages, or an activation
needed by a nearby scalar consumer.

This innovation combines last-use analysis with prefetch: nontemporal eviction
preference protects leased future weights/activations from one-pass output
traffic. It should be evaluated through L2 miss, castout, and prefetch-miss
counters rather than latency alone.

### M8/P1: Store-to-load-distance-aware fusion and buffer rotation

Detect a VMEM store followed by a load from the same physical region. First try
to eliminate the pair by forwarding the producer value or fusing the consumer.
If materialization is required:

- rotate among VTCM/L2 buffers so the consumer reads an older completed tile;
- schedule at least roughly 15 independent packets between L2 store and load;
  or
- use a DMA readiness token and consume from a different bank/slot.

This generalizes the existing two-dimensional pipeline from K tiles to
operator boundaries. Its profitability statistic is not only overlapped time,
but the number of store-to-load hazards removed.

### M9/P2: Tiered layout residency across VTCM, L2/optional L2TCM, and DDR

Treat a transformed layout as a movable resident object:

- hottest, short-live, deterministic-reuse tiles in VTCM;
- medium-reuse tiles protected by an L2 prefetch lease;
- cold or far-future values in DDR in their original precision; and
- optionally, platform-permitting hot read-only data in L2TCM.

Moving a layout between tiers must be justified by future saved transfers.
The optional L2TCM path needs resource-management support and must account for
the corresponding reduction in ordinary L2 cache capacity; it cannot be
assumed available from the compiler alone.

This extends persistent WH caching into a general **layout residency graph**:
nodes are physical layouts in hierarchy tiers and edges are DMA, in-situ
production, zero-copy aliasing, or eviction. A shortest-cost path chooses where
each consumer obtains its bytes.

### M10/P2: PMU-driven hierarchy controller

Replace generic software wait-time adaptation with hardware-specific feedback.
At minimum record:

- `L2FETCH_ACCESS`, `L2FETCH_MISS`, `L2FETCH_COMMAND`,
  `L2FETCH_COMMAND_OVERWRITE`, and `L2FETCH_ACCESS_CREDIT_FAIL`;
- `L2_PIPE_CONFLICT_STALL`, `L2_TAG_ARRAY_CONFLICT`, `L2_CASTOUT`, and
  L2 FIFO replays;
- HVX L2 load/store outstanding cycles and L2 store misses;
- VTCM outstanding/FIFO/scatter-gather stalls; and
- JTLB/micro-TLB-related pressure where exposed by the runtime.

Use these counters per compiler site and model phase to tune:

- L2 lease distance and request size;
- L2 versus direct-DMA selection;
- DMA cache allocation/bypass policy;
- VTCM bank phase and number of pipeline slots; and
- whether a speculative cross-operator prefetch should be suppressed.

The controller must save the selected policy and counter deltas with every
benchmark so that adaptation remains reproducible.

### Lower-priority scalar handoff

When an HVX-produced value is genuinely consumed by scalar code, the manual
recommends storing it to L2, issuing `dcfetch`, and allowing at least about 30
cycles before the scalar load. This is useful for scalar tails, control values,
or runtime metadata. It is lower priority for model tensors because converting
the scalar tail to HVX or keeping the reduction result in registers usually
removes more movement.

### Recommended implementation order and gates

The new memory-hierarchy work should proceed independently of retired item 8:

| Order | Deliverable | Required gate |
|---:|---|---|
| M1 | correct 2D/page-safe, single-flight `l2fetch` scheduler | no overwrite events; all requested rows/pages accounted for |
| M2 | per-tile L2 versus direct-DMA policy and explicit cache-bypass/allocation settings | numerical correctness plus coherent DMA ownership test |
| M3 | physical-layout equivalence and safe reshape/view/copy elimination | IR byte-address equivalence checks and model output gate |
| M4 | producer-side direct layout/VRF forwarding | fewer VMEM bytes and removed transform operations |
| M5 | alignment/page/micro-TLB-aware placement | fewer unaligned accesses and active pages without excess padding traffic |
| M6 | bank-phase VTCM coloring and direct scatter placement | lower VTCM/scatter stall counters |
| M7 | liveness-driven `:nt` | fewer L2 castouts/misses for leased future tiles |
| M8 | store-to-load-aware inter-operator pipeline | fewer hazards and lower model latency |
| M9 | tiered layout residency | saved movement exceeds promotion/eviction traffic |
| M10 | PMU-driven policy tuning | stable improvement across repeated interleaved model runs |

Every stage retains the mandatory rows:

1. HVX / unmodified Hexagon-MLIR;
2. HexKL with all OmniFetch features disabled; and
3. HexKL plus only the non-quantized OmniFetch stages that have passed their
   ordered gates.

Hexagon NN Library remains the external baseline. Report model-level device
latency together with DDR/DMA bytes, L2 request/miss/overwrite/credit-fail
counters, VTCM stall counters, transformed operators removed, and physical
bytes moved. This makes “early movement” and “less movement” independently
measurable rather than inferring both from one latency number.

## Triton and triton-shared dependency boundary

The current repository is build-dependent on both submodules, even though the
OmniFetch algorithm and the torch-mlir model input are not intrinsically
Triton-dependent.

- `triton` is a hard dependency of the current canonical host build. The
  Hexagon backend is registered with `add_triton_plugin`, its Python launcher
  imports `triton._C.libtriton`, and model runners import the backend through
  the Triton Python package. The generated `libtriton.so` contains the active
  Hexagon plugin used to lower torch-mlir/Linalg modules.
- `triton_shared` is required by the canonical combined build and by the
  Triton-kernel frontend path (`Triton IR -> Linalg`). It is included in
  `TRITON_PLUGIN_DIRS`, and the build/test layout places its tools beside the
  Hexagon plugin.
- The current Falcon `.mlirbc`/torch-mlir path does not semantically invoke
  Triton IR or require `triton-shared-opt` to convert the model. After the
  compiler plugin and DSP shared objects have been built, device inference
  does not load either submodule's source tree.
- Therefore the local source edits in `triton/python/src/main.cc` and the two
  `triton_shared` conversion files are not OmniFetch item-5 changes. They are
  compatibility/lowering fixes for the Triton frontend path and remain outside
  the OmniFetch commit scope.

A future standalone `hexagon-mlir-model` build can remove the model path's
source-level dependency by replacing `add_triton_plugin` with a standalone
MLIR library/tool target, moving the torch launcher out of
`triton.backends.*`, and linking only the required MLIR/LLVM and Hexagon
libraries. Until that refactor is done, deleting or deinitializing either
submodule will break the documented build even though the core optimization
passes themselves contain no Triton IR.

## Model-level experiment matrix

Use at least these roles:

| Role | Initial runner | Purpose |
|---|---|---|
| fast mechanism screen | Falcon debug, Mamba debug | compiler/runtime correctness |
| Transformer numerical gate | GPT-2 debug | attention/MLP differential check |
| published LLM topology | GPT-2 full after NaN fix; then Qwen/Falcon | end-to-end claim |
| vision layout diversity | ViT/Swin | transpose/reshape propagation |
| convolution-heavy negative control | SD UNet/VAE, Real-ESRGAN | ensure gating avoids regressions |

For every model, compare:

- unmodified Hexagon-MLIR;
- HexKL;
- HexKL plus layout-aware prefetch;
- plus reshape reuse;
- plus persistent VTCM/layout cache;
- and Hexagon NN Library.

Use identical model, input, sequence/image size, precision, warm-up, timing
boundary, and thermal state.

## First model result

On 2026-07-26, Falcon debug (2 layers, hidden 64, vocab 4096), content-filled
sequence length 128, produced:

| Configuration | Device time | Numerical result |
|---|---:|---|
| HexKL | 1693.275 ms | top-1 match, max abs 0.0239 |
| layout-aware OmniFetch | 1689.758 ms | top-1 match, max abs 0.0239 |
| layout + WH reshape reuse + persistent VTCM | 1701.922 ms | top-1 match, max abs 0.0239 |

The debug runner fixes `torch.manual_seed(0)`, so all independently launched
cases use identical random weights. The layout-aware path is about 0.21% faster
than HexKL. Adding the current column-resident WH reshape reuse and persistent
VTCM is about 0.51% slower than HexKL and about 0.72% slower than layout-aware
OmniFetch. It must remain opt-in.

This negative result supports the proposed direction: reshape reuse cannot be
enabled merely because reuse exists. Loop interchange, VTCM footprint, lost
pipeline opportunities, and transform placement must enter a per-site cost
model. This remains a mechanism-screen result, not a publication claim: the
runner reduces Falcon depth/width/vocabulary and uses deterministic random
weights.

GPT-2 debug at sequence lengths 128 and 32 did not return from the device call
within the manual observation window and was interrupted. These runs are
failures, not performance data. The explicit timeout in
`scripts/run_omnifetch_model_ablation.sh` prevents future unbounded waits.

## Cross-model integration of cumulative items 1–7

### Why runner flags alone were insufficient

The original cumulative experiment was fully wired only in the Falcon runner.
Other runners enabled generic VDAE/layout-aware prefetch, but did not enable
items 4–7 together.  More importantly, item 7 originally relied on semantic
metadata emitted while lowering `tm_tensor.attention`.  Models exported with
eager attention contain ordinary `linalg.batch_matmul` operations, so their
K/V streams remained invisible even when the item-7 option was true.

The cross-model integration adds one common runner option:

```text
--enable-omnifetch-items-1-7
```

It enables the same cumulative configuration in every non-quantized debug
runner:

1. layout-value analysis and reshape memoization;
2. transform-aware profitability;
3. layout-carrying fusion;
4. persistent WH cache;
5. two-dimensional load/reshape/compute pipeline;
6. VTCM lifetime coloring;
7. K/V-aware page-coalesced prefetch.

Item 7 now also has a conservative compiler-side eager-attention recognizer.
It identifies only static rank-3 contractions with the unambiguous shapes
`[B,S,H] x [B,H,S] -> [B,S,S]` (K) and
`[B,S,S] x [B,S,H] -> [B,S,H]` (V), and requires `S != H`.  Explicit metadata
from `tm_tensor.attention` takes precedence.  Ambiguous square batch matmuls
are not annotated.

### Debug-model result on `omnifetch-2x-improvement`

The following measurements were made on 2026-07-28 on the connected v75
device.  The HVX and HexKL columns are the earlier measurements from the same
tag and identical debug model configurations.  The cumulative column was
rebuilt and rerun after the cross-model integration.

| Model | HVX (ms) | HexKL (ms) | HexKL + items 1–7 (ms) | vs HexKL | Actual cumulative hits |
|---|---:|---:|---:|---:|---|
| Falcon debug, seq=128 | 11238.419 | 1619.671 | 590.880 | 2.741x | KV=8, VTCM=5, async=5, persistent=5 |
| SD text encoder | 1.450 | 1.478 | 0.293 | 5.044x | KV=2 |
| SD VAE decoder | 8615.044 | 502.577 | 385.453 | 1.304x | VTCM=12, async=7, persistent=12 |
| Swin debug | 67575.498 | 66860.712 | 20642.078 | 3.239x | KV=6 |
| TinyLlama debug, seq=128 | 8879.842 | 2533.438 | 1299.005 | 1.950x | KV=18, VTCM=1, async=1, persistent=1 |
| ViT debug | 1201.766 | 1204.605 | 718.449 | 1.677x | KV=4 |

Qwen debug also completed once in a clean device state at 600.726 ms with
KV=10, VTCM=9, async=9, persistent=9 and a maximum logit difference of
0.0007.  Its earlier HexKL baseline returned device status 13, so no valid
HexKL speedup is reported.  A later matrix rerun after the GPT-2 device hang
also returned status 13 and is treated as contaminated rather than replacing
the successful clean run.

GPT-2 hit KV=4, VTCM=8, async=8 and persistent=8 at compile time, but its
device execution hung, matching the existing HexKL baseline problem.
GraphSAGE and Mamba likewise hit applicable compiler mechanisms but retained
their existing HexKL device status-13 failure.  Real-ESRGAN is a convolutional
graph with zero HexKL/attention sites, so items 4–7 are not applicable.
SD UNet also had zero cumulative sites in this debug configuration.

### Correctness caveats

- Falcon and TinyLlama matched top-1 and had maximum logit differences of
  0.0231 and 0.0005 respectively.
- SD text encoder matched the CPU reference exactly within the configured
  tolerance.
- Swin and ViT matched the reference top-1 class.
- The SD VAE device run passed, but its debug harness reported a maximum
  elementwise difference of 1.2686 and then continued because debug comparison
  is non-fatal.  Its timing is useful for performance exploration but must not
  be promoted to a correctness-qualified result until the numerical mismatch
  is resolved.
- Device status-13 failures and timeouts are not performance measurements.

The canonical three-way runner is now `scripts/run_debug_matrix.sh`.  One
invocation runs HVX, HexKL, and HexKL + cumulative items 1–7, records
per-mechanism hit counts, retries failed rows while skipping successful rows,
and writes both the attempt history (`results.csv`) and latest-result speedup
table (`summary.csv`).  The older cumulative-only script is not required.

### Completed Debug matrix for the previously blocked models

The blocked models were rerun on 2026-07-28 with one command, a clean device
state, and the same sequence length in all three configurations:

```bash
ANDROID_SERIAL=49d1c7b2 scripts/run_debug_matrix.sh \
  --runtime-root /tmp/omnifetch-2x-improvement \
  --seq-len 32 \
  --timeout 600 \
  --output-dir /tmp/omnifetch-unfinished-debug-seq32 \
  qwen2.5-0.5b graphsage mamba-130m sd_unet real-esrgan gpt2lmheadmodel
```

`--runtime-root` selected the isolated build made from the
`omnifetch-2x-improvement` tag while the runner sources came from this
worktree.  Omit that option after building the current worktree itself.

| Debug model | HVX (ms) | HexKL (ms) | HexKL + items 1–7 (ms) | HexKL / combo | HVX / combo | Applicable item hits |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B, 2L, seq=32 | 1897.422 | 153.888 | 79.475 | 1.936x | 23.875x | prefetch=45, in-situ=49, async=15, persistent=5, KV=4 |
| GraphSAGE/BERT debug, seq=32 | 296.833 | 122.142 | 81.037 | 1.507x | 3.663x | prefetch=36, in-situ=36, async=12, persistent=2 |
| Mamba debug, 1L, seq=32 | 1165.356 | 127.751 | 110.213 | 1.159x | 10.574x | prefetch=9, in-situ=9, async=3, persistent=2 |
| SD UNet debug, no cross-attn | 105.085 | 105.505 | 106.154 | 0.994x | 0.990x | none |
| Real-ESRGAN reduced RRDBNet | 71.064 | 71.041 | 70.614 | 1.006x | 1.006x | none |
| GPT-2, 2L, full LM head, seq=32 | 22496.585 | 109539.127 | 31937.477 | 3.430x | 0.704x | prefetch=24, in-situ=72, async=8, persistent=8, KV=4 |

The former Qwen, GraphSAGE, and Mamba device-status-13 failures did not
reproduce with the clean, parameter-consistent run.  Qwen and Mamba matched
top-1; GraphSAGE matched the CPU tolerance.  All three GPT-2 numerical gates
passed.  GPT-2 demonstrates a narrower claim: items 1–7 recover 3.43x over its
poor HexKL result, but the combination remains 1.42x slower than HVX.  Its
two-layer Debug model still retains the full 50,257-token LM head, which
dominates this configuration.

SD UNet and Real-ESRGAN are negative controls.  They have zero eligible
prefetch/HexKL sites, so no speedup should be claimed.  Their original
cumulative runs replayed a cold/warm/invalidated persistent-WH experiment even
with no WH-cache candidate, exhausting device heap before output emission.
The Debug runners now adaptively disable only that inapplicable replay.  The
normal full-model hooks remain unchanged.  SD UNet remains numerically
unqualified (maximum differences 0.7682–1.4204), and the reduced Real-ESRGAN
combination run also exceeded its 0.05 tolerance (0.0651); their timings are
smoke/negative-control data only.

Build trees, MLIR bytecode, generated shared objects, raw tensors, CSV files,
and logs are generated artifacts and must not be committed.

## Precision diagnosis, corrected speedup attribution, and three-domain plan

### Why the current GPT-2 HexKL baseline is slower than HVX

The current GPT-2 result is not directly comparable with the other FP16 model
runners.  The main GPT-2 harness exports the complete model in FP32.  In
HexKL mode, `rewrite_matmul_inputs_to_f16` inserts runtime `f32 -> f16`
conversions for both operands of every eligible matmul while keeping
LayerNorm, softmax, residuals, and the surrounding graph in FP32.  In
contrast, Qwen, Falcon, TinyLlama, Swin, Mamba, GraphSAGE, and ViT are exported
as FP16 models, so their HexKL rewrite normally bypasses an existing
`f16 -> f32` extension instead of converting full FP32 tensors at inference
time.

This difference is especially expensive in the current two-layer GPT-2 Debug
runner:

- only `n_layer` is reduced from 12 to 2;
- hidden width remains 768;
- vocabulary remains 50,257;
- the wrapper returns full-sequence logits.

The LM-head weight alone contains `50,257 * 768 = 38.6M` elements, or
approximately 154 MB in FP32.  Repeatedly converting this and the intermediate
activations before HMX matmuls can cost more than the HMX compute saves.  This
is consistent with the measured 22,496.585 ms HVX and 109,539.127 ms HexKL
times.

There is also direct historical evidence.  At `b6f5548`, GPT-2 was exported
entirely in FP16 and the recorded full-model result was approximately 24.0 s
HVX versus 12.0 s HexKL.  It was later changed to FP32 because the full
12-layer FP16 path produced NaNs after roughly four layers.  Therefore the
current 3.43x `HexKL / combination` result partly recovers a pathological
conversion-heavy HexKL baseline; it must not yet be presented as an intrinsic
3.43x GPT-2 structural benefit.

The next GPT-2 experiment must respect the decision not to introduce mixed
precision or quantization:

1. use an all-FP16 two-layer Debug GPT-2;
2. normalize its Debug width, heads, FFN, and vocabulary to the same scale as
   the Qwen/Falcon/TinyLlama Debug runners, or make every LLM runner compute
   last-token logits only;
3. run all three backends with byte-identical FP16 weights and inputs;
4. keep full 12-layer GPT-2 out of the qualified result set until the all-FP16
   numerical failure is fixed.

### Corrected cold-versus-warm view of the apparent high-speedup models

The cumulative runner enables persistent-WH mode.  Its reported `Perf` is the
second, warm execution, whereas the ordinary HexKL runner historically
reports one execution.  The combination logs also contain `cold_us`, which
allows a more conservative comparison:

| Model | Reported HexKL / warm combination | HexKL / cold combination | Principal mechanism | Interpretation |
|---|---:|---:|---|---|
| Falcon debug | 2.741x | 2.326x | persistent WH + KV prefetch + VTCM coloring + async pipeline | robust multi-mechanism gain |
| Swin debug | 3.239x | 3.237x | window-attention KV prefetch | strongest clean structure-driven result |
| TinyLlama debug | 1.950x | 2.037x | KV prefetch, with one persistent/VTCM site | robust; warm replay is not the source |
| GPT-2 debug | 3.430x | 3.459x | VTCM coloring + KV prefetch + in-situ/pipeline | robust against warm bias, but the FP32 HexKL baseline is abnormal |
| SD text encoder debug | 5.044x | 3.336x | two KV-prefetch sites | sub-millisecond result; repeated statistical validation required |
| Qwen debug | 1.936x | 1.423x | 15 async sites + persistent WH + eager-attention KV prefetch | structurally suitable, but the current >=1.8x claim depends on warm cache |
| ViT debug | 1.677x | 1.624x | KV-aware prefetch | moderate and relatively stable |
| GraphSAGE debug | 1.507x | 0.986x | no effective persistent hits | apparent gain is warm/device-state dominated |

The mechanism counters explain the differences:

- Falcon has five async/persistent candidates, eight KV-prefetch sites, about
  28 KB of colored VTCM peak reduction, and an approximately 84% incremental
  warm WH-cache hit rate.
- Qwen has 15 async sites, four KV sites, about 36 KB of VTCM peak reduction,
  and an approximately 89% incremental warm WH-cache hit rate.  It is a strong
  cross-invocation-cache target, but its cold gain is currently only 1.42x.
- TinyLlama has 18 KV sites and about 360 KB of KV-prefetch traffic.  Its warm
  execution is slightly slower than cold, so its approximately 2x gain is not
  a warm-cache artifact.
- Swin has no ordinary async/persistent candidate in this run.  Its six
  window-attention KV sites issue about 527 KB of page-coalesced prefetch, and
  cold and warm times are effectively identical.
- GPT-2 reduces colored VTCM peak by about 404 KB and issues about 384 KB of KV
  prefetch.  Its persistent cache has almost no useful hits, so the gain comes
  from data staging/layout mechanisms and from compensating for the poor FP32
  HexKL path.
- SD text encoder and GraphSAGE are too short or too warm-state-sensitive to
  support a strong paper claim without repeated measurements.

### Structural commonality of suitable models

The promising models share a memory-system structure rather than one model
name or application:

1. static sequence/window shapes expose compile-time tile, page, and lookahead
   decisions;
2. attention creates repeated Q/K/V head split, transpose, reshape, and merge
   operations that can be carried in situ instead of materialized;
3. GQA/MQA or window attention creates strong K/V reuse;
4. batch-one mobile inference is frequently memory-bound, so moving weights
   and KV data early can overlap otherwise exposed stalls;
5. matrix dimensions are 32-aligned or can be tiled into HexKL/HMX shapes;
6. the exported graph exposes ordinary matmul/batch-matmul operations rather
   than opaque fused custom operators;
7. immutable projection and FFN weights are reused across tokens or
   invocations, making a generation-safe persistent cache useful.

This predicts good results for GQA decoder models, hierarchical/window vision
transformers, and long-sequence speech transformers.  It predicts little
benefit for convolution-only models, models whose eligible operations are too
small, or graphs whose attention is hidden behind unsupported custom ops.

The current reduced LLM results do not yet prove the GQA/MQA part of this
hypothesis.  The Qwen Debug configuration uses one query and one KV head, and
Falcon/TinyLlama Debug use two query and two KV heads.  Their measured graphs
therefore have MHA-style head counts even though the published full
architectures use GQA/MQA variants.  The present speedups demonstrate
attention-KV movement and persistent-weight opportunities; explicit
query-head sharing must be validated with the new full-ratio candidates.

### Final target set: 15 models in three domains

The final set should not consist only of decoder-only LLM variants.  The
planned set has eight Language/LLM models, four Computer Vision models, and
three Speech/Audio models.  The list preserves the current high-gain vehicles
and adds architectures that exercise the same memory abstractions in different
domains.

#### Domain A: Language and LLM inference (8)

| # | Model | Status | Structural reason |
|---:|---|---|---|
| 1 | Falcon-RW-1B Debug | existing | fused attention/QKV movement, persistent WH, async pipeline |
| 2 | Qwen2.5-0.5B Debug | existing | RoPE/eager attention and many aligned projection/FFN sites |
| 3 | TinyLlama-1.1B Debug | existing | Llama-style attention and large KV-prefetch opportunity |
| 4 | GPT-2 Debug | existing; FP16 baseline must be repaired | classic MHA plus large layout/VTCM opportunity |
| 5 | SD/CLIP text encoder Debug | existing; statistical qualification pending | encoder self-attention and short-sequence KV prefetch |
| 6 | Qwen2.5-Coder-0.5B | new | 0.49B, 24 layers, 14 query heads / 2 KV heads; direct Qwen-family replication |
| 7 | SmolLM2-135M | new | Llama decoder, 30 layers, width 576, 9 query / 3 KV heads |
| 8 | MobileLLM-125M | new | on-device deep-thin decoder with GQA, SwiGLU, and shared embeddings |

Primary model references:

- <https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B>
- <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>
- <https://huggingface.co/facebook/MobileLLM-125M>

Qwen2.5-Coder is useful as a replication, not independent architecture
evidence.  SmolLM2 provides a natively supported Llama/GQA family.  MobileLLM
is particularly relevant to the paper's phone-inference motivation, but its
Hugging Face path uses custom model code, so export feasibility must be
screened before device work.

#### Domain B: Computer Vision (4)

| # | Model | Status | Structural reason |
|---:|---|---|---|
| 9 | Swin-Tiny Debug | existing | hierarchical window attention; current clean 3.24x result |
| 10 | SwinV2-Tiny | new | same window-attention family with changed attention/layout details |
| 11 | SegFormer MiT-B0 | new | hierarchical spatial-reduction attention and heavy feature-layout transitions |
| 12 | DeiT-Small | new | 22M-parameter global ViT attention at 224x224; architecture-diverse control |

Primary model references:

- <https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256>
- <https://huggingface.co/nvidia/mit-b0>
- <https://huggingface.co/facebook/deit-small-patch16-224>

SwinV2 is the highest-confidence extension because the existing Swin gain is
cold/warm invariant and attributable to window-attention KV prefetch.
SegFormer tests whether the abstraction transfers to hierarchical
spatial-reduction attention.  DeiT is a global-attention boundary case: it is
less likely than Swin to exceed 1.8x, but is necessary to avoid selecting only
models already known to match the mechanism.

#### Domain C: Speech and Audio (3)

| # | Model | Status | Structural reason |
|---:|---|---|---|
| 13 | Whisper-tiny | new | 39M encoder-decoder Transformer; self- and cross-attention |
| 14 | Wav2Vec2-base-960h | new | 94.4M long-sequence speech encoder with repeated self-attention |
| 15 | AST AudioSet | new | audio spectrogram converted to a ViT token sequence; direct cross-domain layout test |

Primary model references:

- <https://huggingface.co/openai/whisper-tiny>
- <https://huggingface.co/facebook/wav2vec2-base-960h>
- <https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593>

Whisper covers encoder-decoder and cross-attention, Wav2Vec2 stresses
long-sequence encoder memory traffic, and AST deliberately applies a ViT to
spectrogram patches.  Together they test whether the OmniFetch abstraction
generalizes across data modalities rather than merely across model names.
The installed Transformers 4.52.4 exposes native Whisper, Wav2Vec2, AST,
DeiT, SwinV2, and SegFormer classes.

### Evaluation rules for the 15-model set

Cross-domain runtimes must not be pooled into one average.  Comparisons are
within each model and workload:

- LLM/Text: batch 1, fixed prefill lengths 32 and 128, FP16 throughout, and a
  consistent last-token-logit policy;
- Vision: batch 1 at the checkpoint's native 224/256 resolution;
- Speech/Audio: batch 1 with fixed-duration audio or fixed feature-frame
  length, reported explicitly.

Every backend must execute the same timing protocol:

1. one cold execution after a fresh model/device context;
2. the same number of warm-up executions for HVX, HexKL, and the combination;
3. at least five measured executions, reporting median and p90;
4. cold latency reported separately from steady-state latency;
5. compiler mechanism counts, KV bytes, VTCM peak reduction, persistent
   hit/miss counters, numerical correctness, and thermal/device state retained
   with the result.

A model is a qualified `>=1.8x` result only when:

- `HexKL / (HexKL + items 1-7) >= 1.8` under a symmetric timing protocol;
- correctness passes;
- at least one intended mechanism has a nonzero runtime/compiler counter;
- the result is not explained only by warm-up, DVFS, or an artificially poor
  precision-conversion baseline.

All 15 models, including failures and sub-threshold results, must remain in the
reported matrix.  The scientific claim should be that the system was evaluated
on 15 models from three domains and that the high-gain subset shares the
predicted memory-access structure, not that unsuccessful models were removed
until 15 winners remained.

## First new-candidate implementation and device screening (2026-07-28)

Three deterministic, random-weight FP16 Debug runners were added to screen one
representative per domain before full-checkpoint experiments:

- `run_smollm2-135m_debug.py`: two-layer Llama decoder, hidden 96, FFN 256,
  preserving SmolLM2's 3:1 GQA ratio as 3 query heads / 1 KV head;
- `run_swinv2-tiny_debug.py`: hierarchical shifted-window SwinV2 proxy;
- `run_ast-audioset_debug.py`: two-layer AST over a 64x32 spectrogram, using
  16x16 patches, stride 8, and the original 527-label output space.

These are architecture/performance-screening proxies, not accuracy substitutes
for pretrained checkpoints.  Config and weights are constructed locally in
FP16, so the tests require neither network access nor a model cache.

```bash
ANDROID_SERIAL=49d1c7b2 scripts/run_debug_matrix.sh \
  --runtime-root /tmp/omnifetch-2x-improvement \
  --seq-len 32 --timeout 600 \
  --output-dir /tmp/omnifetch-new-candidates-debug \
  smollm2-135m swinv2-tiny ast-audioset
```

| Debug candidate | HVX (ms) | HexKL (ms) | HexKL + items 1-7 (ms) | HexKL/combo | Outcome |
|---|---:|---:|---:|---:|---|
| SmolLM2-135M proxy | 3145.272 | 204.110 | 120.371 | **1.6957x** | all three pass |
| SwinV2-Tiny proxy, 64x64/4-stage | N/A | N/A | N/A | N/A | device launcher exit 13 |
| AST AudioSet proxy | 235.001 | 236.875 | N/A | N/A | combo compile timeout at 600 s |

SmolLM2 is positive but currently below the 1.8x target.  Its combination run
has direct mechanism evidence: 33 prefetch sites, 37 in-situ operations, 11
async and 5 persistent choices, 53,248 bytes of VTCM peak saved, four
KV-prefetch sites, and four eagerly inferred KV sites.  Maximum absolute logit
error was 0.0006, so this is not an enabled-but-dead option.

The negative results are informative.  AST HexKL is 0.8% slower than HVX; its
combination log has zero prefetch sites, zero async/persistent choices, zero
VTCM saving, and layout elimination reports `no work`.  Four-stage SwinV2
remains expensive to lower after reducing it to 32x32/embed 32: HVX compilation
still exceeded 600 seconds.  This implicates window/control graph complexity
rather than tensor volume alone.  The retry was terminated and its generated
artifacts were not added to Git.

Screening decision: advance SmolLM2 to a full-checkpoint symmetric repeated
experiment; do not count AST without a new audio-specific movement policy such
as spectrogram patch-stream DMA; and first bound/fix SwinV2 compile complexity
before claiming a device performance result.

## Second new-candidate screening (2026-07-28)

The second group adds three offline deterministic FP16 structural proxies:
Qwen2.5-Coder-0.5B with its exact 7:1 GQA grouping ratio, a two-stage SegFormer
MiT-B0, and a one-layer encoder/one-layer decoder Whisper-tiny.  The matrix was:

```bash
ANDROID_SERIAL=49d1c7b2 scripts/run_debug_matrix.sh \
  --runtime-root /tmp/omnifetch-2x-improvement \
  --seq-len 32 --timeout 480 \
  --output-dir /tmp/omnifetch-new-candidates-group2 \
  qwen2.5-coder-0.5b segformer-mit-b0 whisper-tiny
```

Whisper initially exposed two integration defects.  Its convolutional frontend
hard-codes exact GELU and left unsupported `math.erf`; the Debug harness now
uses the standard tanh GELU approximation.  Re-parsing unchanged textual IR
also lost `tm_tensor.attention`; when no HexKL rewrite fires, the harness now
retains original bytecode.  Reducing only Debug width/vocabulary to fit the DSP
then produced a complete three-way result.

| Debug candidate | HVX (ms) | HexKL (ms) | HexKL + items 1-7 (ms) | HexKL/combo | Outcome |
|---|---:|---:|---:|---:|---|
| Qwen2.5-Coder-0.5B proxy | 8486.568 | 424.451 | N/A | N/A | combo compile timeout at 480 s |
| SegFormer MiT-B0 proxy | 324.340 | 275.134 | 112.937 | **2.4362x** | all pass |
| Whisper-tiny proxy | 331.746 | 340.427 | 97.619 | **3.4873x** | all pass after lowering fixes |

SegFormer has five prefetch sites, nine in-situ sites, one persistent choice,
36,864 bytes of VTCM peak saving, and two KV-aware sites.  Its maximum absolute
error is 0.0002 and top-1 matches.  This is the first strong result on
hierarchical spatial-reduction attention and supports generalization beyond
LLM projection graphs.

Whisper's generic prefetch/cost-model counters are zero, while KV metadata
inference finds seven sites and KV-aware prefetch inserts nine sites covering
41 pages / 81,920 bytes.  Maximum absolute error is 0.0005 and top-1 matches.
Therefore its 3.4873x gain is specifically evidence for attention-state-aware
early movement across encoder self-attention, decoder self-attention, and
cross-attention, rather than layout elimination.

Qwen2.5-Coder confirms that its aligned projection/FFN structure strongly
benefits from HexKL (`HVX/HexKL = 19.9942x`).  Its combination compilation
already reports 33 prefetch sites, 11 async choices, 11 persistent choices,
69,632 bytes VTCM saving, and four KV-prefetch sites, but no runtime number was
produced inside 480 seconds.  It must remain a compile-timeout result until a
longer run or pass-complexity fix completes; mechanism counters alone are not a
performance result.

## Third new-candidate screening (2026-07-28)

The third group deliberately uses structural controls: an OPT MHA decoder, a
global-attention DeiT, and a long-sequence Wav2Vec2 encoder.  All are offline
random-weight FP16 Debug proxies.  OPT and Wav2Vec2 required semantic-preserving
export adaptations: eager attention, explicit OPT `position_ids` to avoid
`tm_tensor.scan`, and materializing Wav2Vec2's inference-time weight-normalized
position-convolution weight.

```bash
ANDROID_SERIAL=49d1c7b2 scripts/run_debug_matrix.sh \
  --runtime-root /tmp/omnifetch-2x-improvement \
  --seq-len 32 --timeout 180 \
  --output-dir /tmp/omnifetch-new-candidates-group3 \
  opt-125m deit-small wav2vec2-base
```

OPT was rerun after the explicit-position fix with a 120-second bound.

| Debug candidate | HVX (ms) | HexKL (ms) | HexKL + items 1-7 (ms) | HexKL/combo | Correctness/outcome |
|---|---:|---:|---:|---:|---|
| OPT-125M proxy | 1680.180 | 164.211 | 106.451 | **1.5426x** | pass, max error 0.0012 |
| DeiT-Small proxy | 1472.615 | 1487.593 | 656.140 | nominal 2.2672x | **invalid: NaN output, top-1 mismatch in all configurations** |
| Wav2Vec2-base proxy | 489.284 | 488.338 | N/A | N/A | baselines pass; combo compile timeout at 180 s |

OPT is a valid positive but sub-1.8x result.  It triggers 39 prefetch and 39
in-situ sites, 13 async and 13 persistent choices, and saves 53,248 bytes of
VTCM peak, but has no KV-aware sites.  Compared with GQA models, this suggests
classic MHA benefits from projection/weight movement yet lacks the extra
KV-state opportunity that produced Whisper's larger gain.

DeiT's raw latency ratio must not be used in the paper's successful-model
count.  The identical NaN/top-1 failure in HVX, HexKL, and the combination
points to a pre-existing device numerical/lowering problem rather than an
OmniFetch-only regression, but no backend is correct.  It is retained as a
failed boundary case.

Wav2Vec2's HVX and HexKL baselines are effectively equal and numerically valid
(maximum error 0.0009, top-1 match).  The combination pass exceeded the
180-second compile limit, so this experiment currently establishes neither
speedup nor slowdown.  A longer compile run is lower priority than fixing the
systemic combination-pass compile complexity.

## Current >=1.8x census and domain balance (2026-07-28)

For architecture screening, use the relaxed rule requested for the current
stage: three runtime numbers exist and `HexKL / combination >= 1.8`; DeiT is
temporarily counted despite its NaN correctness failure.  Under that rule
there are **9** models:

| Domain | Count | Models |
|---|---:|---|
| Language/Text | 5 | Falcon, GPT-2, Qwen2.5-0.5B, TinyLlama, SD/CLIP text encoder |
| Computer Vision | 3 | Swin, SegFormer MiT-B0, DeiT-Small (correctness waived) |
| Speech/Audio | 1 | Whisper-tiny |
| **Total** | **9** | target is approximately 15 |

This is a screening count, not yet the final paper-qualified count.  Qwen's
warm result falls below 1.8x, GPT-2 has an abnormal FP32 HexKL baseline,
SD/CLIP is sub-millisecond, and DeiT is numerically invalid.  A strict,
robustness-aware paper count is therefore only five today: Falcon, TinyLlama,
Swin, SegFormer, and Whisper.

To reach 15 while balancing three domains, the target should be **5/5/5**.
Language/Text already has five screening candidates, so the next search should
add two vision and four speech/audio successes:

- Vision: DETR (CNN + encoder/decoder + cross-attention) and BEiT (global ViT
  with relative position embeddings);
- Speech/Audio: Speech2Text (encoder/decoder cross-attention), HuBERT,
  WavLM, and Data2Vec-Audio (long-sequence speech encoders);
- additionally rerun Wav2Vec2 combination with a longer compile bound, but
  retain it as an explicit timeout if it still fails.

All six new model families have native classes in the installed Transformers
4.52.4.  They must all remain in the reported matrix; unsuccessful candidates
must not be silently replaced.  Full, non-reduced checkpoint work should start
only after the Debug screening set reaches the desired count, except that full
runner feasibility work may begin early for the already robust five models.

## Fourth six-candidate Debug screening (2026-07-28)

The planned two vision plus four audio candidates were all retained in one
matrix, with a 90-second per-configuration screening bound:

```bash
ANDROID_SERIAL=49d1c7b2 scripts/run_debug_matrix.sh \
  --runtime-root /tmp/omnifetch-2x-improvement \
  --seq-len 32 --timeout 90 \
  --output-dir /tmp/omnifetch-new-candidates-group4 \
  detr-resnet-50 beit-base speech2text-small \
  hubert-base wavlm-base-plus data2vec-audio-base
```

| Candidate | HVX (ms) | HexKL (ms) | Combination (ms) | HexKL/combo | Screening result |
|---|---:|---:|---:|---:|---|
| DETR | N/A | N/A | N/A | N/A | `tm_tensor.scan` parser failure |
| BEiT | 688.260 | 694.660 | 318.362 | **2.1820x** | nominal pass; all outputs NaN |
| Speech2Text | N/A | N/A | N/A | N/A | `tm_tensor.scan` parser failure |
| HuBERT | 503.813 | 491.712 | N/A | N/A | baselines correct; combo compile timeout |
| WavLM | N/A | N/A | N/A | N/A | torch-mlir heterogeneous-list importer failure |
| Data2Vec-Audio | N/A | N/A | N/A | N/A | baseline device exit 13; combo timeout |

BEiT uses an absolute-position Debug variant because its dynamic relative
position index cannot currently be legalized by torch-mlir.  Its nominal
2.182x is driven by four KV-aware sites covering 16 pages / 33,280 bytes, but
it has the same NaN correctness problem as DeiT.  Under the current
accuracy-waived screening rule it can be counted; it is not paper-qualified.

The relaxed `>=1.8x` census therefore rises from 9 to **10**:
Language/Text 5, Computer Vision 4, Speech/Audio 1.  The target of 15 is still
not reached and domain balance remains poor.  The failures identify concrete
infrastructure work rather than evidence that the optimization is slow:

1. register/lower `tm_tensor.scan`, or eliminate scan through explicit
   position/mask metadata, for DETR and Speech2Text;
2. bound the combination-pass compile complexity for HuBERT/Wav2Vec2-family
   graphs;
3. legalize WavLM relative-position bucket construction;
4. diagnose the shared DeiT/BEiT FP16 device NaN before final paper runs.
