# OmniFetch: Prefetch and In-situ Reshape Innovation Plan

## Scope and evaluation rule

HexKL remains the compute baseline. OmniFetch should optimize how HexKL-ready
data is produced, moved, retained, and scheduled rather than duplicate the
closed-source HMX kernels.

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

### P2: Prefetch plus dequantization/reshape

For W8A16, W8A8, or W4A16, fuse:

```text
packed quantized load -> dequantize -> WH reshape -> HMX-ready tile
```

The scout side can produce a ready WH tile while compute consumes the previous
one. Keeping compressed weights in DDR/L2 reduces bandwidth and makes overlap
more valuable.

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
6. Extend to activation chains, KV cache, and fused dequantization.

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
| 8 | prefetch plus dequantization/reshape | fused compressed load, dequantization, and WH production | pending |

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
