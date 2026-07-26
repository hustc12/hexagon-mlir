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
| 5 | two-dimensional load/reshape/compute pipeline | independently scheduled load and transform readiness | pending |
| 6 | VTCM lifetime coloring | interference-based VTCM offset assignment | pending |
| 7 | KV-cache-aware prefetch and layout | page-aware K/V staging and attention-consumer layout | pending |
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
