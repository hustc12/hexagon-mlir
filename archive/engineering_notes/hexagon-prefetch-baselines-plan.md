# Hexagon prefetching baselines: implementation plan

## Scope and source-method boundary

This task ports two published software-prefetching ideas to the public
Hexagon execution surface:

1. **Prefetch-Kernel-HX**, derived from *Classifying Memory Access Patterns
   for Prefetching* (ASPLOS 2020), decides which future address can be
   reconstructed safely and cheaply.
2. **APT-GET-HX**, derived from *APT-GET: Profile-Guided Timely Software
   Prefetching* (EuroSys 2022), decides the prefetch distance and whether the
   hint belongs in an inner or outer loop.

The source papers use x86 tracing, PEBS/LBR, and—in the most general
Prefetch-Kernel case—a proposed fault-suppressing `specmov`. Those mechanisms
are not claimed to exist on Hexagon. The Hexagon implementations preserve the
papers' decision abstractions while replacing their platform-specific
frontends with MLIR SSA analysis, device cycle profiling, and V73 `l2fetch`.

The baselines are deliberately separate from OmniFetch. They may reuse the
same `omni_fetch.l2_hint` carrier and the page-safe runtime helper, but they do
not use in-situ reshape, shadow-buffer rewiring, DMA-to-VTCM, V-DAE,
persistent caches, adaptive OmniFetch control, or K/V semantic annotations.

## Existing implementation inventory

The repository already provides:

- `omni_fetch.l2_hint`, a destination-free read-only cache-hint operation;
- lowering from a static strided memref region to
  `__omni_fetch_l2_hint_2d(base,width,height,stride)`;
- a V73 runtime helper that emits non-blocking `l2fetch`, limits normal
  requests to the recommended sub-8-KiB range, observes the single-flight PFA
  state, clips requests at the starting virtual page, and records requested,
  issued, clipped, busy-suppressed, and unsupported counts; and
- LWP/function timing plus serial/interleaved execution infrastructure.

The current `PrefetchInsertPass` is not itself a valid external baseline: its
normal paths can allocate/copy shadow buffers, rewrite consumers, fuse HexKL
layout transforms, stage VTCM, and compose with other OmniFetch items.
Baseline lowering therefore uses a dedicated pass and independent options.

## Baseline 1: Prefetch-Kernel-HX

### MVP capability

The first implementation analyzes bufferized `memref.subview` readers inside
static `scf.for` loops. Starting at the future tile address, it performs a
restricted SSA/address classification equivalent to a load-free compacted
prefetch kernel:

- sources: tensor/memref base, loop induction variable, constants;
- operations: induction plus constant distance, static strides and offsets;
- supported patterns: contiguous affine tile (L1) and statically strided 2-D
  tile (L2);
- rejected patterns: dependency loads, dynamic shapes/sizes/strides,
  non-positive trip/step, ambiguous induction dimensions, writes through the
  candidate view, out-of-bounds future tiles, and requests above the
  configured command budget.

For iteration `i`, the pass reconstructs `i + distance * step`, guards the
remaining trip range, creates the future subview, and emits exactly one
`omni_fetch.l2_hint`. It never changes the demand load or consumer buffer.

Each function receives a ledger containing loops, candidate views, admitted
1-D/2-D kernels, emitted hints, estimated requested bytes, and rejection
counts. Every hint is tagged with baseline kind, address class, fixed
distance, and page policy.

### Page boundary status

The existing runtime clips a command to its starting 4-KiB page. This is safe
but may lose coverage and is not the final paper-quality implementation. The
MVP tags this policy as `runtime_clip_v1` and reports it. The next runtime
stage will split a logical tile into page-contained rows/sub-boxes while
respecting V73's single-flight engine; it must not issue a burst whose later
commands are merely busy-suppressed.

### Fixed timing policy

Prefetch-Kernel-HX uses a fixed conservative distance, initially one, with a
bounded sweep over `{1,2,4,8,16}`. This keeps address reconstruction separate
from APT-GET timing and supports the required address-only ablation.

## Baseline 2: APT-GET-HX

APT-GET-HX uses a known/manual-safe address candidate and replaces the fixed
timing policy with a profile-derived plan.

The profile key is `(model, kernel, shape, loop, candidate)` and stores:

```text
iteration-cycle histogram, trip count, warm/near-cache peak,
cold/memory peak, optional L2/DDR stall counters,
row bytes, rows, stride, alignment, and page clipping/split counts
```

The selector computes `d0 = ceil((cold_peak - warm_peak) / warm_peak)`, maps
it to a legal distance, and considers the outer-loop site when
`trip_count * coverage_factor < d0`. The result is projected through remaining
iterations, command budget, estimated residency capacity, and page constraints.
No separable peaks or a mismatched shape produces a no-prefetch plan.

The first version will use explicit cycle-profile JSON and a deterministic
offline selector. It will not pretend that Hexagon LWP is Intel LBR/PEBS.
Later instrumentation can generate the same schema automatically.

## Fair comparison matrix

All rows compile the same model graph, precision, shapes, layouts, input data,
threading, device frequency/thermal protocol, warm-up, and serial measurement
count:

| Row | Address decision | Timing decision | Other OmniFetch mechanisms |
|---|---|---|---|
| No prefetch | none | none | off |
| Manual fixed | manually marked | fixed distance | off |
| Prefetch-Kernel-HX | automatic safe SSA classification | fixed distance | off |
| APT-GET-HX | manually/previously qualified | profile distance/site | off |
| OmniFetch | OmniFetch analysis | OmniFetch policy | only the explicitly named cumulative row |
| Oracle | exhaustive legal distance/site/box sweep | offline best | off |

Primary outcomes are complete-model latency and correctness. Causal metrics
are emitted/issued hints, requested/issued bytes, page clipping/splitting,
busy suppression, extra address instructions, final-object `l2fetch`, and
available L2/DDR stall counters. A zero-hit or device-failed row is not a
performance result.

## Ordered implementation and gates

1. Implement the isolated Prefetch-Kernel-HX pass, option plumbing, ledger,
   positive 1-D/2-D tests, and negative safety tests.
2. Verify MLIR-to-LLVM/runtime calls and final Hexagon object contains
   `l2fetch`; add page-contained splitting rather than relying only on clip.
3. Define APT-GET-HX JSON schema and deterministic distance/site selector;
   test clean bimodal, noisy/unimodal, short-trip, capacity, and shape-mismatch
   cases.
4. Add compiler options for APT plans without enabling any other baseline or
   OmniFetch feature.
5. Add one serial script that runs the comparison matrix and records both
   latency and causal counters.
6. Screen one already device-runnable full model, then expand to the agreed
   model set. A baseline is admitted only after correctness, real `l2fetch`,
   nonzero issued coverage, and repeatable serial timing are all established.

## Implementation checkpoint (2026-08-06)

The first engineering slice is now present, but no device performance result
is claimed yet.

### Prefetch-Kernel-HX implemented surface

- `PrefetchKernelHXPass.cpp` implements the isolated static classifier and
  future-subview reconstruction described above.
- `--prefetch-kernel-hx` exposes the pass in `linalg-hexagon-opt`.
- Model compilation uses independent options
  `enablePrefetchKernelHX`, `prefetchKernelHxDistance`, and
  `prefetchKernelHxMaxCommandBytes`. Enabling them does not set
  `enablePrefetch` or any other OmniFetch flag.
- The pass emits only `omni_fetch.l2_hint`; the existing
  OmniFetch-to-LLVM lowering is enabled solely to translate that carrier to
  `__omni_fetch_l2_hint_2d`.
- Function attributes form the auditable admission/rejection ledger. There is
  no parallel diagnostic printing, so multi-function compilation cannot
  interleave or corrupt the ledger text.
- `prefetch-kernel-hx.mlir` covers affine 1-D, affine 2-D, destination/write
  rejection, and oversized-command rejection.
- The v73 runtime-test object was disassembled with
  `hexagon-llvm-objdump -d`; `omni_l2fetch_2d` contains the real instruction
  `l2fetch(r16,r1:0)`. This proves the shared hint runtime is not a host-only
  stub, although a complete model still needs emitted/issued counter checks.

The current address MVP recognizes direct static `memref.subview` readers in
an `scf.for`. This is deliberately narrower than the paper's dynamic binary
dataflow extraction. Load-bearing indirection remains rejected until a public,
fault-safe Hexagon mechanism or a complete bounds proof exists.

### APT-GET-HX implemented surface

`scripts/apt_get_hx_select.py` is the deterministic offline policy frontend.
It accepts schema version 1 cycle histograms, validates the model/kernel/shape
key, detects separable warm and cold peaks, computes
`ceil((cold-warm)/warm)`, chooses from `{1,2,4,8,16}`, applies the paper's
inner-versus-outer rule, and projects the choice through trip count, command,
page-split, and residency budgets. It records a SHA-256 digest of the complete
input profile in the output plan. A mismatch or unreliable profile yields an
explicit no-prefetch plan rather than a guessed distance.
The machine-readable input contract is
`docs/schemas/apt-get-hx-profile-v1.schema.json`.

The model pipeline also exposes `enableAPTGetHX` and `aptGetHxDistance`.
APT-only mode admits only a `memref.subview` or enclosing `scf.for` carrying
the unit attribute `prefetch_baseline.manual_safe`; its hints and function
ledger are tagged `apt-get-hx`. The compiler rejects any build that enables
APT-GET-HX and Prefetch-Kernel-HX together. These identities are compiler
enforced rather than inferred later from runner names.

Example:

```bash
python scripts/apt_get_hx_select.py profile.json \
  --expected-shape 1x128x768xf16 \
  --output apt_get_hx_plan.json --pretty
```

The unit tests cover a clean paper-like bimodal profile, a unimodal fallback,
outer-loop selection, capacity projection, and shape mismatch. Pass tests also
verify that APT-only mode accepts a manual-safe view and rejects an identical
unmarked view. The remaining compiler gate is consumption of per-candidate
plan IDs and outer-loop slice cloning. Until that lands, the compiler accepts
one selected global distance per APT-GET-HX model build.

## Execution-flow taxonomy and full-model checkpoint (2026-08-13)

### Orthogonal classification

The useful classification has three independent axes rather than treating a
compiler, a compute unit, and a prefetch policy as peers:

| Axis | Choices in this repository | Meaning |
|---|---|---|
| Framework/compiler | Hexagon-MLIR | graph import, optimization, lowering, code generation, and device launcher |
| Compute mapping | scalar; HVX vector; HexKL/HMX-compatible matrix path | where the arithmetic selected by lowering actually executes |
| Data-movement policy | none; Prefetch-Kernel-HX; APT-GET-HX; OmniFetch | how future data movement/cache hints are selected |

Accordingly, “Hexagon-MLIR” is the no-prefetch framework control, while the
two external methods and OmniFetch are data-movement policies compiled inside
that same framework. Scalar/HVX/HMX are compute configurations on which a
policy can be evaluated. A fair prefetch comparison fixes the framework and
compute mapping and changes only the data-movement policy.

HexKL must not be labeled as equivalent to the entire HMX or HTP stack. It is
the available open Hexagon library/lowering route for HMX-compatible matrix
shapes. Every result must record the number of successful HexKL rewrites. A
row with HexKL enabled but zero rewrites is an HVX compute row with HexKL
inactive, not an HMX performance result.

### Implementation-completeness verdict

- **Prefetch-Kernel-HX is complete for the explicitly scoped Hexagon MVP**:
  it has an isolated pass and options, static affine 1-D/2-D admission,
  ownership/bounds rejection, future-address reconstruction, V73 `l2fetch`,
  runtime safety clipping and counters, tests, and full-model execution.
  It is not the paper's unavailable fault-suppressing dynamic-indirection
  hardware and makes no such claim.
- **APT-GET-HX is engineering-complete for the current global-plan MVP**:
  the profile selector, schema, manual-safe enforcement, stable candidate
  allowlist, global selected distance, independent options, tests, runtime
  counters, and full-model execution work. It is not yet a full-fidelity port
  of per-candidate plans and automatic outer-loop cloning; those remain the
  next compiler gate. Therefore papers and plots should call the current row
  “APT-GET-HX global-plan MVP,” not an unrestricted reproduction of APT-GET.

### Full non-Debug DINOv2 and ViT results

The serial script is `scripts/run_prefetch_baseline_full_two_models.sh`. It
uses no timeout and never runs models/configurations concurrently. For each
model it first runs Prefetch-Kernel-HX, extracts that complete graph's stable
admitted candidate IDs, then runs APT-GET-HX using exactly that manual-safe
allowlist. Both methods use distance one in this checkpoint; therefore their
site/traffic counts intentionally match and this run validates execution, not
APT's eventual per-site timing advantage.

Both graphs are full FP16 model structures at 224x224:

- DINOv2-small: 12 layers, hidden 384, 6 heads, intermediate 1536,
  22,825,192 parameters, 257 tokens;
- ViT-Base: 12 layers, hidden 768, 12 heads, patch 16, 86,416,360
  parameters, 197 tokens.

Both compile with `hvx-vector`, VTCM whole-graph tiling disabled, and HexKL
enabled. Each contains 96 batch matmuls and one matmul, but both report zero
HexKL rewrites. These are therefore **true HVX-vector + L2-prefetch** results;
HMX is inactive in all four rows.

| Full model | Policy | Latency | Static hints | Runtime issued | Requested bytes | Issued bytes | Correctness |
|---|---|---:|---:|---:|---:|---:|---|
| DINOv2-small | Prefetch-Kernel-HX | 10,553.097 ms | 360 | 5,446,124 | 1,325,550,848 | 1,273,555,516 | finite; max diff 0.0046; top-1 match |
| DINOv2-small | APT-GET-HX global-plan MVP | 10,428.103 ms | 360 | 5,446,124 | 1,325,550,848 | 1,273,555,516 | finite; max diff 0.0046; top-1 match |
| ViT-Base | Prefetch-Kernel-HX | 20,620.770 ms | 334 | 6,529,693 | 1,578,968,448 | 1,537,553,384 | finite; max diff 0.0046; top-1 match |
| ViT-Base | APT-GET-HX global-plan MVP | 20,683.438 ms | 334 | 6,529,693 | 1,578,968,448 | 1,537,553,384 | finite; max diff 0.0046; top-1 match |

APT is 1.184% faster than Prefetch-Kernel on this single DINOv2 sample and
0.304% slower on ViT. These small differences must not be interpreted as a
speedup claim: one device iteration was requested, both policies selected the
same sites and distance, and a matched no-prefetch row has not yet been run in
this checkpoint. The four rows do establish correctness, full-model support,
real nonzero device issue coverage, and stable serial automation.

During recovery, three baseline-infrastructure defects were fixed: the full
DINOv2 runner now forwards baseline options; public functions are transformed
with the caller-owned out-param ABI used by the wrapper; and FastRPC report and
output paths use the absolute Android execution directory. OmniFetch allocation
lifetime shortening is now gated behind OmniFetch and cannot contaminate the
external-baseline rows. None of these repairs is an optimization or may be
counted as baseline/OmniFetch speedup.

The validated local result root is
`/home/huzq85/2-working/hexagon_npu/run_artifacts/full_prefetch_baselines_20260813_valid`.
The complete 1.7-GiB generated artifacts and logs are backed up at
`nano:/home/huzq85/2-working/working_set/full_prefetch_baselines_20260813_valid`;
generated artifacts remain outside Git.

### Why the two external rows currently have identical traffic

The equality is expected from the current implementation, rather than evidence
that the counters are broken. The full-model script first obtains every stable,
admitted candidate ID from Prefetch-Kernel-HX and then gives exactly that
allowlist to APT-GET-HX. Both select global distance one and ultimately use the
same hint emitter and V73 runtime helper. Consequently, they generate the same
dynamic request schedule for a fixed model: identical runtime-issued requests,
requested bytes, and issued bytes.

The object files need not be byte-identical because method identity attributes,
ledgers, and symbols differ. The relevant point is that the current schedules
are logically equivalent. The 1.184% DINOv2 difference and -0.304% ViT
difference are compatible with one-sample device variation, not a causal
algorithmic difference. To create a meaningful APT row, the compiler must
consume the selector's per-candidate plan (site subset, individual distance,
and eventual outer-loop placement) rather than reducing it to the same global
allowlist and distance as Prefetch-Kernel-HX.

### OmniFetch items 1--7 full-model gate

The preferred next step was evaluated before broadening either external
baseline to all models: run complete OmniFetch items 1--7 on these same two
graphs, with the same FP16 input structure, `hvx-vector` profile, HexKL enabled,
serial execution, and no timeout. The reproducible driver is
`scripts/run_omnifetch_full_two_models.sh`.

The OmniFetch persistent-cache protocol intentionally invokes the device model
three times (cold, warm, then invalidated). Latency below is the measured warm
P50. Runtime traffic counters are cumulative across all three calls, so the raw
traffic must be divided by three before comparison with the one-call external
baseline rows.

| Full model | Policy | Warm latency | Runtime issued | Issued bytes | Correctness |
|---|---|---:|---:|---:|---|
| DINOv2-small | Prefetch-Kernel-HX | 10,553.097 ms | 5,446,124 | 1,273,555,516 | finite; top-1 match |
| DINOv2-small | APT-GET-HX global-plan MVP | 10,428.103 ms | 5,446,124 | 1,273,555,516 | finite; top-1 match |
| DINOv2-small | OmniFetch items 1--7 | 19,061.970 ms | 171,041,329 cumulative | 287,102,337,728 cumulative | finite; max diff 0.0049; top-1 match |
| ViT-Base | Prefetch-Kernel-HX | 20,620.770 ms | 6,529,693 | 1,537,553,384 | finite; top-1 match |
| ViT-Base | APT-GET-HX global-plan MVP | 20,683.438 ms | 6,529,693 | 1,537,553,384 | finite; top-1 match |
| ViT-Base | OmniFetch items 1--7 | 48,344.123 ms | 398,320,594 cumulative | 669,108,840,110 cumulative | finite; max diff 0.0049; top-1 match |

This full-model gate is negative but diagnostically decisive. OmniFetch is
1.806x slower than Prefetch-Kernel-HX and 1.828x slower than APT-GET-HX on
DINOv2; it is 2.344x and 2.337x slower respectively on ViT. Correctness still
passes, so this is data-movement overhead rather than a wrong model result.

After normalizing the three OmniFetch calls, DINOv2 still issues 10.47x as many
requests and 75.14x as many bytes as either external baseline. ViT issues
20.33x as many requests and 145.06x as many bytes. In both models,
`page_clipped == issued`: every runtime request is clipped at a page boundary.
The complete graph activates 72 persistent/async cost-model sites, while the
attention K/V pass reports 24 static sites on DINOv2 and 25 on ViT. Dynamic hot
loop execution turns these apparently selective static sites into a command and
traffic storm. The earlier Debug result—where OmniFetch issued far fewer
requests—therefore does not generalize to the full graph under the current
items-1--7 policy.

The validated local result root is
`/home/huzq85/2-working/hexagon_npu/run_artifacts/full_omnifetch_vs_prefetch_20260813`;
the 840-MiB result backup is
`nano:/home/huzq85/2-working/working_set/full_omnifetch_vs_prefetch_20260813`.
Generated outputs remain outside Git.

The next gate is not all-model expansion. First run a matched full-model
ablation that isolates item 7 (attention K/V stream prefetch) from the generic
persistent/async sites. Then introduce a dynamic per-site command/byte budget,
coalesce repeated page requests, and reject or reshape footprints that are
100% page-clipped. OmniFetch should expand beyond these two models only after
it beats the external rows without increasing normalized traffic pathologically.

### Item-4-disabled causal ablation and convergence (2026-08-13)

`cold prime + warm measurement + invalidated audit` is not an independent
OmniFetch optimization and is not one of the seven items. It is the benchmark
protocol used only when item 4's generation-safe persistent-WH cache is
enabled: cold is a cache miss/fill, warm is a cache hit, and invalidated is a
post-invalidation miss/rebuild. The three calls execute the same compiled
entry point but intentionally take different runtime cache branches. On the
full DINOv2 and ViT checks, cold/warm/invalidated latency differed by less than
one percent, so the following causal ablation disables item 4 and makes exactly
one measured model invocation per row.

The reproducible no-timeout, strictly serial driver is
`scripts/run_omnifetch_full_no_item4_ablation.sh`. Every OmniFetch row uses the
same `hvx-vector` profile and HexKL-enabled lowering as `hexkl-control`; the
Native Hexagon-MLIR HVX (HexKL-off) row is retained only as an additional
framework reference. This
matched control is essential: even with zero reported HexKL matmul rewrites,
`--enable-hexkl` changes upstream lowering/code generation and reduces DINOv2
from 29,885.944 ms to 10,090.480 ms and ViT from 132,967.479 ms to
19,959.873 ms. That improvement is not an OmniFetch or HMX-kernel claim.

| Full model | Scheme | Latency (ms) | Speedup vs matched HexKL | Runtime issued | Issued bytes |
|---|---|---:|---:|---:|---:|
| DINOv2-small | Native Hexagon-MLIR HVX (HexKL off) | 29,885.944 | -- | 0 | 0 |
| DINOv2-small | HexKL control | 10,090.480 | 1.000x | 0 | 0 |
| DINOv2-small | item 7 only | 5,960.990 | 1.693x | 0 | 0 |
| DINOv2-small | items 1--3 | 10,305.507 | 0.979x | 186,624 | 41,840,640 |
| DINOv2-small | items 1--5, item 4 off | 9,612.841 | 1.050x | 186,624 | 41,840,640 |
| DINOv2-small | items 1--6, item 4 off | 9,543.607 | 1.057x | 186,624 | 41,840,640 |
| DINOv2-small | items 1--7, item 4 off | 5,669.315 | 1.780x | 186,624 | 41,171,328 |
| ViT-Base | Native Hexagon-MLIR HVX (HexKL off) | 132,967.479 | -- | 0 | 0 |
| ViT-Base | HexKL control | 19,959.873 | 1.000x | 0 | 0 |
| ViT-Base | item 7 only | 13,872.472 | 1.439x | 0 | 0 |
| ViT-Base | items 1--3 | 19,979.337 | 0.999x | 0 | 0 |
| ViT-Base | items 1--5, item 4 off | 19,901.741 | 1.003x | 0 | 0 |
| ViT-Base | items 1--6, item 4 off | 19,827.296 | 1.007x | 0 | 0 |
| ViT-Base | items 1--7, item 4 off | 19,803.730 | 1.008x | 0 | 0 |

All rows pass finite-output and top-1 correctness checks. The result root is
`/home/huzq85/2-working/hexagon_npu/run_artifacts/full_omnifetch_no_item4_20260813`
and its generated-artifact backup is
`nano:/home/huzq85/2-working/working_set/full_omnifetch_no_item4_20260813`.

The most important causal finding is that the measured item-7 benefit on these
two eager vision encoders is not a hardware K/V-prefetch benefit. DINOv2 has 84
and ViT has 96 candidate K/V reads, but all sources are produced inside the
same invocation immediately before consumption. They have no legal early
prefetch window. The compiler now rejects them (`sites=0`, `issued=0`) instead
of injecting commands into inner vector loops, which previously caused a V73
DSP user-process exception. Item 7 still changes K/V propagation, tiling, and
attention slicing, and that topology change produces the 1.693x/1.439x
item-7-only speedups. Future reporting must name this contribution separately
from runtime K/V-cache prefetch. True cross-invocation K/V prefetch remains for
decoder graphs whose persistent cache arrives as an entry argument.

The DINOv2 items-1--3 row also proves that a small static site count does not
bound dynamic traffic: it issues 186,624 commands, requests 382,205,952 bytes,
and every request is page-clipped. A V73-aware per-invocation policy was
therefore added for OmniFetch only: at most 4,096 issued commands and 8 MiB,
plus a 64-entry recent window that coalesces only requests with identical
address and 2-D geometry. External Prefetch-Kernel-HX and APT-GET-HX rows keep
their original unbounded policy.

One final full DINOv2 validation is sufficient to close this gate:

| Scheme | Latency (ms) | Speedup vs HexKL | Issued | Issued bytes | Budget-suppressed | Duplicate-suppressed |
|---|---:|---:|---:|---:|---:|---:|
| items 1--7, item 4 off, bounded | 5,558.501 | **1.815x** | 4,096 | 903,872 | 178,569 | 3,959 |

Correctness passes. Compared with the unbounded 5,669.315-ms combination, the
traffic envelope adds only 1.99% latency improvement while reducing issued
commands 45.6x and issued bytes 45.5x. Thus it is an important robustness and
data-movement result, but the main speedup still comes from item 7's compiler
topology change. ViT was deliberately not rerun: its validated combination
issues zero requests, so an L2 traffic budget cannot affect it. The bounded
result is stored under
`/home/huzq85/2-working/hexagon_npu/run_artifacts/full_omnifetch_budget_20260813`
and synchronized to
`nano:/home/huzq85/2-working/working_set/full_omnifetch_budget_20260813`.

### Current default policy: item 7 only

The causal evidence above makes item 7 the current default OmniFetch policy.
Items 1--6 remain implemented but are disabled by the normal Debug, full-model,
and external-baseline scripts. Those scripts now pass
`--enable-omnifetch-kv-cache-prefetch --disable-layout-aware
--disable-omnifetch-adaptive`, not `--enable-omnifetch-items-1-7`.
Explicitly named cumulative/ablation scripts are intentionally retained for
reproducibility.

| Full model | Item7-only (ms) | vs Prefetch-Kernel-HX | vs APT-GET-HX | vs matched HexKL-on/0-rewrite HVX |
|---|---:|---:|---:|---:|
| DINOv2-small | 5,953.59 | 1.78x | 1.78x | 1.69x |
| ViT-Base | 14,115.92 | 1.52x | 1.48x | 1.42x |

These are clear improvements, but neither model reaches 1.8x against the fair
external/matched controls. Ratios against Native Hexagon-MLIR HVX (HexKL off)
are 5.04x and 9.21x;
those numbers must be labeled diagnostic rather than causal because the HexKL
flag changes lowering/code generation even with zero reported HMX rewrites.
Also, item7 emits zero runtime hints for these eager encoder graphs. Its speedup
is currently attributable to attention propagation/tiling/slicing topology,
not to issued K/V-prefetch instructions.

### Strict item7-only HVX regression rerun (2026-08-13)

After items 1--6 were disabled in the default scripts, the complete DINOv2-small
and ViT-Base HVX matrix was rerun strictly serially with one measured device
invocation and no timeout. The item7 command also explicitly sets
`--disable-layout-aware --disable-omnifetch-adaptive`. Pass logs confirm
`kvCacheOnly=1`, persistent/two-dimensional/VDAE paths disabled, zero HMX
rewrites, and zero issued K/V hints. All ten rows pass finite-output and top-1
correctness checks.

The ratio in parentheses is configuration latency divided by item7 latency;
therefore item7 is 1.00x and a larger value means a larger item7 speedup. All
reported values use two decimal places.

| Model | HVX configuration | Latency (item7 = 1.00x) |
|---|---|---:|
| DINOv2-small | Native Hexagon-MLIR HVX (HexKL off) | 30,031.10 ms (5.04x) |
| DINOv2-small | Hexagon-MLIR HVX + HexKL pipeline (0 HMX rewrites) | 10,035.87 ms (1.69x) |
| DINOv2-small | Prefetch-Kernel-HX | 10,596.72 ms (1.78x) |
| DINOv2-small | APT-GET-HX global-plan MVP | 10,591.11 ms (1.78x) |
| DINOv2-small | OmniFetch item7-only | **5,953.59 ms (1.00x)** |
| ViT-Base | Native Hexagon-MLIR HVX (HexKL off) | 129,979.04 ms (9.21x) |
| ViT-Base | Hexagon-MLIR HVX + HexKL pipeline (0 HMX rewrites) | 20,088.50 ms (1.42x) |
| ViT-Base | Prefetch-Kernel-HX | 21,489.53 ms (1.52x) |
| ViT-Base | APT-GET-HX global-plan MVP | 20,915.64 ms (1.48x) |
| ViT-Base | OmniFetch item7-only | **14,115.92 ms (1.00x)** |

There is no material switch-disable regression. Against the earlier item7-only
rows, DINOv2 changes from 5,960.99 to 5,953.59 ms (-0.12%) and ViT changes from
13,872.47 to 14,115.92 ms (+1.75%). Both changes are within the observed
single-sample full-model variation, while the main speedups remain intact.

The reproducible one-entry driver is
`scripts/run_full_hvx_regression_two_models.sh`. Generated results are kept out
of Git at
`/home/huzq85/2-working/hexagon_npu/run_artifacts/full_hvx_regression_20260813_item7_strict`
and backed up at
`nano:/home/huzq85/2-working/working_set/full_hvx_regression_20260813_item7_strict`.

The gate is now closed; do not continue blind full-model sweeps. The next code
task is to split item 7 into separately named `attention topology preparation`
and `persistent K/V hint emission` switches, then use the bounded policy as the
default OmniFetch traffic guard. No additional full-model experiment is needed
until that semantic split is implemented or a true decoder-cache ABI is under
test.

### Immediate next gates

1. Add stable candidate/loop IDs and a plan-consuming APT injection pass while
   continuing to require manual-safe candidate annotations.
2. Replace `runtime_clip_v1` with measured page-contained scheduling. Multiple
   fragments must respect the V73 single-flight engine rather than issuing a
   burst that is silently busy-suppressed.
3. Verify the complete path in final Hexagon assembly/object, then add the
   strictly serial model comparison script and run one existing full model.

## Two-model HVX Debug comparison (2026-08-06)

### Scope and controls

The first end-to-end comparison uses DINOv2 Debug and ViT Debug.  Both are
complete reduced model graphs from `benchmark_models/debug_running`, not GEMM
microbenchmarks.  Every row uses the same Hexagon-MLIR build, HexKL overlay,
`hvx-vector-vtcm` backend profile, FP16 graph, one host-launched DSP kernel,
and seven serial in-process device executions.  No configurations run in
parallel.

The exact graph/input identities are stable across all three rows:

- DINOv2 Debug model SHA-256
  `b1369edc7559bf68bb11108bd91a0b2e5e2adc8c95e8eb24b6bb6180aa2afe28`,
  input SHA-256
  `352b527f0be5256826d717e85b672a410f0332941951f1e1c7336683bba8976c`;
- ViT Debug model SHA-256
  `b4b8db84104a253e1905626a827997d086ccead17910fa317dff0dfee6bf180c`,
  input SHA-256
  `b8551e108c4f0fe57fcff10658f58f03572bea3f238808ef07a8ab5062e12edf`.

ViT Debug was repaired to call the current full-runner entry point, uses two
encoder layers, hidden size 64, 64x64 input, patch size 32, and externalizes
its position embedding through the same safe ABI strategy used by DINOv2.
Its model and input RNG are explicitly seeded before every configuration.

APT-GET-HX uses an explicit stable candidate-ID allowlist.  Two serial DINOv2
LWP samples selected the conservative global distance one.  The same distance
is used for the shape-similar ViT proxy.  This is an engineering MVP: the
current compiler consumes one model-global distance, so it does not yet
reproduce per-candidate inner/outer placement.  On these two shapes the APT
plan admits every manually qualified safe candidate and therefore converges
to the same generated request set as Prefetch-Kernel-HX.  Their latency
difference should be treated as run-order/noise, not an algorithmic win.

### Reproduction command

```bash
OUTPUT_DIR=/tmp/omnifetch-prefetch-baseline5-final2 \
DEVICE_ITERATIONS=7 MODEL_TIMEOUT=600 \
scripts/run_prefetch_baseline_two_models.sh
```

The script executes only the following three independent rows, stops on a
device/correctness failure, and rejects a baseline with zero compile-time
hints or zero runtime-issued requests:

1. HexKL + OmniFetch items 1--7;
2. HexKL + APT-GET-HX;
3. HexKL + Prefetch-Kernel-HX.

Logs and CSV files remain under `/tmp` and are not committed.

### Device results

| Model | Scheme | Mean (us) | P50 (us) | P90 (us) | Min (us) | Static hints | Runtime issued | Requested / issued bytes | Correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DINOv2 Debug | OmniFetch items 1--7 | **18,084** | 18,130 | 19,195 | 16,926 | 4 | 36 | 78,336 / 34,432 | finite, max diff 0.0006, top-1 match |
| DINOv2 Debug | APT-GET-HX | 21,052 | 20,498 | 23,096 | 18,681 | 24 | 122,843 | 18,677,120 / 18,355,264 | finite, max diff 0.0006, top-1 match |
| DINOv2 Debug | Prefetch-Kernel-HX | 20,159 | 19,996 | 22,257 | 18,627 | 24 | 122,843 | 18,677,120 / 18,355,264 | finite, max diff 0.0006, top-1 match |
| ViT Debug | OmniFetch items 1--7 | **48,511** | 47,443 | 51,415 | 46,816 | 8 | 72 | 46,080 / 41,408 | finite, max diff 0.0007, top-1 match |
| ViT Debug | APT-GET-HX | 50,073 | 46,772 | 56,425 | 46,566 | 42 | 36,274 | 5,243,392 / 5,175,552 | finite, max diff 0.0007, top-1 match |
| ViT Debug | Prefetch-Kernel-HX | 51,184 | 47,995 | 57,733 | 47,209 | 42 | 36,274 | 5,243,392 / 5,175,552 | finite, max diff 0.0007, top-1 match |

Relative to APT-GET-HX and Prefetch-Kernel-HX respectively, the OmniFetch
mean is 1.164x and 1.115x faster on DINOv2 Debug, and 1.032x and 1.055x faster
on ViT Debug.  This is a positive Debug-screen result, not yet a full-model or
statistical paper claim.

The causal result is more informative than the small ViT timing gap.
OmniFetch issues orders of magnitude fewer requests: 36 rather than 122,843
on DINOv2 and 72 rather than 36,274 on ViT.  Busy suppression is zero in all
rows, so the difference is not hidden by the single-flight guard.  The broad
baseline classifier prefetches every legal future subview in dynamically hot
loops, whereas OmniFetch's semantic/layout-aware policy selects four or eight
sites and controls their footprint.  These measurements support the design
hypothesis that timeliness must be coupled with traffic selectivity and data
movement ownership; indiscriminately issuing every safe request consumes
bandwidth and adds address/guard overhead.

The runtime report generator was extended to activate for either external
baseline, rather than only for OmniFetch V-DAE.  This closed a measurement bug
found in the first run and makes nonzero issued coverage an enforced script
gate.

### Model-screen exclusions

The screening stage deliberately rejected invalid comparisons:

- GPT-2 and Mamba scalar paths produced zero eligible hints;
- Real-ESRGAN produced zero admitted hints and its HVX/VTCM path exited 13;
- Falcon, Whisper, SegFormer, Swin, DeiT, and BEiT vector attempts produced
  candidates but hit the existing device exit-13 backend failure (DeiT's
  matching no-prefetch control failed as well).

These failures are not recorded as performance results.  ViT was admitted
only after nonzero hints, nonzero runtime-issued requests, successful device
execution, finite output, and top-1 agreement were all observed.
