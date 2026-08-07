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

### Immediate next gates

1. Add stable candidate/loop IDs and a plan-consuming APT injection pass while
   continuing to require manual-safe candidate annotations.
2. Replace `runtime_clip_v1` with measured page-contained scheduling. Multiple
   fragments must respect the V73 single-flight engine rather than issuing a
   burst that is silently busy-suppressed.
3. Verify the complete path in final Hexagon assembly/object, then add the
   strictly serial model comparison script and run one existing full model.
