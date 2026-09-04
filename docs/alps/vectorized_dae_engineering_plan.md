# Vectorized Decoupled Access--Execute Engineering Plan

## 1. Definition and scope

ALPS uses **V-DAE** to mean a *Vectorized Decoupled Access--Execute
pipeline*.  The definition follows classical DAE rather than using
"decoupled" as a synonym for ordinary loop prefetching:

1. the original program is partitioned into an **Access stream** and an
   **Execute stream**;
2. the streams have independent progress and may dynamically slip relative to
   one another;
3. they communicate through bounded queues and ready/free state, rather than a
   dependence after every individual instruction;
4. queue full/empty conditions provide backpressure and must not introduce
   deadlock;
5. Access runs sufficiently far ahead to hide memory and transformation
   latency, while Execute consumes only a proven-ready representation.

The word *vectorized* describes the granularity and consumer of the pipeline:
Access supplies vector/matrix tiles and Execute consumes them on HVX or HMX.
It does not require the Access stream itself to occupy HVX.  On Hexagon, doing
so would often make Access compete with the very HVX work it is intended to
overlap.  The preferred mapping is:

```text
Access stream:  scalar/scout address slice + UserDMA
Queue:          bounded VTCM slots + exact ready/free tokens
Execute stream: HVX/HMX consumer of the consumer-ready tile
```

Producer-direct formation remains the first choice.  It removes a layout
movement completely.  V-DAE is used only for a residual movement that cannot
be eliminated and can be overlapped.  If Access also creates the required
layout while supplying the tile, the operation is ALPS's **prefetching in-situ
transformation**.  Asynchronous HMX-result evacuation is useful for freeing
VTCM, but is classified as asynchronous write-back rather than input
prefetching.

Primary conceptual references are James E. Smith's original DAE work
(`https://www.cs.cmu.edu/afs/cs/academic/class/15740-f19/www/papers/1982-smith-dae.pdf`)
and recent compiler work on separating access and execute slices
(`https://doi.org/10.1145/3708493.3712695`).

## 2. Assessment of the current implementation

The present implementation contains useful pieces but does not yet meet the
full definition above.

- `AlpsExactReadiness` already identifies an invocation, value version, tile,
  layout, memory tiers, slot, and slot generation.  This is the correct
  ownership foundation.
- UserDMA and the P5n two-slot VTCM result path provide real asynchronous
  hardware movement.  P5n is output drainage, however, not a general Access
  stream supplying future consumer inputs.
- `AlpsVDAEInsert` inserts one semaphore around an existing loop.  In its
  default path the previous iteration issues work for the current iteration;
  it is software pipelining, not an independently progressing access slice.
- The optional scout worker exists, but the legacy path can enqueue completion
  of DMA initiated on the Execute thread.  UserDMA completion is
  thread-context-sensitive on this stack.  A true scout path must own both DMA
  issue and DMA completion; transferring only the wait is not sufficient.
- The pass does not extract address generation and transfer into a persistent
  Access program, and most consumer-driven formation sites do not emit V-DAE
  requests.
- Runtime admission observes completion lateness.  Once DMA is suppressed it
  currently has no robust periodic probe that can re-open the stream.

Consequently, the measured small residual-overlap gain is evidence for a
narrow asynchronous path, not yet evidence for a complete vectorized DAE
pipeline.

## 3. Two-layer ALPS organization

The engineering work follows the two layers of the paper.

### Layer 1: prefetching in-situ transformation

1. Consumer-driven contracts select the final representation.
2. Producer-direct formation removes avoidable materializations.
3. A residual-movement planner emits a consumer-ready transfer descriptor only
   for unavoidable, stageable movement.
4. V-DAE schedules those descriptors ahead of Execute and places the result in
   bounded VTCM slots.

V-DAE is therefore the execution backbone inside this layer, not an unrelated
top-level optimization.

### Layer 2: runtime traffic admission

Runtime chooses whether to issue, how far ahead to issue, how much VTCM to
reserve, and when to fall back synchronously.  It uses two feedback rates:

- fast, same-invocation feedback from completion latency, poll retries, queue
  occupancy, fallbacks, and wait cycles;
- slow, cross-invocation feedback derived from SDK sysMon samples such as AXI
  traffic, cache misses, VTCM activity, and HVX/HMX activity.

Stock `sysMonApp` writes a profile that is parsed after execution and exposes no
documented in-process sample API to an unsigned DSP process.  It can therefore
guide the next invocation, but it cannot honestly be described as per-tile
same-invocation feedback.  True fine-grained hardware feedback requires an
authorized PMU domain or a privileged service that exports live samples.

## 4. Implementation sequence

Each phase remains independently switchable and is tested before the next one.

### Phase V0 -- contracts and observability

- Freeze terminology above in code comments and user-facing telemetry.
- Add counters for access requests, queue-full backpressure, ready tokens,
  Execute wait cycles, overlapped bytes, synchronous fallbacks, and maximum
  queue occupancy.
- Keep legacy V-DAE disabled unless explicitly requested.

**Gate:** host runtime tests prove exact tile/version/slot ownership; a run can
distinguish issued movement from useful overlapped movement.

### Phase V1 -- scout-owned Access stream

- Introduce a bounded invocation-local SPSC request queue.
- Enqueue a complete transfer descriptor, not a callback that waits for work
  initiated on another thread.
- The scout owns UserDMA start and wait, then publishes the exact ready token.
- Execute consumes the matching tile and publishes the free token.
- Retain synchronous fallback on queue full, unsupported layout, unavailable
  worker, or insufficient lookahead.

**Gate:** a deterministic runtime test forces Access and Execute to slip,
checks backpressure and wraparound, and proves no stale tile can be consumed.

### Phase V2 -- connect consumer-ready residual supply

- Convert post-formation movement obligations into V-DAE descriptors.
- Start with contiguous/2-D FP16 tiles whose transformation is producer-direct
  or expressible by UserDMA; do not put arbitrary HVX gather work on the scout.
- Generate a K-tile prologue, steady state, and epilogue with bounded VTCM
  double buffering.
- Keep HMX result drainage separately identified as asynchronous write-back.

**Gate:** compiler IR tests show that a discharged materialization emits no
request, while a residual eligible movement emits one exact descriptor.

### Phase R1 -- recoverable fast admission

- Make completion window, late-arrival threshold, initial lookahead, and probe
  interval configurable.
- Replace permanent suppression with bounded cooldown and periodic probe.
- Adjust lookahead from useful wait reduction, not from issued bytes alone.

**Gate:** runtime tests cover allow, throttle, cooldown, re-probe, and recovery.

### Phase R2 -- sysMon-guided cross-invocation policy

- Add a host tool that compares matched Hexagon-MLIR and ALPS sysMon summaries
  and emits a versioned policy JSON.
- Validate the policy in the host launcher and embed it in the next run without
  model-specific compiler conditions.
- Record policy provenance and all applied units in `perf.txt`.

**Gate:** replaying a policy changes only admission parameters; missing or
malformed policy fails closed to the documented default.

### Phase E -- complete-model validation

Run serially on complete DINOv2-small first, followed by one model with a large
measured residual stream.  Use matched binaries/configurations for:

1. Hexagon-MLIR;
2. formation only;
3. formation + V-DAE;
4. formation + V-DAE + runtime admission.

Report latency, correctness, residual bytes, asynchronously issued bytes,
usefully overlapped bytes, Execute wait cycles, synchronous fallbacks, maximum
queue occupancy, VTCM peak usage, AXI bytes, and sysMon policy provenance.
Proceed to the full corpus only if V-DAE changes real movement and improves or
correctly rejects work without regression.

## 5. Immediate work boundary

The first code change is Phase V0 plus the recoverability foundation of R1.
It does not change the frozen default ALPS configuration.  Phase V1 follows
only after the new counters and runtime state tests pass; this avoids hiding a
thread-ownership bug inside a long complete-model run.

## 6. Phase V1 result (2026-09-04)

The first complete-model experiment used DINOv2-small with the exact weight
path and a real one-worker scout stream.  Correctness passed, all 174,960
requests were started and completed by the scout, and there were no synchronous
fallbacks.  Nevertheless, no request was READY when Execute reached its
consumer:

```text
latency_ms=11135.33
access_enqueued=174960
scout_completed=174960
ready_before_consume=0
execute_wait_cycles=8408179904
sync_fallbacks=0
```

This is an important negative result.  It proves that thread separation alone
is not DAE latency tolerance.  The compiler emitted a one-tile lead into two WH
ping-pong slots; one HMX micro-matmul was shorter than the scout scheduling,
2-D DMA, and RM-to-WH formation path.  Access therefore remained behind demand
and Execute paid the readiness wait on every asynchronous tile.

The existing `enableAlpsWeightPrepack` path does not directly solve this
problem.  It synchronously forms all K weight tiles for a column before running
its M tiles.  This removes repeated formation through reuse, but it provides no
independent Access/Execute progress and is classified as formation/prepacking,
not V-DAE.

## 7. Phase V2 bounded-lead schedule

V2 replaces the fixed one-tile scheme with a parameterized bounded queue.  For
a lookahead of K, decomposition reserves K+1 non-aliasing HMX WH slots.  The
first K tiles form the safe prologue; iteration `i` then asks Access to produce
tile `i+K`, Execute consumes the exact tile/version token, and releases its slot
only after HMX finishes.  The default experimental depth is two, capped at
seven by the eight-descriptor invocation-local ring.

The single scout serializes UserDMA start, completion, and consumer-ready WH
formation, so its DDR-to-VTCM staging tile may be reused safely.  The HMX-visible
WH destinations cannot be shared while requests are outstanding and are
therefore explicitly expanded.  Dual-stream requests no longer acquire the
legacy Execute-owned single-DMA credit at enqueue time; that credit previously
forced every queued request after the first into synchronous Execute fallback.

The next complete-model gate is not latency alone.  V2 must first produce a
non-zero `ready_before_consume` count and reduce `execute_wait_cycles` while
preserving exact correctness.  If a deeper bounded lead still cannot do so,
the remaining RM-to-WH work is too expensive or resource-conflicting for this
tile size.  The next step would then be compile/load-time consumer-ready weight
formation or a larger supertile descriptor, rather than further increasing K.

## 8. Phase V2 results and consumer-ready supply (2026-09-04)

The bounded schedule was first evaluated without changing the Access work.
Depth two reduced DINOv2-small latency from 11,135.33 ms to 10,757.69 ms, but
no descriptor was ready before demand.  Depth four produced 558 early-ready
descriptors out of 139,968 and reached 9,929.45 ms.  Much of that reduction,
however, came from moving more tiles into the synchronous prologue.  The
remaining scout path still repeated RM-to-WH formation for every M-row reuse
of a weight tile.  Increasing queue depth further would hide the symptom while
increasing VTCM occupancy; it would not repair the Access stream.

V2 therefore connects representation formation to V-DAE at the physical tile
boundary.  The first miss performs the unavoidable RM-to-WH formation and
retains the consumer-ready WH tile in the existing bounded weight cache.
Later requests for the same immutable source tile/version issue a contiguous
UserDMA transfer from that representation into the request's non-aliasing VTCM
queue slot.  Access still owns address generation, DMA start, and completion;
Execute still waits on the exact tile/version/slot token.  This is not a
cross-run warm-up scheme and does not add cold/warm benchmark invocations.  It
is reuse within one model invocation.

The complete DINOv2-small run at depth two passed numerical correctness and
reported:

```text
latency_ms=7531.10
access_enqueued=163296
scout_completed=163296
consumer_ready_hits=123833
consumer_ready_misses=39463
ready_before_consume=17239
ready_before_consume_bytes=35305472
execute_wait_cycles=3051807464
sync_fallbacks=0
```

Thus 75.83% of Access requests reused a previously formed representation,
10.56% completed before Execute demand, and Execute wait cycles fell by 60.74%
relative to the depth-two uncached path.  End-to-end latency improved by
1.43x over that matched V2 run and by 1.48x over V1.  Average wait per request
fell from about 47.6 K to 18.7 K cycles.  This is the first complete-model
result in which independently progressing Access, consumer-ready formation,
bounded VTCM buffering, and exact readiness all make measurable contributions.

The result also sets the boundary for the next work.  The cache is not itself
the final ALPS endpoint: the first occurrence still pays formation, and most
descriptors are not yet ready before demand.  The next experiment must compose
this residual V-DAE supply with the already frozen producer-direct/full
consumer-formation stack.  Runtime admission is then allowed to regulate only
the remaining asynchronous stream.  We will not search additional queue depths
until that matched composition is available.

## 9. Composition gate: formation before residual DAE (2026-09-04)

The matched composition experiment compared the frozen full consumer-formation
endpoint with the same endpoint plus depth-two V-DAE, then with recoverable
same-invocation traffic admission:

| DINOv2-small configuration | Latency (ms) | Relative to formation |
|---|---:|---:|
| Full consumer-ready formation | 3,343.30 | 1.00x |
| Formation + residual V-DAE | 4,842.67 | 0.69x |
| Formation + residual V-DAE + admission | 4,347.16 | 0.77x |

All three outputs passed the same numerical check.  Without traffic control,
V-DAE issued and completed 163,296 scout-owned requests; 21,525 (13.18%) were
ready before demand.  Same-invocation admission suppressed 112,141 requests
and reduced latency by 11.40% relative to unregulated V-DAE, but it did not
recover the formation-only endpoint.

This is not a contradiction in the two-layer design.  It is the admission
decision the design requires.  Full formation has already discharged the
expensive attention and patch-token materializations, leaving a fine-grained
weight stream whose descriptor, queue, DMA, and token costs exceed the latency
it hides.  Such a stream must remain on the native formation path.  V-DAE's
1.43x improvement in Section 8 applies when repeated residual formation is
actually present; it must not be generalized to a stream after that obligation
has already been discharged.

The experiment also exposed an observability bug: synchronous traffic-control
fallbacks entered `consume` in READY state and were incorrectly counted as
`ready_before_consume`.  The runtime now counts that metric only when the exact
descriptor is scout-owned.  Synchronous fallback remains separately visible.

The next admission step is therefore invocation-level, not another tile-depth
search.  Slow sysMon/profile feedback must be able to prevent emission of the
V-DAE residual path for a previously rejected contract, so fallback uses the
original formation implementation rather than paying exact-descriptor overhead
for synchronous work.  Fast runtime feedback remains responsible for temporary
pressure changes within an admitted invocation.  This preserves the intended
division: compiler contracts choose *what can be eliminated or decoupled*;
runtime feedback chooses *whether an eligible decoupled stream should run*.

### R2 native-path replay

The slow-loop policy was then replayed with this measured contract marked as
rejected.  Before lowering, the launcher disabled the residual exact/V-DAE
passes but retained the complete consumer-formation configuration.  The run
issued zero V-DAE requests, passed correctness, and measured 3,360.67 ms--only
0.52% above the independently compiled 3,343.30 ms formation control.  This
closes the native-fallback requirement that same-invocation throttling could
not satisfy.

The replay is a controlled mechanism test, not a claim that sysMon makes a
decision during one invocation.  In deployment the versioned policy is
generated from the preceding matched latency/sysMon summaries; its
`residual_vdae_admitted` field determines the next compilation/invocation.
When admitted, the fast completion controller may still throttle and probe
within that invocation.  When rejected, no residual descriptor machinery is
emitted.  The policy is contract-driven and contains no model-name condition.
