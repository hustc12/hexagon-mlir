# OmniFetch Experimental Data Consolidation

Last consolidated: 2026-08-14

This document consolidates measured latency data scattered across the project
Markdown files. It preserves historical results instead of replacing them with
the newest number, because the compiler revision, runner, model topology,
precision, timing boundary, and actual compute mapping changed during the
project.

## 1. Normalization and interpretation rules

- All latency values below are normalized to **milliseconds** unless a table
  explicitly says otherwise. `NA` means that the corresponding configuration
  has no valid measured latency in the source documents.
- Debug/reduced models and full models are never treated as interchangeable.
  Monolithic full-graph latency and host-staged full-model sums are also kept
  separate.
- `Scalar`, `HVX`, and `HMX` describe the **actual compute mapping**, not merely
  a CLI flag. A HexKL-enabled row with zero successful HexKL rewrites remains
  an HVX row. HMX is recorded only when the source has evidence of successful
  HexKL/HMX lowering.
- Several early rows were originally called `HVX`, but the later audit found
  `enableVectorization=false`. They are preserved with an `original label`
  warning and must not be cited as true-HVX results.
- Prefetch-Kernel-HX and APT-GET-HX are data-movement policies implemented in
  Hexagon-MLIR. They are not frameworks or compute engines. QNN and LiteRT are
  reported separately as vendor/runtime references.
- Approximate values and ranges remain approximate. Failed correctness, exit
  13, compile timeout, or missing `perf.txt` never becomes a latency value.
- The source column names the authoritative document section. Duplicate copies
  in summary/history documents are consolidated into one row.

Primary sources:

- `docs_engineering/plan_todo.md`
- `benchmark_models/PHASE4_STATUS_AND_OMNIFETCH.md`
- `benchmark_models/debug_running/QWEN_HEXKL_EXIT13_DEBUG.md`
- `docs_engineering/omnifetch-prefetch-insitu-innovation.md`
- `docs_engineering/engineering_work.md`
- `docs_engineering/omnifetch_history.md`
- `docs_engineering/hexagon-prefetch-baselines-plan.md`

The generic user guides and `benchmark_models/micro_bench/*` documentation
contain commands, expected performance ranges, or placeholder `Perf: xxxx`
rather than completed measurements. They were inspected but are not turned
into experimental rows. Likewise, projected/target CSV values are excluded;
this file records measured values documented as device results.

## 2. Complete-model parameter and arithmetic-compute census

This table describes the **same 15 full-structure runners** used by the
complete-model corpus below; it is not a count for a debug/reduced graph.
`Runner parameters` is the exact sum of `model.parameters()` in the structure
compiled by the runner (including its deployed prediction head where present).
`Model-card / published parameters` is deliberately retained as a separate
public, usually rounded, reference.  It can differ because a card may count a
pre-training/task head, use a different checkpoint revision, or round the
value.  In particular, the SD card describes the whole diffusion pipeline, so
it is not a valid published parameter count for the CLIP text-encoder component
used here.

`MACs` are the major dense/conv arithmetic operation count measured with
PyTorch `FlopCounterMode` on meta tensors; `FLOPs = 2 x MACs` (one multiply plus
one add).  These counts include `linear`/matrix multiply, batched matrix
multiply and convolution, but exclude elementwise activation, normalization,
softmax, indexing and data movement.  They therefore provide a reproducible
arithmetic-workload indicator rather than a claim about every DSP instruction.
They are independent of FP16/FP32 storage precision.

| Domain | Complete model | Runner input / decoding scope | Runner parameters (exact; M) | Model-card / published parameters | MACs (G) | FLOPs (G) |
|---|---|---|---:|---:|---:|---:|
| Language/text | GPT-2 | B=1, 32 tokens, full vocabulary logits | 124,439,808; 124.44 | [124M](https://huggingface.co/openai-community/gpt2) | 3.97 | 7.94 |
| Language/text | SD/CLIP text encoder | B=1, 77 tokens | 123,060,480; 123.06 | N/A (SD card is whole pipeline) | 6.65 | 13.30 |
| Language/text | Qwen2.5-0.5B | B=1, 32 tokens, full vocabulary logits | 494,032,768; 494.03 | [0.49B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 15.85 | 31.70 |
| Language/text | TinyLlama-1.1B | B=1, 32 tokens, full vocabulary logits | 1,100,048,384; 1,100.05 | [1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 33.19 | 66.39 |
| Language/text | SmolLM2-1.7B | B=1, 32 tokens, full vocabulary logits | 1,711,376,384; 1,711.38 | [1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 54.86 | 109.72 |
| Vision | Swin Transformer (tiny) | B=1, 3x224x224 | 28,288,354; 28.29 | [28.3M](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224) | 4.49 | 8.98 |
| Vision | SegFormer MiT-B0 | B=1, 3x224x224 | 3,576,392; 3.58 | [3.7M](https://arxiv.org/abs/2105.15203) | 0.45 | 0.90 |
| Vision | DeiT-small | B=1, 3x224x224 | 22,051,432; 22.05 | [22M](https://huggingface.co/facebook/deit-small-patch16-224) | 4.62 | 9.25 |
| Vision | BEiT-base | B=1, 3x224x224 | 86,530,984; 86.53 | [87M](https://huggingface.co/microsoft/beit-base-patch16-224) | 17.56 | 35.13 |
| Vision | DINOv2-small | B=1, 3x224x224 | 22,825,192; 22.83 | [22.1M](https://huggingface.co/facebook/dinov2-small) | 6.12 | 12.25 |
| Speech | Whisper-tiny | B=1, 80x3000 mel; 32 decoder tokens | 37,760,640; 37.76 | [39M](https://huggingface.co/openai/whisper-tiny) | 21.29 | 42.58 |
| Speech | HuBERT-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [94.7M](https://www.isca-archive.org/interspeech_2023/zaiem23b_interspeech.pdf) | 9.00 | 17.99 |
| Speech | Wav2Vec2-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [95M](https://arxiv.org/abs/2006.11477) | 9.00 | 17.99 |
| Speech | UniSpeech-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [94.68M](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/UniSpeech_SAT.pdf) | 9.00 | 17.99 |
| Speech | UniSpeech-SAT-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [94.68M](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/UniSpeech_SAT.pdf) | 9.00 | 17.99 |

The five language-model counts include the full-token logits projection; they
are thus forward-pass counts, not one-token autoregressive decode counts.  The
audio runners intentionally benchmark the representation encoder and omit an
ASR/pre-training head, matching the current Hexagon-MLIR benchmark topology.

## 3. Required cross-product matrix for the two prefetch baselines

The following table expands the current **15 complete-model corpus plus the
ViT-Base external-baseline vehicle** across the requested policy-by-engine
dimensions. `HMLIR` means native/no-prefetch Hexagon-MLIR and `OF` means
OmniFetch. All values are device latency in ms. As of 2026-08-16, PK/APT and
item7-only have been measured on all 15 complete models in the primary corpus;
every still-untested processor/policy combination is explicitly `NA`.

| Date / scope | Model | PK Scalar | PK HVX | PK HMX | APT Scalar | APT HVX | APT HMX | HMLIR Scalar | HMLIR HVX | HMLIR HMX | OF Scalar | OF HVX | OF HMX | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-13 strict rerun | DINOv2-small | NA | 10,596.72 (1.78x) | NA | NA | 10,591.11 (1.78x) | NA | 1,281,514.12‡ | 30,031.10 HMLIR HVX (HexKL Off) (5.04x); 10,035.87 HMLIR HVX (HexKL On; 0 HMX rewrites) (1.69x) | NA | 300,484.16‡ | **5,953.59 item7-only (1.00x)** | NA | Ratios are row latency / item7 latency. Items 1–6 are explicitly disabled; all current rows pass correctness. |
| 2026-08-13 strict rerun | ViT-Base | NA | 21,489.53 (1.52x) | NA | NA | 20,915.64 (1.48x) | NA | NA | 129,979.04 HMLIR HVX (HexKL Off) (9.21x); 20,088.50 HMLIR HVX (HexKL On; 0 HMX rewrites) (1.42x) | NA | NA | **14,115.92 item7-only (1.00x)** | NA | Ratios are row latency / item7 latency. Items 1–6 are explicitly disabled; all current rows pass correctness. |
| 2026-08-08+ / 2026-08-15 strict rerun | DeiT-small | NA | 9,203.05 (1.74x) | NA | NA | 8,850.23 (1.67x) | NA | 913,419.276†‡ | 23,188.05 HMLIR HVX (HexKL Off) (4.38x); 8,495.38 HMLIR HVX (HexKL On; 0 HMX rewrites) (1.61x) | NA | 266,291.591†‡ | **5,290.50 item7-only (1.00x)** | NA | Ratios are row latency / item7 latency. Current V73 HVX rows used CDSP 1,478.40 MHz and MEMNOC 2,000 MHz, ran strictly serially, and all pass correctness. PK admitted 334 sites; APT consumed that allowlist. Item7 propagated/tiled 12 K/V pairs but emitted zero runtime L2-scheduler hints; old scalar rows remain invalid NaN history (†‡). |
| 2026-08-08+ / 2026-08-15 strict rerun | SegFormer MiT-B0 | NA | 9,413.14 (1.04x) | NA | NA | 9,393.78 (1.04x) | NA | NA | 27,189.70 HMLIR HVX (HexKL Off) (3.01x); 9,417.02 HMLIR HVX (HexKL On; 8 rewrites) (1.04x) | NA | NA | **9,025.85 item7-only (1.00x)** | NA | Current five HVX rows are strictly serial and pass correctness. PK admitted 171 sites; APT consumed that allowlist. Item7 identified two K/V pairs but emitted zero runtime L2-scheduler hints. Current strict rows supersede the older mixed-generation 27,488.063/9,349.189/19,785.205‡ values. |
| 2026-08-08+ / 2026-08-15 strict rerun | Swin Transformer | NA | 77,737.95 (1.52x) | NA | NA | 73,318.92 (1.43x) | NA | NA | 121,829.86 HMLIR HVX (HexKL Off) (2.38x); 73,457.69 HMLIR HVX (HexKL On; 4 rewrites) (1.43x) | NA | NA | **51,221.83 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and each passed its CPU correctness gate. PK admitted 310 sites; APT consumed that allowlist. Item7 identified 12 K/V pairs but emitted zero runtime L2-scheduler hints. Current strict rows supersede the older mixed-generation values. |
| 2026-08-08+ / 2026-08-15 strict rerun | BEiT-base | NA | 15,804.82 (1.08x) | NA | NA | 15,837.16 (1.08x) | NA | NA | 132,908.25 HMLIR HVX (HexKL Off) (9.07x); 14,737.02 HMLIR HVX (HexKL On; 0 HMX rewrites) (1.01x) | NA | NA | **14,656.99 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and pass correctness. PK admitted 382 sites; APT consumed that allowlist. Item7 identified 12 K/V pairs but emitted zero runtime L2-scheduler hints; its gain over HexKL On is only 1.01x. |
| 2026-08-08+ / 2026-08-15 strict rerun | Whisper-tiny | NA | 115,763.93 (1.66x) | NA | NA | 115,908.13 (1.66x) | NA | NA | 161,702.15 HMLIR HVX (HexKL Off) (2.32x); 115,663.98 HMLIR HVX (HexKL On; 32 rewrites) (1.66x) | NA | NA | **69,743.50 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and pass correctness. PK admitted 283 sites; APT consumed that allowlist. Item7 identified eight K/V pairs but emitted zero runtime L2-scheduler hints. Current strict rows supersede the older mixed-generation rows, including the invalid 1,093,076.414‡ result. |
| 2026-08-08+ / 2026-08-15 strict rerun | HuBERT-base | NA | 174,798.20 (1.01x) | NA | NA | 176,416.16 (1.02x) | NA | NA | 216,965.03 HMLIR HVX (HexKL Off) (1.25x); 176,744.13 HMLIR HVX (HexKL On; 74 rewrites) (1.02x) | NA | NA | **173,783.08 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and pass correctness. PK admitted 387 sites; APT consumed that allowlist. Item7 identified 12 K/V pairs but emitted zero runtime L2-scheduler hints; its gain over HexKL On is only 1.02x. |
| 2026-08-08+ / 2026-08-15 strict rerun | Wav2Vec2-base | NA | 191,710.61 (1.03x) | NA | NA | 174,880.18 (0.94x) | NA | NA | 216,862.85 HMLIR HVX (HexKL Off) (1.17x); 177,055.12 HMLIR HVX (HexKL On; 74 rewrites) (0.96x) | NA | NA | **185,248.95 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and pass correctness. PK admitted 387 sites; APT consumed that allowlist. Item7 identified 12 K/V pairs but emitted zero runtime L2-scheduler hints. Ratios below 1.00x mean item7 is slower than that comparison; it regresses versus HexKL On here. |
| 2026-08-08+ / 2026-08-15 strict rerun | UniSpeech-base | NA | 174,208.93 (0.93x) | NA | NA | 174,355.22 (0.93x) | NA | NA | 217,191.90 HMLIR HVX (HexKL Off) (1.16x); 178,052.40 HMLIR HVX (HexKL On; 74 rewrites) (0.95x) | NA | NA | **186,487.32 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and pass correctness. PK admitted 387 sites; APT consumed that allowlist. Item7 identified 12 K/V pairs but emitted zero runtime L2-scheduler hints. Ratios below 1.00x expose an item7 regression. |
| 2026-08-08+ / 2026-08-15 strict rerun | UniSpeech-SAT-base | NA | 175,383.30 (0.93x) | NA | NA | 173,985.61 (0.92x) | NA | NA | 216,746.78 HMLIR HVX (HexKL Off) (1.15x); 174,227.26 HMLIR HVX (HexKL On; 74 rewrites) (0.93x) | NA | NA | **188,149.68 item7-only (1.00x)** | NA | Current five HVX rows ran strictly serially and pass correctness. PK admitted 387 sites; APT consumed that allowlist. Item7 identified 12 K/V pairs but emitted zero runtime L2-scheduler hints. Ratios below 1.00x expose an item7 regression. |
| 2026-08-15--16 strict staged FP16 rerun | GPT-2, full 12L | NA | 3,714.22 (0.97x) | NA | NA | 3,808.27 (0.99x) | NA | 24,027†‡ | 20,771.46 HMLIR HVX (HexKL Off) (5.43x); 3,705.82 HMLIR HVX (HexKL On) (0.97x) | 3,359.199*‡ | NA | **3,827.61 item7-only (1.00x)** | 11,673†‡ | Uniform FP16; sum of embedding + 12 blocks + full-vocabulary head device Perf. All five current rows are finite and preserve last-token Top-1. Ratios below 1.00x show that item7 regresses slightly versus PK/APT/HexKL On. The earlier monolithic OOM and invalid scalar rows remain history only. |
| 2026-08-15--16 strict staged FP16 rerun | SD/CLIP text encoder | NA | 4,118.21 (1.06x) | NA | NA | 4,106.20 (1.06x) | NA | NA | 243,370.18 HMLIR HVX (HexKL Off) (62.57x); 3,963.37 HMLIR HVX (HexKL On) (1.02x) | 5,440.120*‡ | NA | **3,889.70 item7-only (1.00x)** | NA | Uniform FP16; sum of embedding + 12 encoder layers + final norm device Perf. All rows are finite; item7 max abs is 0.015625. Item7 improves only 1.02x over the matched HexKL-On control. |
| 2026-08-15--16 strict staged FP16 rerun | Qwen2.5-0.5B | NA | 10,989.09 (1.88x) | NA | NA | 10,994.89 (1.88x) | NA | NA | 414,934.20 HMLIR HVX (HexKL Off) (70.81x); 10,929.40 HMLIR HVX (HexKL On) (1.87x) | 13,537.873*‡ | NA | **5,859.60 item7-only (1.00x)** | NA | Uniform FP16; sum of embedding + 24 complete blocks + full-vocabulary head device Perf. All rows are finite and preserve last-token Top-5. This is a clean 1.87x item7 gain over matched HexKL On. |
| 2026-08-15--16 strict staged FP16 rerun | TinyLlama-1.1B | NA | 27,611.75 (1.62x) | NA | NA | 27,406.97 (1.61x) | NA | NA | 976,720.43 HMLIR HVX (HexKL Off) (57.32x); 27,228.38 HMLIR HVX (HexKL On) (1.60x) | 32,908.259*‡ | NA | **17,038.99 item7-only (1.00x)** | NA | Uniform FP16; sum of embedding + 22 complete blocks + full-vocabulary head device Perf. All rows are finite and preserve last-token Top-5; item7 is 1.60x faster than matched HexKL On. |
| 2026-08-15--16 strict staged FP16 rerun | SmolLM2-1.7B | NA | 39,463.94 (1.40x) | NA | NA | 39,396.33 (1.39x) | NA | NA | 1,559,447.10 HMLIR HVX (HexKL Off) (55.18x); 39,348.33 HMLIR HVX (HexKL On) (1.39x) | 42,315.577*‡ | NA | **28,262.23 item7-only (1.00x)** | NA | Uniform FP16; sum of embedding + 24 complete blocks + full-vocabulary head device Perf. All rows are finite and preserve last-token Top-5; item7 is 1.39x faster than matched HexKL On. |

The five 2026-08-15--16 language/text rows use matched host-staged execution
because monolithic MLIR-to-LLVM code generation exceeds the 15-GiB host memory
limit. Every policy compiles and runs the identical uniform-FP16 embedding,
complete transformer layers, and final norm/head boundaries. The reported
latency is the sum of every stage's device `Perf`; compilation, ADB transfer,
and host round trips are excluded symmetrically. These are complete-model
device-kernel sums, not single-executable end-to-end latency.

| Model | PK runtime issued / bytes | APT runtime issued / bytes | Item7 compiler K/V pairs / runtime sites | Item7 vs HexKL On |
|---|---:|---:|---:|---:|
| GPT-2 | 27,200 / 6,494,272 | 27,200 / 6,494,272 | 1 / 0 | 0.97x (regression) |
| SD/CLIP | 3,388 / 433,664 | 3,388 / 433,664 | 1 / 0 | 1.02x |
| Qwen2.5-0.5B | 77,632 / 19,398,912 | 77,632 / 19,398,912 | 1 / 0 | **1.87x** |
| TinyLlama-1.1B | 19,968 / 4,652,544 | 19,968 / 4,652,544 | 1 / 0 | **1.60x** |
| SmolLM2-1.7B | 28,544 / 6,805,504 | 28,544 / 6,805,504 | 1 / 0 | **1.39x** |

Item7 emitted no runtime L2-scheduler prefetch sites in these staged runs.
Consequently, the observed Qwen/TinyLlama/SmolLM2 latency changes are real
matched measurements but must not yet be attributed to executed hardware
prefetch commands; an IR/object-level differential audit is required to locate
the responsible code-generation change. PK and APT again issue identical
traffic because APT consumes the exact PK-admitted per-stage candidate union.
The complete generated corpus and logs are stored on `nano` under
`/home/huzq85/2-working/working_set/full_hvx_five_way_20260815_layered_fp16`;
the local `/tmp` tree retains only logs, status, CSV, and Markdown summaries.

`*` denotes a measured HexKL-enabled full-model latency placed in the HMX-route
column because that is the repository's available HMX lowering route. The
consolidated source table does not record a final-object rewrite audit for
every model; these are therefore **HexKL/HMX-route measurements, not proof
that every MatMul ran on HMX**. DINOv2 and ViT are stricter counterexamples:
their current logs explicitly report zero rewrites, so their HMX cells remain
`NA`.

`†` marks a correctness-invalid historical latency. `‡` marks a historical
full-model result from a different compiler/timing generation. Such values are
retained because the user requested all available full-model data, but they
must not be used as a matched speedup against the 2026-08-08+ native corpus.

Important comparison rule: the 2026-08-13 prefetch-baseline builds enabled
HexKL but produced zero rewrites. Their closest no-prefetch control is therefore
the `HMLIR HVX (HexKL On; 0 HMX rewrites)` row, not the much slower `HMLIR HVX (HexKL Off)`
(HexKL-off) compilation.
Both are retained because the large difference proves that the flag changed
other lowering/code-generation behavior even without HMX coverage.

With the strict item-7-only policy, DINOv2-small is 1.78x faster than
Prefetch-Kernel-HX, 1.78x faster than APT-GET-HX, and 1.69x faster than the
matched HexKL-on/zero-rewrite control. ViT-Base is respectively 1.52x,
1.48x, and 1.42x faster. The much larger 5.04x and 9.21x ratios against
HMLIR HVX (HexKL Off) are real timing differences but are not clean OmniFetch attribution,
because merely enabling HexKL changes lowering even when it reports zero HMX
rewrites.

For the matched uniform-FP16 staged language rows, item7 is 1.87x faster than
HexKL On on Qwen2.5-0.5B, 1.60x on TinyLlama-1.1B, and 1.39x on
SmolLM2-1.7B. It is only 1.02x faster on SD/CLIP and 0.97x on GPT-2. The much
larger 5.43--70.81x ratios against HexKL Off primarily expose HexKL/HMX-path
acceleration and must not be claimed as OmniFetch speedup.

## 4. Chronological experimental record

### 2026-07-23 — first Phase-1/2 and model three-way measurements

#### Model/debug measurements

| Model / topology | Original HVX label | HexKL | HexKL + OmniFetch | Correctness / qualification |
|---|---:|---:|---:|---|
| GPT-2 Debug, 2L, seq 32, fair | 13,332 | 11,110 | 11,111 | Pass; OF approximately equal to HexKL. Early `HVX` label predates the true-vector audit. |
| Qwen2.5 Debug, 2L, seq 32, first fair run | 1,919 | 155.8 | 153.0 | Pass; top-1 on HexKL/OF. |
| Qwen2.5 Debug, 2L, seq 32, fresh three-way | 1,951.7 | 156.2 | 150.6 | Pass; OF 1.037x over HexKL. |
| Falcon Debug, 2L, seq 32 | 2,751 | 113.5 | 109.6 | Pass; OF 1.036x over HexKL. |
| TinyLlama Debug, 2L, seq 32 | 1,910 | 195.3 | 194.6 | Pass; approximately equal. |
| Mamba Debug, 1L, seq 32 | 1,135 | 130.6 | 127.9 | Pass; OF about 1.02x. |
| ViT Debug, 2L, patch 32 | 1,220 | 1,222 | 1,222 | Pass; zero relevant sites. |
| Swin Debug, `[1,1,1,1]`, width 48 | 67,066 | 67,400–67,700 | 67,400–67,700 | Pass; approximately equal. |
| GraphSAGE/BERT Debug, 2L | 304 | 124 | 123 | Pass; OF approximately equal. |
| SD/CLIP text encoder Debug | ~1.47 | ~1.47 | ~1.47 | Pass; all approximately equal. |
| Real-ESRGAN reduced 8x8 | 8,000 | 7,894 | 7,920 | Device pass; loose numerical comparison. |
| SD-VAE Debug `[32,64]` | 8,984 | 488 | 495 | Device pass; loose numerical comparison. |
| SD-UNet Debug, no cross-attention | 106 | 106 | 105 | Device pass; loose numerical comparison. |
| GPT-2 full 12L, seq 32 | 24,027 | 12,049 | 11,673 | Device ran, but all paths produced NaN logits; performance-only historical evidence. |

The Phase-4 audit later established that these early `HVX` configurations often
had vectorization disabled. They must not be mixed with the true-HVX data from
2026-07-30 onward.

#### Kernel/mechanism measurements

| Vehicle | Configuration | Latency | Result |
|---|---|---:|---|
| Attention Q/K `1x8x128x64` | HexKL | 41.45–41.70 | Pass; no MicroHMX conversion. |
| Attention Q/K | HexKL + VTCM | 332–364 | Pass; roughly 8x slower. |
| Attention Q/K | HexKL + OF, layout off | ~41.6 | Pass; no meaningful insert. |
| GEMM `256^3` FP16 | HexKL | 14.152 | Pass. |
| GEMM `256^3` FP16 | HexKL + OF, layout off | 14.286 | Pass; two L2 hints. |
| GEMM `256^3` FP16, post-reboot | HexKL | 14.092 | Pass. |
| GEMM `256^3` FP16, post-reboot | HexKL + OF, layout off | 14.358 | Pass. |
| GEMM `256^3` FP16 | HexKL + OF, layout on | 14.033 | Pass; one HMXWeight fusion. |
| GEMM `256^3` FP16 | HexKL + OF, bad gather | ~92–105 | Incorrect/NaN; invalid. |
| GEMM `256^3` FP16 | HexKL + OF, idle synchronous fill | 15.731 | Pass; slower. |
| GEMM `256^3` FP16 | HexKL + OF, layout on + async L2 | **13.745** | Pass; about 3% over ~14.2-ms HexKL. |

Source: `docs_engineering/plan_todo.md`, results log; duplicated summaries in
`benchmark_models/PHASE4_STATUS_AND_OMNIFETCH.md` and the Qwen exit-13 notes.

### 2026-07-24 — resumed GEMM and GPT-2 variants

| Vehicle | Configuration | Latency | Qualification |
|---|---|---:|---|
| GPT-2 full 12L | HexKL + last-token LM head | 5,413 | Not comparable to the earlier full-sequence 12,049-ms HexKL row because the output boundary changed. |
| GEMM `64x128x256` | HexKL | 4.292 | Pass. |
| GEMM `64x128x256` | OF layout off | 4.191 | Pass. |
| GEMM `64x128x256` | OF layout on + activation fusion + L2 async | 4.283 | Pass. |
| GEMM `64x128x256` | OF dma2d + WH-on-signal | 4.701 | Pass. |
| GEMM `64x128x256` | OF DMA-to-VTCM stage | 9.420 | Pass; about 2x slower than DDR staging. |

Source: `docs_engineering/plan_todo.md`.

### 2026-07-25 — GPT-2 Debug snapshot

These values are preserved exactly as documented; they are unusually large
for a 2-layer Debug model and should not replace the matched 2026-07-23 rows.

| Configuration | Seq 32 | Seq 128 |
|---|---:|---:|
| HexKL | 307,500 | 1,335,900 |
| HexKL + OF | 345,700 | 1,337,200 |
| HexKL + prepack | 334,500 | 1,278,200 |
| OF + dual thread | 316,400 | NA |
| OF + inter-layer | 322,600 | NA |
| Attention-HMX | 314,200 | NA |

Source: `benchmark_models/PHASE4_STATUS_AND_OMNIFETCH.md`, section 7.7.

### 2026-07-26 — Falcon Debug cost-model screen

| Model | HexKL | Layout-aware OF | Layout + WH reuse + persistent VTCM | Correctness |
|---|---:|---:|---:|---|
| Falcon Debug, 2L, hidden 64, vocab 4096, seq 128 | 1,693.275 | 1,689.758 | 1,701.922 | All top-1 match; max abs 0.0239. |

Source: `docs_engineering/omnifetch-prefetch-insitu-innovation.md`, “First
model result”.

The same date's compiler/runtime validation also recorded:

| Vehicle | Configuration | Latency | Qualification |
|---|---|---:|---|
| GEMM `64x128x256` | HexKL | 3.044 | Reference quoted by the short-loop analysis. |
| GEMM `64x128x256` | Layout-aware OF, short-loop gate | 3.057 | Correct; about 0.4% slower. |
| GEMM `64x256x512` | HexKL | 8.169 | Correct. |
| GEMM `64x256x512` | Plain OF | 8.227 | Correct. |
| GEMM `64x256x512` | Async layout-aware OF | 8.442 | Correct; about 3.3% slower than HexKL. |
| GPT-2 full, seq 128 | original `HVX` row | 127,401.804 | Finite/top-1 qualified; matched HexKL host OOM, so no speedup. |

Source: `docs_engineering/omnifetch-design-audit-and-roadmap.md` and the M1
section of `docs_engineering/omnifetch-prefetch-insitu-innovation.md`.

### 2026-07-28 — `omnifetch-2x-improvement` Debug matrices

#### Initial cross-model cumulative integration

| Debug model | HVX | HexKL | HexKL + items 1–7 | HexKL / combo | Qualification |
|---|---:|---:|---:|---:|---|
| Falcon, seq 128 | 11,238.419 | 1,619.671 | 590.880 | 2.741x | Correct. |
| SD text encoder | 1.450 | 1.478 | 0.293 | 5.044x | Correct; sub-ms result. |
| SD-VAE decoder | 8,615.044 | 502.577 | 385.453 | 1.304x | Timing only; max difference 1.2686. |
| Swin Debug | 67,575.498 | 66,860.712 | 20,642.078 | 3.239x | Correct. |
| TinyLlama Debug, seq 128 | 8,879.842 | 2,533.438 | 1,299.005 | 1.950x | Correct. |
| ViT Debug | 1,201.766 | 1,204.605 | 718.449 | 1.677x | Correct. |
| Qwen Debug, clean cumulative-only run | NA | NA | 600.726 | NA | Correct; no valid matched HexKL row in this run. |

#### Previously blocked models, clean seq-32 run

| Debug model | HVX | HexKL | Items 1–7 | HexKL / combo | Qualification |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B, 2L | 1,897.422 | 153.888 | 79.475 | 1.936x | Correct. |
| GraphSAGE/BERT | 296.833 | 122.142 | 81.037 | 1.507x | Correct. |
| Mamba, 1L | 1,165.356 | 127.751 | 110.213 | 1.159x | Correct. |
| SD-UNet, no cross-attention | 105.085 | 105.505 | 106.154 | 0.994x | Numerical result not paper-qualified. |
| Real-ESRGAN reduced | 71.064 | 71.041 | 70.614 | 1.006x | Combination exceeded configured tolerance. |
| GPT-2, 2L, full LM head | 22,496.585 | 109,539.127 | 31,937.477 | 3.430x | Correct, but combo remains slower than HVX. |

#### New candidate screens

| Candidate | HVX | HexKL | Items 1–7 | HexKL / combo | Outcome |
|---|---:|---:|---:|---:|---|
| SmolLM2-135M proxy | 3,145.272 | 204.110 | 120.371 | 1.6957x | Correct. |
| SwinV2-Tiny proxy | NA | NA | NA | NA | Device exit 13. |
| AST AudioSet proxy | 235.001 | 236.875 | NA | NA | Combo compile timeout. |
| Qwen2.5-Coder proxy | 8,486.568 | 424.451 | NA | NA | Combo compile timeout. |
| SegFormer MiT-B0 proxy | 324.340 | 275.134 | 112.937 | 2.4362x | Correct. |
| Whisper-tiny proxy | 331.746 | 340.427 | 97.619 | 3.4873x | Correct. |
| OPT-125M proxy | 1,680.180 | 164.211 | 106.451 | 1.5426x | Correct. |
| DeiT-Small proxy | 1,472.615 | 1,487.593 | 656.140 | nominal 2.2672x | Invalid: all paths NaN/top-1 mismatch. |
| Wav2Vec2-base proxy | 489.284 | 488.338 | NA | NA | Combo compile timeout. |
| BEiT proxy | 688.260 | 694.660 | 318.362 | nominal 2.1820x | Invalid: all outputs NaN. |
| HuBERT proxy | 503.813 | 491.712 | NA | NA | Combo compile timeout. |
| HuBERT 1L reduced | 294.892 | 293.522 | NA | NA | Combo compile timeout. |
| Wav2Vec2 1L reduced | 303.805 | 295.438 | NA | NA | Combo compile timeout. |
| DETR | NA | NA | NA | NA | Parser/device failure; no latency. |
| Speech2Text | NA | NA | NA | NA | Parser/device failure; no latency. |
| WavLM | NA | NA | NA | NA | Import failure. |
| Data2Vec-Audio | NA | NA | NA | NA | Device exit 13 / combo timeout. |

#### Item-7-off causal ablation and restored attention K/V policy

| Model | HVX | HexKL | Item-7-off combo | Restored combo | HexKL / restored | Qualification |
|---|---:|---:|---:|---:|---:|---|
| Whisper-tiny | 331.281 | 325.680 | 294.560 from adjacent run | 109.109 | 2.9849x | Correct. |
| SegFormer MiT-B0 | 323.039 | 275.901 | 188.807 from adjacent run | 119.245 | 2.3137x | Correct. |
| BEiT proxy | 683.859 | 690.560 | 652.237 from adjacent run | 318.992 | 2.1648x | Invalid NaN; relaxed screen only. |
| HuBERT 1L | 299.490 | 294.864 | 271.113 from adjacent run | 150.069 | 1.9649x | Correct. |
| Wav2Vec2 1L | 304.530 | 294.956 | 278.363 from adjacent run | 150.063 | 1.9655x | Correct. |

The item-7-off baselines were separate serial invocations with slightly
different HVX/HexKL values: Whisper 330.888/327.800, SegFormer
326.840/275.611, BEiT 688.364/685.684, HuBERT 294.892/293.522, and Wav2Vec2
303.805/295.438 ms. They are not silently averaged with the restored run.

#### Balanced-set additions

| Candidate | HVX | HexKL | Items 1–7 | HexKL / combo | Result |
|---|---:|---:|---:|---:|---|
| UniSpeech-base 1L | 293.102 | 293.799 | 150.528 | 1.9518x | Correct. |
| UniSpeech-SAT-base 1L | 296.251 | 306.903 | 150.127 | 2.0443x | Correct. |
| DINOv2-small proxy | 173.867 | 174.867 | 60.337 | 2.8982x | Correct. |
| SEW-base 1L | 222.662 | 212.712 | NA | NA | Combo codegen timeout. |
| SD-UNet CrossAttn proxy | NA | NA | NA | NA | HVX codegen timeout. |

#### Direct QNN and LiteRT references

| Model/scope | Path | Latency | Notes |
|---|---|---:|---|
| DINOv2 Debug, controlled 20-run | Hexagon-MLIR original `HVX` path | 148.022 | Later audit found this path was not true-vector HVX. |
| DINOv2 Debug, controlled 20-run | Hexagon-MLIR HexKL | 148.568 | No useful HMX coverage. |
| DINOv2 Debug, controlled 20-run | HexKL + items 1–7 warm | 51.712 | One DAE scout. |
| DINOv2 Debug, controlled 20-run | QNN NetRun execute | **0.867** | One HVX thread, default vote; comparable broad QNN scope. |
| DINOv2 Debug, controlled 20-run | QNN execute | 0.854 | Nested scope. |
| DINOv2 Debug, controlled 20-run | QNN accelerator execute | 0.341 | Nested scope. |
| DINOv2 Debug, controlled 20-run | accelerator execute / excluding wait | 0.306 / 0.250 | Not directly comparable to model-body timing. |
| DINOv2 Debug, LiteRT + QNN HTP | 0.979 steady mean | Three serial trials; one HVX thread, default vote. |
| DINOv2 Debug, LiteRT XNNPACK CPU | 0.350 steady mean | Mean of trial means 0.355/0.366/0.330 ms. |

The earlier exploratory QNN run used the burst profile and four selected HVX
threads. Its nested 20-run averages were: NetRun 1.017, QNN execute 1.002, RPC
execute 0.955, QNN accelerator execute 0.472, accelerator execute 0.418, and
accelerator execute excluding wait 0.304 ms. It is retained as bring-up data,
not as the controlled cross-framework comparison.

The three LiteRT trials were:

| LiteRT path | Trial | First | Steady mean | Median | P90 |
|---|---:|---:|---:|---:|---:|
| Qualcomm HTP, one HVX, default | 1 | 4.965 | 1.004 | 0.907 | 1.222 |
| Qualcomm HTP, one HVX, default | 2 | 4.687 | 0.969 | 0.901 | 1.250 |
| Qualcomm HTP, one HVX, default | 3 | 6.363 | 0.964 | 0.893 | 1.223 |
| XNNPACK CPU | 1 | 1.171 | 0.355 | 0.304 | 0.416 |
| XNNPACK CPU | 2 | 1.047 | 0.366 | 0.316 | 0.438 |
| XNNPACK CPU | 3 | 0.845 | 0.330 | 0.312 | 0.429 |

Source: `docs_engineering/omnifetch-prefetch-insitu-innovation.md`, external
baseline and LiteRT sections.

### 2026-07-29 section — first monolithic full-structure screens

The source groups these rows under the full-model plan dated 2026-07-29; the
individual subheadings do not provide separate run dates.

| Full model | HVX | HexKL | Items 1–7 warm | Qualification |
|---|---:|---:|---:|---|
| Swin-Tiny | NA | NA | 206,326.114 | OF passed; both baselines exited 13. |
| SegFormer MiT-B0 | NA | NA | 19,785.205 | OF passed; both baselines exited 13. |
| DeiT-Small | 913,419.276 | 912,734.553 | 266,291.591 | All outputs NaN; invalid performance claim. |
| BEiT-base | NA | NA | NA | Baselines exited 13; combo host exit 137. |
| DINOv2-small | 1,281,514.120 | 1,284,233.665 | 300,484.161 | All correct; HexKL had zero rewrites, combo 4.2739x over it. |
| Whisper-tiny | NA | NA | 1,093,076.414 | OF correct; HVX eventually exited 13 and HexKL immediately exited 13. |

#### Full DINOv2 direct-QNN checkpoint

| Path | Latency | Qualification |
|---|---:|---|
| Hexagon-MLIR HVX | 1,281,514.120 | Same monolithic graph above. |
| Hexagon-MLIR HexKL | 1,284,233.665 | Zero legal HMX rewrites. |
| HexKL + items 1–7 | 300,484.161 | Correct. |
| QNN CPU | 472.936 | FP32, default scheduler; not single-thread controlled. |
| QNN HTP | **99.520** | FP16, one HVX thread, default profile. |

Source: `docs_engineering/omnifetch-prefetch-insitu-innovation.md`, full-model
screening and Direct-QNN checkpoint.

### 2026-07-30 — true-vector audit and interleaved percentiles

#### DINOv2 Debug, initial interleaved run (p50)

| Actual mapping/policy | P50 | P90 | P99 | Qualification |
|---|---:|---:|---:|---|
| Native scalar | 187.334 | 192.557 | 193.359 | Correct. |
| Native HVX vector | 20.844 | 20.967 | 21.084 | Correct; about 9x over scalar. |
| Native HVX + VTCM | 22.001 | 22.629 | 23.162 | Correct. |
| HexKL flag, zero HMX rewrites | 21.804 | 22.010 | 22.297 | Actual HVX mapping. |
| Items 1–7, zero HMX rewrites | 20.889 | 20.977 | 20.995 | Actual HVX mapping. |

#### DINOv2 Debug after C1+B1 movement fixes (p50)

| Actual mapping/policy | P50 | P90 | P99 |
|---|---:|---:|---:|
| Native HVX | 21.764 | 22.041 | 22.309 |
| Native HVX + VTCM | 20.882 | 21.008 | 21.019 |
| HexKL flag, zero rewrites | 21.709 | 21.895 | 21.988 |
| Items 1–7 on HVX | 20.876 | 20.914 | 20.966 |

#### HMX anchor, aligned non-square GEMM (p50)

| Mapping/policy | P50 | P90 | P99 |
|---|---:|---:|---:|
| Native HVX | 67.974 | 68.001 | 68.110 |
| Native HVX + VTCM | 67.583 | 67.603 | 67.607 |
| HexKL/HMX | **1.218** | 1.261 | 1.280 |
| HexKL/HMX + items 1–7 | 1.320 | 2.365 | 2.382 |

Source: `docs_engineering/plan_todo.md` and
`docs_engineering/omnifetch-design-audit-and-roadmap.md`.

### Date not stated; recorded by the 2026-08-07 history snapshot

| Experiment | Baseline | Optimized | Reported result / caveat |
|---|---:|---:|---|
| Falcon Debug item 4 persistent-WH | HexKL 1,628.410 | 1,612.315 | 1.010x; 99.31% warm hit. |
| Falcon Debug item 5 2-D pipeline | HexKL 1,622.670 | 1,586.580 | 1.023x. |
| Falcon Debug item 6 VTCM coloring | HexKL 1,619.855 | 1,598.756 | 1.013x; static peak 45,056 to 16,384 bytes. |
| Old `omnifetch-2x-improvement` Falcon | original `HVX` 11,742.816; HexKL 1,614.896 | items 1–7 599.969 | Large mixed-policy signal; old `HVX` is scalar-like, so not a true-HVX claim. |
| DINOv2 Debug audit | scalar 174.832 | true HVX 21.258 | 8.224x baseline correction. |
| DINOv2 Debug cumulative item 7 | true HVX 21.258 | scalar items 1–7 60.927 | Different compute mappings; OF is 2.866x slower than true HVX. |
| DINOv2 Debug K/V L2 hint | true HVX 21.258 | HVX + hint 20.663 | 1.029x; four hints. |
| Falcon Debug seq 128 K/V hint | HVX 509.019 | HVX + hint 510.438 | 0.997x; no benefit. |
| DINOv2 Debug whole-stream VTCM | HVX 21.258 | synchronous 40.150; asynchronous 41.552 | Both regress by about 2x; no useful overlap window. |
| Falcon Debug N1 stationary prototype | HVX 509.019 | N1 661.809 | Correct but 1.300x slower. |
| Falcon Debug N2 candidate run | HVX reference 509.019 | N2-enabled 491.355 | Zero N2 candidates; difference is noise and not attributable to N2. |

Sources: `docs_engineering/omnifetch_history.md` and
`docs_engineering/engineering_work.md`.

### 2026-08-06 — full HuBERT and two-prefetch-baseline checkpoint

#### Full HuBERT true-vector and K/V-prefetch measurements

| Configuration | Latency | Correctness |
|---|---:|---|
| Native HVX | 212,074.277 | Finite; top-1 match. |
| HexKL + HVX | 268,255.667 | Finite; top-1 match. |
| HexKL + items 1–7 + HVX | 620,050.837 | Finite; top-1 match; regression. |
| Native HVX + repaired K/V L2-only | 217,146.200 | Finite; top-1 match; 2.39% slower than HVX. |

The two-model Prefetch-Kernel-HX/APT-GET-HX Debug results are in the required
cross-product matrix in Section 2.

Sources: `docs_engineering/engineering_work.md`, sections 15–16, and
`docs_engineering/hexagon-prefetch-baselines-plan.md`.

### 2026-08-08 and later in the same engineering section — latest upstream v73 corpus

All values are full-model, FP16, one device iteration, serial execution on the
same SM8550/v73 phone. Language models using staged execution report the sum of
device `Perf` only and exclude compilation, ADB transfer, and host round trips.

| Domain | Complete model | Native HVX | HexKL-enabled path | HVX / HexKL | Qualification |
|---|---|---:|---:|---:|---|
| Vision | DINOv2-small | 30,396.932 | 9,873.013 | 3.078x | Correct. |
| Vision | DeiT-small | 24,075.094 | 8,342.079 | 2.886x | Correct. |
| Vision | SegFormer MiT-B0 | 27,488.063 | 9,349.189 | 2.940x | Correct. |
| Vision | Swin Transformer | 122,531.193 | 73,079.566 | 1.677x | Correct. |
| Vision | BEiT-base | 129,376.464 | 13,495.335 | 9.587x | Correct. |
| Speech/audio | Whisper-tiny | 160,182.939 | 112,048.671 | 1.430x | Correct. |
| Speech/audio | HuBERT-base | 216,665.328 | 190,385.382 | 1.138x | Correct. |
| Speech/audio | Wav2Vec2-base | 216,404.775 | 174,978.035 | 1.237x | Correct. |
| Speech/audio | UniSpeech-base | 215,996.232 | 174,108.521 | 1.241x | Correct. |
| Speech/audio | UniSpeech-SAT-base | 215,612.362 | 174,006.382 | 1.239x | Correct. |
| Language/text | GPT-2 | 21,324.308 | 3,359.199 | 6.348x | Uniform FP16; finite, top-1 exact. |
| Language/text | SD/CLIP text encoder | 244,487.748 | 5,440.120 | 44.942x | Uniform FP16; max abs 0.015625. |
| Language/text | Qwen2.5-0.5B | 412,891.312 | 13,537.873 | 30.499x | FP16; top-5 exact. |
| Language/text | TinyLlama-1.1B | 975,472.962 | 32,908.259 | 29.642x | FP16; top-5 exact. |
| Language/text | SmolLM2-1.7B | 1,549,937.550 | 42,315.577 | 36.628x | FP16; top-5 exact. |

The earlier matched staged rows for GPT-2 and CLIP used FP32 and are retained
for reproducibility but are superseded by uniform FP16 for the primary corpus:

| Model | FP32 HVX | FP32 HexKL | HVX / HexKL |
|---|---:|---:|---:|
| GPT-2 full 12L | 29,133.859 | 28,661.474 | 1.016x |
| SD/CLIP full text encoder | 109,656.923 | 107,141.040 | 1.023x |

Source: `docs_engineering/engineering_work.md`, section 17.

### 2026-08-13 — full-model prefetch baselines and item-7-only convergence

The required PK/APT/native/OmniFetch engine matrix is in Section 2. The raw
OmniFetch ablation is reproduced here because it determines the current
conclusion.

| Full model | Configuration | Latency | Runtime issued | Issued bytes | Correctness |
|---|---|---:|---:|---:|---|
| DINOv2-small | HMLIR HVX (HexKL Off) | 29,885.944 | 0 | 0 | Pass. |
| DINOv2-small | Matched HexKL-on/zero-rewrite HVX control | 10,090.480 | 0 | 0 | Pass. |
| DINOv2-small | Item7-only | 5,960.990 | 0 | 0 | Pass. |
| DINOv2-small | Items 1–3 | 10,305.507 | 186,624 | 41,840,640 | Pass. |
| DINOv2-small | Items 1–5, item4 off | 9,612.841 | 186,624 | 41,840,640 | Pass. |
| DINOv2-small | Items 1–6, item4 off | 9,543.607 | 186,624 | 41,840,640 | Pass. |
| DINOv2-small | Items 1–7, item4 off | 5,669.315 | 186,624 | 41,171,328 | Pass. |
| DINOv2-small | Items 1–7, item4 off, bounded traffic (historical combination) | 5,558.501 | 4,096 | 903,872 | Pass; 1.815x over matched control. |
| ViT-Base | HMLIR HVX (HexKL Off) | 132,967.479 | 0 | 0 | Pass. |
| ViT-Base | Matched HexKL-on/zero-rewrite HVX control | 19,959.873 | 0 | 0 | Pass. |
| ViT-Base | Item7-only | **13,872.472** | 0 | 0 | Pass; 1.439x over matched control. |
| ViT-Base | Items 1–3 | 19,979.337 | 0 | 0 | Pass. |
| ViT-Base | Items 1–5, item4 off | 19,901.741 | 0 | 0 | Pass. |
| ViT-Base | Items 1–6, item4 off | 19,827.296 | 0 | 0 | Pass. |
| ViT-Base | Items 1–7, item4 off | 19,803.730 | 0 | 0 | Pass. |

Before item 4 was disabled, the original cumulative cold/warm/invalidated
protocol reported DINOv2 19,061.970 ms and ViT 48,344.123 ms. Those rows issued
171,041,329 / 398,320,594 cumulative requests and 287,102,337,728 /
669,108,840,110 cumulative bytes respectively across the three calls. They are
historical evidence of a traffic storm, not the current OmniFetch result.

For these eager vision graphs, item 7 rejected all internally produced K/V
sources and emitted zero hardware K/V hints. Its measured benefit therefore
comes from attention propagation/tiling/slicing topology changes, not from
runtime K/V prefetch. The bounded DINO policy reduces issued commands 45.6x
and issued bytes 45.5x, but improves latency only 1.99% relative to the
unbounded combination.

Source: `docs_engineering/hexagon-prefetch-baselines-plan.md`.

The default experiment scripts now select **item7-only** and keep items 1–6
disabled via command-line switches, including explicit disabling of the
otherwise-default layout-aware and adaptive options. They do not delete any
implementation; the explicitly named cumulative and no-item4 ablation scripts
remain available to reproduce the historical combination rows.

## 5. Supersession and paper-use guidance

1. Use the **2026-08-08+ uniform-FP16 15-model table** for the current native
   HVX-versus-HexKL full-model corpus. Do not combine it with the much slower
   2026-07-29 monolithic screening numbers.
2. Use the **2026-08-13 matched HexKL-on/zero-rewrite control** when computing
   OmniFetch speedup for the full DINOv2/ViT external-prefetch comparison.
   Pure HVX remains useful as a framework diagnostic, not the causal OF base.
3. The 2026-08-06 PK/APT Debug rows show traffic selectivity but only small
   latency differences. The 2026-08-13 full rows make PK and APT logically
   equivalent because both consume the same allowlist and distance-one plan;
   their small latency difference is noise/run order, not an algorithmic win.
4. Early `omnifetch-2x-improvement` results remain valuable historical signals
   but are not true-HVX evidence. Later audits showed that the old baseline was
   scalar-like and that multiple cumulative policies were entangled.
5. DeiT/BEiT NaN rows, SD-VAE/SD-UNet loose-comparison rows, timeouts, and exit
   13 cases must remain visible as negative or unqualified results and must not
   enter correctness-qualified speedup counts.
6. QNN HTP and LiteRT/QNN are closed vendor-stack upper bounds, not independent
   open prefetch baselines. Their graph fusion, layout, HMX/HVX mapping, memory
   planning, and runtime scheduling differ fundamentally from the current
   open Hexagon-MLIR pipeline.
