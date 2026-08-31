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
| Vision | ViT-Base | B=1, 3x224x224 | 86,567,656; 86.57 | [86.6M](https://huggingface.co/google/vit-base-patch16-224) | 17.56 | 35.13 |
| Vision | DINOv2-small | B=1, 3x224x224 | 22,825,192; 22.83 | [22.1M](https://huggingface.co/facebook/dinov2-small) | 6.12 | 12.25 |
| Speech | Whisper-tiny | B=1, 80x3000 mel; 32 decoder tokens | 37,760,640; 37.76 | [39M](https://huggingface.co/openai/whisper-tiny) | 21.29 | 42.58 |
| Speech | HuBERT-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [94.7M](https://www.isca-archive.org/interspeech_2023/zaiem23b_interspeech.pdf) | 9.00 | 17.99 |
| Speech | Wav2Vec2-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [95M](https://arxiv.org/abs/2006.11477) | 9.00 | 17.99 |
| Speech | UniSpeech-base | B=1, 20,560 waveform samples | 94,396,320; 94.40 | [94.68M](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/05/UniSpeech_SAT.pdf) | 9.00 | 17.99 |

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
limit. Every policy compiles and runs the identical FP16-model/storage embedding,
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

For the matched FP16-model/storage staged language rows, item7 is 1.87x faster than
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
for reproducibility but are superseded by the FP16-model/storage protocol for the primary corpus:

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

### 2026-08-28 — ALPS P5m/P5n full-model screening

These are complete, non-Debug FP16 models, run strictly serially with one device
measurement per configuration. Both P5m and P5n are cumulative configurations
that include P5h and P5i; P5m is analysis-only and executes the same synchronous
P5k path, while P5n adds VTCM ping-pong plus UserDMA asynchronous HMX-result
drain. Therefore each speedup below isolates only the P5n increment and must not
be attributed to P5h or P5i individually. Formal component ablations are deferred
until model screening is complete.

| Domain | Complete model | P5m/P5k matched control (ms) | P5n (ms) | Control / P5n | Latency reduction | P5m admitted / overlap bytes | P5n DMA issued / completed; bytes; fallback | Correctness / qualification |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Vision | DINOv2-small | 3,234.81 | **2,993.49** | **1.0806x** | **7.46%** | 21,233,664 / 21,086,208 | 10,368 / 10,368; 21,233,664; 0 | PASS; max diff 0.0046, top-1 match. Positive case. |
| Vision | BEiT-base | 8,864.83 | **8,292.04** | **1.0691x** | **6.46%** | 28,311,552 / 28,188,672 | 13,824 / 13,824; 28,311,552; 0 | PASS; max diff 0.0056, top-1 match. Positive case. |
| Vision | Swin Transformer | 41,401.57 | 40,898.92 | 1.0123x | 1.21% | 14,770,176 / 14,698,496 | 7,212 / 7,212; 14,770,176; 0 | PASS; top-1 match. Below the 3% continuation threshold; inconclusive. |
| Language/text | Qwen2.5-0.5B | 10,582.80 | **10,364.01** | 1.0211x | 2.07% | 19,464,192 / 19,120,128 (24 legal layers; vocabulary head rejected) | 9,504 / 9,504; 19,464,192; 0 | PASS after descriptor-range fix; finite, top-5 match. The 303,872 B vocabulary-head stride is now rejected and synchronously drained instead of being silently truncated. Correctness-qualified Language weak-performance case, below 3%. |
| Language/text | GPT-2 | 3,583.47 | NA | NA | NA | 0 / 0 | 0 / 0; 0; 0 | PASS control after descriptor-range fix; all 48 block sites have no residual direct-output drain and the 100,514 B vocabulary-head stride is correctly rejected. P5n is not run because it would issue no asynchronous work. Static admission negative. |
| Speech/audio | Whisper-tiny | 71,240.91 | **70,414.99** | 1.0117x | 1.16% | 41,680,896 / 41,574,400 | 20,352 / 20,352; 41,680,896; 0 | PASS; max diff 0.0039, top-1 match. P2g-c exit 13 was isolated to a 1500xf16 source row not divisible by the 64-lane/128 B HVX load and fixed with a source-tail admission gate. P5n is a correctness-qualified Speech weak-performance case, below the 3% continuation threshold. |
| Speech/audio | HuBERT-base | 180,829.05 | **176,891.35** | 1.0223x | 2.18% | 10,719,232 / 10,567,680 | 5,234 / 5,234; 10,719,232; 0 | PASS after rank-contract admission fix; max diff 0.0083, last-frame top-1 match. Correctness-qualified Speech weak-performance case, below 3%. |

#### Matched item7 composition follow-up

The earlier Qwen P5m/P5n row did not include item7 and therefore cannot explain
the historical item7 latency. With battery saver and device idle disabled, the
same complete 24-layer FP16 model was rerun with matched composition:

| Complete model | Item7-only (ms) | Item7 + P5m/P5k (ms) | Item7 + P5n (ms) | P5m / P5n | Item7 / P5n | P5n DMA | Correctness |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-0.5B | 5,911.34 | 5,639.33 | **5,328.06** | **1.0584x** | **1.1095x** | 9,504/9,504; 19,464,192 B; fallback=0 | PASS; finite, top-5 match, max abs 0.58984375. |

Item7 issued zero runtime L2 hints, so its recovered gain is a topology effect.
P5n's 5.52% reduction over matched P5m is the isolated asynchronous-drain
increment. A 20-repeat matched representative-layer sysMon run found 2.52%
fewer processor cycles but 5.81% more total AXI bytes under P5n; the result is
consistent with overlap/critical-path shortening rather than traffic removal.
The phone remained at 100%, 27.4--27.6 C, with low-power off and device-idle
disabled. Raw PMU and per-stage data are archived at:

```text
nano:/home/huzq85/2-working/working_set/alps_qwen_item7_p5n_sysmon_20260828
```

For the original **without-item7** screening matrix, the
correctness-qualified P5n positive set is DINOv2-small and BEiT-base. The
matched composition follow-up adds Qwen as a Language positive candidate, but
its formal repeated measurement remains part of the final ablation campaign.
Swin proves that high admitted/overlap bytes alone are insufficient:
the residual drain must also occupy a meaningful fraction of the critical path.
Qwen additionally proves that `issued == completed` is insufficient when the
compiler/runtime have not validated every hardware descriptor field width.

Raw artifacts and logs were moved directly to:

```text
nano:/home/huzq85/2-working/working_set/alps_p5k_matched_control_dinov2_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_dinov2_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_beit_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_beit_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_swin_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_swin_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_qwen_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hmx_async_drain_qwen_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_gpt2_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_whisper_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hmx_async_drain_analysis_hubert_20260828
nano:/home/huzq85/2-working/working_set/alps_p2g_whisper_rootcause_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_rootcause_20260828
nano:/home/huzq85/2-working/working_set/alps_p5gg_whisper_rootcause_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_site_diag_all_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_site_0_20260828
nano:/home/huzq85/2-working/working_set/alps_p2gc_whisper_tailfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_whisper_tailfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_whisper_tailfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_qwen_stridefix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_hubert_rankfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5n_hubert_rankfix_20260828
nano:/home/huzq85/2-working/working_set/alps_p5m_gpt2_stridefix_20260828
```

## 5. Supersession and paper-use guidance

1. Use the **2026-08-08+ FP16-model/storage, mixed-kernel 15-model table** for the current native
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

### Frozen 15-model ALPS measurement schema (2026-08-29)

The next complete-model table must use the frozen post-`3b90cd4` code and run
models serially in this order of configurations: PK HVX, APT HVX, Hexagon-MLIR
HVX (HexKL Off), Hexagon-MLIR HVX (HexKL On), and ALPS C+E+P+R. All models are
non-Debug and use matched model/storage precision, input shape, layer count,
device settings, and timing windows. A globally enabled ALPS stage that does
no work is reported as `not admitted`, not as an active optimization.

This run creates a new table headed **Frozen ALPS 15-model complete-model
matrix (post-3b90cd4)**. It does not replace any historical table in this file.
The generated `frozen_full_matrix.csv`, `frozen_full_matrix_long.csv`, and
`frozen_full_matrix.md` are the authoritative machine-readable and rendered
sources for that new table.

Correction recorded on 2026-08-30: the current `UniSpeechForCTC` and
`UniSpeechSatForCTC` default-config runners do **not** constitute two
independent model structures.  They have the same 12/768/12/3072 encoder,
identical convolution frontend, identical 94,396,320 parameters, identical
213 parameter-shape signatures, and export the same 98-batch-matmul graph.
Their PK ledgers are also identical (387 hints, 1,925,676 runtime issues,
401,762,008 issued bytes and 24,625,664 B materialization); measured PK latency
is 172,840.50 versus 174,607.95 ms, a 1.02% run-to-run difference.  The latter
row is retained only as a duplicate/reproducibility check.  It is excluded from
the independent-model count, its remaining four configurations were cancelled,
and complete ViT-Base replaces it in the frozen 15-unique-model matrix.

In addition to latency, speedup, and correctness, every row must record the
following movement fields. They are deliberately separate because they answer
different questions:

| Field | Source | Interpretation |
|---|---|---|
| Baseline static materialization bytes | P1 post-bufferization ledger | Logical bytes explicitly materialized by copies/physical transforms before ALPS |
| ALPS static materialization bytes | Matched P1 post-bufferization ledger | Remaining logical materialization after admitted ALPS rewrites |
| Logical materialization reduction | Baseline minus ALPS, bytes and percent | Compiler-caused reduction; not a hardware traffic counter |
| Rewrite-attributed eliminated bytes | P2e/P5h/P5i ledgers | Bytes causally discharged by actual direct/in-situ formations |
| Runtime DMA issued/completed/suppressed bytes | P5n/R runtime ledger | Real asynchronous traffic introduced or rejected by P/R |
| sysMon AXI read/write/total bytes | Hardware PMU kernel window | External traffic caused by L2 misses, including demand and prefetch |
| sysMon pcycles/packets/HVX/HMX/L2fetch | Hardware PMU kernel window | Compute coverage, efficiency, and prefetch/cache behavior |
| sysMon clock/DCVS/thermal/BLC summary | Raw sysMon samples | Run comparability and memory/bus-pressure diagnostics |

For layered Language runners, compiler bytes must be weighted by each stage's
actual invocation count. Dynamic sites with no trustworthy runtime extent are
reported as `NA`, never silently treated as zero. The final paper table should
show logical materialization reduction beside physical AXI change so that a
compiler estimate is never presented as measured DDR traffic.

The device process currently cannot access HAP user PMU counters from its
Unsigned PD, but sysMon itself samples hardware PMU through the system profiler
service. Default-mode sysMon is therefore used for offline evidence and
cross-invocation profile-guided admission. It cannot implement descriptor-level
within-invocation feedback because its samples are out-of-process, roughly
1 ms granularity, and parsed after execution. Custom sysMon User-mode events and
STID/marker filtering may be used in separate diagnostic runs; they must not be
mixed into formal latency because four-counter mode can perturb DCVS and
eight-counter mode disables DCVS.

### 2026-08-29 — complete-model bottleneck corpus

This corpus profiles correctness-qualified item7 synchronous device code. LWP
is a separate instrumented run used only for ranking; sysMon replays the
non-instrumented, already-compiled complete model after all files are placed on
the phone. P5m/P5k compiler-ledger evidence is attached where the graph can be
compiled within the host-memory budget, but is not required by sysMon.

| Domain | Complete model | Formal non-LWP latency | LWP top stage/operator | sysMon processor cycles | AXI read / write / total | Status |
|---|---|---:|---|---:|---:|---|
| Language/text | GPT-2 | 3,671.84 ms | vocabulary head 60.26%; 12 blocks 39.66%; embedding 0.08% | 8,541,263,621 | 1,443,787,648 / 562,182,144 / 2,005,969,792 B | PASS; standard item7 14-stage sysMon; LWP distribution from matched synchronous analysis. |
| Language/text | SD/CLIP | 3,438.01 ms | HMX outer chains 66.09%; f16→f32 HMX copy 8.26%; GELU chain 7.75%; reductions 4.11% | 7,637,741,754 | 1,699,881,984 / 467,926,016 / 2,167,808,000 B | LWP + sysMon PASS; 12 layers 99.65% of formal Perf. |
| Language/text | Qwen2.5-0.5B | 5,875.53 ms | head 21.69%; HMX outer 63.50%; extf/mul/add 16.22%; HMX output copy 6.18%; SiLU 5.94% | 17,332,438,202 | 4,317,510,144 / 1,607,163,264 / 5,924,673,408 B | LWP + sysMon PASS; full 24-layer item7. |
| Language/text | TinyLlama-1.1B | 17,769.19 ms | 22 layers 96.69%; HMX outer 79.48%; extf chain 11.08%; output copy 2.55% | 35,041,832,364 | 9,049,282,304 / 2,558,079,488 / 11,607,361,792 B | LWP + sysMon PASS. |
| Language/text | SmolLM2-1.7B | 29,725.99 ms | 24 layers 94.42%; head 5.58%; HMX outer 84.60%; extf chain 7.89%; output copy 2.12% | 56,332,486,863 | 14,924,729,472 / 4,065,359,232 / 18,990,088,704 B | LWP + sysMon PASS. |
| Vision | Swin Transformer | 51,098.04 ms | full-graph item7-only LWP: extf/mulf/addf 94.00%; HMX 2.22%; softmax/reduction 1.87% | 75,793,607,744 | 783,823,104 / 311,505,664 / 1,095,328,768 B | LWP + sysMon PASS; LWP Perf 50,973.58 ms, max diff 0.0015/top-1 match; scalar/codegen dominated. |
| Vision | SegFormer MiT-B0 | 8,930.05 ms | full-graph item7-only LWP: extf/mulf/addf 85.68%; other elementwise 5.92%; HMX 2.63% | 13,364,473,801 | 236,403,200 / 80,740,096 / 317,143,296 B | LWP + sysMon PASS; LWP Perf 8,954.01 ms, max diff 0.0013/top-1 match; scalar/codegen dominated. |
| Vision | DeiT-small | 5,086.94 ms | full-graph item7-only LWP: extf/mulf/addf 57.64%; HMX chain 26.21%; softmax/reduction 4.23% | 7,754,132,341 | 620,727,168 / 248,425,984 / 869,153,152 B | LWP + sysMon PASS; LWP Perf 5,006.24 ms, finite/max diff 0.0035/top-1 match. Earlier P1--P5m instrumentation builds OOMed; lightweight matched build passed. |
| Vision | BEiT-base | 13,824.45 ms | full-graph item7-only LWP: HMX chain 47.57%; extf/mulf/addf 34.90%; softmax/reduction 11.34% | 24,635,320,296 | 2,980,995,200 / 554,533,888 / 3,535,529,088 B | LWP + sysMon PASS; LWP Perf 13,910.97 ms, finite/max diff 0.0049/top-1 match. |
| Vision | DINOv2-small | 6,071.57 ms | prior LWP: patch conv 39.09% before P5i; residual HMX outer/attention after P5i | 9,187,880,356 | 842,610,944 / 338,963,712 / 1,181,574,656 B | sysMon PASS by SSH-to-ADB remote replay; low-power off, 28.2 C. |
| Speech/audio | Whisper-tiny | 70,067.02 ms | full-shape local LWP: frontend 88.37%; one encoder layer 8.38%; head 2.30%; conv accumulation chain 92.43% | 104,747,049,461 | 5,064,919,680 / 1,790,385,408 / 6,855,305,088 B | LWP + sysMon PASS; frontend convolution/codegen dominated. |
| Speech/audio | HuBERT-base | 172,773.84 ms | full-shape local LWP: frontend 90.74%; position conv 9.17%; one layer 0.09% | 255,846,585,000 | 5,036,792,192 / 418,441,600 / 5,455,233,792 B | LWP + sysMon PASS; convolution codegen dominated. |
| Speech/audio | Wav2Vec2-base | 184,932.30 ms | full-shape local LWP: frontend 90.96%; position conv 8.95%; one layer 0.09% | 273,842,295,628 | 5,067,176,448 / 424,789,376 / 5,491,965,824 B | LWP + sysMon PASS; convolution codegen dominated. |
| Speech/audio | UniSpeech-base | 184,300.14 ms | full-shape local LWP: frontend 90.73%; position conv 9.17%; one layer 0.09% | 272,912,050,151 | 5,053,952,896 / 425,125,376 / 5,479,078,272 B | LWP + sysMon PASS; shared convolution bottleneck. |
| Speech/audio | UniSpeech-SAT-base | 179,391.43 ms | full-shape local LWP: frontend 90.03%; position conv 9.88%; one layer 0.09% | 265,650,670,274 | 5,059,622,912 / 421,796,864 / 5,481,419,776 B | LWP + sysMon PASS; shared convolution bottleneck. |

Raw GPT-2 evidence:

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/gpt2
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/sysmon/gpt2
```

DeiT and DINO evidence (the earlier DeiT P1--P5m whole-graph LWP OOM builds
never reached the phone; the lightweight item7-only whole-graph LWP did pass):

```text
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/sysmon/deit-small/item7-archived-replay
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/lwp/deit-small/item7-only
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/sysmon/dinov2-small/item7-remote-replay
nano:/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829/hotspot_lwp_whisper
```

Precision qualification for this complete-model corpus: all 15 rows use FP16
floating-point weights, primary activations and floating inputs. Integer token
IDs remain integer. HMX matmul consumes FP16 operands, but some current HVX
convolution/elementwise lowering widens both operands to FP32 before multiply
and accumulation. The precise label is therefore **FP16 model/storage with
mixed FP16-HMX and FP32-HVX kernel arithmetic**, not pure FP16 compute and not
an FP32 model baseline. No runtime quantize/dequantize pass is enabled. The
archived measurements do not need a model-dtype-only rerun; affected models
must be rerun only after a helper-free FP16-operand HVX lowering changes their
actual object code.

### 2026-08-29 — SegFormer FP16-HVX precision-policy ablation

This is a complete-model, matched HVX-vector + HexKL-on experiment. The only
treatment is the independently switchable FP16 convolution/elementwise
precision pass; item7, prefetch and layout passes are off in both arms.

| Complete model | Configuration | Non-LWP latency | Relative to treatment | Processor cycles | AXI read / write / total | Correctness |
|---|---|---:|---:|---:|---:|---|
| SegFormer MiT-B0 | HMLIR HVX, HexKL on | **9,262.15 ms** | **0.83x** | 13,928,266,519 | 237,068,544 / 82,174,464 / 319,243,008 B | PASS; finite, max diff 0.0016, top-1 match |
| SegFormer MiT-B0 | ALPS FP16-HVX arithmetic | 11,209.29 ms | 1.00x | 16,808,990,332 | 220,965,760 / 71,278,720 / 292,244,480 B | PASS; finite, max diff 0.0017, top-1 match |

The treatment is **21.03% slower despite 8.46% less AXI traffic**. Object
inspection explains the result: instruction count rises from 107,523 to
150,896, HVX-like share falls from 4.54% to 2.52%, and static relocation sites
for `__extendhfsf2` / `__truncsfhf2` rise from 7,636 / 5,111 to 13,294 /
10,530. This broad FP16-accumulator policy is a correctness-qualified negative
ablation and remains default-off. The V73 baseline-engineering target is
helper-free FP16 operand loads with vector FP32 accumulation, followed by ALPS
E/P/R optimization of the residual representation and transfers.

```text
nano:/home/huzq85/2-working/working_set/alps_fp16_hvx_segformer_lwp_20260829
nano:/home/huzq85/2-working/working_set/alps_fp16_hvx_segformer_sysmon_20260829
```

### 2026-08-29 — ALPS C selective HVX widening convolution

Both arms use complete FP16 models, HVX vector execution and HexKL on. The
only treatment is the independently switchable native-width widening
convolution pass; item7, layout, prefetch, DMA and PMU admission are off.

| Domain | Complete model | Configuration | Latency | Speedup over treatment | Correctness |
|---|---|---|---:|---:|---|
| Vision | SegFormer MiT-B0 | HMLIR HVX, HexKL on | 9,313.73 ms | 1.47x | PASS; finite, max diff 0.0016, top-1 match |
| Vision | SegFormer MiT-B0 | ALPS C widening conv | **6,352.36 ms** | 1.00x | PASS; finite, max diff 0.0015, top-1 match |
| Speech/audio | Whisper-tiny | HMLIR HVX, HexKL on | 113,415.80 ms | 1.05x | PASS; finite, max diff 0.0049, last-token top-1 match |
| Speech/audio | Whisper-tiny | ALPS C widening conv | **108,246.36 ms** | 1.00x | PASS; finite, max diff 0.0044, last-token top-1 match |

For SegFormer, matched sysMon processor cycles fall from 13,917,700,995 to
9,642,931,398 and committed packets from 4,970,712,872 to 3,190,233,004,
while AXI traffic stays essentially constant (318,152,704 vs 317,474,688 B).
The result is therefore a compute/codegen improvement, not a data-prefetch or
DRAM-traffic reduction. Matched LWP root cycles fall from 13.723 B to 9.540 B.

The first post-transfer Whisper run is excluded: its log reported
`alps_hvx_widening_conv=0` because the runner omitted CLI-to-backend parameter
propagation, and its object was byte-for-codegen identical to control. The row
above is the corrected run with the backend flag explicitly verified as 1.

```text
nano:/home/huzq85/2-working/working_set/alps_c64_segformer_full_20260829
nano:/home/huzq85/2-working/working_set/alps_c64_segformer_lwp_20260829
/tmp/alps_c64_segformer_sysmon_control_20260829
/tmp/alps_c64_segformer_sysmon_candidate_20260829
nano:/home/huzq85/2-working/working_set/alps_c_whisper_full_20260829
```

### 2026-08-29 — ALPS E on top of C

The treatment changes only consumer-driven direct layout formation. Both arms
retain identical complete FP16 models, HVX vector execution, HexKL-on and the
selective C widening-convolution lowering.

| Domain | Complete model | Configuration | Latency | Speedup over treatment | P2e direct / demands | Eliminated materialization | Correctness |
|---|---|---|---:|---:|---:|---:|---|
| Vision | SegFormer MiT-B0 | ALPS C | 6,352.36 ms | 1.16x | 0 / 0 | 0 B | PASS; finite, max diff 0.0015, top-1 match |
| Vision | SegFormer MiT-B0 | **ALPS C + E** | **5,489.21 ms** | 1.00x | 24 / 116 | 1,655,808 B | PASS; finite, max diff 0.0015, top-1 match |
| Speech/audio | Whisper-tiny | ALPS C | 108,246.36 ms | 1.51x | 0 / 0 | 0 B | PASS; finite, max diff 0.0044, last-token top-1 match |
| Speech/audio | Whisper-tiny | **ALPS C + E** | **71,512.90 ms** | 1.00x | 36 / 114 | 18,923,520 B | PASS; finite, max diff 0.0044, last-token top-1 match |

Relative to the original matched HexKL-on controls, cumulative C+E speedup is
1.70x for SegFormer (9,313.73 / 5,489.21) and 1.59x for Whisper
(113,415.80 / 71,512.90). These cumulative ratios are not attributed to
prefetch: C is compute/codegen baseline enablement and E removes/directly forms
representations. P will be measured separately on residual movement.

```text
nano:/home/huzq85/2-working/working_set/alps_ce_segformer_full_20260829
nano:/home/huzq85/2-working/working_set/alps_ce_whisper_full_20260829
```

### 2026-08-29 — ALPS P residual async drain on top of C+E

Both complete-model arms enable identical HVX vector, HexKL, C, P2e, HMX F16
direct epilogue/output formation and P5m analysis. Only the treatment enables
the P5n VTCM ping-pong asynchronous UserDMA drain. The old cumulative P1--P5
bundle is absent.

| Domain | Complete model | Configuration | Latency | Speedup over treatment | P5m admitted / overlap bytes | Runtime DMA | Correctness |
|---|---|---|---:|---:|---:|---|---|
| Vision | DINOv2-small | **ALPS C+E direct-output control** | **5,768.58 ms** | **0.99x** | 72 / 21,086,208 B | 0 | PASS; finite, max diff 0.0049, allclose/top-1 |
| Vision | DINOv2-small | ALPS C+E+P async drain | 5,828.72 ms | 1.00x | 72 / 21,086,208 B | 10,368/10,368; 21,233,664 B; fallback=0 | PASS; finite, max diff 0.0051, allclose/top-1 |

P is a correctness-qualified **negative ablation** here: real DMA executes,
but latency regresses 1.04%. This case is the reject target for R/PMU traffic
admission; static overlap legality alone is insufficient.

```text
nano:/home/huzq85/2-working/working_set/alps_cep_dinov2_valid2_20260829
```

### 2026-08-29 — ALPS R traffic admission on top of C+E+P

Both complete-model arms execute the same P5n asynchronous drain. Only the R
treatment monitors 64-completion windows and may restore the native HexKL
synchronous drain after sustained DMPoll pressure.

| Domain | Complete model | Configuration | Latency | Relative to R | DMA issued/completed; bytes | R windows / hold / throttle / suppressed | PMU / software fallback | Correctness |
|---|---|---|---:|---:|---|---|---|---|
| Vision | DINOv2-small | ALPS C+E+P | 5,825.25 ms | 1.06x | 10,368/10,368; 21,233,664 B | NA | NA | PASS; finite, max diff 0.0049, allclose/top-1 |
| Vision | DINOv2-small | **ALPS C+E+P+R** | **5,487.22 ms** | 1.00x | 10,368/10,368; 21,233,664 B | 162 / 162 / 0 / 0 | HAP PMU unavailable; 0 poll retries | PASS; identical gate |

R made no admission change in this run, so the observed 1.0616x single-sample
difference is **not attributed to R**. The row validates correct monitoring,
zero-regression execution and honest PMU-unavailable fallback. A PMU-triggered
performance claim remains pending an authorized-PMU or sysMon-guided final
ablation.

```text
nano:/home/huzq85/2-working/working_set/alps_cepr_dinov2_final_20260829
```

## 2026-08-29--30 — Frozen ALPS 15-unique-model complete-model matrix

This is the new post-`3b90cd4` main table.  Every primary row is a complete,
non-Debug FP16 model, all five configurations passed their model-specific
correctness gate, execution was strictly serial, and no compile/device timeout
was used.  Ratios in parentheses are latency divided by ALPS C+E+P+R, so ALPS
is fixed at 1.00x.  PK means Prefetch-Kernel-HX and APT means APT-GET-HX.

UniSpeech-SAT-Base is not counted as an independent model.  The structural
audit above showed that its current default-config ForCTC export is identical
to UniSpeech-Base except for class-name prefixes and random seed.  Its completed
PK row (`174,607.95 ms`) is retained only as a reproducibility check; the other
four duplicate runs were cancelled.  Full ViT-Base replaces it, yielding 15
structurally independent models and 75/75 primary PASS rows.

### Latency

| Domain | Complete model | PK HVX | APT HVX | HMLIR HVX (HexKL Off) | HMLIR HVX (HexKL On) | ALPS C+E+P+R |
|---|---|---:|---:|---:|---:|---:|
| Vision | DINOv2-small | 10,363.19 ms (1.88x) | 10,325.86 ms (1.88x) | 30,024.51 ms (5.46x) | 9,878.12 ms (1.80x) | **5,499.60 ms (1.00x)** |
| Language/text | GPT-2 | 3,659.70 ms (1.04x) | 3,602.06 ms (1.03x) | 17,587.14 ms (5.02x) | 3,584.55 ms (1.02x) | **3,505.38 ms (1.00x)** |
| Language/text | SD/CLIP | 3,677.41 ms (1.20x) | 3,689.75 ms (1.20x) | 234,816.52 ms (76.63x) | 3,507.08 ms (1.14x) | **3,064.11 ms (1.00x)** |
| Language/text | Qwen2.5-0.5B | 11,079.34 ms (1.07x) | 11,204.97 ms (1.09x) | 384,424.67 ms (37.29x) | 10,879.13 ms (1.06x) | **10,307.88 ms (1.00x)** |
| Language/text | TinyLlama-1.1B | 27,392.68 ms (1.01x) | 27,376.92 ms (1.01x) | 917,036.32 ms (33.90x) | 27,117.70 ms (1.00x) | **27,053.75 ms (1.00x)** |
| Language/text | SmolLM2-1.7B | 39,182.95 ms (1.02x) | 39,215.91 ms (1.02x) | 1,448,959.95 ms (37.78x) | 39,351.79 ms (1.03x) | **38,349.59 ms (1.00x)** |
| Vision | Swin Transformer | 73,193.61 ms (1.54x) | 73,176.63 ms (1.54x) | 120,881.83 ms (2.54x) | 73,608.59 ms (1.55x) | **47,500.03 ms (1.00x)** |
| Vision | SegFormer MiT-B0 | 9,270.92 ms (1.73x) | 9,248.58 ms (1.73x) | 27,392.48 ms (5.11x) | 9,221.61 ms (1.72x) | **5,358.76 ms (1.00x)** |
| Vision | DeiT-Small | 8,580.89 ms (1.71x) | 8,553.92 ms (1.71x) | 23,960.79 ms (4.78x) | 8,278.09 ms (1.65x) | **5,016.40 ms (1.00x)** |
| Vision | BEiT-Base | 14,796.46 ms (1.19x) | 14,762.74 ms (1.19x) | 113,698.07 ms (9.16x) | 13,506.50 ms (1.09x) | **12,411.58 ms (1.00x)** |
| Vision | ViT-Base | 20,313.51 ms (1.54x) | 20,391.71 ms (1.55x) | 119,657.63 ms (9.08x) | 19,474.96 ms (1.48x) | **13,180.53 ms (1.00x)** |
| Speech/audio | Whisper-Tiny | 114,142.31 ms (1.67x) | 114,667.72 ms (1.68x) | 157,107.57 ms (2.30x) | 112,410.63 ms (1.65x) | **68,177.63 ms (1.00x)** |
| Speech/audio | HuBERT-Base | 172,586.28 ms (0.98x) | 172,448.59 ms (0.98x) | 209,056.42 ms (1.19x) | 172,764.15 ms (0.98x) | 176,251.20 ms (1.00x) |
| Speech/audio | Wav2Vec2-Base | 175,435.16 ms (1.00x) | 172,905.90 ms (0.99x) | 210,208.72 ms (1.20x) | 176,979.80 ms (1.01x) | **174,954.37 ms (1.00x)** |
| Speech/audio | UniSpeech-Base | 172,840.50 ms (0.98x) | 175,592.14 ms (1.00x) | 209,262.92 ms (1.19x) | 177,101.66 ms (1.01x) | **176,065.25 ms (1.00x)** |

Relative to the matched HexKL-On control, ALPS reaches at least 1.50x on five
models: DINOv2 (1.80x), Swin (1.55x), SegFormer (1.72x), DeiT (1.65x), and
Whisper (1.65x).  ViT is close at 1.48x.  It is effectively neutral on the
three large LLMs and Wav2Vec2, improves GPT-2/CLIP/Qwen/BEiT modestly, and
regresses HuBERT by 2.02%; these negative/neutral cases must remain visible.
Relative to PK, six models are at least 1.50x: DINOv2, Swin, SegFormer, DeiT,
ViT, and Whisper.

### Materialization and measured external traffic

Post-bufferization materialization and rewrite-attributed elimination are kept
separate.  A zero in the first delta does not negate a P2e rewrite: it means the
common post-bufferization ledger still sees the same total physical-copy class,
while P2e records the tensor-level materialization causally discharged.  AXI is
the independent sysMon hardware-PMU measurement.

| Domain | Model | HMLIR-On materialization | ALPS materialization | Ledger delta | P2e eliminated | ALPS DMA bytes | HMLIR-On AXI | ALPS AXI | AXI reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vision | DINOv2-small | 66,290,754 | 66,290,754 | 0 (0.00%) | 7,105,536 | 21,233,664 | 1,158,877,184 | 948,935,040 | 209,942,144 |
| Language/text | GPT-2 | 3,857,792 | 3,857,792 | 0 (0.00%) | 589,824 | 0 | 2,004,345,600 | 1,964,631,040 | 39,714,560 |
| Language/text | SD/CLIP | 4,501,570 | 4,501,570 | 0 (0.00%) | 4,257,792 | 10,616,832 | 2,162,537,856 | 2,158,678,656 | 3,859,200 |
| Language/text | Qwen2.5-0.5B | 11,947,008 | 11,947,008 | 0 (0.00%) | 1,572,864 | 19,464,192 | 5,934,923,264 | 6,031,726,336 | -96,803,072 |
| Language/text | TinyLlama-1.1B | 6,687,488 | 6,693,120 | -5,632 (-0.08%) | 3,244,032 | 26,918,912 | 11,630,026,496 | 11,717,683,328 | -87,656,832 |
| Language/text | SmolLM2-1.7B | 11,144,192 | 11,150,336 | -6,144 (-0.06%) | 6,291,456 | 37,748,736 | 19,012,453,248 | 19,052,500,480 | -40,047,232 |
| Vision | Swin Transformer | 38,927,630 | 41,637,134 | -2,709,504 (-6.96%) | 8,580,096 | 14,770,176 | 1,043,488,512 | 943,997,696 | 99,490,816 |
| Vision | SegFormer MiT-B0 | 5,727,486 | 5,727,486 | 0 (0.00%) | 1,655,808 | 6,422,784 | 318,393,088 | 184,964,480 | 133,428,608 |
| Vision | DeiT-Small | 39,470,060 | 39,470,060 | 0 (0.00%) | 5,474,304 | 15,925,248 | 844,810,880 | 695,464,064 | 149,346,816 |
| Vision | BEiT-Base | 25,808,448 | 25,808,448 | 0 (0.00%) | 10,893,312 | 28,311,552 | 3,512,182,912 | 3,208,813,952 | 303,368,960 |
| Vision | ViT-Base | 78,416,154 | 78,416,154 | 0 (0.00%) | 10,893,312 | 31,850,496 | 3,627,590,912 | 3,299,069,440 | 328,521,472 |
| Speech/audio | Whisper-Tiny | 558,061,304 | 558,062,328 | -1,024 (-0.00%) | 18,923,520 | 50,823,168 | 6,942,770,816 | 6,016,917,888 | 925,852,928 |
| Speech/audio | HuBERT-Base | 24,625,664 | 24,625,664 | 0 (0.00%) | 4,718,592 | 10,719,232 | 5,460,724,608 | 5,426,191,104 | 34,533,504 |
| Speech/audio | Wav2Vec2-Base | 24,625,664 | 24,625,664 | 0 (0.00%) | 4,718,592 | 10,719,232 | 5,450,981,888 | 5,431,344,640 | 19,637,248 |
| Speech/audio | UniSpeech-Base | 24,625,664 | 24,625,664 | 0 (0.00%) | 4,718,592 | 10,719,232 | 5,459,425,536 | 5,436,072,192 | 23,353,344 |

Measured AXI traffic falls on 12/15 models and rises on Qwen2.5, TinyLlama,
and SmolLM2.  The largest reductions occur on Whisper (925.85 MB), ViT
(328.52 MB), BEiT (303.37 MB), DINOv2 (209.94 MB), DeiT (149.35 MB), and
SegFormer (133.43 MB).  The three LLM traffic increases align with their weak
latency gains and make them important rejection/admission cases for P/R.

### ALPS admission/runtime audit

| Domain | Model | P2e direct/demands | P5j formed | P5m admitted | DMA issued/bytes | R windows/hold/throttle/suppressed | Poll retries |
|---|---|---:|---:|---:|---:|---:|---:|
| Vision | DINOv2-small | 36/122 | 72 | 72 | 10,368 / 21,233,664 | 162/162/0/0 | 0 |
| Language/text | GPT-2 | 12/49 | 1 | 0 | 0 / 0 | 0/0/0/0 | 0 |
| Language/text | SD/CLIP | 36/132 | 72 | 72 | 5,184 / 10,616,832 | 72/72/0/0 | 0 |
| Language/text | Qwen2.5-0.5B | 48/289 | 169 | 168 | 9,504 / 19,464,192 | 144/144/0/0 | 8 |
| Language/text | TinyLlama-1.1B | 44/265 | 133 | 133 | 13,144 / 26,918,912 | 191/191/0/0 | 29 |
| Language/text | SmolLM2-1.7B | 48/289 | 145 | 144 | 18,432 / 37,748,736 | 288/288/0/0 | 0 |
| Vision | Swin Transformer | 36/165 | 35 | 35 | 7,212 / 14,770,176 | 112/112/0/0 | 1 |
| Vision | SegFormer MiT-B0 | 24/116 | 52 | 48 | 3,228 / 6,422,784 | 50/50/0/0 | 2 |
| Vision | DeiT-Small | 36/122 | 72 | 72 | 7,776 / 15,925,248 | 121/121/0/0 | 3 |
| Vision | BEiT-Base | 36/134 | 60 | 60 | 13,824 / 28,311,552 | 216/216/0/0 | 6 |
| Vision | ViT-Base | 36/122 | 72 | 72 | 15,552 / 31,850,496 | 243/243/0/0 | 0 |
| Speech/audio | Whisper-Tiny | 36/114 | 65 | 64 | 24,816 / 50,823,168 | 387/387/0/0 | 9 |
| Speech/audio | HuBERT-Base | 48/137 | 74 | 74 | 5,234 / 10,719,232 | 81/81/0/0 | 0 |
| Speech/audio | Wav2Vec2-Base | 48/137 | 74 | 74 | 5,234 / 10,719,232 | 81/81/0/0 | 1 |
| Speech/audio | UniSpeech-Base | 48/137 | 74 | 74 | 5,234 / 10,719,232 | 81/81/0/0 | 0 |

R observed only hold windows in this frozen run; no window throttled or
suppressed DMA and the in-process PMU status remained unavailable.  Therefore
none of the latency change may be attributed to an R policy decision.  R is a
validated monitoring/safety path here; a traffic-control performance claim
requires the planned cross-invocation sysMon policy or an authorized-PMU run.

Authoritative generated sources and all moved compile/runtime artifacts:

```text
nano:/home/huzq85/2-working/working_set/alps_frozen_full_matrix_20260829/results.csv
nano:/home/huzq85/2-working/working_set/alps_frozen_full_matrix_20260829/frozen_full_matrix.csv
nano:/home/huzq85/2-working/working_set/alps_frozen_full_matrix_20260829/frozen_full_matrix_long.csv
nano:/home/huzq85/2-working/working_set/alps_frozen_full_matrix_20260829/frozen_full_matrix.md
```

## Formal ALPS component ablation — selected complete models

This table is deliberately separate from the 15-model baseline matrix.  Model
selection is pre-registered from that matrix rather than chosen after inspecting
component results: every model whose frozen ALPS latency is at least 1.50x faster
than the matched HexKL-On control is included.  This yields DINOv2-small, Swin,
SegFormer, DeiT, and Whisper.  A0 and A4 are reused verbatim from the frozen main
experiment; only A1--A3 require new runs.

| Domain | Complete model | A0: HexKL On | A1: +C | A2: +E | A3: +P | A4: +R / final ALPS | A0/A4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Vision | DINOv2-small | 9,878.12 ms | 9,829.81 ms (1.00x) | 5,812.16 ms (1.69x) | 5,503.90 ms (1.06x) | **5,499.60 ms (1.00x)** | **1.80x** |
| Vision | Swin Transformer | 73,608.59 ms | 75,721.32 ms (0.97x) | 47,928.50 ms (1.58x) | 47,764.16 ms (1.00x) | **47,500.03 ms (1.01x)** | **1.55x** |
| Vision | SegFormer MiT-B0 | 9,221.61 ms | 7,038.80 ms (1.31x) | 5,427.76 ms (1.30x) | 5,326.17 ms (1.02x) | **5,358.76 ms (0.99x)** | **1.72x** |
| Vision | DeiT-Small | 8,278.09 ms | 8,273.45 ms (1.00x) | 5,290.15 ms (1.56x) | 5,007.50 ms (1.06x) | **5,016.40 ms (1.00x)** | **1.65x** |
| Speech/audio | Whisper-Tiny | 112,410.63 ms | 107,996.01 ms (1.04x) | 69,066.98 ms (1.56x) | 68,019.59 ms (1.02x) | **68,177.63 ms (1.00x)** | **1.65x** |

Ratios inside A1--A4 cells are adjacent-stage speedups `A(i-1)/Ai`; the final
column is cumulative `A0/A4`.  All 15 newly run A1--A3 cases passed their
model-specific correctness gate.  Execution was strictly serial, used complete
non-Debug FP16 models, and had no timeout or automatic retry.

### Ablation mechanism and physical-traffic audit

| Model | E eliminated bytes | P DMA issued / bytes | A1 sysMon AXI | A2 sysMon AXI | A3 sysMon AXI | A1→A3 AXI reduction |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2-small | 7,105,536 | 10,368 / 21,233,664 | 1,158,515,200 | 977,842,304 | 946,680,832 | 211,834,368 (18.28%) |
| Swin Transformer | 8,580,096 | 7,212 / 14,770,176 | 1,043,097,600 | 960,236,672 | 943,701,248 | 99,396,352 (9.53%) |
| SegFormer MiT-B0 | 1,655,808 | 3,228 / 6,422,784 | 317,420,672 | 181,221,632 | 185,138,048 | 132,282,624 (41.67%) |
| DeiT-Small | 5,474,304 | 7,776 / 15,925,248 | 845,373,696 | 714,644,864 | 695,897,728 | 149,475,968 (17.68%) |
| Whisper-Tiny | 18,923,520 | 24,816 / 50,823,168 | 6,901,641,088 | 6,112,085,760 | 6,014,351,616 | 887,289,472 (12.86%) |

The causal result is consistent across all five models.  E is the dominant
stage: its adjacent gain is 1.30x--1.69x and it reduces measured AXI traffic in
every model.  P completes every issued DMA without fallback and contributes
1.06x on DINOv2/DeiT, about 1.02x on SegFormer/Whisper, and 1.00x on Swin.
C is topology dependent: it contributes 1.31x on SegFormer and 1.04x on
Whisper, is neutral on DINOv2/DeiT, and regresses Swin by about 2.8% before E
recovers the loss.  R made no throttle/suppression decision in the frozen A4
runs; A3-to-A4 differences are within -0.6%--+0.6% and carry no causal R claim.

The following older complete-model results can be reused as supporting evidence
and sanity bounds, but not mixed with the frozen table to claim incremental
speedup:

| Model | Historical staged evidence | Interpretation |
|---|---|---|
| SegFormer MiT-B0 | A0 9,313.73; C 6,352.36; C+E 5,489.21 ms | C and E both positive; rerun frozen A1--A3 |
| Whisper-Tiny | A0 113,415.80; C 108,246.36; C+E 71,512.90 ms | E is the dominant historical increment; rerun frozen A1--A3 |
| DINOv2-small | C+E 5,768.58; C+E+P 5,828.72 ms | P issued/completed 10,368 DMA operations but slightly regressed; useful negative check |
| DINOv2-small | C+E+P 5,825.25; C+E+P+R 5,487.22 ms | all 162 R windows held; the difference is not a causal R gain |

Swin and DeiT have frozen endpoint gains but no clean matched C/E/P intermediate
rows in the historical corpus; the formal A1--A3 rows above now provide those
measurements rather than inferring them from item7.

Authoritative raw logs, compiler artifacts, movement ledgers, UserDMA telemetry,
and sysMon summaries were moved to:

```text
nano:/home/huzq85/2-working/working_set/alps_component_ablation_20260830
```
