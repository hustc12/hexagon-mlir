# Debug runners (reduced topology)

These scripts override the parent harness `customize_*` / `load_*` hooks for
fast DSP smoke and Phase-4 A/B. **Do not use them for fair full-structure numbers.**

```bash
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 --enable-hexkl
python benchmark_models/debug_running/run_qwen2.5-0.5b_debug.py --seq-len 32 --enable-hexkl --enable-alps-vdae
```

Full published topology: `benchmark_models/run_*.py` (parent directory).

## One-command three-way matrix

Use the repository script rather than invoking the runners separately:

```bash
ANDROID_SERIAL=49d1c7b2 scripts/script_legacy/run_debug_matrix.sh \
  --seq-len 32 \
  --timeout 600 \
  --output-dir /tmp/alps-debug-matrix \
  qwen2.5-0.5b graphsage mamba-130m gpt2lmheadmodel
```

Each selected model runs in the fixed order HVX, HexKL, and HexKL + cumulative
Alps items 1–7.  `results.csv` retains every attempt and `summary.csv`
contains the latest statuses, timings, speedups, and compiler-mechanism hit
counts.  Re-running the same command skips successful rows and retries failed
ones; use `--force` to repeat successful rows.

The matrix also includes offline, deterministic cross-domain candidate screens:
`smollm2-135m` (Llama/GQA, 3Q/1KV), `swinv2-tiny` (hierarchical window
attention), and `ast-audioset` (spectrogram Transformer).  They use random FP16
weights and reduced topology, so they screen compiler/runtime structure rather
than pretrained-model accuracy.

The second candidate group adds `qwen2.5-coder-0.5b`,
`segformer-mit-b0`, and `whisper-tiny`.  Whisper includes a process-local tanh
GELU replacement for its hard-coded convolutional `F.gelu`, because exact GELU
otherwise leaves unsupported `math.erf` in the Hexagon lowering path.

The third structural-control group adds `opt-125m`, `deit-small`, and
`wav2vec2-base`.  OPT supplies explicit position IDs for export, while
Wav2Vec2 materializes its weight-normalized positional-convolution weight
before torch-mlir export.

The fourth candidate group adds DETR, BEiT, Speech2Text, HuBERT, WavLM, and
Data2Vec-Audio Debug runners.  Their first matrix intentionally retains
torch-mlir/parser/device failures as screening results rather than silently
dropping unsupported models.
