#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${OMNIFETCH_STAGED_RUN_ROOT:-/tmp/omnifetch-native-v73-staged}

mkdir -p "${RUN_ROOT}"

echo "[1/5] full GPT-2"
"${SCRIPT_DIR}/run_gpt2_layered_probe.sh" \
  --output-dir "${RUN_ROOT}/gpt2-full"

echo "[2/5] full SD/CLIP text encoder"
"${SCRIPT_DIR}/run_clip_layered_probe.sh" \
  --output-dir "${RUN_ROOT}/sd-clip-text-full"

echo "[3/5] full Qwen2.5-0.5B"
"${SCRIPT_DIR}/run_qwen_layered_probe.sh" \
  --output-dir "${RUN_ROOT}/qwen2.5-0.5b-full"

echo "[4/5] full TinyLlama-1.1B"
"${SCRIPT_DIR}/run_tinyllama_layered_probe.sh" \
  --output-dir "${RUN_ROOT}/tinyllama-1.1b-full"

echo "[5/5] Falcon-RW-1B-4L cropped with final-LayerNorm host fallback"
"${SCRIPT_DIR}/run_falcon_layered_probe.sh" \
  --effective-layers 4 \
  --split-head \
  --output-dir "${RUN_ROOT}/falcon-rw-1b-4l-cropped"

echo "All staged model runs completed serially: ${RUN_ROOT}"
