#!/usr/bin/env bash
#
# Model-level HexKL/OmniFetch ablation. This intentionally does not run GEMM
# microbenchmarks. Logs are kept per configuration for later parsing.
#
set -euo pipefail

usage() {
  echo "Usage: $0 [--model falcon-full|gpt2-full|falcon-debug|gpt2-debug]"
  echo "          [--seq-len N]"
  echo "          [--device-iterations N] [--timeout SEC] [--serial SERIAL]"
  echo "          [--output-dir DIR]"
  echo "          [--m1-only]"
  echo "          [--include-experimental]"
}

MODEL="falcon-debug"
SEQ_LEN=128
RUN_TIMEOUT=300
DEVICE_ITERATIONS=3
DEVICE_SERIAL="${ANDROID_SERIAL:-49d1c7b2}"
OUTPUT_DIR=""
INCLUDE_EXPERIMENTAL=0
M1_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --device-iterations) DEVICE_ITERATIONS="$2"; shift 2 ;;
    --timeout) RUN_TIMEOUT="$2"; shift 2 ;;
    --serial) DEVICE_SERIAL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --m1-only) M1_ONLY=1; shift ;;
    --include-experimental) INCLUDE_EXPERIMENTAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "${MODEL}" in
  falcon-full)
    RUNNER="benchmark_models/run_falcon_rw_1b.py"
    ;;
  gpt2-full)
    RUNNER="benchmark_models/run_gpt2lmheadmodel.py"
    ;;
  falcon-debug)
    RUNNER="benchmark_models/debug_running/run_falcon_rw_1b_debug.py"
    ;;
  gpt2-debug)
    RUNNER="benchmark_models/debug_running/run_gpt2lmheadmodel_debug.py"
    ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    usage
    exit 2
    ;;
esac

REPO_DIR="$(git rev-parse --show-toplevel)"
PARENT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
VENV_DIR="${CONDA_ENV:-${PARENT_DIR}/mlir-env}"
[[ -f "${VENV_DIR}/bin/activate" ]] || {
  echo "Python environment not found: ${VENV_DIR}" >&2
  exit 1
}

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_DIR}/benchmark_models/results/omnifetch_model_ablation"
fi
mkdir -p "${OUTPUT_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
export ANDROID_SERIAL="${DEVICE_SERIAL}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

run_case() {
  local name="$1"
  shift
  local log="${OUTPUT_DIR}/${MODEL}_seq${SEQ_LEN}_${name}.log"
  local iteration_args=()
  if [[ "${MODEL}" == "falcon-debug" || "${MODEL}" == "falcon-full" ]]; then
    iteration_args=(--device-iterations "${DEVICE_ITERATIONS}")
  fi
  echo "=== ${name}: ${MODEL}, seq_len=${SEQ_LEN} ==="
  if timeout --foreground "${RUN_TIMEOUT}" \
      python "${RUNNER}" --seq-len "${SEQ_LEN}" \
      "${iteration_args[@]}" "$@" \
      2>&1 | tee "${log}"; then
    echo "RESULT ${name}=PASS log=${log}"
  else
    local status="${PIPESTATUS[0]}"
    echo "RESULT ${name}=FAIL status=${status} log=${log}" >&2
    return "${status}"
  fi
}

# Mandatory primary rows. Keep this order stable in logs and reports.
run_case hvx
run_case hexkl --enable-hexkl
CUMULATIVE_ARGS=(--enable-hexkl --enable-omnifetch-vdae)
if [[ "${M1_ONLY}" -eq 0 ]] &&
   [[ "${MODEL}" == "falcon-debug" || "${MODEL}" == "falcon-full" ]]; then
  CUMULATIVE_ARGS+=(--enable-omnifetch-persistent-wh-cache)
  CUMULATIVE_ARGS+=(--enable-omnifetch-two-dim-pipeline)
  CUMULATIVE_ARGS+=(--enable-omnifetch-vtcm-coloring)
  CUMULATIVE_ARGS+=(--enable-omnifetch-kv-cache-prefetch)
fi
run_case hexkl_omnifetch_cumulative "${CUMULATIVE_ARGS[@]}"

# Negative or not-yet-gated mechanisms never enter the cumulative row.
if [[ "${INCLUDE_EXPERIMENTAL}" -eq 1 ]]; then
  run_case experimental_reshape_reuse \
    --enable-hexkl \
    --enable-omnifetch-vdae \
    --enable-omnifetch-weight-prepack \
    --enable-hexkl-persistent-vtcm
  if [[ "${MODEL}" == "falcon-debug" ]]; then
    run_case experimental_dequant_reshape \
      "${CUMULATIVE_ARGS[@]}" \
      --enable-omnifetch-dequant-reshape
  fi
fi
