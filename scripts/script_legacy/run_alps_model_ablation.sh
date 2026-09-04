#!/usr/bin/env bash
#
# Model-level HexKL/Alps ablation. This intentionally does not run GEMM
# microbenchmarks. Logs are kept per configuration for later parsing.
#
set -euo pipefail

usage() {
  echo "Usage: $0 [--model falcon-full|gpt2-full|falcon-debug|gpt2-debug]"
  echo "          [--seq-len N]"
  echo "          [--device-iterations N] [--timeout SEC] [--serial SERIAL]"
  echo "          [--output-dir DIR]"
  echo "          [--m1-only]"
  echo "          [--cumulative-only|--hvx-item7-only|--hvx-n1-only|--hvx-n2-only]"
  echo "          [--enable-hvx-vector]"
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
CUMULATIVE_ONLY=0
ENABLE_HVX_VECTOR=0
HVX_ITEM7_ONLY=0
HVX_N1_ONLY=0
HVX_N2_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --device-iterations) DEVICE_ITERATIONS="$2"; shift 2 ;;
    --timeout) RUN_TIMEOUT="$2"; shift 2 ;;
    --serial) DEVICE_SERIAL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --m1-only) M1_ONLY=1; shift ;;
    --cumulative-only) CUMULATIVE_ONLY=1; shift ;;
    --hvx-item7-only) HVX_ITEM7_ONLY=1; shift ;;
    --hvx-n1-only) HVX_N1_ONLY=1; shift ;;
    --hvx-n2-only) HVX_N2_ONLY=1; shift ;;
    --enable-hvx-vector) ENABLE_HVX_VECTOR=1; shift ;;
    --include-experimental) INCLUDE_EXPERIMENTAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

exclusive_modes=$((CUMULATIVE_ONLY + HVX_ITEM7_ONLY + HVX_N1_ONLY + HVX_N2_ONLY))
if [[ "${exclusive_modes}" -gt 1 ]]; then
  echo "Only one of --cumulative-only, --hvx-item7-only, --hvx-n1-only and --hvx-n2-only may be used" >&2
  exit 2
fi

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
  OUTPUT_DIR="${REPO_DIR}/benchmark_models/results/alps_model_ablation"
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

if [[ "${HVX_ITEM7_ONLY}" -eq 1 ]]; then
  if [[ "${MODEL}" != "falcon-debug" && "${MODEL}" != "falcon-full" ]]; then
    echo "--hvx-item7-only currently requires a Falcon runner" >&2
    exit 2
  fi
  run_case hvx_vector_item7 \
    --enable-hvx-vector \
    --enable-alps-kv-cache-prefetch \
    --disable-layout-aware --disable-alps-adaptive
  exit 0
fi

if [[ "${HVX_N2_ONLY}" -eq 1 ]]; then
  if [[ "${MODEL}" != "falcon-debug" && "${MODEL}" != "falcon-full" ]]; then
    echo "--hvx-n2-only currently requires a Falcon runner" >&2
    exit 2
  fi
  run_case hvx_vector_n2 \
    --enable-hvx-vector \
    --enable-alps-activation-multicast
  exit 0
fi

if [[ "${HVX_N1_ONLY}" -eq 1 ]]; then
  if [[ "${MODEL}" != "falcon-debug" && "${MODEL}" != "falcon-full" ]]; then
    echo "--hvx-n1-only currently requires a Falcon runner" >&2
    exit 2
  fi
  run_case hvx_vector_n1 \
    --enable-hvx-vector \
    --enable-alps-weight-stationary
  exit 0
fi

CUMULATIVE_ARGS=(--enable-hexkl --enable-alps-vdae)
if [[ "${ENABLE_HVX_VECTOR}" -eq 1 ]]; then
  CUMULATIVE_ARGS+=(--enable-hvx-vector)
fi
if [[ "${M1_ONLY}" -eq 0 ]] &&
   [[ "${MODEL}" == "falcon-debug" || "${MODEL}" == "falcon-full" ]]; then
  CUMULATIVE_ARGS+=(--enable-alps-persistent-wh-cache)
  CUMULATIVE_ARGS+=(--enable-alps-two-dim-pipeline)
  CUMULATIVE_ARGS+=(--enable-alps-vtcm-coloring)
  CUMULATIVE_ARGS+=(--enable-alps-kv-cache-prefetch)
fi

# Mandatory primary rows. Keep this order stable in logs and reports.  The
# cumulative-only mode is useful when matched baselines already exist.
if [[ "${CUMULATIVE_ONLY}" -eq 0 ]]; then
  HVX_ARGS=()
  HEXKL_ARGS=(--enable-hexkl)
  if [[ "${ENABLE_HVX_VECTOR}" -eq 1 ]]; then
    HVX_ARGS+=(--enable-hvx-vector)
    HEXKL_ARGS+=(--enable-hvx-vector)
  fi
  run_case hvx "${HVX_ARGS[@]}"
  run_case hexkl "${HEXKL_ARGS[@]}"
fi
run_case hexkl_alps_cumulative "${CUMULATIVE_ARGS[@]}"

# Negative or not-yet-gated mechanisms never enter the cumulative row.
if [[ "${INCLUDE_EXPERIMENTAL}" -eq 1 ]]; then
  run_case experimental_reshape_reuse \
    --enable-hexkl \
    --enable-alps-vdae \
    --enable-alps-weight-prepack \
    --enable-hexkl-persistent-vtcm
  if [[ "${MODEL}" == "falcon-debug" ]]; then
    run_case experimental_dequant_reshape \
      "${CUMULATIVE_ARGS[@]}" \
      --enable-alps-dequant-reshape
  fi
fi
