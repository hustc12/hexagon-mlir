#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUT="${OMNIFETCH_RESULTS_DIR:-${ROOT}/benchmark_models/results/debug_matrix}"
VENV="${OMNIFETCH_VENV:-/home/huzq85/2-working/hexagon_npu/mlir-env}"

mkdir -p "${OUT}"
source "${VENV}/bin/activate"

export HEXAGON_MLIR_ROOT="${ROOT}"
export TRITON_ROOT="${ROOT}/triton"
export TRITON_HOME="${ROOT}"
export TRITON_PLUGIN_DIRS="${ROOT}/triton_shared;${ROOT}/qcom_hexagon_backend"
export TRITON_BUILD_DIR="${TRITON_BUILD_DIR:-${ROOT}/triton-build}"
export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_OPT_PATH:-${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}"
export PATH="${TRITON_BUILD_DIR}/third_party/qcom_hexagon_backend/bin:${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export PYTHONPATH="${ROOT}/triton/python"
export HOST_TOOLCHAIN="${HOST_TOOLCHAIN:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}"
export HEXAGON_TOOLS="${HEXAGON_TOOLS:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools}"
export HEXKL_ROOT="${HEXKL_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXKL_DIR/hexkl_addon}"
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH_VERSION:-75}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-49d1c7b2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

CSV="${OUT}/results.csv"
if [[ ! -f "${CSV}" ]]; then
  printf 'model,config,status,perf_us,perf_ms,actual_args,log\n' >"${CSV}"
fi

models=(
  falcon_rw_1b
  gpt2lmheadmodel
  graphsage
  mamba-130m
  qwen2.5-0.5b
  real-esrgan
  sd_text_encoder
  sd_unet
  sd_vae_decoder
  swin_transformer
  tinyllama
  vit
)

base_args_for() {
  case "$1" in
    falcon_rw_1b)
      printf '%s\n' '--seq-len' '128' '--device-iterations' '3'
      ;;
    gpt2lmheadmodel|qwen2.5-0.5b|tinyllama)
      printf '%s\n' '--seq-len' '128'
      ;;
  esac
}

config_args_for() {
  case "$1" in
    hvx) ;;
    hexkl) printf '%s\n' '--enable-hexkl' ;;
    hexkl_omnifetch_1_7)
      printf '%s\n' '--enable-hexkl' '--enable-omnifetch-items-1-7'
      ;;
  esac
}

already_recorded() {
  local model=$1
  local config=$2
  awk -F, -v model="${model}" -v config="${config}" \
    'NR > 1 && $1 == model && $2 == config { found=1 } END { exit !found }' \
    "${CSV}"
}

run_one() {
  local model=$1
  local config=$2
  local runner="${ROOT}/benchmark_models/debug_running/run_${model}_debug.py"
  local log="${OUT}/${model}_${config}.log"
  local -a args=()
  local status perf_us perf_ms arg_text

  if already_recorded "${model}" "${config}"; then
    echo "SKIP ${model} ${config}: already recorded"
    return
  fi

  mapfile -t args < <(
    base_args_for "${model}"
    config_args_for "${config}"
  )
  printf -v arg_text '%q ' "${args[@]}"

  echo "START ${model} ${config} $(date --iso-8601=seconds)"
  if timeout --foreground "${OMNIFETCH_TIMEOUT:-900}" \
      python "${runner}" "${args[@]}" >"${log}" 2>&1; then
    status=PASS
  else
    status="FAIL_$?"
  fi

  perf_us=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/, "", $2); value=$2} END{print value}' "${log}")
  if [[ -n "${perf_us}" ]]; then
    perf_ms=$(awk -v us="${perf_us}" 'BEGIN { printf "%.6f", us / 1000.0 }')
  else
    perf_us=NA
    perf_ms=NA
  fi

  printf '%s,%s,%s,%s,%s,"%s",%s\n' \
    "${model}" "${config}" "${status}" "${perf_us}" "${perf_ms}" \
    "${arg_text}" "${log}" >>"${CSV}"
  echo "DONE ${model} ${config} status=${status} perf_ms=${perf_ms}"
}

cd "${ROOT}"
for model in "${models[@]}"; do
  for config in hvx hexkl hexkl_omnifetch_1_7; do
    run_one "${model}" "${config}"
  done
done

echo "MATRIX_COMPLETE csv=${CSV}"
