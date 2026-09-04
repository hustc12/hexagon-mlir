#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
OUT="${ALPS_RESULTS_DIR:-${ROOT}/benchmark_models/results/debug_matrix_full_1_7}"
VENV="${ALPS_VENV:-/home/huzq85/2-working/hexagon_npu/mlir-env}"

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
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH_VERSION:-73}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-49d1c7b2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

CSV="${OUT}/results.csv"
if [[ ! -f "${CSV}" ]]; then
  printf '%s\n' \
    'model,status,perf_us,perf_ms,eager_kv,kv_sites,kv_hints,vtcm_sites,cost_native,cost_sync,cost_async,cost_persistent,layout_sites,layout_reusable,log' \
    >"${CSV}"
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

last_field() {
  local pattern=$1
  local key=$2
  local log=$3
  awk -v pattern="${pattern}" -v key="${key}" '
    index($0, pattern) {
      for (i = 1; i <= NF; ++i) {
        split($i, pair, "=")
        if (pair[1] == key) value = pair[2]
      }
    }
    END {
      gsub(/[^0-9-]/, "", value)
      print value == "" ? 0 : value
    }
  ' "${log}"
}

already_recorded() {
  local model=$1
  awk -F, -v model="${model}" \
    'NR > 1 && $1 == model { found=1 } END { exit !found }' "${CSV}"
}

run_one() {
  local model=$1
  local runner="${ROOT}/benchmark_models/debug_running/run_${model}_debug.py"
  local log="${OUT}/${model}_hexkl_alps_items_1_7.log"
  local -a args=()
  local status perf_us perf_ms
  local eager_kv kv_sites kv_hints vtcm_sites
  local cost_native cost_sync cost_async cost_persistent
  local layout_sites layout_reusable

  if already_recorded "${model}"; then
    echo "SKIP ${model}: already recorded"
    return
  fi

  mapfile -t args < <(base_args_for "${model}")
  args+=(--enable-hexkl --enable-alps-items-1-7)

  echo "START ${model} $(date --iso-8601=seconds)"
  if timeout --foreground "${ALPS_TIMEOUT:-900}" \
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

  eager_kv=$(last_field '[KVCacheMetadata]' eager_inferred "${log}")
  kv_sites=$(last_field '[KVCachePrefetch]' sites "${log}")
  kv_hints=$(last_field '[KVCachePrefetch]' hints "${log}")
  vtcm_sites=$(last_field '[VTCMLifetimeColoring]' sites "${log}")
  cost_native=$(last_field '[TransformCostModel]' native "${log}")
  cost_sync=$(last_field '[TransformCostModel]' sync "${log}")
  cost_async=$(last_field '[TransformCostModel]' async "${log}")
  cost_persistent=$(last_field '[TransformCostModel]' persistent "${log}")
  layout_sites=$(last_field '[LayoutValueAnalysis]' sites "${log}")
  layout_reusable=$(last_field '[LayoutValueAnalysis]' reusable_sites "${log}")

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "${status}" "${perf_us}" "${perf_ms}" \
    "${eager_kv}" "${kv_sites}" "${kv_hints}" "${vtcm_sites}" \
    "${cost_native}" "${cost_sync}" "${cost_async}" "${cost_persistent}" \
    "${layout_sites}" "${layout_reusable}" "${log}" >>"${CSV}"
  echo "DONE ${model} status=${status} perf_ms=${perf_ms} kv=${kv_sites} vtcm=${vtcm_sites} async=${cost_async} persistent=${cost_persistent}"
}

cd "${ROOT}"
if (($#)); then
  for model in "$@"; do
    run_one "${model}"
  done
else
  for model in "${models[@]}"; do
    run_one "${model}"
  done
fi

echo "MATRIX_COMPLETE csv=${CSV}"
