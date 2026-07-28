#!/usr/bin/env bash
# Run the Debug model matrix with one reproducible entry point.
#
# Every selected model is run in exactly these three configurations:
#   1. HVX
#   2. HexKL
#   3. HexKL + cumulative OmniFetch items 1-7
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_ROOT="${OMNIFETCH_RUNTIME_ROOT:-${ROOT}}"
OUT="${OMNIFETCH_RESULTS_DIR:-${ROOT}/benchmark_models/results/debug_matrix_items_1_7}"
VENV="${OMNIFETCH_VENV:-/home/huzq85/2-working/hexagon_npu/mlir-env}"
SEQ_LEN="${OMNIFETCH_SEQ_LEN:-32}"
RUN_TIMEOUT="${OMNIFETCH_TIMEOUT:-600}"
REAL_ESRGAN_INPUT_SIZE="${OMNIFETCH_REAL_ESRGAN_INPUT_SIZE:-8}"
FORCE=0

all_models=(
  falcon_rw_1b
  gpt2lmheadmodel
  graphsage
  mamba-130m
  qwen2.5-0.5b
  qwen2.5-coder-0.5b
  real-esrgan
  sd_text_encoder
  sd_unet
  sd_vae_decoder
  segformer-mit-b0
  smollm2-135m
  swin_transformer
  swinv2-tiny
  tinyllama
  vit
  ast-audioset
  whisper-tiny
  opt-125m
  deit-small
  wav2vec2-base
  detr-resnet-50
  beit-base
  speech2text-small
  hubert-base
  wavlm-base-plus
  data2vec-audio-base
  speecht5-asr
  clap-htsat
)
models=()

usage() {
  cat <<EOF
Usage: scripts/run_debug_matrix.sh [options] [model ...]

Run HVX, HexKL, and HexKL+OmniFetch-items-1-7 for every selected Debug model.
With no model arguments, all Debug models are selected.

Options:
  --seq-len N       Sequence length for every sequence model (default: ${SEQ_LEN})
  --timeout SEC     Per model/configuration timeout (default: ${RUN_TIMEOUT})
  --output-dir DIR  Results directory (default: ${OUT})
  --runtime-root DIR
                    Tree containing a previously built triton-build and Triton
                    Python extension (default: source tree)
  --real-esrgan-input-size N
                    Debug Real-ESRGAN spatial input (default: ${REAL_ESRGAN_INPUT_SIZE})
  --force           Rerun PASS rows too; failed rows are retried by default
  -h, --help        Show this help

Models:
  ${all_models[*]}
EOF
}

is_known_model() {
  local candidate=$1
  local known
  for known in "${all_models[@]}"; do
    [[ "${candidate}" == "${known}" ]] && return 0
  done
  return 1
}

while (($#)); do
  case "$1" in
    --seq-len)
      [[ $# -ge 2 ]] || { echo "ERROR: --seq-len needs a value" >&2; exit 2; }
      SEQ_LEN=$2
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || { echo "ERROR: --timeout needs a value" >&2; exit 2; }
      RUN_TIMEOUT=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --output-dir needs a value" >&2; exit 2; }
      OUT=$2
      shift 2
      ;;
    --runtime-root)
      [[ $# -ge 2 ]] || { echo "ERROR: --runtime-root needs a value" >&2; exit 2; }
      RUNTIME_ROOT=$2
      shift 2
      ;;
    --real-esrgan-input-size)
      [[ $# -ge 2 ]] || {
        echo "ERROR: --real-esrgan-input-size needs a value" >&2
        exit 2
      }
      REAL_ESRGAN_INPUT_SIZE=$2
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        models+=("$1")
        shift
      done
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      models+=("$1")
      shift
      ;;
  esac
done

[[ "${SEQ_LEN}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: sequence length must be a positive integer" >&2
  exit 2
}
[[ "${RUN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: timeout must be a positive integer" >&2
  exit 2
}
[[ "${REAL_ESRGAN_INPUT_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: Real-ESRGAN input size must be a positive integer" >&2
  exit 2
}

if ((${#models[@]} == 0)); then
  models=("${all_models[@]}")
fi
for model in "${models[@]}"; do
  is_known_model "${model}" || {
    echo "ERROR: unknown model: ${model}" >&2
    usage >&2
    exit 2
  }
done

mkdir -p "${OUT}"
source "${VENV}/bin/activate"

export HEXAGON_MLIR_ROOT="${ROOT}"
export TRITON_ROOT="${RUNTIME_ROOT}/triton"
export TRITON_HOME="${RUNTIME_ROOT}"
export TRITON_PLUGIN_DIRS="${RUNTIME_ROOT}/triton_shared;${RUNTIME_ROOT}/qcom_hexagon_backend"
export TRITON_BUILD_DIR="${TRITON_BUILD_DIR:-${RUNTIME_ROOT}/triton-build}"
export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_OPT_PATH:-${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}"
export PATH="${TRITON_BUILD_DIR}/third_party/qcom_hexagon_backend/bin:${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export PYTHONPATH="${RUNTIME_ROOT}/triton/python"
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
CSV_HEADER='model,config,attempt,status,perf_us,perf_ms,seq_len,prefetch_sites,in_situ_ops,async_choices,persistent_choices,vtcm_saved_bytes,kv_prefetch_sites,eager_kv_inferred,actual_args,log'
if [[ -f "${CSV}" ]] && [[ "$(head -n 1 "${CSV}")" != "${CSV_HEADER}" ]]; then
  CSV="${OUT}/results_v2.csv"
fi
if [[ ! -f "${CSV}" ]]; then
  printf '%s\n' "${CSV_HEADER}" >"${CSV}"
fi

base_args_for() {
  case "$1" in
    falcon_rw_1b)
      printf '%s\n' '--seq-len' "${SEQ_LEN}" '--device-iterations' '3'
      ;;
    gpt2lmheadmodel|graphsage|mamba-130m|opt-125m|qwen2.5-0.5b|qwen2.5-coder-0.5b|smollm2-135m|tinyllama|whisper-tiny)
      printf '%s\n' '--seq-len' "${SEQ_LEN}"
      ;;
    real-esrgan)
      printf '%s\n' '--input-size' "${REAL_ESRGAN_INPUT_SIZE}"
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

last_status() {
  local model=$1
  local config=$2
  awk -F, -v model="${model}" -v config="${config}" \
    'NR > 1 && $1 == model && $2 == config { status=$4 } END { print status }' \
    "${CSV}"
}

next_attempt() {
  local model=$1
  local config=$2
  awk -F, -v model="${model}" -v config="${config}" \
    'NR > 1 && $1 == model && $2 == config { count++ } END { print count + 1 }' \
    "${CSV}"
}

sum_assignment() {
  local key=$1
  local pattern=$2
  local log=$3
  awk -v key="${key}" -v pattern="${pattern}" '
    index($0, pattern) {
      count=split($0, fields, /[[:space:]]+/)
      for (i=1; i<=count; ++i) {
        if (index(fields[i], key "=") == 1) {
          value=fields[i]
          sub("^" key "=", "", value)
          gsub(/[^0-9].*$/, "", value)
          sum += value + 0
        }
      }
    }
    END { print sum + 0 }
  ' "${log}"
}

sum_prefetch_sites() {
  local log=$1
  awk -F': ' '/\[PrefetchInsert\] Total prefetch sites:/ { sum += $NF } END { print sum + 0 }' \
    "${log}"
}

run_one() {
  local model=$1
  local config=$2
  local runner="${ROOT}/benchmark_models/debug_running/run_${model}_debug.py"
  local attempt status perf_us perf_ms arg_text recorded_seq_len
  local prefetch_sites in_situ_ops async_choices persistent_choices
  local vtcm_saved_bytes kv_prefetch_sites eager_kv_inferred
  local -a args=()

  status=$(last_status "${model}" "${config}")
  if [[ "${FORCE}" -eq 0 && "${status}" == PASS ]]; then
    echo "SKIP ${model} ${config}: PASS already recorded"
    return
  fi

  attempt=$(next_attempt "${model}" "${config}")
  local log="${OUT}/${model}_${config}_attempt${attempt}.log"
  mapfile -t args < <(
    base_args_for "${model}"
    config_args_for "${config}"
  )
  if ((${#args[@]})); then
    printf -v arg_text '%q ' "${args[@]}"
  else
    arg_text=""
  fi
  case "${model}" in
    falcon_rw_1b|gpt2lmheadmodel|graphsage|mamba-130m|opt-125m|qwen2.5-0.5b|qwen2.5-coder-0.5b|smollm2-135m|tinyllama|whisper-tiny)
      recorded_seq_len=${SEQ_LEN}
      ;;
    *)
      recorded_seq_len=NA
      ;;
  esac

  echo "START ${model} ${config} attempt=${attempt} $(date --iso-8601=seconds)"
  if timeout --foreground "${RUN_TIMEOUT}" \
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

  prefetch_sites=$(sum_prefetch_sites "${log}")
  in_situ_ops=$(awk '/\[LayoutOpsElimination\] Found prefetch_in_situ operation:/ { count++ } END { print count + 0 }' "${log}")
  async_choices=$(sum_assignment async TransformCostModel "${log}")
  persistent_choices=$(sum_assignment persistent TransformCostModel "${log}")
  vtcm_saved_bytes=$(sum_assignment saved_peak VTCMLifetimeColoring "${log}")
  kv_prefetch_sites=$(sum_assignment sites KVCachePrefetch "${log}")
  eager_kv_inferred=$(sum_assignment eager_inferred KVCacheMetadata "${log}")

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s\n' \
    "${model}" "${config}" "${attempt}" "${status}" "${perf_us}" "${perf_ms}" \
    "${recorded_seq_len}" "${prefetch_sites}" "${in_situ_ops}" "${async_choices}" \
    "${persistent_choices}" "${vtcm_saved_bytes}" "${kv_prefetch_sites}" \
    "${eager_kv_inferred}" "${arg_text}" "${log}" >>"${CSV}"
  echo "DONE ${model} ${config} status=${status} perf_ms=${perf_ms} log=${log}"
}

last_result_field() {
  local model=$1
  local config=$2
  local field=$3
  awk -F, -v model="${model}" -v config="${config}" -v field="${field}" \
    'NR > 1 && $1 == model && $2 == config { value=$field } END { print value }' \
    "${CSV}"
}

speedup() {
  local numerator_status=$1
  local numerator_ms=$2
  local denominator_status=$3
  local denominator_ms=$4
  if [[ "${numerator_status}" == PASS && "${denominator_status}" == PASS ]] &&
      [[ "${numerator_ms}" != NA && "${denominator_ms}" != NA ]]; then
    awk -v numerator="${numerator_ms}" -v denominator="${denominator_ms}" \
      'BEGIN { if (denominator > 0) printf "%.4f", numerator / denominator; else print "NA" }'
  else
    printf 'NA\n'
  fi
}

write_summary() {
  local summary="${OUT}/summary.csv"
  local model hvx_status hexkl_status combo_status
  local hvx_ms hexkl_ms combo_ms hvx_over_hexkl hexkl_over_combo hvx_over_combo
  local prefetch_sites in_situ_ops async_choices persistent_choices
  local vtcm_saved_bytes kv_prefetch_sites eager_kv_inferred

  printf '%s\n' \
    'model,hvx_status,hvx_ms,hexkl_status,hexkl_ms,combo_status,combo_ms,hvx_over_hexkl,hexkl_over_combo,hvx_over_combo,prefetch_sites,in_situ_ops,async_choices,persistent_choices,vtcm_saved_bytes,kv_prefetch_sites,eager_kv_inferred' \
    >"${summary}"
  for model in "${all_models[@]}"; do
    hvx_status=$(last_result_field "${model}" hvx 4)
    hexkl_status=$(last_result_field "${model}" hexkl 4)
    combo_status=$(last_result_field "${model}" hexkl_omnifetch_1_7 4)
    [[ -n "${hvx_status}${hexkl_status}${combo_status}" ]] || continue
    hvx_ms=$(last_result_field "${model}" hvx 6)
    hexkl_ms=$(last_result_field "${model}" hexkl 6)
    combo_ms=$(last_result_field "${model}" hexkl_omnifetch_1_7 6)
    hvx_over_hexkl=$(speedup "${hvx_status}" "${hvx_ms}" "${hexkl_status}" "${hexkl_ms}")
    hexkl_over_combo=$(speedup "${hexkl_status}" "${hexkl_ms}" "${combo_status}" "${combo_ms}")
    hvx_over_combo=$(speedup "${hvx_status}" "${hvx_ms}" "${combo_status}" "${combo_ms}")
    prefetch_sites=$(last_result_field "${model}" hexkl_omnifetch_1_7 8)
    in_situ_ops=$(last_result_field "${model}" hexkl_omnifetch_1_7 9)
    async_choices=$(last_result_field "${model}" hexkl_omnifetch_1_7 10)
    persistent_choices=$(last_result_field "${model}" hexkl_omnifetch_1_7 11)
    vtcm_saved_bytes=$(last_result_field "${model}" hexkl_omnifetch_1_7 12)
    kv_prefetch_sites=$(last_result_field "${model}" hexkl_omnifetch_1_7 13)
    eager_kv_inferred=$(last_result_field "${model}" hexkl_omnifetch_1_7 14)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${model}" "${hvx_status:-NA}" "${hvx_ms:-NA}" \
      "${hexkl_status:-NA}" "${hexkl_ms:-NA}" \
      "${combo_status:-NA}" "${combo_ms:-NA}" \
      "${hvx_over_hexkl}" "${hexkl_over_combo}" "${hvx_over_combo}" \
      "${prefetch_sites:-0}" "${in_situ_ops:-0}" "${async_choices:-0}" \
      "${persistent_choices:-0}" "${vtcm_saved_bytes:-0}" \
      "${kv_prefetch_sites:-0}" "${eager_kv_inferred:-0}" >>"${summary}"
  done
  echo "SUMMARY_COMPLETE csv=${summary}"
}

cd "${ROOT}"
echo "MATRIX_START models=${models[*]} seq_len=${SEQ_LEN} timeout=${RUN_TIMEOUT}s"
echo "SOURCE_ROOT=${ROOT}"
echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "RESULTS=${OUT}"
for model in "${models[@]}"; do
  for config in hvx hexkl hexkl_omnifetch_1_7; do
    run_one "${model}" "${config}"
  done
done

write_summary
echo "MATRIX_COMPLETE csv=${CSV}"
