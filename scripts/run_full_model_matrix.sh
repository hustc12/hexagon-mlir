#!/usr/bin/env bash
# Serial full-model screening for the balanced OmniFetch 15-model set.
#
# This is intentionally distinct from run_debug_matrix.sh:
# - no layer/width/input reduction is allowed;
# - every model is run in HVX, HexKL, then HexKL+OmniFetch-items-1-7 order;
# - missing runners/checkpoints are recorded instead of silently substituted
#   with a Debug proxy.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_ROOT="${OMNIFETCH_RUNTIME_ROOT:-${ROOT}}"
OUT="${OMNIFETCH_RESULTS_DIR:-${ROOT}/benchmark_models/results/full_matrix_items_1_7}"
VENV="${OMNIFETCH_VENV:-/home/huzq85/2-working/hexagon_npu/mlir-env}"
SEQ_LEN="${OMNIFETCH_SEQ_LEN:-32}"
RUN_TIMEOUT="${OMNIFETCH_TIMEOUT:-3600}"
NO_TIMEOUT="${OMNIFETCH_NO_TIMEOUT:-0}"
DSP_HEAP_MB="${OMNIFETCH_DSP_HEAP_MB:-512}"
DEVICE_ITERATIONS="${OMNIFETCH_DEVICE_ITERATIONS:-1}"
FORCE=0
LIST_ONLY=0
NATIVE_ONLY=0
CONFIG_EXPLICIT=0

all_models=(
  falcon_rw_1b
  gpt2lmheadmodel
  qwen2.5-0.5b
  tinyllama
  sd_text_encoder
  swin_transformer
  segformer-mit-b0
  deit-small
  beit-base
  dinov2-small
  whisper-tiny
  hubert-base
  wav2vec2-base
  unispeech-base
  unispeech-sat-base
)
models=()
configs=(hvx hexkl hexkl_omnifetch_1_7)

runner_for() {
  case "$1" in
    falcon_rw_1b) printf '%s\n' benchmark_models/run_falcon_rw_1b.py ;;
    gpt2lmheadmodel) printf '%s\n' benchmark_models/run_gpt2lmheadmodel.py ;;
    qwen2.5-0.5b) printf '%s\n' benchmark_models/run_qwen2.5-0.5b.py ;;
    tinyllama) printf '%s\n' benchmark_models/run_tinyllama.py ;;
    sd_text_encoder) printf '%s\n' benchmark_models/run_sd_text_encoder.py ;;
    swin_transformer) printf '%s\n' benchmark_models/run_swin_transformer.py ;;
    segformer-mit-b0) printf '%s\n' benchmark_models/run_segformer-mit-b0.py ;;
    deit-small) printf '%s\n' benchmark_models/run_deit-small.py ;;
    beit-base) printf '%s\n' benchmark_models/run_beit-base.py ;;
    dinov2-small) printf '%s\n' benchmark_models/run_dinov2-small.py ;;
    whisper-tiny) printf '%s\n' benchmark_models/run_whisper-tiny.py ;;
    hubert-base) printf '%s\n' benchmark_models/run_hubert-base.py ;;
    wav2vec2-base) printf '%s\n' benchmark_models/run_wav2vec2-base.py ;;
    unispeech-base) printf '%s\n' benchmark_models/run_unispeech-base.py ;;
    unispeech-sat-base) printf '%s\n' benchmark_models/run_unispeech-sat-base.py ;;
  esac
}

domain_for() {
  case "$1" in
    falcon_rw_1b|gpt2lmheadmodel|qwen2.5-0.5b|tinyllama|sd_text_encoder)
      printf '%s\n' language_text ;;
    swin_transformer|segformer-mit-b0|deit-small|beit-base|dinov2-small)
      printf '%s\n' vision ;;
    *) printf '%s\n' speech_audio ;;
  esac
}

weight_status_for() {
  case "$1" in
    gpt2lmheadmodel|qwen2.5-0.5b) printf '%s\n' cached_checkpoint ;;
    swin_transformer|sd_text_encoder) printf '%s\n' random_full_structure ;;
    falcon_rw_1b|tinyllama) printf '%s\n' checkpoint_missing ;;
    segformer-mit-b0|deit-small|beit-base|dinov2-small|whisper-tiny|hubert-base|wav2vec2-base|unispeech-base|unispeech-sat-base)
      printf '%s\n' random_full_structure
      ;;
    *) printf '%s\n' runner_pending ;;
  esac
}

usage() {
  cat <<EOF
Usage: scripts/run_full_model_matrix.sh [options] [model ...]

Options:
  --list            Print the ordered 15-model implementation plan and exit
  --seq-len N       Full LLM prefill length (default: ${SEQ_LEN})
  --timeout SEC     Per model/configuration timeout (default: ${RUN_TIMEOUT})
  --no-timeout      Do not impose a host-side deadline (recommended for full graphs)
  --dsp-heap-mb N   QuRT heap reservation for full graphs (default: ${DSP_HEAP_MB})
  --device-iterations N
                    Serial in-process measured calls (screening default: ${DEVICE_ITERATIONS})
  --config NAME     Run hvx, hexkl, hexkl_omnifetch_1_7, or hvx_kv_prefetch
  --native-only     Run only strict upstream HVX and HexKL configurations
  --output-dir DIR  Result directory (default: ${OUT})
  --runtime-root DIR
                    Built hexagon-mlir runtime tree (default: source tree)
  --force           Rerun configurations already recorded as PASS
  -h, --help        Show this help

With no model arguments all 15 models are attempted, strictly serially.
EOF
}

is_known_model() {
  local candidate=$1 known
  for known in "${all_models[@]}"; do
    [[ "${candidate}" == "${known}" ]] && return 0
  done
  return 1
}

while (($#)); do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --native-only) NATIVE_ONLY=1; configs=(hvx hexkl); shift ;;
    --seq-len) SEQ_LEN=$2; shift 2 ;;
    --timeout) RUN_TIMEOUT=$2; shift 2 ;;
    --no-timeout) NO_TIMEOUT=1; shift ;;
    --dsp-heap-mb) DSP_HEAP_MB=$2; shift 2 ;;
    --device-iterations) DEVICE_ITERATIONS=$2; shift 2 ;;
    --config)
      case "$2" in
        hvx|hexkl|hexkl_omnifetch_1_7|hvx_kv_prefetch)
          configs=("$2")
          CONFIG_EXPLICIT=1
          ;;
        *) echo "ERROR: invalid configuration: $2" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --output-dir) OUT=$2; shift 2 ;;
    --runtime-root) RUNTIME_ROOT=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) models+=("$1"); shift ;;
  esac
done

if [[ "${NATIVE_ONLY}" -eq 1 && "${CONFIG_EXPLICIT}" -eq 0 ]]; then
  configs=(hvx hexkl)
fi
if [[ "${NATIVE_ONLY}" -eq 1 && "${CONFIG_EXPLICIT}" -eq 1 ]]; then
  case "${configs[0]}" in
    hvx|hexkl) ;;
    *)
      echo "ERROR: --native-only only permits an explicit hvx or hexkl config" >&2
      exit 2
      ;;
  esac
fi

[[ "${SEQ_LEN}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: sequence length must be a positive integer" >&2
  exit 2
}
[[ "${RUN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: timeout must be a positive integer" >&2
  exit 2
}
[[ "${NO_TIMEOUT}" == 0 || "${NO_TIMEOUT}" == 1 ]] || {
  echo "ERROR: OMNIFETCH_NO_TIMEOUT must be 0 or 1" >&2
  exit 2
}
[[ "${DSP_HEAP_MB}" =~ ^[1-9][0-9]*$ ]] && ((DSP_HEAP_MB < 1024)) || {
  echo "ERROR: DSP heap must be an integer between 1 and 1023 MiB" >&2
  exit 2
}
[[ "${DEVICE_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: device iterations must be a positive integer" >&2
  exit 2
}

if ((${#models[@]} == 0)); then
  models=("${all_models[@]}")
fi
for model in "${models[@]}"; do
  is_known_model "${model}" || {
    echo "ERROR: unknown model: ${model}" >&2
    exit 2
  }
done

print_plan() {
  printf '%-3s %-24s %-15s %-11s %s\n' \
    '#' model domain runner weight_status
  local i=0 model runner state
  for model in "${all_models[@]}"; do
    i=$((i + 1))
    runner=$(runner_for "${model}")
    if [[ -f "${ROOT}/${runner}" ]]; then state=READY; else state=PENDING; fi
    printf '%-3s %-24s %-15s %-11s %s\n' \
      "${i}" "${model}" "$(domain_for "${model}")" "${state}" \
      "$(weight_status_for "${model}")"
  done
}

if [[ "${LIST_ONLY}" -eq 1 ]]; then
  print_plan
  exit 0
fi

mkdir -p "${OUT}"
source "${VENV}/bin/activate"
export HEXAGON_MLIR_ROOT="${RUNTIME_ROOT}"
export TRITON_ROOT="${RUNTIME_ROOT}/triton"
export TRITON_HOME="${RUNTIME_ROOT}"
export TRITON_PLUGIN_DIRS="${RUNTIME_ROOT}/triton_shared;${RUNTIME_ROOT}/qcom_hexagon_backend"
PYTHON_VERSION="$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
export TRITON_BUILD_DIR="${TRITON_BUILD_DIR:-${RUNTIME_ROOT}/triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}}"
export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_OPT_PATH:-${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}"
export PATH="${TRITON_BUILD_DIR}/third_party/qcom_hexagon_backend/bin:${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export PYTHONPATH="${RUNTIME_ROOT}/triton/python"
export HOST_TOOLCHAIN="${HOST_TOOLCHAIN:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}"
export HEXAGON_TOOLS="${HEXAGON_TOOLS:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools}"
export HEXKL_ROOT="${HEXKL_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXKL_DIR/hexkl_addon}"
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH_VERSION:-73}"
export ANDROID_HOST="${ANDROID_HOST:-}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-49d1c7b2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMNIFETCH_DSP_HEAP_MB="${DSP_HEAP_MB}"
export PYTHONUNBUFFERED=1
# Full graphs can run for tens of minutes.  Never repeat a failed device run;
# preserve the first failure for diagnosis.  A known FastRPC teardown quirk may
# return AEE_EBADSTATE (13) after main() has written all outputs and perf.txt;
# in that exact case the runner pulls the files and the model correctness gate
# still decides whether the row passes.
export HEXAGON_RUN_RETRIES="${HEXAGON_RUN_RETRIES:-1}"
export HEXAGON_TOLERATE_TEARDOWN_FAULT="${HEXAGON_TOLERATE_TEARDOWN_FAULT:-1}"
if [[ "${NATIVE_ONLY}" -eq 1 ]]; then
  export HEXAGON_BASELINE_MODE=upstream-strict
fi

CSV="${OUT}/results.csv"
CSV_HEADER='model,domain,weight_status,config,attempt,status,perf_us,perf_ms,seq_len,dsp_heap_mb,device_iterations,compile_s,prefetch_sites,in_situ_ops,async_choices,persistent_choices,vtcm_saved_bytes,kv_prefetch_sites,log'
if [[ -f "${CSV}" ]] && [[ "$(head -n 1 "${CSV}")" != "${CSV_HEADER}" ]]; then
  CSV="${OUT}/results_v2.csv"
fi
if [[ ! -f "${CSV}" ]]; then
  printf '%s\n' \
    "${CSV_HEADER}" >"${CSV}"
fi

last_status() {
  awk -F, -v m="$1" -v c="$2" \
    'NR>1 && $1==m && $4==c {s=$6} END {print s}' "${CSV}"
}

next_attempt() {
  awk -F, -v m="$1" -v c="$2" \
    'NR>1 && $1==m && $4==c {n++} END {print n+1}' "${CSV}"
}

base_args_for() {
  case "$1" in
    falcon_rw_1b|gpt2lmheadmodel|qwen2.5-0.5b|tinyllama)
      printf '%s\n' --seq-len "${SEQ_LEN}"
      ;;
  esac
  case "$1" in
    falcon_rw_1b|gpt2lmheadmodel|qwen2.5-0.5b|tinyllama|sd_text_encoder)
      printf '%s\n' --enable-hvx-vector
      ;;
    *)
      printf '%s\n' --backend-profile hvx-vector
      ;;
  esac
  printf '%s\n' --device-iterations "${DEVICE_ITERATIONS}"
}

config_args_for() {
  case "$1" in
    hvx) ;;
    hexkl) printf '%s\n' --enable-hexkl ;;
    hexkl_omnifetch_1_7)
      printf '%s\n' --enable-hexkl --enable-omnifetch-items-1-7
      ;;
    hvx_kv_prefetch)
      printf '%s\n' --enable-omnifetch-kv-cache-prefetch
      ;;
  esac
}

run_one() {
  local model=$1 config=$2 runner attempt status log perf_us perf_p50_us perf_ms recorded_seq
  local compile_s prefetch_sites in_situ_ops async_choices persistent_choices
  local vtcm_saved_bytes kv_prefetch_sites
  local -a args=()
  runner="${ROOT}/$(runner_for "${model}")"
  attempt=$(next_attempt "${model}" "${config}")
  log="${OUT}/${model}_${config}_attempt${attempt}.log"

  if [[ ! -f "${runner}" ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${model}" "$(domain_for "${model}")" "$(weight_status_for "${model}")" \
      "${config}" "${attempt}" NOT_IMPLEMENTED NA NA NA "${DSP_HEAP_MB}" \
      "${DEVICE_ITERATIONS}" NA 0 0 0 0 0 0 \
      "${log}" >>"${CSV}"
    echo "NOT_IMPLEMENTED ${model}: ${runner}"
    return
  fi
  if [[ "${FORCE}" -eq 0 && "$(last_status "${model}" "${config}")" == PASS ]]; then
    echo "SKIP ${model} ${config}: PASS already recorded"
    return
  fi

  mapfile -t args < <(base_args_for "${model}"; config_args_for "${config}")
  echo "START ${model} ${config} attempt=${attempt} $(date --iso-8601=seconds)"
  if [[ "${NO_TIMEOUT}" -eq 1 ]]; then
    if python "${runner}" "${args[@]}" >"${log}" 2>&1; then
      status=PASS
    else
      status="FAIL_$?"
    fi
  else
    if timeout --foreground "${RUN_TIMEOUT}" \
        python "${runner}" "${args[@]}" >"${log}" 2>&1; then
      status=PASS
    else
      status="FAIL_$?"
    fi
  fi
  perf_us=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  # The persistent-WH wrapper records the measured warm invocation in
  # PerfP50.  For a one-sample full-model screen this is also the only timer
  # that is consistent with cold_us + warm + invalidated_us and the enclosing
  # ADB wall time.  Some long V73 runs return a corrupted outer warm_avg timer,
  # so do not use that duplicate for the combination row.
  if [[ "${config}" == hexkl_omnifetch_1_7 ]] && \
      grep -q '^OmniFetchWHCache:' "${log}"; then
    perf_p50_us=$(awk -F: '/^[[:space:]]*PerfP50:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
    if [[ -n "${perf_p50_us}" ]]; then
      if [[ -n "${perf_us}" && "${perf_us}" != "${perf_p50_us}" ]]; then
        echo "METRIC_OVERRIDE ${model} ${config}: Perf=${perf_us}us PerfP50=${perf_p50_us}us (persistent warm sample)"
      fi
      perf_us=${perf_p50_us}
    fi
  fi
  if [[ -n "${perf_us}" ]]; then
    perf_ms=$(awk -v us="${perf_us}" 'BEGIN{printf "%.6f",us/1000.0}')
  else
    perf_us=NA
    perf_ms=NA
  fi
  case "${model}" in
    falcon_rw_1b|gpt2lmheadmodel|qwen2.5-0.5b|tinyllama) recorded_seq=${SEQ_LEN} ;;
    *) recorded_seq=NA ;;
  esac
  compile_s=$(awk '/Compilation from initial MLIR to .so took/{v=$(NF-1)}END{print v}' "${log}")
  prefetch_sites=$(awk -F': ' '/\[PrefetchInsert\] Total prefetch sites:/{s+=$NF}END{print s+0}' "${log}")
  in_situ_ops=$(awk '/\[LayoutOpsElimination\] Found prefetch_in_situ operation:/{n++}END{print n+0}' "${log}")
  async_choices=$(awk '/TransformCostModel/{
    for(i=1;i<=NF;i++)if(index($i,"async=")==1){v=$i;sub("async=","",v);gsub(/[^0-9].*/,"",v);s+=v}
  }END{print s+0}' "${log}")
  persistent_choices=$(awk '/TransformCostModel/{
    for(i=1;i<=NF;i++)if(index($i,"persistent=")==1){v=$i;sub("persistent=","",v);gsub(/[^0-9].*/,"",v);s+=v}
  }END{print s+0}' "${log}")
  vtcm_saved_bytes=$(awk '/VTCMLifetimeColoring/{
    for(i=1;i<=NF;i++)if(index($i,"saved_peak=")==1){v=$i;sub("saved_peak=","",v);gsub(/[^0-9].*/,"",v);s+=v}
  }END{print s+0}' "${log}")
  kv_prefetch_sites=$(awk '/KVCachePrefetch/{
    for(i=1;i<=NF;i++)if(index($i,"sites=")==1){v=$i;sub("sites=","",v);gsub(/[^0-9].*/,"",v);s+=v}
  }END{print s+0}' "${log}")
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "$(domain_for "${model}")" "$(weight_status_for "${model}")" \
    "${config}" "${attempt}" "${status}" "${perf_us}" "${perf_ms}" \
    "${recorded_seq}" "${DSP_HEAP_MB}" "${DEVICE_ITERATIONS}" \
    "${compile_s:-NA}" "${prefetch_sites}" "${in_situ_ops}" "${async_choices}" \
    "${persistent_choices}" "${vtcm_saved_bytes}" "${kv_prefetch_sites}" \
    "${log}" >>"${CSV}"
  echo "DONE ${model} ${config} status=${status} perf_ms=${perf_ms} log=${log}"
}

write_summary() {
  local summary="${OUT}/summary.csv"
  printf '%s\n' \
    'model,hvx_status,hvx_ms,hexkl_status,hexkl_ms,combo_status,combo_ms,hexkl_over_combo' \
    >"${summary}"
  local model hs hm ks km cs cm speedup
  for model in "${all_models[@]}"; do
    hs=$(awk -F, -v m="${model}" '$1==m&&$4=="hvx"{v=$6}END{print v}' "${CSV}")
    hm=$(awk -F, -v m="${model}" '$1==m&&$4=="hvx"{v=$8}END{print v}' "${CSV}")
    ks=$(awk -F, -v m="${model}" '$1==m&&$4=="hexkl"{v=$6}END{print v}' "${CSV}")
    km=$(awk -F, -v m="${model}" '$1==m&&$4=="hexkl"{v=$8}END{print v}' "${CSV}")
    cs=$(awk -F, -v m="${model}" '$1==m&&$4=="hexkl_omnifetch_1_7"{v=$6}END{print v}' "${CSV}")
    cm=$(awk -F, -v m="${model}" '$1==m&&$4=="hexkl_omnifetch_1_7"{v=$8}END{print v}' "${CSV}")
    [[ -n "${hs}${ks}${cs}" ]] || continue
    speedup=NA
    if [[ "${ks}" == PASS && "${cs}" == PASS && "${km}" != NA && "${cm}" != NA ]]; then
      speedup=$(awk -v a="${km}" -v b="${cm}" 'BEGIN{if(b>0)printf "%.4f",a/b;else print "NA"}')
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${model}" "${hs:-NA}" "${hm:-NA}" "${ks:-NA}" "${km:-NA}" \
      "${cs:-NA}" "${cm:-NA}" "${speedup}" >>"${summary}"
  done
  echo "SUMMARY_COMPLETE csv=${summary}"
}

if [[ "${NO_TIMEOUT}" -eq 1 ]]; then
  timeout_description=disabled
else
  timeout_description="${RUN_TIMEOUT}s"
fi
echo "FULL_MATRIX_START models=${models[*]} seq_len=${SEQ_LEN} timeout=${timeout_description} dsp_heap_mb=${DSP_HEAP_MB} device_iterations=${DEVICE_ITERATIONS}"
print_plan
for model in "${models[@]}"; do
  for config in "${configs[@]}"; do
    run_one "${model}" "${config}"
  done
done
write_summary
echo "FULL_MATRIX_COMPLETE csv=${CSV}"
