#!/usr/bin/env bash
# Full-model causal ablation for Alps with item 4 disabled.
# DINOv2-small and ViT-Base are executed strictly serially with no timeout.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
parent_dir=$(cd -- "${repo_root}/.." && pwd)
venv=${ALPS_VENV:-${parent_dir}/mlir-env}
python_version=$("${venv}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
triton_build_dir=${TRITON_BUILD_DIR:-${repo_root}/triton/build/cmake.linux-x86_64-cpython-${python_version}}
output_dir=${OUTPUT_DIR:-${parent_dir}/run_artifacts/full_alps_no_item4_$(date +%Y%m%d_%H%M%S)}
iterations=${DEVICE_ITERATIONS:-1}
remote_results_dir=${REMOTE_RESULTS_DIR:-}
reuse_valid_logs=${REUSE_VALID_LOGS:-1}
only_models=${ONLY_MODELS:-}
only_schemes=${ONLY_SCHEMES:-}

export PYTHONPATH="${repo_root}/triton/python:${repo_root}/benchmark_models"
export TRITON_PLUGIN_DIRS="${repo_root}/triton_shared;${repo_root}/qcom_hexagon_backend"
export PATH="${triton_build_dir}/third_party/qcom_hexagon_backend/bin:${triton_build_dir}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export HEXAGON_MLIR_ROOT=${HEXAGON_MLIR_ROOT:-${repo_root}}
export HEXAGON_ARCH_VERSION=${HEXAGON_ARCH_VERSION:-73}
export ANDROID_SERIAL=${ANDROID_SERIAL:-49d1c7b2}
export ANDROID_HOST=${ANDROID_HOST:-}
export HOST_TOOLCHAIN=${HOST_TOOLCHAIN:-${parent_dir}/HOST_TOOLCHAIN}
export HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-${parent_dir}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}
export HEXAGON_TOOLS=${HEXAGON_TOOLS:-${parent_dir}/HEXAGON_TOOLS/Tools}
export HEXKL_ROOT=${HEXKL_ROOT:-${parent_dir}/HEXKL_DIR/hexkl_addon}
export HEXAGON_RUNTIME_LIBS_DIR=${HEXAGON_RUNTIME_LIBS_DIR:-${triton_build_dir}/third_party/qcom_hexagon_backend/bin/runtime}
export ALPS_DSP_HEAP_MB=${ALPS_DSP_HEAP_MB:-512}

mkdir -p "${output_dir}"
printf 'model,scheme,status,perf_us,p50_us,static_kv_sites,issued,busy_suppressed,page_clipped,requested_bytes,issued_bytes,budget_suppressed,duplicate_suppressed,correctness,log\n' > "${output_dir}/results.csv"

sync_remote() {
  [[ -n "${remote_results_dir}" ]] || return 0
  ssh nano "mkdir -p '${remote_results_dir}'"
  rsync -a --partial "${output_dir}/" "nano:${remote_results_dir}/"
}

metric() {
  local pattern=$1 log=$2 offset=$3
  awk -v p="${pattern}" -v o="${offset}" 'match($0,p){v=substr($0,RSTART+o,RLENGTH-o)}END{print v}' "${log}"
}

record_result() {
  local model=$1 scheme=$2 log=$3
  local perf p50 kv_sites issued busy clipped requested issued_bytes budget duplicate correctness
  perf=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p50=$(awk -F: '/^[[:space:]]*PerfP50:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  kv_sites=$(awk '/KVCachePrefetch/{for(i=1;i<=NF;i++)if(index($i,"sites=")==1){v=$i;sub("sites=","",v);gsub(/[^0-9].*/,"",v)}}END{print v+0}' "${log}")
  issued=$(metric 'issued=[0-9]+' "${log}" 7)
  busy=$(metric 'busy_suppressed=[0-9]+' "${log}" 16)
  clipped=$(metric 'page_clipped=[0-9]+' "${log}" 13)
  requested=$(metric 'requested_bytes=[0-9]+' "${log}" 16)
  issued_bytes=$(metric 'issued_bytes=[0-9]+' "${log}" 13)
  budget=$(metric 'budget_suppressed=[0-9]+' "${log}" 18)
  duplicate=$(metric 'duplicate_suppressed=[0-9]+' "${log}" 21)
  correctness=$(awk '/\[Compare\]/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  [[ -n "${perf}" ]] || perf=${p50}
  if [[ -z "${perf}" || -z "${correctness}" ]]; then
    echo "Invalid result: model=${model} scheme=${scheme} log=${log}" >&2
    return 1
  fi
  printf '%s,%s,pass,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "${scheme}" "${perf}" "${p50}" "${kv_sites}" \
    "${issued:-0}" "${busy:-0}" "${clipped:-0}" "${requested:-0}" \
    "${issued_bytes:-0}" "${budget:-0}" "${duplicate:-0}" \
    "${correctness}" "${log}" >> "${output_dir}/results.csv"
}

run_case() {
  local model=$1 runner=$2 scheme=$3
  shift 3
  if [[ -n "${only_models}" && " ${only_models} " != *" ${model} "* ]]; then
    return 0
  fi
  if [[ -n "${only_schemes}" && " ${only_schemes} " != *" ${scheme} "* ]]; then
    return 0
  fi
  local case_dir="${output_dir}/${model}/${scheme}"
  local log="${case_dir}/run.log"
  mkdir -p "${case_dir}/artifacts"
  if [[ "${reuse_valid_logs}" == 1 && -f "${log}" ]] &&
      grep -q '^[[:space:]]*Perf' "${log}" &&
      grep -q '^\[Compare\].*finite=True.*top1_match=True' "${log}"; then
    echo "[SerialAblation] reuse model=${model} scheme=${scheme}"
  else
    echo "[SerialAblation] run model=${model} scheme=${scheme}"
    HEXAGON_MLIR_DUMP_DIR="${case_dir}/artifacts" \
      "${venv}/bin/python" "${runner}" \
        --backend-profile hvx-vector --device-iterations "${iterations}" \
        "$@" >"${log}" 2>&1
  fi
  record_result "${model}" "${scheme}" "${log}"
  sync_remote
}

run_model() {
  local model=$1 runner=$2
  run_case "${model}" "${runner}" hvx
  run_case "${model}" "${runner}" hexkl-control \
    --enable-hexkl
  run_case "${model}" "${runner}" item7-only \
    --enable-hexkl --enable-alps-kv-cache-prefetch \
    --disable-layout-aware --disable-alps-adaptive
  run_case "${model}" "${runner}" items1-3 \
    --enable-hexkl --alps-items-through 3
  run_case "${model}" "${runner}" items1-5-no4 \
    --enable-hexkl --alps-items-through 5 \
    --disable-alps-persistent-wh-cache
  run_case "${model}" "${runner}" items1-6-no4 \
    --enable-hexkl --alps-items-through 6 \
    --disable-alps-persistent-wh-cache
  run_case "${model}" "${runner}" items1-7-no4 \
    --enable-hexkl --alps-items-through 7 \
    --disable-alps-persistent-wh-cache
}

run_model dinov2-small "${repo_root}/benchmark_models/run_dinov2-small.py"
run_model vit-base "${repo_root}/benchmark_models/run_vit.py"

sync_remote
echo "[SerialAblation] complete: ${output_dir}/results.csv"
