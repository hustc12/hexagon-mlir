#!/usr/bin/env bash
# Run full DINOv2-small and ViT-Base with the two isolated prefetch baselines.
#
# The four device cases are strictly serial and have no timeout.  Each model's
# Prefetch-Kernel-HX run discovers the complete graph's admitted stable IDs;
# the matched APT-GET-HX run consumes exactly that explicit allowlist.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
parent_dir=$(cd -- "${repo_root}/.." && pwd)
venv=${ALPS_VENV:-${parent_dir}/mlir-env}
python_version=$("${venv}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
triton_build_dir=${TRITON_BUILD_DIR:-${repo_root}/triton/build/cmake.linux-x86_64-cpython-${python_version}}
output_dir=${OUTPUT_DIR:-${parent_dir}/run_artifacts/full_prefetch_baselines_$(date +%Y%m%d_%H%M%S)}
iterations=${DEVICE_ITERATIONS:-1}
remote_results_dir=${REMOTE_RESULTS_DIR:-}

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
printf 'model,scheme,status,perf_us,p50_us,p90_us,min_us,hints,issued,requested_bytes,issued_bytes,correctness\n' > "${output_dir}/results.csv"

sync_remote() {
  [[ -n "${remote_results_dir}" ]] || return 0
  ssh nano "mkdir -p '${remote_results_dir}'"
  rsync -a --partial "${output_dir}/" "nano:${remote_results_dir}/"
}

extract_admitted_ids() {
  local log=$1
  "${venv}/bin/python" - "${log}" <<'PY'
import re
import sys

ids = []
seen = set()
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = re.search(r"\[prefetch-kernel-hx\].*?admitted_ids=(.*?)\s*$", line)
    if not match or match.group(1) == "none":
        continue
    for candidate in (item.strip() for item in match.group(1).split(",")):
        if candidate and candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
print(",".join(ids))
PY
}

record_result() {
  local model=$1
  local scheme=$2
  local log=$3
  local perf p50 p90 min hints issued requested issued_bytes correctness
  perf=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p50=$(awk -F: '/^[[:space:]]*PerfP50:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p90=$(awk -F: '/^[[:space:]]*PerfP90:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  min=$(awk -F: '/^[[:space:]]*PerfMin:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  hints=$(awk 'match($0,/hints=[0-9]+/){v=substr($0,RSTART+6,RLENGTH-6)}END{print v}' "${log}")
  issued=$(awk 'match($0,/issued=[0-9]+/){v=substr($0,RSTART+7,RLENGTH-7)}END{print v}' "${log}")
  requested=$(awk 'match($0,/requested_bytes=[0-9]+/){v=substr($0,RSTART+16,RLENGTH-16)}END{print v}' "${log}")
  issued_bytes=$(awk 'match($0,/issued_bytes=[0-9]+/){v=substr($0,RSTART+13,RLENGTH-13)}END{print v}' "${log}")
  correctness=$(awk '/\[Compare\]/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  if [[ -z "${perf}" || -z "${hints}" || "${hints}" == 0 || -z "${issued}" || "${issued}" == 0 ]]; then
    echo "Invalid baseline result in ${log}: perf=${perf:-missing} hints=${hints:-missing} issued=${issued:-missing}" >&2
    return 1
  fi
  printf '%s,%s,pass,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "${scheme}" "${perf}" "${p50}" "${p90}" "${min}" \
    "${hints}" "${issued}" "${requested}" "${issued_bytes}" "${correctness}" \
    >> "${output_dir}/results.csv"
}

run_model() {
  local model=$1
  local runner=$2
  local model_dir="${output_dir}/${model}"
  mkdir -p "${model_dir}"

  local pk_log="${model_dir}/prefetch-kernel-hx.log"
  mkdir -p "${model_dir}/prefetch-kernel-hx-artifacts"
  echo "[SerialRun] model=${model} scheme=prefetch-kernel-hx"
  HEXAGON_MLIR_DUMP_DIR="${model_dir}/prefetch-kernel-hx-artifacts" \
    "${venv}/bin/python" "${runner}" \
      --enable-hexkl --backend-profile hvx-vector \
      --prefetch-baseline prefetch-kernel-hx \
      --prefetch-baseline-distance 1 --device-iterations "${iterations}" \
      >"${pk_log}" 2>&1
  record_result "${model}" prefetch-kernel-hx "${pk_log}"
  sync_remote

  local candidate_ids
  candidate_ids=$(extract_admitted_ids "${pk_log}")
  if [[ -z "${candidate_ids}" ]]; then
    echo "No admitted full-model candidate IDs found in ${pk_log}" >&2
    return 1
  fi
  printf '%s\n' "${candidate_ids}" > "${model_dir}/apt-get-hx-candidate-ids.txt"

  local apt_log="${model_dir}/apt-get-hx.log"
  mkdir -p "${model_dir}/apt-get-hx-artifacts"
  echo "[SerialRun] model=${model} scheme=apt-get-hx"
  HEXAGON_MLIR_DUMP_DIR="${model_dir}/apt-get-hx-artifacts" \
    "${venv}/bin/python" "${runner}" \
      --enable-hexkl --backend-profile hvx-vector \
      --prefetch-baseline apt-get-hx --prefetch-baseline-distance 1 \
      --apt-get-hx-manual-candidate-ids "${candidate_ids}" \
      --device-iterations "${iterations}" >"${apt_log}" 2>&1
  record_result "${model}" apt-get-hx "${apt_log}"
  sync_remote
}

# Do not background these calls.  One complete model/configuration owns the
# compiler and phone until it has passed before the next starts.
run_model dinov2-small "${repo_root}/benchmark_models/run_dinov2-small.py"
run_model vit-base "${repo_root}/benchmark_models/run_vit.py"

echo "[SerialRun] complete: ${output_dir}/results.csv"
