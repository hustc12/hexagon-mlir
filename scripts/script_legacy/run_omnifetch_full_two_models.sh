#!/usr/bin/env bash
# Run full DINOv2-small and ViT-Base with the current OmniFetch item-7-only
# policy, then join
# those rows with an already validated Prefetch-Kernel-HX/APT-GET-HX CSV.
# Device cases are strictly serial and have no timeout.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
parent_dir=$(cd -- "${repo_root}/.." && pwd)
venv=${OMNIFETCH_VENV:-${parent_dir}/mlir-env}
python_version=$("${venv}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
triton_build_dir=${TRITON_BUILD_DIR:-${repo_root}/triton/build/cmake.linux-x86_64-cpython-${python_version}}
output_dir=${OUTPUT_DIR:-${parent_dir}/run_artifacts/full_omnifetch_vs_prefetch_$(date +%Y%m%d_%H%M%S)}
baseline_csv=${BASELINE_CSV:-${parent_dir}/run_artifacts/full_prefetch_baselines_20260813_valid/results.csv}
iterations=${DEVICE_ITERATIONS:-1}
remote_results_dir=${REMOTE_RESULTS_DIR:-}
reuse_valid_logs=${REUSE_VALID_LOGS:-1}

[[ -f "${baseline_csv}" ]] || {
  echo "Missing validated baseline CSV: ${baseline_csv}" >&2
  exit 2
}

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
export OMNIFETCH_DSP_HEAP_MB=${OMNIFETCH_DSP_HEAP_MB:-512}

mkdir -p "${output_dir}"
printf 'model,scheme,status,perf_us,p50_us,p90_us,min_us,static_sites,issued,busy_suppressed,page_clipped,requested_bytes,issued_bytes,correctness\n' > "${output_dir}/omnifetch_results.csv"

sync_remote() {
  [[ -n "${remote_results_dir}" ]] || return 0
  ssh nano "mkdir -p '${remote_results_dir}'"
  rsync -a --partial "${output_dir}/" "nano:${remote_results_dir}/"
}

record_result() {
  local model=$1 log=$2
  local perf p50 p90 min sites issued busy clipped requested issued_bytes correctness
  perf=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p50=$(awk -F: '/^[[:space:]]*PerfP50:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p90=$(awk -F: '/^[[:space:]]*PerfP90:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  min=$(awk -F: '/^[[:space:]]*PerfMin:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  sites=$(awk '/KVCachePrefetch/{for(i=1;i<=NF;i++)if(index($i,"sites=")==1){v=$i;sub("sites=","",v);gsub(/[^0-9].*/,"",v)}}END{print v+0}' "${log}")
  issued=$(awk 'match($0,/issued=[0-9]+/){v=substr($0,RSTART+7,RLENGTH-7)}END{print v}' "${log}")
  busy=$(awk 'match($0,/busy_suppressed=[0-9]+/){v=substr($0,RSTART+16,RLENGTH-16)}END{print v}' "${log}")
  clipped=$(awk 'match($0,/page_clipped=[0-9]+/){v=substr($0,RSTART+13,RLENGTH-13)}END{print v}' "${log}")
  requested=$(awk 'match($0,/requested_bytes=[0-9]+/){v=substr($0,RSTART+16,RLENGTH-16)}END{print v}' "${log}")
  issued_bytes=$(awk 'match($0,/issued_bytes=[0-9]+/){v=substr($0,RSTART+13,RLENGTH-13)}END{print v}' "${log}")
  correctness=$(awk '/\[Compare\]/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  # Older wrappers may omit the duplicate Test_Info Perf line; never discard
  # an otherwise complete device result for that reason.
  [[ -n "${perf}" ]] || perf=${p50}
  if [[ -z "${perf}" || -z "${correctness}" ]]; then
    echo "Invalid OmniFetch result in ${log}: perf=${perf:-missing} correctness=${correctness:-missing}" >&2
    return 1
  fi
  printf '%s,omnifetch-item7-only,pass,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "${perf}" "${p50}" "${p90}" "${min}" "${sites}" \
    "${issued}" "${busy}" "${clipped}" "${requested}" "${issued_bytes}" \
    "${correctness}" >> "${output_dir}/omnifetch_results.csv"
}

run_model() {
  local model=$1 runner=$2
  local model_dir="${output_dir}/${model}"
  local log="${model_dir}/omnifetch-item7-only.log"
  mkdir -p "${model_dir}/artifacts"
  if [[ "${reuse_valid_logs}" == 1 && -f "${log}" ]] && \
      grep -q '^[[:space:]]*PerfP50:' "${log}" && \
      grep -q '^\[Compare\].*finite=True.*top1_match=True' "${log}"; then
    echo "[SerialRun] reuse-valid model=${model} scheme=omnifetch-item7-only"
    record_result "${model}" "${log}"
    sync_remote
    return 0
  fi
  echo "[SerialRun] model=${model} scheme=omnifetch-item7-only"
  HEXAGON_MLIR_DUMP_DIR="${model_dir}/artifacts" \
    "${venv}/bin/python" "${runner}" \
      --enable-hexkl --backend-profile hvx-vector \
      --enable-omnifetch-kv-cache-prefetch \
      --disable-layout-aware --disable-omnifetch-adaptive \
      --device-iterations "${iterations}" >"${log}" 2>&1
  record_result "${model}" "${log}"
  sync_remote
}

run_model dinov2-small "${repo_root}/benchmark_models/run_dinov2-small.py"
run_model vit-base "${repo_root}/benchmark_models/run_vit.py"

"${venv}/bin/python" - "${baseline_csv}" "${output_dir}/omnifetch_results.csv" "${output_dir}/comparison.csv" <<'PY'
import csv
import sys

baseline_path, omni_path, output_path = sys.argv[1:]
rows = []
with open(baseline_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rows.append({
            "model": row["model"],
            "scheme": row["scheme"],
            "perf_us": row["perf_us"],
            "issued": row["issued"],
            "requested_bytes": row["requested_bytes"],
            "issued_bytes": row["issued_bytes"],
            "correctness": row["correctness"],
        })
with open(omni_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rows.append({key: row[key] for key in (
            "model", "scheme", "perf_us", "issued", "requested_bytes",
            "issued_bytes", "correctness")})

for model in {row["model"] for row in rows}:
    omni = next(float(row["perf_us"]) for row in rows
                if row["model"] == model and row["scheme"] == "omnifetch-item7-only")
    for row in rows:
        if row["model"] == model:
            scheme_latency = float(row["perf_us"])
            # Values greater than one unambiguously mean that OmniFetch is
            # slower than the row's scheme.  Keep the reciprocal ratio too so
            # downstream plotting never has to guess the numerator.
            row["omnifetch_slowdown_vs_scheme"] = f'{omni / scheme_latency:.6f}'
            row["scheme_latency_over_omnifetch"] = f'{scheme_latency / omni:.6f}'

fields = ["model", "scheme", "perf_us", "omnifetch_slowdown_vs_scheme",
          "scheme_latency_over_omnifetch", "issued", "requested_bytes",
          "issued_bytes", "correctness"]
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

sync_remote
echo "[SerialRun] complete: ${output_dir}/comparison.csv"
