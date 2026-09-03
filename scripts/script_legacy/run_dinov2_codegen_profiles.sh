#!/usr/bin/env bash
# Strictly serial DINOv2 Debug screening for repaired backend configurations.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "${script_dir}/../.." && pwd)
runtime_root=${OMNIFETCH_RUNTIME_ROOT:-${project_root}}
output_dir=${OUTPUT_DIR:-/tmp/omnifetch-dinov2-codegen-profiles}
venv=${OMNIFETCH_VENV:-/home/huzq85/2-working/hexagon_npu/mlir-env}
run_timeout=${OMNIFETCH_TIMEOUT:-600}
device_iterations=${OMNIFETCH_DEVICE_ITERATIONS:-1}
selected_cases=${OMNIFETCH_CASES:-all}
model_variant=${OMNIFETCH_DINOV2_VARIANT:-debug}
case "${model_variant}" in
  debug)
    runner="${project_root}/benchmark_models/debug_running/run_dinov2-small_debug.py"
    ;;
  full)
    runner="${project_root}/benchmark_models/run_dinov2-small.py"
    ;;
  *)
    echo "ERROR: OMNIFETCH_DINOV2_VARIANT must be debug or full" >&2
    exit 2
    ;;
esac

[[ "${run_timeout}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: OMNIFETCH_TIMEOUT must be a positive integer" >&2
  exit 2
}
[[ "${device_iterations}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: OMNIFETCH_DEVICE_ITERATIONS must be a positive integer" >&2
  exit 2
}

mkdir -p "${output_dir}"
source "${venv}/bin/activate"

export HEXAGON_MLIR_ROOT="${project_root}"
export TRITON_ROOT="${runtime_root}/triton"
export TRITON_HOME="${runtime_root}"
export TRITON_PLUGIN_DIRS="${runtime_root}/triton_shared;${runtime_root}/qcom_hexagon_backend"
export TRITON_BUILD_DIR="${TRITON_BUILD_DIR:-${runtime_root}/triton-build}"
export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_OPT_PATH:-${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}"
export PATH="${TRITON_BUILD_DIR}/third_party/qcom_hexagon_backend/bin:${TRITON_BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export PYTHONPATH="${runtime_root}/triton/python"
export HOST_TOOLCHAIN="${HOST_TOOLCHAIN:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}"
export HEXAGON_TOOLS="${HEXAGON_TOOLS:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools}"
export HEXKL_ROOT="${HEXKL_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXKL_DIR/hexkl_addon}"
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH_VERSION:-73}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-49d1c7b2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

results_csv="${output_dir}/results.csv"
audit_csv="${output_dir}/codegen_audit.csv"
printf '%s\n' \
  'config,status,perf_us,perf_ms,finite,max_abs_diff,top1_match,artifact_dir,log' \
  >"${results_csv}"

case_selected() {
  local name=$1
  [[ "${selected_cases}" == all ]] ||
    [[ " ${selected_cases//,/ } " == *" ${name} "* ]]
}

run_case() {
  local config=$1
  shift
  local log="${output_dir}/${config}.log"
  local status=PASS
  local perf_us=NA
  local perf_ms=NA
  local finite=NA
  local max_abs_diff=NA
  local top1_match=NA
  local artifact_dir=NA

  echo "START ${config} $(date --iso-8601=seconds)"
  if timeout --foreground "${run_timeout}" \
      python "${runner}" --device-iterations "${device_iterations}" "$@" \
      >"${log}" 2>&1; then
    status=PASS
  else
    rc=$?
    status="FAIL_${rc}"
  fi

  perf_us=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  if [[ -n "${perf_us}" ]]; then
    perf_ms=$(awk -v us="${perf_us}" 'BEGIN { printf "%.6f", us / 1000.0 }')
  else
    perf_us=NA
  fi
  finite=$(sed -n 's/.*\[Compare\].*finite=\([^ ]*\).*/\1/p' "${log}" | tail -1)
  max_abs_diff=$(sed -n 's/.*max_abs_diff=\([^ ]*\).*/\1/p' "${log}" | tail -1)
  top1_match=$(sed -n 's/.*top1_match=\([^ ]*\).*/\1/p' "${log}" | tail -1)
  finite=${finite:-NA}
  max_abs_diff=${max_abs_diff:-NA}
  top1_match=${top1_match:-NA}
  artifact_dir=$(sed -n "s/.*Folder '\\([^']*\\)' created.*/\\/tmp\\/\\1/p" "${log}" | tail -1)
  artifact_dir=${artifact_dir:-NA}

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${config}" "${status}" "${perf_us}" "${perf_ms}" "${finite}" \
    "${max_abs_diff}" "${top1_match}" "${artifact_dir}" "${log}" \
    >>"${results_csv}"

  if [[ "${artifact_dir}" != NA && -d "${artifact_dir}" ]]; then
    if ! "${script_dir}/audit_hexagon_codegen.sh" \
        "${artifact_dir}" "${audit_csv}" \
        >"${output_dir}/${config}_audit.txt" 2>&1; then
      echo "WARN ${config}: static codegen audit unavailable" >&2
    fi
  fi
  echo "DONE ${config} status=${status} perf_ms=${perf_ms}"
}

# The order is deliberate and strictly serial.
case_selected legacy_scalar &&
  run_case legacy_scalar --backend-profile legacy-scalar
case_selected hvx_vector &&
  run_case hvx_vector --backend-profile hvx-vector
case_selected hvx_vector_lwp &&
  run_case hvx_vector_lwp --backend-profile hvx-vector --enable-lwp
case_selected hvx_vector_vtcm &&
  run_case hvx_vector_vtcm --backend-profile hvx-vector-vtcm
case_selected hexkl_vector_vtcm &&
  run_case hexkl_vector_vtcm \
    --backend-profile hvx-vector-vtcm --enable-hexkl
case_selected hexkl_prefetch_vdae_vector &&
  run_case hexkl_prefetch_vdae_vector \
    --backend-profile hvx-vector --enable-hexkl --enable-omnifetch-vdae
case_selected hexkl_items_4_vector &&
  run_case hexkl_items_4_vector \
    --backend-profile hvx-vector --enable-hexkl --omnifetch-items-through 4
case_selected hexkl_items_5_vector &&
  run_case hexkl_items_5_vector \
    --backend-profile hvx-vector --enable-hexkl --omnifetch-items-through 5
case_selected hexkl_items_6_vector &&
  run_case hexkl_items_6_vector \
    --backend-profile hvx-vector --enable-hexkl --omnifetch-items-through 6
case_selected hexkl_items_7_vector &&
  run_case hexkl_items_7_vector \
    --backend-profile hvx-vector --enable-hexkl --omnifetch-items-through 7
case_selected hexkl_items_7_vector_vtcm_stage &&
  run_case hexkl_items_7_vector_vtcm_stage \
    --backend-profile hvx-vector --enable-hexkl --omnifetch-items-through 7 \
    --enable-omnifetch-kv-vtcm
case_selected hexkl_items_7_scalar &&
  run_case hexkl_items_7_scalar \
    --backend-profile legacy-scalar --enable-hexkl \
    --omnifetch-items-through 7
case_selected hexkl_omnifetch_1_7_vector_vtcm &&
  run_case hexkl_omnifetch_1_7_vector_vtcm \
    --backend-profile hvx-vector-vtcm --enable-hexkl \
    --enable-omnifetch-items-1-7

echo "COMPLETE results=${results_csv} audit=${audit_csv}"
