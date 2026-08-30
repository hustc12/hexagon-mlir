#!/usr/bin/env bash
# Profile one already-compiled Hexagon stage with SDK sysMon.  This is intended
# for matched control/treatment attribution without recompiling a full model.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: profile_archived_hexagon_stage.sh ARTIFACT_DIR OUTPUT_DIR

ARTIFACT_DIR must contain one input *_t0.raw, the principal
lib_mlir_ciface_*.so, and any lib_mlir_ciface_*-consts-*.so files.

Optional environment:
  ALPS_STAGE_DEVICE_NAME  Original artifact basename when the wrapper embeds it.
  ALPS_PROFILE_REPEATS    Serial invocations in one PMU window (default: 1).
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

artifact_dir=$(realpath "$1")
if [[ -d "$2" ]] && find "$2" -mindepth 1 -print -quit | grep -q .; then
  echo "OUTPUT_DIR must be new or empty: $2" >&2
  exit 2
fi
output_dir=$(mkdir -p "$2" && realpath "$2")
serial=${ANDROID_SERIAL:-49d1c7b2}
profile_repeats=${ALPS_PROFILE_REPEATS:-1}
if ! [[ "${profile_repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ALPS_PROFILE_REPEATS must be a positive integer" >&2
  exit 2
fi
sdk_root=${HEXAGON_SDK_ROOT:?HEXAGON_SDK_ROOT is not set}
tools_root=${HEXAGON_TOOLS:?HEXAGON_TOOLS is not set}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sysmon_bin="${sdk_root}/tools/utils/sysmon/sysMonApp"
sysmon_parser="${sdk_root}/tools/utils/sysmon/parser_linux_v2/HTML_Parser/sysmon_parser"
runner="${sdk_root}/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
skel="${sdk_root}/libs/run_main_on_hexagon/ship/hexagon_toolv87_v73/librun_main_on_hexagon_skel.so"
libcpp="${tools_root}/target/hexagon/lib/v73/G0/pic/libc++.so.1"
libcppabi="${tools_root}/target/hexagon/lib/v73/G0/pic/libc++abi.so.1"

for path in "${artifact_dir}" "${sysmon_bin}" "${sysmon_parser}" \
  "${runner}" "${skel}" "${libcpp}" "${libcppabi}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done

mapfile -t inputs < <(find "${artifact_dir}" -maxdepth 1 -type f -name '*_t0.raw' -print)
mapfile -t principal_libs < <(find "${artifact_dir}" -maxdepth 1 -type f \
  -name 'lib_mlir_ciface_*.so' ! -name '*-consts-*' -print)
mapfile -t const_libs < <(find "${artifact_dir}" -maxdepth 1 -type f \
  -name 'lib_mlir_ciface_*-consts-*.so' -print)
if [[ ${#inputs[@]} -ne 1 || ${#principal_libs[@]} -ne 1 ]]; then
  echo "Expected exactly one input and one principal library in ${artifact_dir}" >&2
  exit 1
fi

# Generated wrappers embed the original device directory in input/output paths.
# ALPS_STAGE_DEVICE_NAME lets an artifact copied under a shorter local name keep
# that original basename and is therefore a correctness requirement, not merely
# a cosmetic label.
stage_name=${ALPS_STAGE_DEVICE_NAME:-$(basename "${artifact_dir}")}
if [[ "${stage_name}" == */* || -z "${stage_name}" ]]; then
  echo "ALPS_STAGE_DEVICE_NAME must be a non-empty directory basename" >&2
  exit 2
fi
device_dir="/data/local/tmp/${stage_name}"
adb_cmd=(adb -s "${serial}")
sysmon_pid=

cleanup() {
  if [[ -n "${sysmon_pid}" ]]; then
    kill "${sysmon_pid}" 2>/dev/null || true
  fi
  "${adb_cmd[@]}" shell "rm -rf '${device_dir}'; rm -f /sdcard/sysmon_cdsp.bin /data/sysmon_cdsp.bin /tmp/sysmon_cdsp.bin" \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${repo_root}/scripts/prepare_phone_benchmark.sh" apply >"${output_dir}/phone_before.txt"
"${adb_cmd[@]}" shell "rm -rf '${device_dir}'; mkdir -p '${device_dir}/lib'; rm -f /sdcard/sysmon_cdsp.bin /data/sysmon_cdsp.bin /tmp/sysmon_cdsp.bin"
"${adb_cmd[@]}" push "${inputs[0]}" "${runner}" "${principal_libs[0]}" \
  "${const_libs[@]}" "${device_dir}" >/dev/null
"${adb_cmd[@]}" push "${skel}" "${libcpp}" "${libcppabi}" "${device_dir}/lib" >/dev/null
"${adb_cmd[@]}" push "${sysmon_bin}" /data/local/tmp/sysMonApp >/dev/null
"${adb_cmd[@]}" shell chmod 777 /data/local/tmp/sysMonApp

coproc SYSMON_PROC {
  "${adb_cmd[@]}" shell "/data/local/tmp/sysMonApp profiler --debugLevel 1 --q6 cdsp --samplingPeriodUs 1000" \
    >"${output_dir}/sysmon_host.log" 2>&1
}
sysmon_pid=${SYSMON_PROC_PID}
sleep 1

start_ns=$(date +%s%N)
run_status=0
for ((repeat = 1; repeat <= profile_repeats; ++repeat)); do
  printf '=== repeat %d/%d ===\n' "${repeat}" "${profile_repeats}" \
    >>"${output_dir}/run_host.log"
  set +e
  "${adb_cmd[@]}" shell "cd '${device_dir}'; touch /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf; export DSP_LIBRARY_PATH='${device_dir}/lib'; export ADSP_LIBRARY_PATH='${device_dir}/lib;/vendor/lib/rfsa/adsp/'; ./run_main_on_hexagon 3 '${device_dir}/$(basename "${principal_libs[0]}")'" \
    >>"${output_dir}/run_host.log" 2>&1
  run_status=$?
  set -e
  [[ ${run_status} -eq 0 ]] || break
done
end_ns=$(date +%s%N)

printf '\n' >&"${SYSMON_PROC[1]}"
wait "${sysmon_pid}"
sysmon_pid=
if [[ ${run_status} -ne 0 ]]; then
  echo "Hexagon stage failed with status ${run_status}" >&2
  exit "${run_status}"
fi

"${adb_cmd[@]}" pull "${device_dir}/perf.txt" "${output_dir}/perf.txt" >/dev/null
raw_profile="${output_dir}/sysmon_cdsp.bin"
if ! "${adb_cmd[@]}" pull /sdcard/sysmon_cdsp.bin "${raw_profile}" >/dev/null 2>&1; then
  if ! "${adb_cmd[@]}" pull /data/sysmon_cdsp.bin "${raw_profile}" >/dev/null 2>&1; then
    "${adb_cmd[@]}" pull /tmp/sysmon_cdsp.bin "${raw_profile}" >/dev/null
  fi
fi

elapsed=$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN { printf "%.9f", (end-start)/1000000000 }')
printf '{\n  "kernel_elapsed_seconds": %s\n}\n' "${elapsed}" >"${output_dir}/kernel_window.json"
"${sysmon_parser}" "${raw_profile}" --outdir "${output_dir}/parsed" >>"${output_dir}/sysmon_host.log"
python "${repo_root}/scripts/summarize_sysmon_profile.py" \
  --raw-pmu "${output_dir}/parsed/raw_pmu.csv" \
  --kernel-window "${output_dir}/kernel_window.json" \
  --json "${output_dir}/kernel_window_summary.json" \
  --markdown "${output_dir}/kernel_window_summary.md"
"${repo_root}/scripts/prepare_phone_benchmark.sh" status >"${output_dir}/phone_after.txt"
cat "${output_dir}/perf.txt"
cat "${output_dir}/kernel_window_summary.md"
