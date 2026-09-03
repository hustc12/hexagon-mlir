#!/usr/bin/env bash
# Replay every already-compiled stage in one continuous SDK sysMon window.
# All files are pre-positioned before sampling, so host compilation/ADB upload
# is excluded. Stages execute strictly serially and are never retried.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 ARTIFACT_ROOT OUTPUT_DIR [default|memory]" >&2
  exit 2
fi

artifact_root=$(realpath "$1")
if [[ -d "$2" ]] && find "$2" -mindepth 1 -print -quit | grep -q .; then
  echo "OUTPUT_DIR must be new or empty: $2" >&2
  exit 2
fi
output_dir=$(mkdir -p "$2" && realpath "$2")
profile_mode=${3:-default}
case "${profile_mode}" in
  default) sysmon_debug_level=1 ;;
  memory) sysmon_debug_level=0 ;;
  *) echo "Unknown sysMon profile mode: ${profile_mode}" >&2; exit 2 ;;
esac
serial=${ANDROID_SERIAL:-49d1c7b2}
sdk_root=${HEXAGON_SDK_ROOT:?HEXAGON_SDK_ROOT is not set}
tools_root=${HEXAGON_TOOLS:?HEXAGON_TOOLS is not set}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
sysmon_bin="${sdk_root}/tools/utils/sysmon/sysMonApp"
sysmon_parser="${sdk_root}/tools/utils/sysmon/parser_linux_v2/HTML_Parser/sysmon_parser"
runner="${sdk_root}/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
skel="${sdk_root}/libs/run_main_on_hexagon/ship/hexagon_toolv87_v73/librun_main_on_hexagon_skel.so"
libcpp="${tools_root}/target/hexagon/lib/v73/G0/pic/libc++.so.1"
libcppabi="${tools_root}/target/hexagon/lib/v73/G0/pic/libc++abi.so.1"
adb_cmd=(adb -s "${serial}")

mapfile -d '' -t stage_dirs < <(
  find "${artifact_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z | \
    while IFS= read -r -d '' candidate; do
      find "${candidate}" -maxdepth 1 -type f -name 'lib_mlir_ciface_*.so' \
        ! -name '*-consts-*' -print -quit | grep -q . && printf '%s\0' "${candidate}"
    done
)
if [[ ${#stage_dirs[@]} -eq 0 ]]; then
  echo "No compiled Hexagon stage directories under ${artifact_root}" >&2
  exit 1
fi

device_dirs=()
sysmon_pid=
cleanup() {
  if [[ -n "${sysmon_pid}" ]]; then
    kill "${sysmon_pid}" 2>/dev/null || true
  fi
  for device_dir in "${device_dirs[@]}"; do
    "${adb_cmd[@]}" shell "rm -rf '${device_dir}'" >/dev/null 2>&1 || true
  done
  "${adb_cmd[@]}" shell \
    'rm -f /sdcard/sysmon_cdsp.bin /data/sysmon_cdsp.bin /tmp/sysmon_cdsp.bin' \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" apply >"${output_dir}/phone_before.txt"
"${adb_cmd[@]}" push "${sysmon_bin}" /data/local/tmp/sysMonApp >/dev/null
"${adb_cmd[@]}" shell \
  'chmod 777 /data/local/tmp/sysMonApp; rm -f /sdcard/sysmon_cdsp.bin /data/sysmon_cdsp.bin /tmp/sysmon_cdsp.bin'

# Pre-position all stages before the PMU window. Generated wrappers embed the
# artifact basename as their absolute /data/local/tmp input/output directory.
for stage_dir in "${stage_dirs[@]}"; do
  stage_name=$(basename "${stage_dir}")
  device_dir="/data/local/tmp/${stage_name}"
  device_dirs+=("${device_dir}")
  mapfile -t inputs < <(find "${stage_dir}" -maxdepth 1 -type f -name '*_t*.raw' -print | sort)
  mapfile -t principal < <(find "${stage_dir}" -maxdepth 1 -type f \
    -name 'lib_mlir_ciface_*.so' ! -name '*-consts-*' -print)
  mapfile -t constants < <(find "${stage_dir}" -maxdepth 1 -type f \
    -name 'lib_mlir_ciface_*-consts-*.so' -print | sort)
  if [[ ${#inputs[@]} -eq 0 || ${#principal[@]} -ne 1 ]]; then
    echo "Invalid archived stage: ${stage_dir}" >&2
    exit 1
  fi
  "${adb_cmd[@]}" shell "rm -rf '${device_dir}'; mkdir -p '${device_dir}/lib'"
  "${adb_cmd[@]}" push "${inputs[@]}" "${runner}" "${principal[0]}" \
    "${constants[@]}" "${device_dir}" >/dev/null
  "${adb_cmd[@]}" push "${skel}" "${libcpp}" "${libcppabi}" \
    "${device_dir}/lib" >/dev/null
done

coproc SYSMON_PROC {
  "${adb_cmd[@]}" shell "/data/local/tmp/sysMonApp profiler --debugLevel ${sysmon_debug_level} --q6 cdsp --samplingPeriodUs 1000" \
    >"${output_dir}/sysmon_host.log" 2>&1
}
sysmon_pid=${SYSMON_PROC_PID}
sleep 1

mkdir -p "${output_dir}/stage_perf"
start_ns=$(date +%s%N)
for stage_dir in "${stage_dirs[@]}"; do
  stage_name=$(basename "${stage_dir}")
  device_dir="/data/local/tmp/${stage_name}"
  principal=$(find "${stage_dir}" -maxdepth 1 -type f \
    -name 'lib_mlir_ciface_*.so' ! -name '*-consts-*' -printf '%f\n')
  printf '=== stage %s ===\n' "${stage_name}" >>"${output_dir}/run_host.log"
  set +e
  "${adb_cmd[@]}" shell "cd '${device_dir}'; touch /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf; export DSP_LIBRARY_PATH='${device_dir}/lib'; export ADSP_LIBRARY_PATH='${device_dir}/lib;/vendor/lib/rfsa/adsp/'; ./run_main_on_hexagon 3 '${device_dir}/${principal}'" \
    >>"${output_dir}/run_host.log" 2>&1
  run_status=$?
  set -e
  if [[ ${run_status} -ne 0 ]]; then
    printf '\n' >&"${SYSMON_PROC[1]}"
    wait "${sysmon_pid}" || true
    sysmon_pid=
    echo "Stage ${stage_name} failed with status ${run_status}; not retrying" >&2
    exit "${run_status}"
  fi
  "${adb_cmd[@]}" pull "${device_dir}/perf.txt" \
    "${output_dir}/stage_perf/${stage_name}.txt" >/dev/null
done
end_ns=$(date +%s%N)

printf '\n' >&"${SYSMON_PROC[1]}"
wait "${sysmon_pid}"
sysmon_pid=

raw_profile="${output_dir}/sysmon_cdsp.bin"
if ! "${adb_cmd[@]}" pull /sdcard/sysmon_cdsp.bin "${raw_profile}" >/dev/null 2>&1; then
  if ! "${adb_cmd[@]}" pull /data/sysmon_cdsp.bin "${raw_profile}" >/dev/null 2>&1; then
    "${adb_cmd[@]}" pull /tmp/sysmon_cdsp.bin "${raw_profile}" >/dev/null
  fi
fi
elapsed=$(awk -v start="${start_ns}" -v end="${end_ns}" \
  'BEGIN { printf "%.9f", (end-start)/1000000000 }')
printf '{\n  "kernel_elapsed_seconds": %s\n}\n' "${elapsed}" \
  >"${output_dir}/kernel_window.json"
"${sysmon_parser}" "${raw_profile}" --outdir "${output_dir}/parsed" \
  >>"${output_dir}/sysmon_host.log"
python "${repo_root}/scripts/script_release/internal/summarize_sysmon_profile.py" \
  --raw-pmu "${output_dir}/parsed/raw_pmu.csv" \
  --kernel-window "${output_dir}/kernel_window.json" \
  --run-log "${output_dir}/run_host.log" \
  --profile-mode "${profile_mode}" \
  --json "${output_dir}/kernel_window_summary.json" \
  --markdown "${output_dir}/kernel_window_summary.md"
"${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" status >"${output_dir}/phone_after.txt"
cat "${output_dir}/kernel_window_summary.md"
