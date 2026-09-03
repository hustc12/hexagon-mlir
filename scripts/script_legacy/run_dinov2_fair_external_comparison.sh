#!/usr/bin/env bash
set -euo pipefail

# Run every DINOv2 Debug backend serially with the same logical model/input,
# one HVX execution thread, default performance mode, and 20 measured calls.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result_root="${OUTPUT_DIR:-/tmp/omnifetch-baselines/dinov2-debug-fair}"
iterations="${ITERATIONS:-20}"
device_serial="${ANDROID_SERIAL:-49d1c7b2}"
python_bin="${PROJECT_PYTHON:-${project_root}/../mlir-env/bin/python}"
runner="${project_root}/benchmark_models/debug_running/run_dinov2-small_debug.py"

mkdir -p "${result_root}"
export ANDROID_SERIAL="${device_serial}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_hexagon_case() {
  local name="$1"
  shift
  echo "===== ${name}: serial ${iterations}-iteration run ====="
  "${python_bin}" "${runner}" "$@" --device-iterations "${iterations}" \
    2>&1 | tee "${result_root}/${name}.log"
}

run_hexagon_case hvx
run_hexagon_case hexkl --enable-hexkl
run_hexagon_case hexkl_omnifetch_item7 \
  --enable-hexkl --enable-omnifetch-kv-cache-prefetch \
  --disable-layout-aware --disable-omnifetch-adaptive

echo "===== qnn: serial ${iterations}-iteration run ====="
QNN_ITERATIONS="${iterations}" \
QNN_HVX_THREADS=1 \
QNN_PERF_PROFILE=default \
OUTPUT_DIR="${result_root}/qnn" \
  "${project_root}/scripts/script_legacy/run_dinov2_qnn_baseline.sh" \
  2>&1 | tee "${result_root}/qnn.log"

echo "===== litert-qnn: serial ${iterations}-iteration runs ====="
OUTPUT_DIR="${result_root}/litert" \
LITERT_ITERATIONS="${iterations}" \
LITERT_TRIALS="${LITERT_TRIALS:-3}" \
ANDROID_SERIAL="${device_serial}" \
  "${project_root}/scripts/script_legacy/run_dinov2_litert_baseline.sh" \
  2>&1 | tee "${result_root}/litert.log"

echo "===== Summary (microseconds) ====="
for name in hvx hexkl hexkl_omnifetch_item7; do
  value="$(
    awk '/Perf:/{gsub(/[^0-9.]/, "", $0); print $0; exit}' \
      "${result_root}/${name}.log"
  )"
  printf '%-24s %s\n' "${name}" "${value}"
done
qnn_value="$(
  awk '
    /Execute Stats \(Average\):/ { average=1 }
    average && /NetRun:/ {
      gsub(/[^0-9.]/, "", $0)
      print $0
      exit
    }
  ' "${result_root}/qnn/qnn_profile.txt"
)"
printf '%-24s %s\n' "qnn_netrun" "${qnn_value}"
litert_value="$(
  awk '
    /Steady-state runs excluding first took average/ {
      sum += $(NF-1)
      count += 1
    }
    END {
      if (count) printf "%.3f", sum / count
    }
  ' "${result_root}"/litert/npu_trial_*.log
)"
printf '%-24s %s\n' "litert_qnn_steady" "${litert_value}"
echo "Logs and generated artifacts: ${result_root}"
