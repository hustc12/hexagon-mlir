#!/usr/bin/env bash
# Build the OmniFetch overlay and run the complete 15-model evaluation matrix.
#
# The matrix is deliberately serial: one model and one backend configuration
# are active at a time. No `timeout` process or host-side deadline is used.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
BUILD_JOBS="${BUILD_JOBS:-4}"
RESULTS_DIR="${OMNIFETCH_RESULTS_DIR:-/tmp/omnifetch-upstream-full-models}"

usage() {
  cat <<EOF
Usage: scripts/script_legacy/run_all_models_no_timeout.sh [--skip-build] [matrix options/model ...]

Builds the official-upstream-based OmniFetch overlay, runs its test suite, and
then invokes run_full_model_matrix.sh with --no-timeout. Remaining arguments
are passed to the matrix runner. Successful rows are skipped on restart unless
--force is supplied.

Environment:
  BUILD_JOBS=N              Parallelism for compilation only (default: 4)
  OMNIFETCH_RESULTS_DIR=DIR Persistent matrix logs/CSV (default: ${RESULTS_DIR})
EOF
}

skip_build=0
forward=()
while (($#)); do
  case "$1" in
    --skip-build) skip_build=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) forward+=("$1"); shift ;;
  esac
done

if [[ "${skip_build}" -eq 0 ]]; then
  BUILD_JOBS="${BUILD_JOBS}" LLVM_BUILD_JOBS="${LLVM_BUILD_JOBS:-${BUILD_JOBS}}" \
    "${SCRIPT_DIR}/build_omnifetch_upstream.sh"
fi

adb devices | awk 'NR>1 && $2=="device" {found=1} END {exit !found}' || {
  echo "ERROR: no Android device is connected and authorized" >&2
  exit 1
}

OMNIFETCH_RESULTS_DIR="${RESULTS_DIR}" \
  "${SCRIPT_DIR}/run_full_model_matrix.sh" --no-timeout "${forward[@]}"
