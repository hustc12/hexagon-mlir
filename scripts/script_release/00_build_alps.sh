#!/usr/bin/env bash
# Build the V73 ALPS/Hexagon-MLIR toolchain once for the release experiments.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "${script_dir}/lib.sh"

if alps_build_is_ready && [[ ${ALPS_FORCE_REBUILD:-0} != 1 ]]; then
  echo "REUSE existing ALPS build: $(alps_plugin_build_dir)"
  exit 0
fi

if [[ -n ${ALPS_BUILD_JOBS:-} ]]; then
  jobs=${ALPS_BUILD_JOBS}
else
  online=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
  available_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
  memory_jobs=$((available_kib / 3145728))
  ((memory_jobs < 1)) && memory_jobs=1
  jobs=${online}
  ((jobs > memory_jobs)) && jobs=${memory_jobs}
  ((jobs > 8)) && jobs=8
fi

alps_stage_banner "Build ALPS for Hexagon v73 with ${jobs} job(s)"
cd "${ALPS_REPO_ROOT}"
"${ALPS_REPO_ROOT}/scripts/script_release/setup/build_hexagon_mlir_incremental.sh" \
  --arch 73 --jobs "${jobs}"
