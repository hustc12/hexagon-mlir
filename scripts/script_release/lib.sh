#!/usr/bin/env bash
# Shared release-reproduction helpers. This file is sourced, not run directly.

set -euo pipefail

ALPS_REPRO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALPS_REPO_ROOT=$(cd "${ALPS_REPRO_DIR}/../.." && pwd)
ALPS_PARENT_DIR=$(cd "${ALPS_REPO_ROOT}/.." && pwd)

if [[ -z ${ALPS_RUN_ID:-} ]]; then
  ALPS_RUN_ID="$(git -C "${ALPS_REPO_ROOT}" rev-parse --short HEAD)"
fi
export ALPS_RUN_ID

export ALPS_REPRO_LOCAL_ROOT=${ALPS_REPRO_LOCAL_ROOT:-/tmp/alps_reproduce_${ALPS_RUN_ID}}
export ALPS_REPRO_REMOTE_ROOT=${ALPS_REPRO_REMOTE_ROOT:-/home/huzq85/2-working/working_set/alps_reproduce_${ALPS_RUN_ID}}

alps_python_version() {
  "${ALPS_PARENT_DIR}/mlir-env/bin/python" -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}

alps_plugin_build_dir() {
  local py_version
  py_version=$(alps_python_version)
  printf '%s\n' "${ALPS_REPO_ROOT}/triton/build/cmake.linux-x86_64-cpython-${py_version}/third_party/qcom_hexagon_backend"
}

alps_build_is_ready() {
  local plugin_dir
  plugin_dir=$(alps_plugin_build_dir)
  [[ -x ${plugin_dir}/bin/linalg-hexagon-opt &&
     -x ${plugin_dir}/bin/linalg-hexagon-translate &&
     -f ${plugin_dir}/bin/runtime/libhexagon_runtime.a ]]
}

alps_ensure_build() {
  if ! alps_build_is_ready; then
    echo "ALPS build products are missing; running 00_build_alps.sh first."
    "${ALPS_REPRO_DIR}/00_build_alps.sh"
  fi
}

alps_make_roots() {
  mkdir -p "${ALPS_REPRO_LOCAL_ROOT}"
  ssh nano "mkdir -p '${ALPS_REPRO_REMOTE_ROOT}'"
}

alps_stage_banner() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$1"
}
