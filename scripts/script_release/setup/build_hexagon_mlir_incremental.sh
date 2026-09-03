#!/usr/bin/env bash
#
# Incremental Hexagon-MLIR/Triton plugin build for an already provisioned tree.
# The required directory layout and variables follow docs/user-guide.md.
#
# Usage:
#   bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh
#   bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh --full
#   bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh --tests
#   bash scripts/script_release/setup/build_hexagon_mlir_incremental.sh --full --clean --tests
#
set -euo pipefail

usage() {
  echo "Usage: $0 [--full] [--tests] [--clean] [--arch VERSION] [--jobs N]"
}

RUN_TESTS=0
CLEAN_BUILD=0
FULL_BUILD=0
# The connected OnePlus CPH2449 uses the SM8550/kalama CDSP (Hexagon v73).
# Keep this overridable for other devices, but never inherit upstream's v75
# default silently for the experiment scripts.
HEXAGON_ARCH="${HEXAGON_ARCH_VERSION:-73}"
BUILD_JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tests)
      RUN_TESTS=1
      shift
      ;;
    --full)
      FULL_BUILD=1
      shift
      ;;
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --arch)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      HEXAGON_ARCH="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      BUILD_JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

REPO_DIR="$(git rev-parse --show-toplevel)"
PARENT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

export HEXAGON_MLIR_ROOT="${REPO_DIR}"
export TRITON_ROOT="${REPO_DIR}/triton"
export TRITON_HOME="${REPO_DIR}"
export TRITON_PLUGIN_DIRS="${REPO_DIR}/triton_shared;${REPO_DIR}/qcom_hexagon_backend"
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH}"

export HOST_TOOLCHAIN="${HOST_TOOLCHAIN:-${PARENT_DIR}/HOST_TOOLCHAIN}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-${PARENT_DIR}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}"
export HEXAGON_TOOLS="${HEXAGON_TOOLS:-${PARENT_DIR}/HEXAGON_TOOLS/Tools}"
export HEXKL_ROOT="${HEXKL_ROOT:-${PARENT_DIR}/HEXKL_DIR/hexkl_addon}"
DEFAULT_LLVM_PROJECT_BUILD_DIR="${PARENT_DIR}/LLVM_DIR/llvm-project/build"
# LLVM_DIR is the canonical checkout name. Treat a missing inherited path like
# an unset override so stale shells cannot select a removed toolchain.
if [[ -z "${LLVM_PROJECT_BUILD_DIR:-}" ||
      ! -d "${LLVM_PROJECT_BUILD_DIR}" ]]; then
  export LLVM_PROJECT_BUILD_DIR="${DEFAULT_LLVM_PROJECT_BUILD_DIR}"
fi
export CONDA_ENV="${CONDA_ENV:-${PARENT_DIR}/mlir-env}"

require_dir() {
  [[ -d "$1" ]] || {
    echo "Required directory is missing: $1" >&2
    echo "Provision dependencies with scripts/script_release/setup/build_hexagon_mlir.sh or follow docs/user-guide.md." >&2
    exit 1
  }
}

require_file() {
  [[ -f "$1" ]] || {
    echo "Required file is missing: $1" >&2
    exit 1
  }
}

require_dir "${TRITON_ROOT}"
require_dir "${REPO_DIR}/triton_shared"
require_dir "${HEXAGON_SDK_ROOT}"
require_dir "${HEXAGON_TOOLS}"
require_dir "${HEXKL_ROOT}"
require_dir "${LLVM_PROJECT_BUILD_DIR}"
require_file "${CONDA_ENV}/bin/activate"

LLVM_PREFIX="${LLVM_PROJECT_BUILD_DIR}"
if [[ -d "${LLVM_PROJECT_BUILD_DIR}/install/include" &&
      -d "${LLVM_PROJECT_BUILD_DIR}/install/lib" ]]; then
  LLVM_PREFIX="${LLVM_PROJECT_BUILD_DIR}/install"
fi
require_dir "${LLVM_PREFIX}/include"
require_dir "${LLVM_PREFIX}/lib"

if [[ -x "${HOST_TOOLCHAIN}/bin/clang" &&
      -x "${HOST_TOOLCHAIN}/bin/clang++" ]]; then
  export CC="${HOST_TOOLCHAIN}/bin/clang"
  export CXX="${HOST_TOOLCHAIN}/bin/clang++"
  export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"
fi

# shellcheck disable=SC1090
source "${CONDA_ENV}/bin/activate"

PYTHON_VERSION="$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
BUILD_DIR="${TRITON_ROOT}/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}"
PLUGIN_BUILD_DIR="${BUILD_DIR}/third_party/qcom_hexagon_backend"

export LLVM_SYSPATH="${LLVM_PREFIX}"
export LLVM_INCLUDE_DIRS="${LLVM_PREFIX}/include"
export LLVM_LIBRARY_DIR="${LLVM_PREFIX}/lib"
export TRITON_SHARED_OPT_PATH="${BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
export PATH="${PLUGIN_BUILD_DIR}/bin:${BUILD_DIR}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export PYTHONPATH="${TRITON_ROOT}/python:${PYTHONPATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
export MAX_JOBS="${BUILD_JOBS}"

# Managed/sandboxed environments commonly expose $HOME as read-only. Keep
# ccache's writable state in /tmp while retaining it across incremental builds.
export CCACHE_DIR="${CCACHE_DIR:-/tmp/hexagon_mlir_ccache_${UID}}"
mkdir -p "${CCACHE_DIR}"

if [[ "${CLEAN_BUILD}" -eq 1 ]]; then
  if [[ -d "${BUILD_DIR}" ]]; then
    echo "Removing the selected generated build directory: ${BUILD_DIR}"
    rm -rf -- "${BUILD_DIR}"
  fi
fi

echo "Repository       : ${REPO_DIR}"
echo "Python env       : ${CONDA_ENV}"
echo "Python version   : ${PYTHON_VERSION}"
echo "LLVM prefix      : ${LLVM_PREFIX}"
echo "Hexagon SDK      : ${HEXAGON_SDK_ROOT}"
echo "Hexagon Tools    : ${HEXAGON_TOOLS}"
echo "HexKL            : ${HEXKL_ROOT}"
echo "Hexagon arch     : v${HEXAGON_ARCH_VERSION}"
echo "Parallel jobs    : ${BUILD_JOBS}"

if [[ "${FULL_BUILD}" -eq 0 && -f "${BUILD_DIR}/build.ninja" ]]; then
  echo "Incremental plugin/runtime build (use --full after Python/Triton changes)"
  cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" \
    --target linalg-hexagon-opt linalg-hexagon-translate hexagon_runtime libtriton.so
else
  echo "Full editable Triton + Hexagon plugin build"
  cd "${TRITON_ROOT}"
  TRITON_BUILD_WITH_CLANG_LLD=1 \
  TRITON_BUILD_WITH_CCACHE=true \
  TRITON_IN_TREE_BACKENDS= \
  TRITON_BUILD_PROTON=OFF \
  TRITON_BUILD_EXAMPLES=OFF \
  TRITON_BUILD_TESTS=OFF \
  TRITON_BUILD_TEST_ANALYSIS=ON \
  TRITON_BUILD_TOOLS=OFF \
  TRITON_OFFLINE_BUILD=1 \
  LLVM_INCLUDE_DIRS="${LLVM_INCLUDE_DIRS}" \
  LLVM_LIBRARY_DIR="${LLVM_LIBRARY_DIR}" \
  LLVM_SYSPATH="${LLVM_SYSPATH}" \
  python3 -m pip install -e . --no-build-isolation --no-deps --verbose
fi

require_file "${PLUGIN_BUILD_DIR}/bin/linalg-hexagon-opt"
require_file "${PLUGIN_BUILD_DIR}/bin/linalg-hexagon-translate"
echo "Build completed: ${PLUGIN_BUILD_DIR}/bin/linalg-hexagon-opt"

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  if command -v lit >/dev/null 2>&1; then
    lit "${PLUGIN_BUILD_DIR}/test"
  else
    echo "LIT is not installed in ${CONDA_ENV}; build succeeded but tests cannot run." >&2
    exit 1
  fi
fi
