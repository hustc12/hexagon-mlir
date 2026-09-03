#!/usr/bin/env bash
# Activate the project-local Hexagon-MLIR/ALPS environment in the current shell.
#
# Usage:
#   source scripts/script_release/setup/set_local_env.sh
#   source scripts/script_release/setup/set_local_env.sh --quiet
#
# This follows Qualcomm's set_local_env.sh workflow, but derives every path
# from this checkout and keeps project settings out of ~/.bashrc.

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  echo "This script must be sourced so its environment survives:" >&2
  echo "  source scripts/script_release/setup/set_local_env.sh" >&2
  exit 2
fi

_alps_env_quiet=0
case ${1:-} in
  "") ;;
  --quiet) _alps_env_quiet=1 ;;
  -h|--help)
    cat <<'EOF'
Usage: source scripts/script_release/setup/set_local_env.sh [--quiet]

Activate the repository's Python environment and export the paths required by
Hexagon-MLIR, Triton, LLVM, Hexagon SDK/Tools, HexKL, and the v73 ALPS target.
The changes apply only to the current shell and disappear when it exits.
EOF
    return 0
    ;;
  *)
    echo "Unknown environment option: ${1}" >&2
    return 2
    ;;
esac

_alps_env_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_alps_env_repo_root=$(cd "${_alps_env_script_dir}/../../.." && pwd)
_alps_env_parent=$(cd "${_alps_env_repo_root}/.." && pwd)

export HEXAGON_MLIR_ROOT="${HEXAGON_MLIR_ROOT:-${_alps_env_repo_root}}"
export TRITON_ROOT="${TRITON_ROOT:-${HEXAGON_MLIR_ROOT}/triton}"
export TRITON_HOME="${TRITON_HOME:-${HEXAGON_MLIR_ROOT}}"
export TRITON_PLUGIN_DIRS="${TRITON_PLUGIN_DIRS:-${HEXAGON_MLIR_ROOT}/triton_shared;${HEXAGON_MLIR_ROOT}/qcom_hexagon_backend}"

export HEXAGON_SDK_VERSION="${HEXAGON_SDK_VERSION:-6.4.0.2}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-${_alps_env_parent}/HEXAGON_SDK/Hexagon_SDK/${HEXAGON_SDK_VERSION}}"
export HEXAGON_TOOLS="${HEXAGON_TOOLS:-${_alps_env_parent}/HEXAGON_TOOLS/Tools}"
export HEXKL_ROOT="${HEXKL_ROOT:-${_alps_env_parent}/HEXKL_DIR/hexkl_addon}"
export HOST_TOOLCHAIN="${HOST_TOOLCHAIN:-${_alps_env_parent}/HOST_TOOLCHAIN}"
export LLVM_PROJECT_BUILD_DIR="${LLVM_PROJECT_BUILD_DIR:-${_alps_env_parent}/LLVM_DIR/llvm-project/build}"
export CONDA_ENV="${CONDA_ENV:-${_alps_env_parent}/mlir-env}"
export HEXAGON_ARCH_VERSION="${HEXAGON_ARCH_VERSION:-73}"
export ANDROID_HOST="${ANDROID_HOST:-}"

_alps_env_missing=0
for _alps_env_path in \
  "${TRITON_ROOT}" \
  "${HEXAGON_MLIR_ROOT}/triton_shared" \
  "${HEXAGON_MLIR_ROOT}/qcom_hexagon_backend" \
  "${HEXAGON_SDK_ROOT}" \
  "${HEXAGON_TOOLS}" \
  "${HEXKL_ROOT}" \
  "${HOST_TOOLCHAIN}" \
  "${LLVM_PROJECT_BUILD_DIR}" \
  "${CONDA_ENV}"; do
  if [[ ! -d ${_alps_env_path} ]]; then
    echo "Missing ALPS dependency: ${_alps_env_path}" >&2
    _alps_env_missing=1
  fi
done
if ((_alps_env_missing)); then
  echo "Provision dependencies with scripts/script_release/setup/build_hexagon_mlir.sh." >&2
  unset _alps_env_quiet _alps_env_script_dir _alps_env_repo_root \
    _alps_env_parent _alps_env_missing _alps_env_path
  return 1
fi

if [[ ! -f ${CONDA_ENV}/bin/activate || ! -x ${CONDA_ENV}/bin/python3 ]]; then
  echo "Invalid project Python environment: ${CONDA_ENV}" >&2
  unset _alps_env_quiet _alps_env_script_dir _alps_env_repo_root \
    _alps_env_parent _alps_env_missing _alps_env_path
  return 1
fi

# Activating the dedicated venv is intentional and local to this shell. It is
# not performed by ~/.bashrc and no python/python3 alias is installed.
# shellcheck disable=SC1090
source "${CONDA_ENV}/bin/activate"
export PYTHON_VERSION
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

_alps_env_build_dir=${TRITON_BUILD_DIR:-${TRITON_ROOT}/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}}
_alps_env_plugin_bin=${_alps_env_build_dir}/third_party/qcom_hexagon_backend/bin
_alps_env_shared_bin=${_alps_env_build_dir}/third_party/triton_shared/tools/triton-shared-opt
export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_OPT_PATH:-${_alps_env_shared_bin}/triton-shared-opt}"

export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"

_alps_prepend_path() {
  [[ -d $1 ]] || return 0
  case :${PATH}: in
    *:"$1":*) ;;
    *) PATH=$1:${PATH} ;;
  esac
}
_alps_prepend_pythonpath() {
  [[ -d $1 ]] || return 0
  case :${PYTHONPATH:-}: in
    *:"$1":*) ;;
    *) PYTHONPATH=$1${PYTHONPATH:+:${PYTHONPATH}} ;;
  esac
}

_alps_prepend_path "${HOST_TOOLCHAIN}/bin"
_alps_prepend_path "${_alps_env_shared_bin}"
_alps_prepend_path "${_alps_env_plugin_bin}"
_alps_prepend_pythonpath "${TRITON_ROOT}/python"
export PATH PYTHONPATH

# Reuse an explicitly selected device. Otherwise choose it only when adb sees
# exactly one device; never bake a developer-specific serial into the shell.
if [[ -z ${ANDROID_SERIAL:-} ]] && command -v adb >/dev/null 2>&1; then
  mapfile -t _alps_env_devices < <(adb devices 2>/dev/null | awk 'NR > 1 && $2 == "device" {print $1}')
  if ((${#_alps_env_devices[@]} == 1)); then
    export ANDROID_SERIAL="${_alps_env_devices[0]}"
  fi
fi

if ((!_alps_env_quiet)); then
  echo "ALPS environment activated in the current shell"
  echo "  repository : ${HEXAGON_MLIR_ROOT}"
  echo "  Python     : $(command -v python3) (${PYTHON_VERSION})"
  echo "  LLVM       : ${LLVM_PROJECT_BUILD_DIR}"
  echo "  SDK/Tools  : ${HEXAGON_SDK_ROOT} / ${HEXAGON_TOOLS}"
  echo "  HexKL      : ${HEXKL_ROOT}"
  echo "  target     : Hexagon v${HEXAGON_ARCH_VERSION}"
  echo "  device     : ${ANDROID_SERIAL:-unset (set ANDROID_SERIAL before device tests)}"
fi

unset -f _alps_prepend_path _alps_prepend_pythonpath
unset _alps_env_quiet _alps_env_script_dir _alps_env_repo_root \
  _alps_env_parent _alps_env_missing _alps_env_path _alps_env_build_dir \
  _alps_env_plugin_bin _alps_env_shared_bin _alps_env_devices
