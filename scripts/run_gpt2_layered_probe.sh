#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_ROOT=${OMNIFETCH_RUNTIME_ROOT:-/home/huzq85/2-working/hexagon_npu/hexagon-mlir-native}
VENV=${OMNIFETCH_VENV:-/home/huzq85/2-working/hexagon_npu/mlir-env}
PYTHON_VERSION=$(
  "${VENV}/bin/python" -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
)
TRITON_BUILD_DIR=${TRITON_BUILD_DIR:-${RUNTIME_ROOT}/triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="${RUNTIME_ROOT}/triton/python:${ROOT}/benchmark_models"
export OMNIFETCH_DSP_HEAP_MB=${OMNIFETCH_DSP_HEAP_MB:-512}
export ANDROID_SERIAL=${ANDROID_SERIAL:-49d1c7b2}
export ANDROID_HOST=${ANDROID_HOST:-}
export HEXAGON_ARCH_VERSION=${HEXAGON_ARCH_VERSION:-73}
export HEXAGON_MLIR_ROOT=${HEXAGON_MLIR_ROOT:-${RUNTIME_ROOT}}
export HOST_TOOLCHAIN=${HOST_TOOLCHAIN:-/home/huzq85/2-working/hexagon_npu/HOST_TOOLCHAIN}
export HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}
export HEXAGON_TOOLS=${HEXAGON_TOOLS:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools}
export HEXKL_ROOT=${HEXKL_ROOT:-/home/huzq85/2-working/hexagon_npu/HEXKL_DIR/hexkl_addon}
export HEXAGON_RUNTIME_LIBS_DIR=${HEXAGON_RUNTIME_LIBS_DIR:-${TRITON_BUILD_DIR}/third_party/qcom_hexagon_backend/bin/runtime}
export PATH="${TRITON_BUILD_DIR}/third_party/qcom_hexagon_backend/bin:${PATH}"

exec "${VENV}/bin/python" "${SCRIPT_DIR}/probe_gpt2_layered_export.py" "$@"
