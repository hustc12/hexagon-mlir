#!/usr/bin/env bash
# Reproducible full build for Qualcomm upstream + the OmniFetch overlay.
set -euo pipefail

repo_dir="$(git rev-parse --show-toplevel)"
parent_dir="$(cd "${repo_dir}/.." && pwd)"
jobs="${BUILD_JOBS:-4}"
hexagon_arch="${HEXAGON_ARCH_VERSION:-73}"

if [[ ! -d "${repo_dir}/triton/.git" || ! -d "${repo_dir}/triton_shared/.git" ]]; then
  bash "${repo_dir}/ci/setup_submodules.sh"
fi
bash "${repo_dir}/scripts/script_legacy/apply_hexagon_only_triton_patch.sh"

bash "${repo_dir}/scripts/script_legacy/build_upstream_llvm.sh"

export LLVM_PROJECT_BUILD_DIR="${parent_dir}/LLVM_DIR/llvm-project/build"
export HOST_TOOLCHAIN="${parent_dir}/HOST_TOOLCHAIN"
export HEXAGON_SDK_ROOT="${parent_dir}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2"
export HEXAGON_TOOLS="${parent_dir}/HEXAGON_TOOLS/Tools"
export HEXKL_ROOT="${parent_dir}/HEXKL_DIR/hexkl_addon"
export CONDA_ENV="${parent_dir}/mlir-env"

bash "${repo_dir}/scripts/script_release/setup/build_hexagon_mlir_incremental.sh" \
  --full --tests --arch "${hexagon_arch}" --jobs "${jobs}"
