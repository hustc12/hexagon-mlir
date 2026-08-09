#!/usr/bin/env bash
# Build the exact LLVM revision required by the freshly checked out Triton.
# This deliberately uses LLVM_DIR_upstream so the legacy toolchain remains
# byte-for-byte recoverable.
set -euo pipefail

repo_dir="$(git rev-parse --show-toplevel)"
parent_dir="$(cd "${repo_dir}/.." && pwd)"
llvm_src="${LLVM_UPSTREAM_SRC:-${parent_dir}/LLVM_DIR_upstream/llvm-project}"
llvm_build="${LLVM_UPSTREAM_BUILD:-${llvm_src}/build}"
host_toolchain="${HOST_TOOLCHAIN:-${parent_dir}/HOST_TOOLCHAIN}"
jobs="${LLVM_BUILD_JOBS:-4}"
expected="$(tr -d '[:space:]' < "${repo_dir}/triton/cmake/llvm-hash.txt")"

[[ -d "${llvm_src}/.git" ]] || {
  echo "Missing upstream LLVM source: ${llvm_src}" >&2
  exit 1
}
actual="$(git -C "${llvm_src}" rev-parse HEAD)"
[[ "${actual}" == "${expected}" ]] || {
  echo "LLVM revision mismatch: expected ${expected}, found ${actual}" >&2
  exit 1
}
[[ -x "${host_toolchain}/bin/clang" ]] || {
  echo "Missing host clang: ${host_toolchain}/bin/clang" >&2
  exit 1
}

export CC="${host_toolchain}/bin/clang"
export CXX="${host_toolchain}/bin/clang++"
export PATH="${host_toolchain}/bin:${PATH}"
export CCACHE_DIR="${CCACHE_DIR:-/tmp/omnifetch_upstream_llvm_ccache_${UID}}"
mkdir -p "${CCACHE_DIR}" "${llvm_build}"

cmake -G Ninja -S "${llvm_src}/llvm" -B "${llvm_build}" \
  -DLLVM_ENABLE_PROJECTS="llvm;mlir" \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_ASM_COMPILER="${CC}" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;Hexagon" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu \
  -DCMAKE_INSTALL_PREFIX="${llvm_build}/install"

cmake --build "${llvm_build}" --parallel "${jobs}"
cmake --install "${llvm_build}"

echo "LLVM ${actual} installed at ${llvm_build}/install"
