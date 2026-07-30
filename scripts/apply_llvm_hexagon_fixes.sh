#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${REPO_ROOT}/../LLVM_DIR/llvm-project}"
PATCH_DIR="${REPO_ROOT}/patches/llvm"

if [[ ! -d "${LLVM_SOURCE_DIR}/.git" ]]; then
  echo "error: LLVM source repository not found: ${LLVM_SOURCE_DIR}" >&2
  exit 2
fi

for patch_file in "${PATCH_DIR}"/*.patch; do
  if git -C "${LLVM_SOURCE_DIR}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "Already applied: $(basename -- "${patch_file}")"
    continue
  fi
  git -C "${LLVM_SOURCE_DIR}" apply --check "${patch_file}"
  git -C "${LLVM_SOURCE_DIR}" apply "${patch_file}"
  echo "Applied: $(basename -- "${patch_file}")"
done

echo "LLVM Hexagon fixes are present in ${LLVM_SOURCE_DIR}"
