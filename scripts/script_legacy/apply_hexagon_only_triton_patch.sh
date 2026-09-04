#!/usr/bin/env bash
# Apply the tracked, reversible Hexagon-only build adaptation to untracked Triton.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
TRITON_ROOT="${ROOT}/triton"
PATCH="${ROOT}/alps/patches/triton/0001-make-gpu-backends-optional.patch"

[[ -d "${TRITON_ROOT}/.git" ]] || {
  echo "ERROR: Triton checkout is missing; run ci/setup_submodules.sh first" >&2
  exit 1
}

if git -C "${TRITON_ROOT}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "Hexagon-only Triton patch is already applied."
elif git -C "${TRITON_ROOT}" apply --check "${PATCH}"; then
  git -C "${TRITON_ROOT}" apply "${PATCH}"
  echo "Applied Hexagon-only Triton patch."
else
  echo "ERROR: Hexagon-only Triton patch conflicts with this Triton revision" >&2
  exit 1
fi
