#!/usr/bin/env bash
# Verify that OmniFetch modifies only the explicitly reviewed upstream files.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
BASE_REF="${OMNIFETCH_UPSTREAM_REF:-upstream-snapshot-20260808}"
MANIFEST="${ROOT}/omnifetch/manifest.txt"

git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null

actual=$(mktemp)
expected=$(mktemp)
trap 'rm -f "${actual}" "${expected}"' EXIT

git -C "${ROOT}" diff --name-only --diff-filter=MD "${BASE_REF}" -- \
  | LC_ALL=C sort -u >"${actual}"
sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "${MANIFEST}" \
  | LC_ALL=C sort -u >"${expected}"

if ! diff -u "${expected}" "${actual}"; then
  echo "ERROR: upstream integration surface differs from omnifetch/manifest.txt" >&2
  exit 1
fi

echo "OmniFetch overlay integration surface verified against ${BASE_REF}."
