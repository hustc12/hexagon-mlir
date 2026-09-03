#!/usr/bin/env bash
# Detect only device mappings that this repository has verified explicitly.
set -euo pipefail

serial="${ANDROID_SERIAL:-}"
adb_args=()
if [[ -n "${serial}" ]]; then
  adb_args=(-s "${serial}")
fi

soc="$(adb "${adb_args[@]}" shell getprop ro.soc.model | tr -d '\r' | tr '[:lower:]' '[:upper:]')"
board="$(adb "${adb_args[@]}" shell getprop ro.board.platform | tr -d '\r' | tr '[:upper:]' '[:lower:]')"

case "${soc}:${board}" in
  SM8550:*|*:kalama)
    printf '%s\n' 73
    ;;
  *)
    echo "Unknown Hexagon architecture for SoC=${soc:-unknown}, board=${board:-unknown}." >&2
    echo "Set HEXAGON_ARCH_VERSION explicitly after checking the device documentation." >&2
    exit 2
    ;;
esac
