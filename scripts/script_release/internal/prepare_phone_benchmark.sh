#!/usr/bin/env bash
# Put the connected Android device in a reproducible, non-battery-saving state
# for Hexagon benchmarks.  Thermal protection deliberately remains enabled.
set -euo pipefail

serial=${ANDROID_SERIAL:-49d1c7b2}
mode=${1:-apply}

adb_cmd=(adb -s "${serial}")

status() {
  "${adb_cmd[@]}" shell '
    echo "low_power=$(settings get global low_power)"
    echo "low_power_sticky=$(settings get global low_power_sticky)"
    echo "adaptive_power_saver=$(settings get global adaptive_battery_management_enabled)"
    echo "app_standby=$(settings get global app_standby_enabled)"
    echo "device_idle_enabled=$(cmd deviceidle enabled all)"
    echo "device_idle_deep=$(cmd deviceidle get deep)"
    echo "device_idle_light=$(cmd deviceidle get light)"
    dumpsys power | grep -E "mWakefulness=|mIsPowered=|mStayOn=" | head -n 4
    dumpsys battery | grep -E "level:|temperature:|USB powered:|AC powered:" | head -n 8
  '
}

case "${mode}" in
  apply)
    "${adb_cmd[@]}" wait-for-device
    "${adb_cmd[@]}" shell '
      cmd power set-mode 0
      cmd power set-adaptive-power-saver-enabled false
      cmd deviceidle unforce
      cmd deviceidle disable all
      input keyevent WAKEUP
      wm dismiss-keyguard >/dev/null 2>&1 || true
    '
    status
    ;;
  status)
    "${adb_cmd[@]}" wait-for-device
    status
    ;;
  restore)
    "${adb_cmd[@]}" wait-for-device
    "${adb_cmd[@]}" shell '
      cmd deviceidle enable all
    '
    status
    ;;
  *)
    echo "Usage: $0 [apply|status|restore]" >&2
    exit 2
    ;;
esac
