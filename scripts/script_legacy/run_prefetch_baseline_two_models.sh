#!/usr/bin/env bash
# Strictly serial comparison of OmniFetch and two independent prefetch baselines.
#
# This is a Debug-model engineering screen, not a full-model paper result.  APT
# currently consumes a model-global distance and an explicit manual allowlist.
# Two serial DINOv2 LWP samples selected distance=1; the same conservative
# distance is used for the shape-similar ViT proxy until per-candidate profile
# ingestion is implemented.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

output_dir=${OUTPUT_DIR:-/tmp/omnifetch-prefetch-baselines-$(date +%Y%m%d-%H%M%S)}
iterations=${DEVICE_ITERATIONS:-7}
timeout_seconds=${MODEL_TIMEOUT:-600}
mkdir -p "${output_dir}"

dino_ids="Dinov2DebugWrapper:loop2:view0,Dinov2DebugWrapper:loop2:view1,Dinov2DebugWrapper:loop3:view0,Dinov2DebugWrapper:loop4:view0,Dinov2DebugWrapper:loop9:view0,Dinov2DebugWrapper:loop11:view0,Dinov2DebugWrapper:loop12:view0,Dinov2DebugWrapper:loop14:view0,Dinov2DebugWrapper:loop15:view0,Dinov2DebugWrapper:loop17:view0,Dinov2DebugWrapper:loop20:view0,Dinov2DebugWrapper:loop20:view1,Dinov2DebugWrapper:loop21:view0,Dinov2DebugWrapper:loop24:view0,Dinov2DebugWrapper:loop25:view0,Dinov2DebugWrapper:loop26:view0,Dinov2DebugWrapper:loop26:view1,Dinov2DebugWrapper:loop26:view2,Dinov2DebugWrapper:loop29:view0,Dinov2DebugWrapper:loop29:view1,Dinov2DebugWrapper:loop30:view0,Dinov2DebugWrapper:loop31:view0,Dinov2DebugWrapper:loop32:view0,Dinov2DebugWrapper:loop32:view1"
vit_ids="ViTWrapper:loop2:view0,ViTWrapper:loop2:view1,ViTWrapper:loop3:view0,ViTWrapper:loop4:view0,ViTWrapper:loop9:view0,ViTWrapper:loop11:view0,ViTWrapper:loop12:view0,ViTWrapper:loop14:view0,ViTWrapper:loop15:view0,ViTWrapper:loop17:view0,ViTWrapper:loop20:view0,ViTWrapper:loop20:view1,ViTWrapper:loop21:view0,ViTWrapper:loop24:view0,ViTWrapper:loop25:view0,ViTWrapper:loop26:view0,ViTWrapper:loop26:view1,ViTWrapper:loop26:view2,ViTWrapper:loop29:view0,ViTWrapper:loop29:view1,ViTWrapper:loop30:view0,ViTWrapper:loop31:view0,ViTWrapper:loop32:view0,ViTWrapper:loop34:view0,ViTWrapper:loop35:view0,ViTWrapper:loop37:view0,ViTWrapper:loop38:view0,ViTWrapper:loop40:view0,ViTWrapper:loop41:view0,ViTWrapper:loop41:view1,ViTWrapper:loop42:view0,ViTWrapper:loop43:view0,ViTWrapper:loop44:view0,ViTWrapper:loop45:view0,ViTWrapper:loop45:view1,ViTWrapper:loop45:view2,ViTWrapper:loop48:view0,ViTWrapper:loop48:view1,ViTWrapper:loop49:view0,ViTWrapper:loop50:view0,ViTWrapper:loop52:view0,ViTWrapper:loop52:view1"

printf 'model,scheme,status,perf_us,p50_us,p90_us,min_us,hints,issued,busy_suppressed,page_clipped,requested_bytes,issued_bytes,correctness\n' > "${output_dir}/results.csv"

run_case() {
  local model=$1
  local runner=$2
  local apt_ids=$3
  local scheme=$4
  local log="${output_dir}/${model}_${scheme}.log"
  local -a args=(
    python "${runner}"
    --enable-hexkl
    --backend-profile hvx-vector-vtcm
    --device-iterations "${iterations}"
  )

  case "${scheme}" in
    omnifetch-item7-only)
      args+=(
        --enable-omnifetch-kv-cache-prefetch
        --disable-layout-aware
        --disable-omnifetch-adaptive
      )
      ;;
    apt-get-hx)
      args+=(
        --prefetch-baseline apt-get-hx
        --prefetch-baseline-distance 1
        --apt-get-hx-manual-candidate-ids "${apt_ids}"
      )
      ;;
    prefetch-kernel-hx)
      args+=(
        --prefetch-baseline prefetch-kernel-hx
        --prefetch-baseline-distance 1
      )
      ;;
    *)
      echo "unknown scheme: ${scheme}" >&2
      return 2
      ;;
  esac

  echo "[SerialRun] model=${model} scheme=${scheme}"
  local status=pass
  if ! timeout --foreground "${timeout_seconds}" "${args[@]}" >"${log}" 2>&1; then
    status=fail
  fi

  local perf p50 p90 min hints issued busy clipped requested issued_bytes correctness
  perf=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p50=$(awk -F: '/^[[:space:]]*PerfP50:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  p90=$(awk -F: '/^[[:space:]]*PerfP90:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  min=$(awk -F: '/^[[:space:]]*PerfMin:/{gsub(/[[:space:]]/,"",$2);v=$2}END{print v}' "${log}")
  hints=$(awk 'match($0,/hints=[0-9]+/){v=substr($0,RSTART+6,RLENGTH-6)}END{print v}' "${log}")
  issued=$(awk 'match($0,/issued=[0-9]+/){v=substr($0,RSTART+7,RLENGTH-7)}END{print v}' "${log}")
  busy=$(awk 'match($0,/busy_suppressed=[0-9]+/){v=substr($0,RSTART+16,RLENGTH-16)}END{print v}' "${log}")
  clipped=$(awk 'match($0,/page_clipped=[0-9]+/){v=substr($0,RSTART+13,RLENGTH-13)}END{print v}' "${log}")
  requested=$(awk 'match($0,/requested_bytes=[0-9]+/){v=substr($0,RSTART+16,RLENGTH-16)}END{print v}' "${log}")
  issued_bytes=$(awk 'match($0,/issued_bytes=[0-9]+/){v=substr($0,RSTART+13,RLENGTH-13)}END{print v}' "${log}")
  correctness=$(awk '/\[Compare\]/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "${scheme}" "${status}" "${perf}" "${p50}" "${p90}" \
    "${min}" "${hints}" "${issued}" "${busy}" "${clipped}" "${requested}" \
    "${issued_bytes}" "${correctness}" >> "${output_dir}/results.csv"

  if [[ "${status}" != pass || -z "${perf}" ]]; then
    echo "[SerialRun] failed: ${log}" >&2
    return 1
  fi
  if [[ "${scheme}" != omnifetch-item7-only && ( -z "${hints}" || "${hints}" == 0 ) ]]; then
    echo "[SerialRun] invalid zero-hint baseline: ${log}" >&2
    return 1
  fi
  if [[ "${scheme}" != omnifetch-item7-only && ( -z "${issued}" || "${issued}" == 0 ) ]]; then
    echo "[SerialRun] invalid zero-issued prefetch row: ${log}" >&2
    return 1
  fi
}

# Intentionally no background jobs: each model/scheme completes before the next.
for scheme in omnifetch-item7-only apt-get-hx prefetch-kernel-hx; do
  run_case dinov2-debug benchmark_models/debug_running/run_dinov2-small_debug.py "${dino_ids}" "${scheme}"
done
for scheme in omnifetch-item7-only apt-get-hx prefetch-kernel-hx; do
  run_case vit-debug benchmark_models/debug_running/run_vit_debug.py "${vit_ids}" "${scheme}"
done

echo "[SerialRun] complete: ${output_dir}/results.csv"
