#!/usr/bin/env bash
# Build the complete 15-model LWP + sysMon bottleneck corpus. Models and phases
# are strictly serial; there is no timeout and no automatic retry after failure.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
date_label=${ALPS_PROFILE_DATE:-20260829}
output_root=${ALPS_PROFILE_OUTPUT:-/tmp/alps_full_bottleneck_corpus_${date_label}}
remote_root=${ALPS_PROFILE_REMOTE:-/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_${date_label}}
phase=both
models=()
all_models=(
  gpt2 sd-clip qwen2.5-0.5b tinyllama-1.1b smollm2-1.7b
  swin-transformer segformer-mit-b0 deit-small beit-base dinov2-small
  whisper-tiny hubert-base wav2vec2-base unispeech-base unispeech-sat-base
)

usage() {
  cat <<EOF
Usage: $0 [--phase lwp|sysmon|both] [model ...]

Profiles item7 + P5m/P5k synchronous control. LWP and non-instrumented sysMon
are separate compilations. With no models, all 15 complete FP16 models run in
the declared 5-language/5-vision/5-speech order.

Environment:
  ALPS_PROFILE_OUTPUT  local resumable state (default: ${output_root})
  ALPS_PROFILE_REMOTE  nano working_set destination (default: ${remote_root})
  ALPS_PROFILE_DATE    archive label (default: ${date_label})
EOF
}

while (($#)); do
  case "$1" in
    --phase)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      phase=$2; shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) models+=("$1"); shift ;;
  esac
done
[[ "${phase}" == lwp || "${phase}" == sysmon || "${phase}" == both ]] || {
  echo "Invalid phase: ${phase}" >&2
  exit 2
}
((${#models[@]})) || models=("${all_models[@]}")

mkdir -p "${output_root}"
state=${output_root}/corpus_state.csv
[[ -f "${state}" ]] || printf '%s\n' 'model,phase,status,timestamp' >"${state}"

record() {
  printf '%s,%s,%s,%s\n' "$1" "$2" "$3" "$(date --iso-8601=seconds)" >>"${state}"
}

lwp_depth_for() {
  case "$1" in
    gpt2|sd-clip|qwen2.5-0.5b|tinyllama-1.1b|smollm2-1.7b) echo 1 ;;
    *) echo 0 ;;
  esac
}

lwp_compile_threads_for() {
  case "$1" in
    gpt2|sd-clip|qwen2.5-0.5b|tinyllama-1.1b|smollm2-1.7b) echo 4 ;;
    *) echo 1 ;;
  esac
}

run_lwp() {
  local model=$1 rc depth compile_threads
  depth=$(lwp_depth_for "${model}")
  compile_threads=$(lwp_compile_threads_for "${model}")
  "${repo_root}/scripts/prepare_phone_benchmark.sh" apply
  record "${model}" lwp START
  set +e
  ALPS_ENABLE_SYSMON_PROFILE=0 ALPS_ENABLE_MODEL_SYSMON_REPLAY=0 \
  ALPS_DISABLE_MOVEMENT_LEDGER=1 \
    "${repo_root}/scripts/run_full_hvx_five_way.sh" \
      --item7-only --alps-p1-hexkl-profile \
      --profile-lwp-loop-depth "${depth}" \
      --compile-threads "${compile_threads}" \
      --output-dir "${output_root}/lwp" \
      --remote-dir "${remote_root}/lwp" \
      "${model}"
  rc=$?
  set -e
  if ((rc)); then
    record "${model}" lwp "FAIL_${rc}"
    return "${rc}"
  fi
  record "${model}" lwp PASS
}

run_sysmon() {
  local model=$1 rc
  "${repo_root}/scripts/prepare_phone_benchmark.sh" apply
  record "${model}" sysmon START
  set +e
  ALPS_ENABLE_SYSMON_PROFILE=0 ALPS_ENABLE_MODEL_SYSMON_REPLAY=1 \
    "${repo_root}/scripts/run_full_hvx_five_way.sh" \
      --alps-p5m --with-item7 \
      --compile-threads 4 \
      --output-dir "${output_root}/sysmon" \
      --remote-dir "${remote_root}/sysmon" \
      "${model}"
  rc=$?
  set -e
  if ((rc)); then
    record "${model}" sysmon "FAIL_${rc}"
    return "${rc}"
  fi
  record "${model}" sysmon PASS
}

for model in "${models[@]}"; do
  if [[ "${phase}" == lwp || "${phase}" == both ]]; then
    run_lwp "${model}"
  fi
  if [[ "${phase}" == sysmon || "${phase}" == both ]]; then
    run_sysmon "${model}"
  fi
done
