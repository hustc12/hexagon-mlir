#!/usr/bin/env bash
# Seed compact, configuration-identical evidence into a two-way output tree.
# No compiler artifacts are copied. The main runner validates PASS + Perf before
# reuse and synchronizes the compact evidence into the new authoritative root.
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT_ROOT" >&2; exit 2; }
output_root=$1
old_root=${ALPS_TWO_WAY_OLD_ROOT:-/home/huzq85/2-working/working_set/alps_frozen_full_matrix_20260829}
full_e_root=${ALPS_TWO_WAY_FULL_E_ROOT:-/home/huzq85/2-working/working_set/alps_frozen_full_matrix_20260831_full_e_selected}
seed_finals=${ALPS_TWO_WAY_SEED_FINALS:-1}

models=(
  gpt2 sd-clip qwen2.5-0.5b tinyllama-1.1b smollm2-1.7b
  swin-transformer segformer-mit-b0 deit-small beit-base vit-base
  dinov2-small whisper-tiny hubert-base wav2vec2-base unispeech-base
)
full_e_models=(
  dinov2-small swin-transformer segformer-mit-b0 deit-small whisper-tiny
)

is_full_e_model() {
  local candidate=$1 item
  for item in "${full_e_models[@]}"; do
    [[ ${candidate} == "${item}" ]] && return 0
  done
  return 1
}

seed_case() {
  local model=$1 scheme=$2 source_root=$3
  local source=${source_root}/${model}/${scheme}
  local destination=${output_root}/${model}/${scheme}
  ssh nano "test -f '${source}/run.log' -a -f '${source}/status.txt'" || return 0
  mkdir -p "${destination}/sysmon_model_replay"
  rsync -a --partial "nano:${source}/run.log" "nano:${source}/status.txt" "${destination}/"
  for name in kernel_window_summary.json kernel_window_summary.md; do
    if ssh nano "test -f '${source}/sysmon_model_replay/${name}'"; then
      rsync -a --partial \
        "nano:${source}/sysmon_model_replay/${name}" \
        "${destination}/sysmon_model_replay/"
    fi
  done
  printf '%s\n' "nano:${source}" >"${destination}/.reused-from"
}

for model in "${models[@]}"; do
  if is_full_e_model "${model}"; then
    seed_case "${model}" hmlir-hvx-hexkl-on "${full_e_root}"
    if [[ ${seed_finals} == 1 ]]; then
      seed_case "${model}" alps-final "${full_e_root}"
    fi
  else
    seed_case "${model}" hmlir-hvx-hexkl-on "${old_root}"
  fi
done
