#!/usr/bin/env bash
# Profile correctness-qualified complete item7 archives without localizing the
# compiled model. Models execute strictly serially; failures are recorded and
# never retried automatically.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
local_root=${ALPS_PROFILE_OUTPUT:-/tmp/alps_full_bottleneck_corpus_20260829}/archived_sysmon
remote_result_root=${ALPS_PROFILE_REMOTE:-/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829}/archived_sysmon
archive_root=/home/huzq85/2-working/working_set
models=(
  gpt2 sd-clip qwen2.5-0.5b tinyllama-1.1b smollm2-1.7b
  swin-transformer segformer-mit-b0 deit-small beit-base dinov2-small
  whisper-tiny hubert-base wav2vec2-base unispeech-base unispeech-sat-base
)
if (($#)); then models=("$@"); fi

artifact_root_for() {
  case "$1" in
    gpt2|sd-clip|qwen2.5-0.5b|tinyllama-1.1b|smollm2-1.7b)
      echo "${archive_root}/full_hvx_five_way_20260815_layered_fp16/$1/item7-only/artifacts" ;;
    swin-transformer|segformer-mit-b0|beit-base|whisper-tiny|hubert-base|wav2vec2-base|unispeech-base|unispeech-sat-base)
      echo "${archive_root}/full_hvx_five_way_20260815/$1/item7-only/artifacts" ;;
    deit-small)
      echo "${archive_root}/deit_hvx_retest_20260815/item7-only/artifacts/DeiTSmallWrapper-2026-08-15_08-41-50-434862" ;;
    dinov2-small)
      echo "${archive_root}/full_hvx_four_way_20260815/dinov2-small/alps-item7/artifacts/Dinov2SmallWrapper-2026-08-15_06-46-01-152125" ;;
    *) echo "Unknown model: $1" >&2; return 2 ;;
  esac
}

mkdir -p "${local_root}"
state=${local_root}/state.csv
[[ -f "${state}" ]] || printf '%s\n' 'model,status,timestamp,artifact_root' >"${state}"
for model in "${models[@]}"; do
  output=${local_root}/${model}
  artifact=$(artifact_root_for "${model}")
  if [[ -s "${output}/kernel_window_summary.json" ]]; then
    printf 'SKIP model=%s existing=%s\n' "${model}" "${output}"
    continue
  fi
  rm -rf "${output}"
  printf '%s,START,%s,%s\n' "${model}" "$(date --iso-8601=seconds)" "${artifact}" >>"${state}"
  set +e
  "${repo_root}/scripts/script_legacy/profile_remote_archived_hexagon_model.sh" "${artifact}" "${output}"
  rc=$?
  set -e
  if ((rc)); then
    printf '%s,FAIL_%s,%s,%s\n' "${model}" "${rc}" "$(date --iso-8601=seconds)" "${artifact}" >>"${state}"
    echo "model=${model} failed status=${rc}; not retrying" >&2
    exit "${rc}"
  fi
  ssh nano mkdir -p "${remote_result_root}/${model}"
  rsync -a "${output}/" "nano:${remote_result_root}/${model}/"
  printf '%s,PASS,%s,%s\n' "${model}" "$(date --iso-8601=seconds)" "${artifact}" >>"${state}"
done
