#!/usr/bin/env bash
# Profile full-shape Speech components serially. This is an operator-local LWP
# vehicle, not a reduced model and not a formal complete-model latency run.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export ANDROID_HOST=${ANDROID_HOST:-}
local_root=${ALPS_PROFILE_OUTPUT:-/tmp/alps_full_bottleneck_corpus_20260829}/hotspot_lwp_audio
remote_root=${ALPS_PROFILE_REMOTE:-/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829}/hotspot_lwp_audio
models=(wav2vec2-base hubert-base unispeech-base unispeech-sat-base)
if (($#)); then models=("$@"); fi
mkdir -p "${local_root}"

for model in "${models[@]}"; do
  output=${local_root}/${model}
  if [[ -s "${output}/lwp_summary.md" ]]; then
    echo "SKIP model=${model} existing=${output}/lwp_summary.md"
    continue
  fi
  rm -rf "${output}"
  mkdir -p "${output}/artifacts"
  "${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" apply >"${output}/phone_before.txt"
  echo "START model=${model} $(date --iso-8601=seconds)"
  set +e
  HEXAGON_MLIR_COMPILE_THREADS=4 \
  HEXAGON_MLIR_DUMP_DIR="${output}/artifacts" \
    python "${repo_root}/scripts/script_legacy/probe_full_audio_hotspots.py" "${model}" \
      >"${output}/run.log" 2>&1
  rc=$?
  set -e
  if ((rc)); then
    printf 'FAIL_%s\n' "${rc}" >"${output}/status.txt"
    echo "model=${model} failed status=${rc}; not retrying" >&2
    exit "${rc}"
  fi
  python "${repo_root}/scripts/script_release/internal/summarize_alps_lwp.py" \
    --artifact-root "${output}/artifacts" \
    --info-dump /tmp/lwp_infodump.txt \
    --ledger "${output}/run.log" \
    --csv "${output}/lwp_regions.csv" \
    --markdown "${output}/lwp_summary.md"
  "${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" status >"${output}/phone_after.txt"
  printf 'PASS\n' >"${output}/status.txt"
  ssh nano mkdir -p "${remote_root}/${model}"
  rsync -a "${output}/" "nano:${remote_root}/${model}/"
  rm -rf "${output}/artifacts"
  echo "PASS model=${model} summary=${output}/lwp_summary.md"
done
