#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export ANDROID_HOST=${ANDROID_HOST:-}
local_root=${ALPS_PROFILE_OUTPUT:-/tmp/alps_full_bottleneck_corpus_20260829}/hotspot_lwp_whisper
remote_root=${ALPS_PROFILE_REMOTE:-/home/huzq85/2-working/working_set/alps_full_bottleneck_corpus_20260829}/hotspot_lwp_whisper
rm -rf "${local_root}"; mkdir -p "${local_root}/artifacts"
"${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" apply >"${local_root}/phone_before.txt"
set +e
HEXAGON_MLIR_COMPILE_THREADS=4 HEXAGON_MLIR_DUMP_DIR="${local_root}/artifacts" \
  python "${repo_root}/scripts/script_legacy/probe_whisper_hotspots.py" >"${local_root}/run.log" 2>&1
rc=$?; set -e
if ((rc)); then printf 'FAIL_%s\n' "$rc" >"${local_root}/status.txt"; exit "$rc"; fi
python "${repo_root}/scripts/script_release/internal/summarize_alps_lwp.py" --artifact-root "${local_root}/artifacts" \
  --info-dump /tmp/lwp_infodump.txt --ledger "${local_root}/run.log" \
  --csv "${local_root}/lwp_regions.csv" --markdown "${local_root}/lwp_summary.md"
"${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" status >"${local_root}/phone_after.txt"
echo PASS >"${local_root}/status.txt"
ssh nano mkdir -p "${remote_root}"; rsync -a "${local_root}/" "nano:${remote_root}/"
rm -rf "${local_root}/artifacts"
