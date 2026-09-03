#!/usr/bin/env bash
# Strictly serial complete-model HexKL-On versus ALPS C+full-E+P+R matrix.
# Compact evidence from prior matched runs is seeded by default; only missing
# cases compile and execute. There is no timeout and no automatic retry.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
label=${ALPS_TWO_WAY_LABEL:-20260831}
output_root=${ALPS_TWO_WAY_OUTPUT:-/tmp/alps_full_e_two_way_${label}}
remote_root=${ALPS_TWO_WAY_REMOTE:-/home/huzq85/2-working/working_set/alps_full_e_two_way_${label}}
compile_threads=${ALPS_TWO_WAY_COMPILE_THREADS:-auto}
device_iterations=${ALPS_TWO_WAY_DEVICE_ITERATIONS:-1}

if [[ ${ALPS_TWO_WAY_SEED_PRIOR:-1} == 1 ]]; then
  "${repo_root}/scripts/script_legacy/seed_alps_two_way_results.sh" "${output_root}"
fi

"${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" apply

ALPS_ENABLE_MODEL_SYSMON_REPLAY=1 \
  exec "${repo_root}/scripts/script_release/internal/run_full_hvx_five_way.sh" \
    --alps-two-way \
    --compile-threads "${compile_threads}" \
    --device-iterations "${device_iterations}" \
    --output-dir "${output_root}" \
    --remote-dir "${remote_root}" \
    "$@"
