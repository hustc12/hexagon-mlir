#!/usr/bin/env bash
# One-command, strictly serial, complete non-Debug 15-unique-model ALPS table.
# The frozen ALPS endpoint uses full E: P2e/P2g plus legality-gated P5h/P5i.
# UniSpeech-SAT is diagnostic-only because its current ForCTC export duplicates
# UniSpeech-Base; ViT-Base occupies that primary-matrix slot.
# There is deliberately no timeout and no automatic retry after a failed case.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
label=${ALPS_FROZEN_LABEL:-20260829}
output_root=${ALPS_FROZEN_OUTPUT:-/tmp/alps_frozen_full_matrix_${label}}
remote_root=${ALPS_FROZEN_REMOTE:-/home/huzq85/2-working/working_set/alps_frozen_full_matrix_${label}}
compile_threads=${ALPS_FROZEN_COMPILE_THREADS:-auto}
device_iterations=${ALPS_FROZEN_DEVICE_ITERATIONS:-1}

"${repo_root}/scripts/script_release/internal/prepare_phone_benchmark.sh" apply

ALPS_ENABLE_MODEL_SYSMON_REPLAY=1 \
  exec "${repo_root}/scripts/script_release/internal/run_full_hvx_five_way.sh" \
    --alps-full-matrix \
    --compile-threads "${compile_threads}" \
    --device-iterations "${device_iterations}" \
    --output-dir "${output_root}" \
    --remote-dir "${remote_root}" \
    "$@"
