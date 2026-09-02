#!/usr/bin/env bash
# Formal complete-model ALPS A0--A4 ablation.
# Full E includes generic P2e formation, P2g continuity/register-tile support,
# and the legality-gated P5h/P5i attention/patch realizers.
# A0 and A4 are rebuilt in this campaign as well so fixed-event and multiplexed
# sysMon profiles come from the same binary/configuration generation.
# Models and cases run strictly serially, without a timeout or automatic retry.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
label=${ALPS_ABLATION_LABEL:-20260901_memory_a0_a4}
output_root=${ALPS_ABLATION_OUTPUT:-/tmp/alps_component_ablation_${label}}
remote_root=${ALPS_ABLATION_REMOTE:-/home/huzq85/2-working/working_set/alps_component_ablation_${label}}
compile_threads=${ALPS_ABLATION_COMPILE_THREADS:-auto}
device_iterations=${ALPS_ABLATION_DEVICE_ITERATIONS:-1}

models=(
  dinov2-small
  swin-transformer
  segformer-mit-b0
  deit-small
  whisper-tiny
)

"${repo_root}/scripts/prepare_phone_benchmark.sh" apply

ALPS_ENABLE_MODEL_SYSMON_REPLAY=1 \
ALPS_ENABLE_MEMORY_SYSMON_REPLAY=1 \
  exec "${repo_root}/scripts/run_full_hvx_five_way.sh" \
    --alps-component-ablation \
    --compile-threads "${compile_threads}" \
    --device-iterations "${device_iterations}" \
    --output-dir "${output_root}" \
    --remote-dir "${remote_root}" \
    "${models[@]}" \
    "$@"
