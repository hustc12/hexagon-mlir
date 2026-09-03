#!/usr/bin/env bash
# Complete public ALPS reproduction workflow. Every stage is strictly serial.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "${script_dir}/lib.sh"

stages=(
  00_build_alps.sh
  01_end_to_end_15_models.sh
  02_ablation_selected_models.sh
  03_movement_and_traffic.sh
  04_portability_v75_v79.sh
)

echo "ALPS run ID     : ${ALPS_RUN_ID}"
echo "Local summaries: ${ALPS_REPRO_LOCAL_ROOT}"
echo "Remote results : nano:${ALPS_REPRO_REMOTE_ROOT}"
echo "Execution      : strictly serial"

for stage in "${stages[@]}"; do
  alps_stage_banner "Begin ${stage}"
  "${script_dir}/${stage}"
done

alps_stage_banner "All ALPS release experiments completed"
