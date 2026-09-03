#!/usr/bin/env bash
# Generate the full-corpus logical-movement and physical-traffic audit.
# The measurements are deliberately reused from experiment 01: recompiling the
# same 75 cases would change neither the compiler ledger nor the sysMon source.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "${script_dir}/lib.sh"

e2e_root=${ALPS_REPRO_LOCAL_ROOT}/01_end_to_end_15_models
audit_root=${ALPS_REPRO_LOCAL_ROOT}/03_movement_and_traffic
audit_remote=${ALPS_REPRO_REMOTE_ROOT}/03_movement_and_traffic

if [[ ! -f ${e2e_root}/results.csv ]]; then
  echo "Experiment 01 data are absent; collecting them once before auditing."
  "${script_dir}/01_end_to_end_15_models.sh"
fi

mkdir -p "${audit_root}"
alps_stage_banner "Full-corpus movement and physical-traffic audit"
"${ALPS_PARENT_DIR}/mlir-env/bin/python" \
  "${ALPS_REPO_ROOT}/scripts/script_release/internal/summarize_alps_full_matrix.py" \
  --output-root "${e2e_root}" \
  --results "${e2e_root}/results.csv" \
  --long-csv "${audit_root}/movement_traffic_long.csv" \
  --wide-csv "${audit_root}/movement_traffic.csv" \
  --markdown "${audit_root}/movement_traffic.md"

ssh nano "mkdir -p '${audit_remote}'"
rsync -a --partial "${audit_root}/" "nano:${audit_remote}/"
echo "Results: ${audit_root}/movement_traffic.md"
