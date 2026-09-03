#!/usr/bin/env bash
# Run the frozen five-way comparison on all 15 complete FP16 models.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "${script_dir}/lib.sh"
alps_ensure_build
alps_make_roots

output_root=${ALPS_REPRO_LOCAL_ROOT}/01_end_to_end_15_models
remote_root=${ALPS_REPRO_REMOTE_ROOT}/01_end_to_end_15_models
alps_stage_banner "15-model complete-graph five-way end-to-end matrix"

ALPS_FROZEN_LABEL=${ALPS_RUN_ID} \
ALPS_FROZEN_OUTPUT=${output_root} \
ALPS_FROZEN_REMOTE=${remote_root} \
  exec "${ALPS_REPO_ROOT}/scripts/script_release/internal/run_alps_frozen_full_matrix.sh"
