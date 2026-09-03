#!/usr/bin/env bash
# Run the frozen A0--A4 study on five complete FP16 models.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "${script_dir}/lib.sh"
alps_ensure_build
alps_make_roots

output_root=${ALPS_REPRO_LOCAL_ROOT}/02_ablation_selected_models
remote_root=${ALPS_REPRO_REMOTE_ROOT}/02_ablation_selected_models
alps_stage_banner "Five-model complete-graph ALPS component ablation"

ALPS_ABLATION_LABEL=${ALPS_RUN_ID} \
ALPS_ABLATION_OUTPUT=${output_root} \
ALPS_ABLATION_REMOTE=${remote_root} \
  exec "${ALPS_REPO_ROOT}/scripts/run_alps_component_ablation.sh"
