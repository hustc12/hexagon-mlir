#!/usr/bin/env bash
# Run dynamic V75/V79 proxies and compile complete ALPS graphs for both cores.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "${script_dir}/lib.sh"
alps_ensure_build
alps_make_roots

output_root=${ALPS_REPRO_LOCAL_ROOT}/04_portability_v75_v79
remote_root=${ALPS_REPRO_REMOTE_ROOT}/04_portability_v75_v79

alps_stage_banner "V75/V79 dynamic portability proxies"
ALPS_SIM_OUTPUT=${output_root}/dynamic \
  "${ALPS_REPO_ROOT}/scripts/run_hexagon_sim_portability.sh" \
    --no-timing --workload proxy

for model in dinov2-small swin-transformer; do
  alps_stage_banner "V75/V79 complete ${model} ALPS lowering/codegen/link"
  ALPS_SIM_OUTPUT=${output_root}/complete_compile/${model} \
    "${ALPS_REPO_ROOT}/scripts/run_hexagon_sim_portability.sh" \
      --no-timing --compile-only --model-layers 12 \
      --model "${model}" --scheme alps-final
done

ssh nano "mkdir -p '${remote_root}'"
rsync -a --partial "${output_root}/" "nano:${remote_root}/"
echo "Results: ${output_root}"
