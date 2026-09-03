#!/usr/bin/env bash
# Public entry point for the ALPS release evaluation suite.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
release_dir=${repo_root}/scripts/script_release
# shellcheck source=scripts/script_release/lib.sh
source "${release_dir}/lib.sh"

usage() {
  cat <<'EOF'
Usage: ./run_alps.sh [OPTIONS]

Build ALPS once and run complete-model release experiments strictly serially.
With no options, all four experiment classes run in the order shown below.

Experiment selection:
  -e, --end-to-end     Run the 15-model, five-configuration end-to-end study
  -a, --ablation       Run the A0--A4 ablation on five selected models
  -m, --movement       Generate the data-movement and physical-traffic audit
  -p, --portability    Run the Hexagon V75/V79 portability study
      --all            Run all four experiment classes (the default)
  -b, --build-only     Build or reuse the V73 ALPS toolchain; run no experiment
  -h, --help           Show this help and exit

Selection options may be combined. Selected experiments always run in the
canonical order: end-to-end, ablation, movement/traffic, then portability.
The build stage is incremental and is checked once before the selected tests.

Examples:
  ./run_alps.sh                       # build, then run all four classes
  ./run_alps.sh --end-to-end          # only the complete-model end-to-end study
  ./run_alps.sh --ablation            # only the selected-model ablation
  ./run_alps.sh --movement            # only the movement/traffic audit
  ./run_alps.sh --portability         # only V75/V79 portability
  ./run_alps.sh -e -a                 # end-to-end followed by ablation
  ./run_alps.sh --build-only          # compile/reuse the toolchain only

Important runtime notes:
  * The complete default suite is intentionally exhaustive and can take many
    hours or longer, depending on compilation, device, and simulator speed.
  * Models and configurations are serial; there is no model-level parallelism.
  * Runs have no timeout and failed cases are not automatically retried.
  * Experiment 03 reuses experiment 01 data when available. If those data are
    absent, it automatically runs experiment 01 before producing the audit.
  * Results resume under /tmp and are moved/synchronized to host nano as
    described in scripts/script_release/README.md.

Optional environment variables:
  ALPS_BUILD_JOBS=N     Limit parallel jobs within compilation of one model
  ALPS_RUN_ID=NAME      Select a stable result/resume directory name
  ALPS_FORCE_REBUILD=1  Rebuild instead of reusing a ready toolchain
EOF
}

run_end_to_end=0
run_ablation=0
run_movement=0
run_portability=0
build_only=0
explicit_selection=0
explicit_all=0

while (($#)); do
  case "$1" in
    -e|--end-to-end)
      run_end_to_end=1
      explicit_selection=1
      ;;
    -a|--ablation)
      run_ablation=1
      explicit_selection=1
      ;;
    -m|--movement|--movement-and-traffic)
      run_movement=1
      explicit_selection=1
      ;;
    -p|--portability)
      run_portability=1
      explicit_selection=1
      ;;
    --all)
      explicit_all=1
      ;;
    -b|--build-only)
      build_only=1
      explicit_selection=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run './run_alps.sh --help' for usage." >&2
      exit 2
      ;;
  esac
  shift
done

if ((explicit_all && explicit_selection)); then
  echo "--all cannot be combined with individual selections or --build-only." >&2
  exit 2
fi

if ((build_only && (run_end_to_end || run_ablation || run_movement || run_portability))); then
  echo "--build-only cannot be combined with experiment selections." >&2
  exit 2
fi

if ((!explicit_selection || explicit_all)); then
  run_end_to_end=1
  run_ablation=1
  run_movement=1
  run_portability=1
  cat >&2 <<'EOF'

WARNING: no individual experiment was selected; the complete ALPS suite will
run. It covers all four release experiment classes and may take many hours or
longer. Use --end-to-end, --ablation, --movement, or --portability to run only
the required class. Run with --help for the complete interface.
EOF
fi

stages=(00_build_alps.sh)
((run_end_to_end)) && stages+=(01_end_to_end_15_models.sh)
((run_ablation)) && stages+=(02_ablation_selected_models.sh)
((run_movement)) && stages+=(03_movement_and_traffic.sh)
((run_portability)) && stages+=(04_portability_v75_v79.sh)

echo "ALPS run ID     : ${ALPS_RUN_ID}"
echo "Local summaries: ${ALPS_REPRO_LOCAL_ROOT}"
echo "Remote results : nano:${ALPS_REPRO_REMOTE_ROOT}"
echo "Execution      : strictly serial"
printf 'Stages         :'
printf ' %s' "${stages[@]}"
printf '\n'

for stage in "${stages[@]}"; do
  alps_stage_banner "Begin ${stage}"
  "${release_dir}/${stage}"
done

if ((build_only)); then
  alps_stage_banner "ALPS build completed"
else
  alps_stage_banner "Selected ALPS release experiments completed"
fi
