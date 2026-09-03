#!/usr/bin/env bash
# Serial full-model portability matrix for Qualcomm hexagon-sim.
#
# The reported Perf value is simulated target time, not host wall time.  These
# results are diagnostic because the simulator memory system is not calibrated
# to a physical V75/V79 SoC.  No timeout is applied.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
output_root=${ALPS_SIM_OUTPUT:-/tmp/alps_hexsim_portability_20260902}
timing=${ALPS_SIM_TIMING:-1}
host_lib_root=${HEXAGON_SIM_HOST_LIB_ROOT:-/tmp/hexagon-sim-host-libs/root}
dsp_clock_mhz=${HEXAGON_SIM_DSP_CLOCK_MHZ:-}
bus_ratio=${HEXAGON_SIM_BUS_RATIO:-}
bus_penalty=${HEXAGON_SIM_BUS_PENALTY:-}
model_layers=${ALPS_SIM_MODEL_LAYERS:-1}
workload=${ALPS_SIM_WORKLOAD:-proxy}
compile_only=${ALPS_SIM_COMPILE_ONLY:-0}

arches=(75 79)
models=(dinov2-small swin-transformer)
schemes=(hmlir-hvx-hexkl-off hmlir-hvx-hexkl-on alps-final)

while (($#)); do
  case "$1" in
    --arch)
      arches=("$2")
      shift 2
      ;;
    --model)
      models=("$2")
      shift 2
      ;;
    --scheme)
      schemes=("$2")
      shift 2
      ;;
    --output-dir)
      output_root=$2
      shift 2
      ;;
    --no-timing)
      timing=0
      shift
      ;;
    --dsp-clock-mhz)
      dsp_clock_mhz=$2
      shift 2
      ;;
    --bus-ratio)
      bus_ratio=$2
      shift 2
      ;;
    --bus-penalty)
      bus_penalty=$2
      shift 2
      ;;
    --model-layers)
      model_layers=$2
      shift 2
      ;;
    --workload)
      workload=$2
      shift 2
      ;;
    --compile-only)
      compile_only=1
      workload=full
      shift
      ;;
    -h|--help)
      sed -n '1,45p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${repo_root}"
source "${repo_root}/scripts/script_release/setup/set_local_env.sh" >/dev/null 2>&1
export RUN_ON_SIM=1
export TRITON_BACKENDS_IN_TREE=1
export HEXAGON_SIM_TIMING=${timing}
export HEXAGON_SIM_BYPASS_IDLE=${HEXAGON_SIM_BYPASS_IDLE:-1}
export HEXAGON_MLIR_COMPILE_ONLY=${compile_only}
[[ -z ${dsp_clock_mhz} ]] || export HEXAGON_SIM_DSP_CLOCK_MHZ=${dsp_clock_mhz}
[[ -z ${bus_ratio} ]] || export HEXAGON_SIM_BUS_RATIO=${bus_ratio}
[[ -z ${bus_penalty} ]] || export HEXAGON_SIM_BUS_PENALTY=${bus_penalty}
if [[ -d ${host_lib_root}/lib/x86_64-linux-gnu ]]; then
  export LD_LIBRARY_PATH="${host_lib_root}/lib/x86_64-linux-gnu:${host_lib_root}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

simulator=${HEXAGON_TOOLS}/bin/hexagon-sim
if ! ldd "${simulator}" | grep -qv 'not found'; then
  :
elif ldd "${simulator}" | grep -q 'not found'; then
  echo "hexagon-sim has unresolved host libraries:" >&2
  ldd "${simulator}" | grep 'not found' >&2
  exit 1
fi

runner_for() {
  case "${workload}:$1" in
    full:dinov2-small) echo benchmark_models/run_dinov2-small.py ;;
    full:swin-transformer) echo benchmark_models/run_swin_transformer.py ;;
    proxy:dinov2-small) echo benchmark_models/debug_running/run_dinov2-small_debug.py ;;
    proxy:swin-transformer) echo benchmark_models/debug_running/run_swin_transformer_debug.py ;;
    *) return 1 ;;
  esac
}

core_for() {
  case "$1" in
    75) echo v75na_1 ;;
    79) echo v79na_1 ;;
    *) return 1 ;;
  esac
}

run_case() {
  local arch=$1 model=$2 scheme=$3
  local runner case_dir log status workload_id
  runner=${repo_root}/$(runner_for "${model}")
  if ((compile_only)); then
    workload_id="full-${model_layers}-compile"
  else
    workload_id=$([[ ${workload} == full ]] && echo "full-${model_layers}" || echo proxy)
  fi
  case_dir=${output_root}/v${arch}/${model}/${workload_id}/${scheme}
  log=${case_dir}/run.log
  status=${case_dir}/status.txt
  mkdir -p "${case_dir}/artifacts"

  if [[ -f ${status} && $(<"${status}") == PASS && -f ${log} ]]; then
    echo "REUSE arch=v${arch} model=${model} scheme=${scheme}"
    return 0
  fi

  local -a args=(--backend-profile hvx-vector --device-iterations 1)
  if [[ ${workload} == full ]]; then
    args+=(--model-layers "${model_layers}")
  fi
  case "${scheme}" in
    hmlir-hvx-hexkl-off) ;;
    hmlir-hvx-hexkl-on) args+=(--enable-hexkl) ;;
    alps-final)
      args+=(--enable-hexkl --enable-alps-hvx-widening-conv
             --enable-omnifetch-kv-cache-prefetch
             --disable-layout-aware --disable-omnifetch-adaptive)
      ;;
    *) return 2 ;;
  esac

  echo "START arch=v${arch} model=${model} scheme=${scheme} $(date --iso-8601=seconds)"
  set +e
  HEXAGON_ARCH_VERSION=${arch} \
  HEXAGON_SIM_CORE=$(core_for "${arch}") \
  HEXAGON_MLIR_DUMP_DIR=${case_dir}/artifacts \
  ALPS_ENABLE_CONSUMER_DRIVEN_LAYOUT="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_CONTINUITY_AUDIT="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_REGISTER_TILE_FORMATION="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_CRP_SUPPLY_ANALYSIS="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_CRP_PRODUCER_DIRECT_HEAD_MAJOR="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_CRP_PRODUCER_LOOP_FORMATION="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_ATTENTION_DESTINATION_FORMATION="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_PATCH_CONV_FORMATION="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_HMX_F16_EPILOGUE_FORMATION="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_HMX_DIRECT_OUTPUT_FORMATION="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_HMX_ASYNC_DRAIN_ANALYSIS="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_HMX_ASYNC_DRAIN="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  ALPS_ENABLE_TRAFFIC_CONTROL="$([[ ${scheme} == alps-final ]] && echo 1 || echo 0)" \
  /usr/bin/time -f 'Host wall time: %e s; host CPU: %P; max RSS: %M KiB' \
    "${repo_root}/../mlir-env/bin/python" "${runner}" "${args[@]}" \
    >"${log}" 2>&1
  local rc=$?
  set -e
  if ((rc == 0)) && grep -Eq '\[CompileOnly\]|\[Compare\].*(finite=True|matched)|matched within the specified tolerance|Top-1 class matched' "${log}"; then
    printf '%s\n' PASS >"${status}"
    echo "DONE arch=v${arch} model=${model} scheme=${scheme}"
  else
    printf 'FAIL_%s\n' "${rc}" >"${status}"
    echo "FAIL arch=v${arch} model=${model} scheme=${scheme} rc=${rc}" >&2
    tail -100 "${log}" >&2
    return "$((rc == 0 ? 1 : rc))"
  fi
}

mkdir -p "${output_root}"
for arch in "${arches[@]}"; do
  export HEXAGON_ARCH_VERSION=${arch}
  for model in "${models[@]}"; do
    for scheme in "${schemes[@]}"; do
      run_case "${arch}" "${model}" "${scheme}"
    done
  done
done

results=${output_root}/results.csv
printf '%s\n' 'arch,model,workload,model_layers,scheme,status,perf_us,latency_ms,host_wall_s,sim_pcycles,correctness,log' >"${results}"
for arch in "${arches[@]}"; do
  for model in "${models[@]}"; do
    for scheme in "${schemes[@]}"; do
      if ((compile_only)); then
        workload_id="full-${model_layers}-compile"
      else
        workload_id=$([[ ${workload} == full ]] && echo "full-${model_layers}" || echo proxy)
      fi
      reported_layers=$([[ ${workload} == full ]] && echo "${model_layers}" || echo NA)
      case_dir=${output_root}/v${arch}/${model}/${workload_id}/${scheme}
      log=${case_dir}/run.log
      status=$(<"${case_dir}/status.txt")
      perf=$(awk -F: '
        /^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);s+=$2;n++}
        /^[[:space:]]*PerfP50:/{gsub(/[[:space:]]/,"",$2);p50=$2}
        END{if(n)printf "%.0f",s;else if(p50!="")printf "%.0f",p50}
      ' "${log}")
      latency=$(awk -v value="${perf:-0}" 'BEGIN{if(value)printf "%.2f",value/1000;else printf "NA"}')
      host_wall=$(awk '/Host wall time:/{print $(NF-8)}' "${log}" | tail -1)
      pcycles=$(awk '/Total:.*Pcycles=/{sub(/^.*Pcycles=/,"");print $1}' "${log}" | tail -1)
      correctness=$(awk '/\[Compare\]|matched within the specified tolerance|Top-1 class matched/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "v${arch}" "${model}" "${workload}" "${reported_layers}" "${scheme}" "${status}" "${perf:-NA}" \
        "${latency}" "${host_wall:-NA}" "${pcycles:-NA}" \
        "${correctness:-NA}" "${log}" >>"${results}"
    done
  done
done
echo "Results: ${results}"
