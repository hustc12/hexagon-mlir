#!/usr/bin/env bash
# Serial five-way full-model HVX comparison.
#
# One maintained driver covers the complete corpus. Model-specific behavior is
# restricted to runner_for(), cli_style_for(), and extra_args_for(); adding a
# model should not require another shell script.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
parent_dir=$(cd -- "${repo_root}/.." && pwd)
venv=${OMNIFETCH_VENV:-${parent_dir}/mlir-env}
python_version=$("${venv}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
triton_build=${TRITON_BUILD_DIR:-${repo_root}/triton/build/cmake.linux-x86_64-cpython-${python_version}}

output_dir=${OUTPUT_DIR:-${parent_dir}/run_artifacts/full_hvx_five_way_$(date +%Y%m%d_%H%M%S)}
remote_dir=${REMOTE_RESULTS_DIR:-/home/huzq85/2-working/working_set/full_hvx_five_way_$(date +%Y%m%d_%H%M%S)}
iterations=${DEVICE_ITERATIONS:-1}
seq_len=${OMNIFETCH_SEQ_LEN:-32}
compile_threads=${HEXAGON_MLIR_COMPILE_THREADS:-auto}
reuse_valid=${REUSE_VALID_LOGS:-1}
list_only=0
alps_p0=0
alps_p0b=0
alps_p1=0
alps_p1_profile=0
alps_p1_hexkl_profile=0
alps_p2a=0
alps_p2b=0
alps_p2c=0
alps_p2d=0
alps_p2e=0
alps_p2f=0
alps_p2g=0
alps_p2gb=0
alps_p2gc=0
alps_p5a=0
alps_p5b=0
alps_p5c=0
alps_p5fa=0
alps_p5fb=0
alps_p5fc=0
alps_p5g=0
alps_p5gb=0
alps_p5gc=0
alps_p5gd=0
alps_p5ge=0
alps_p5gf=0
alps_p5gg=0
alps_p5h=0
alps_p5i=0
alps_p5j=0
alps_p5k=0
alps_p5l=0
alps_p5m=0
alps_p5n=0
alps_p3a=0
alps_p3b=0
alps_p4a=0
models=()

all_models=(
  gpt2
  sd-clip
  qwen2.5-0.5b
  tinyllama-1.1b
  smollm2-1.7b
  swin-transformer
  segformer-mit-b0
  deit-small
  beit-base
  dinov2-small
  whisper-tiny
  hubert-base
  wav2vec2-base
  unispeech-base
  unispeech-sat-base
)

schemes=(pk-hvx apt-hvx hmlir-hvx-hexkl-off hmlir-hvx-hexkl-on item7-only)

usage() {
  cat <<EOF
Usage: scripts/run_full_hvx_five_way.sh [options] [model ...]

Runs complete (non-Debug) models strictly serially with no timeout:
  pk-hvx, apt-hvx, hmlir-hvx-hexkl-off,
  hmlir-hvx-hexkl-on, item7-only.

Options:
  --list                  List supported models and runners
  --alps-p0               Run the six-way ALPS P0 causal matrix instead
  --alps-p0b              Run the five-way ALPS P0b topology matrix instead
  --alps-p1               Run P1 control/elementwise analysis-only ledgers
  --alps-p1-profile       Add loop-level LWP to P1 (monolithic models only)
  --alps-p1-hexkl-profile Add per-phase HexKL/HMX LWP to loop-level profiling
  --alps-p2a              Run P2a elementwise + zero-copy attention candidate
  --alps-p2b              Run P2b producer-direct + P2a cumulative candidate
  --alps-p2c              Run P2c sync fused transform-transfer + stable P2a
  --alps-p2d              Run P2d minimal static admission + stable P2a
  --alps-p2e              Run consumer-driven layout contracts and direct formation
  --alps-p2f              Run P2e plus HVX tiling/vectorization contract propagation
  --alps-p2g              Run P2e plus analysis-only final continuity audit
  --alps-p2g-b            Run strict loop-interchanged direct formation plus P2g audit
  --alps-p2g-c            Run bounded HVX register-tile formation plus P2g audit
  --alps-p5a              Run P2e plus analysis-only contract discharge ledger
  --alps-p5b              Run P5a plus representation-aware input supply analysis
  --alps-p5c              Run P2e plus admitted next-tile layout-supply prefetch
  --alps-p5f-a            Analyze future supply only for P2g-c continuous tiles
  --alps-p5f-b            Prefetch admitted future P2g-c continuous tiles
  --alps-p5f-c            Use physical contiguous/row-segmented P5f-b hints
  --alps-p5g-a            Form exact strided P2g-c source tiles in VTCM
  --alps-p5g-b            Coalesce narrow P2g-c tiles into reusable VTCM windows
  --alps-p5g-c            Async-prefetch coalesced CRP windows with ping-pong VTCM
  --alps-p5g-d            Analyze producer-direct VTCM formation feasibility
  --alps-p5g-e            Redirect proven HVX-only producer epochs directly into VTCM
  --alps-p5g-f            Form proven HVX-only epochs in head-major VTCM layout
  --alps-p5g-g            Also form head-outer/token-inner producer traversal
  --alps-p5h              Redirect padded attention formation to final destination
  --alps-p5i              Form patch convolution directly in contiguous token layout
  --alps-p5j              Form HMX output directly for an immediate FP16 consumer
  --alps-p5k              Clip HMX stores directly into the final non-aligned output
  --alps-p5l              Apply rank-2 broadcast bias during the HMX F16 result drain
  --alps-p5m              Analyze admission for ping-pong async HMX result drain (P5k execution unchanged)
  --alps-p5n              Execute ping-pong VTCM + async DMA HMX result drain
  --alps-p3a              Run P3a exact-readiness contract + P2d + stable P2a
  --alps-p3b              Run P3b descriptor-bound issuer-owned weight DMA overlap
  --alps-p4a              Run P4A telemetry + within-path DMA traffic control
  --output-dir DIR        Local working/result directory
  --remote-dir DIR        nano working_set destination
  --device-iterations N   Device samples per configuration (default: ${iterations})
  --seq-len N             Full LLM prefill length (default: ${seq_len})
  --compile-threads N     CPU cores for the one active model compile
                          (default: auto; memory-aware, capped at 4)
  --no-reuse              Recompile passing configurations
  -h, --help

With no model arguments, all models are run in the declared order. Large
artifacts are synchronized to nano immediately after each configuration.
EOF
}

runner_for() {
  case "$1" in
    gpt2) echo scripts/probe_gpt2_layered_export.py ;;
    sd-clip) echo scripts/probe_clip_layered_export.py ;;
    qwen2.5-0.5b|tinyllama-1.1b|smollm2-1.7b)
      echo scripts/probe_qwen_layered_export.py
      ;;
    swin-transformer) echo benchmark_models/run_swin_transformer.py ;;
    segformer-mit-b0) echo benchmark_models/run_segformer-mit-b0.py ;;
    deit-small) echo benchmark_models/run_deit-small.py ;;
    beit-base) echo benchmark_models/run_beit-base.py ;;
    dinov2-small) echo benchmark_models/run_dinov2-small.py ;;
    whisper-tiny) echo benchmark_models/run_whisper-tiny.py ;;
    hubert-base) echo benchmark_models/run_hubert-base.py ;;
    wav2vec2-base) echo benchmark_models/run_wav2vec2-base.py ;;
    unispeech-base) echo benchmark_models/run_unispeech-base.py ;;
    unispeech-sat-base) echo benchmark_models/run_unispeech-sat-base.py ;;
    *) return 1 ;;
  esac
}

cli_style_for() {
  case "$1" in
    gpt2|sd-clip|qwen2.5-0.5b|tinyllama-1.1b|smollm2-1.7b) echo layered-fp16 ;;
    *) echo phase4 ;;
  esac
}

domain_for() {
  case "$1" in
    gpt2|sd-clip|qwen2.5-0.5b|tinyllama-1.1b|smollm2-1.7b) echo language-text ;;
    swin-transformer|segformer-mit-b0|deit-small|beit-base|dinov2-small) echo vision ;;
    *) echo speech ;;
  esac
}

extra_args_for() {
  case "$1" in
    gpt2)
      printf '%s\n' --seq-len "${seq_len}" --dtype fp16 \
        --device-stage safe_full_model --skip-full-export
      ;;
    sd-clip) printf '%s\n' --dtype fp16 ;;
    qwen2.5-0.5b)
      printf '%s\n' --seq-len "${seq_len}" --model-name Qwen/Qwen2.5-0.5B
      ;;
    tinyllama-1.1b)
      printf '%s\n' --seq-len "${seq_len}" \
        --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
      ;;
    smollm2-1.7b)
      printf '%s\n' --seq-len "${seq_len}" --model-name HuggingFaceTB/SmolLM2-1.7B
      ;;
  esac
}

known_model() {
  local candidate=$1 model
  for model in "${all_models[@]}"; do
    [[ "${candidate}" == "${model}" ]] && return 0
  done
  return 1
}

while (($#)); do
  case "$1" in
    --list) list_only=1; shift ;;
    --alps-p0) alps_p0=1; shift ;;
    --alps-p0b) alps_p0=1; alps_p0b=1; shift ;;
    --alps-p1) alps_p0=1; alps_p1=1; shift ;;
    --alps-p1-profile)
      alps_p0=1; alps_p1=1; alps_p1_profile=1; shift
      ;;
    --alps-p1-hexkl-profile)
      alps_p0=1; alps_p1=1; alps_p1_profile=1; alps_p1_hexkl_profile=1; shift
      ;;
    --alps-p2a) alps_p0=1; alps_p2a=1; shift ;;
    --alps-p2b) alps_p0=1; alps_p2a=1; alps_p2b=1; shift ;;
    --alps-p2c) alps_p0=1; alps_p2a=1; alps_p2c=1; shift ;;
    --alps-p2d) alps_p0=1; alps_p2a=1; alps_p2d=1; shift ;;
    --alps-p2e) alps_p0=1; alps_p2e=1; shift ;;
    --alps-p2f) alps_p0=1; alps_p2e=1; alps_p2f=1; shift ;;
    --alps-p2g) alps_p0=1; alps_p2e=1; alps_p2g=1; shift ;;
    --alps-p2g-b)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gb=1; shift
      ;;
    --alps-p2g-c)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1; shift
      ;;
    --alps-p5a) alps_p0=1; alps_p2e=1; alps_p5a=1; shift ;;
    --alps-p5b) alps_p0=1; alps_p2e=1; alps_p5a=1; alps_p5b=1; shift ;;
    --alps-p5c) alps_p0=1; alps_p2e=1; alps_p5a=1; alps_p5b=1; alps_p5c=1; shift ;;
    --alps-p5f-a)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1; alps_p5fa=1; shift
      ;;
    --alps-p5f-b)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5fb=1; shift
      ;;
    --alps-p5f-c)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5fb=1; alps_p5fc=1; shift
      ;;
    --alps-p5g-a)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5g=1; shift
      ;;
    --alps-p5g-b)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gb=1; shift
      ;;
    --alps-p5g-c)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gc=1; shift
      ;;
    --alps-p5g-d)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; shift
      ;;
    --alps-p5g-e)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5ge=1; shift
      ;;
    --alps-p5g-f)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; shift
      ;;
    --alps-p5g-g)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; shift
      ;;
    --alps-p5h)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; shift
      ;;
    --alps-p5i)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; alps_p5i=1; shift
      ;;
    --alps-p5j)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; alps_p5i=1; alps_p5j=1; shift
      ;;
    --alps-p5k)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; alps_p5i=1; alps_p5j=1; alps_p5k=1; shift
      ;;
    --alps-p5l)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; alps_p5i=1; alps_p5j=1; alps_p5k=1; alps_p5l=1; shift
      ;;
    --alps-p5m)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; alps_p5i=1; alps_p5j=1; alps_p5k=1; alps_p5m=1; shift
      ;;
    --alps-p5n)
      alps_p0=1; alps_p2e=1; alps_p2g=1; alps_p2gc=1
      alps_p5fa=1; alps_p5gd=1; alps_p5gf=1; alps_p5gg=1; alps_p5h=1; alps_p5i=1; alps_p5j=1; alps_p5k=1; alps_p5m=1; alps_p5n=1; shift
      ;;
    --alps-p3a) alps_p0=1; alps_p2a=1; alps_p2d=1; alps_p3a=1; shift ;;
    --alps-p3b)
      alps_p0=1; alps_p2a=1; alps_p2d=1; alps_p3a=1; alps_p3b=1; shift
      ;;
    --alps-p4a)
      alps_p0=1; alps_p2a=1; alps_p2d=1; alps_p3a=1; alps_p3b=1; alps_p4a=1; shift
      ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --remote-dir) remote_dir=$2; shift 2 ;;
    --device-iterations) iterations=$2; shift 2 ;;
    --seq-len) seq_len=$2; shift 2 ;;
    --compile-threads) compile_threads=$2; shift 2 ;;
    --no-reuse) reuse_valid=0; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) models+=("$1"); shift ;;
  esac
done

if ((alps_p5n)); then
  schemes=(alps-hmx-async-drain)
elif ((alps_p5m)); then
  schemes=(alps-hmx-async-drain-analysis)
elif ((alps_p5l)); then
  schemes=(alps-hmx-f16-bias-epilogue-formation)
elif ((alps_p5k)); then
  schemes=(alps-hmx-direct-output-formation)
elif ((alps_p5j)); then
  schemes=(alps-hmx-f16-epilogue-formation)
elif ((alps_p5i)); then
  schemes=(alps-patch-conv-formation)
elif ((alps_p5h)); then
  schemes=(alps-attention-destination-formation)
elif ((alps_p5gg)); then
  schemes=(alps-crp-producer-head-major-contiguous)
elif ((alps_p5gf)); then
  schemes=(alps-crp-producer-direct-head-major)
elif ((alps_p5ge)); then
  schemes=(alps-crp-producer-direct-vtcm)
elif ((alps_p5gd)); then
  schemes=(alps-crp-producer-direct-analysis)
elif ((alps_p5gc)); then
  schemes=(alps-crp-vtcm-async-window)
elif ((alps_p5gb)); then
  schemes=(alps-crp-vtcm-window)
elif ((alps_p5g)); then
  schemes=(alps-crp-vtcm-formation)
elif ((alps_p5fc)); then
  schemes=(alps-crp-segmented-supply)
elif ((alps_p5fb)); then
  schemes=(alps-crp-supply-prefetch)
elif ((alps_p5fa)); then
  schemes=(alps-crp-supply-analysis)
elif ((alps_p5c)); then
  schemes=(alps-layout-supply-prefetch)
elif ((alps_p2gc)); then
  schemes=(alps-register-tile-direct)
elif ((alps_p2gb)); then
  schemes=(alps-loop-interchanged-direct)
elif ((alps_p2g)); then
  schemes=(alps-continuity-audit)
elif ((alps_p4a)); then
  schemes=(alps-elementwise-traffic-control)
elif ((alps_p3b)); then
  schemes=(alps-elementwise-exact-overlap)
elif ((alps_p3a)); then
  schemes=(alps-elementwise-exact-readiness)
elif ((alps_p5b)); then
  schemes=(alps-representation-supply-analysis)
elif ((alps_p5a)); then
  schemes=(alps-consumer-driven-layout alps-contract-discharge-ledger)
elif ((alps_p2f)); then
  schemes=(hmlir-hvx-hexkl-on alps-consumer-driven-layout alps-consumer-layout-propagation)
elif ((alps_p2e)); then
  schemes=(hmlir-hvx-hexkl-on alps-consumer-driven-layout)
elif ((alps_p2d)); then
  schemes=(alps-elementwise-admission)
elif ((alps_p2c)); then
  schemes=(alps-elementwise-fused-transfer)
elif ((alps_p2b)); then
  schemes=(alps-elementwise-producer-direct)
elif ((alps_p2a)); then
  schemes=(alps-elementwise-zero-copy)
elif ((alps_p1)); then
  schemes=(hmlir-hvx-hexkl-on alps-elementwise-fusion)
elif ((alps_p0b)); then
  schemes=(hmlir-hvx-hexkl-on alps-elementwise-fusion alps-multi-use-fusion alps-split-reduction alps-fusion)
elif ((alps_p0)); then
  schemes=(hmlir-hvx-hexkl-on alps-semantic alps-fusion alps-slicing alps-runtime alps-legacy-all)
fi

[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid iterations" >&2; exit 2; }
[[ "${seq_len}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid sequence length" >&2; exit 2; }
if [[ "${compile_threads}" == auto ]]; then
  available_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
  cpu_count=$(getconf _NPROCESSORS_ONLN)
  # Full-model lowering can consume several GiB per simultaneously active pass.
  # Preserve 4 GiB for the OS/toolchain and budget 2 GiB per compile thread.
  memory_threads=$((available_kib > 4194304 ? (available_kib - 4194304) / 2097152 : 1))
  ((memory_threads < 1)) && memory_threads=1
  compile_threads=${cpu_count}
  ((compile_threads > 4)) && compile_threads=4
  ((compile_threads > memory_threads)) && compile_threads=${memory_threads}
fi
[[ "${compile_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid compile thread count" >&2; exit 2; }
if ((${#models[@]} == 0)); then models=("${all_models[@]}"); fi
for model in "${models[@]}"; do
  known_model "${model}" || { echo "Unknown model: ${model}" >&2; exit 2; }
  if ((alps_p1_profile)) && [[ "$(cli_style_for "${model}")" == layered-fp16 ]]; then
    echo "P1 LWP profiling currently supports monolithic models only: ${model}" >&2
    exit 2
  fi
done

if ((list_only)); then
  printf '%-3s %-23s %-14s %s\n' '#' model domain runner
  index=0
  for model in "${all_models[@]}"; do
    index=$((index + 1))
    printf '%-3s %-23s %-14s %s\n' \
      "${index}" "${model}" "$(domain_for "${model}")" "$(runner_for "${model}")"
  done
  exit 0
fi

export PYTHONPATH="${repo_root}/triton/python:${repo_root}/benchmark_models"
export TRITON_PLUGIN_DIRS="${repo_root}/triton_shared;${repo_root}/qcom_hexagon_backend"
export PATH="${triton_build}/third_party/qcom_hexagon_backend/bin:${triton_build}/third_party/triton_shared/tools/triton-shared-opt:${PATH}"
export HEXAGON_MLIR_ROOT=${HEXAGON_MLIR_ROOT:-${repo_root}}
export HEXAGON_ARCH_VERSION=${HEXAGON_ARCH_VERSION:-73}
export ANDROID_SERIAL=${ANDROID_SERIAL:-49d1c7b2}
export ANDROID_HOST=${ANDROID_HOST:-}
export HOST_TOOLCHAIN=${HOST_TOOLCHAIN:-${parent_dir}/HOST_TOOLCHAIN}
export HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-${parent_dir}/HEXAGON_SDK/Hexagon_SDK/6.4.0.2}
export HEXAGON_TOOLS=${HEXAGON_TOOLS:-${parent_dir}/HEXAGON_TOOLS/Tools}
export HEXKL_ROOT=${HEXKL_ROOT:-${parent_dir}/HEXKL_DIR/hexkl_addon}
export HEXAGON_RUNTIME_LIBS_DIR=${HEXAGON_RUNTIME_LIBS_DIR:-${triton_build}/third_party/qcom_hexagon_backend/bin/runtime}
export OMNIFETCH_DSP_HEAP_MB=${OMNIFETCH_DSP_HEAP_MB:-512}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONUNBUFFERED=1
export HEXAGON_RUN_RETRIES=1
export HEXAGON_MLIR_COMPILE_THREADS=${compile_threads}
echo "HOST_COMPILE threads=${compile_threads} online_cpus=$(getconf _NPROCESSORS_ONLN) mem_available_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"

mkdir -p "${output_dir}"
# A tool/session interruption does not necessarily terminate its child process.
# Hold one advisory lock for the complete serial matrix so a resumed dialogue
# cannot accidentally launch a duplicate compiler against the same case tree.
runner_lock=${output_dir}/.runner.lock
exec {runner_lock_fd}>"${runner_lock}"
if ! flock --nonblock "${runner_lock_fd}"; then
  echo "Another full-model runner still owns ${runner_lock}; refusing duplicate run" >&2
  exit 75
fi
ssh nano "mkdir -p '${remote_dir}'"
results=${output_dir}/results.csv
if [[ ! -f "${results}" ]]; then
  printf '%s\n' 'model,domain,scheme,status,perf_us,latency_ms,ratio_over_item7,compile_s,prefetch_hints,runtime_issued,runtime_issued_bytes,kv_pairs,kv_sites,correctness,log' > "${results}"
fi

sync_case() {
  local model=$1 scheme=$2 case_dir=$3
  ssh nano "mkdir -p '${remote_dir}/${model}/${scheme}'"
  rsync -a --partial "${case_dir}/" "nano:${remote_dir}/${model}/${scheme}/"
}

cleanup_case_generated() {
  local case_dir=$1 generated
  # Preserve the compact evidence needed for reuse and reporting. Only remove
  # compiler/launcher products after sync_case has returned successfully.
  [[ -n "${output_dir}" && "${case_dir}" == "${output_dir}/"* ]] || {
    echo "Refusing generated-file cleanup outside output_dir: ${case_dir}" >&2
    return 2
  }
  for generated in "${case_dir}/artifacts" "${case_dir}/mlirbc"; do
    [[ -e "${generated}" ]] && rm -rf -- "${generated}"
  done
}

move_case_to_remote() {
  local model=$1 scheme=$2 case_dir=$3
  sync_case "${model}" "${scheme}" "${case_dir}"
  # ALPS P0 can generate many full-model objects. The remote copy is the
  # authoritative result: after rsync succeeds, remove the complete local case
  # rather than retaining a second "backup" on the constrained local disk.
  if ((alps_p0)); then
    [[ -n "${output_dir}" && "${case_dir}" == "${output_dir}/"* ]] || {
      echo "Refusing case removal outside output_dir: ${case_dir}" >&2
      return 2
    }
    rm -rf -- "${case_dir}"
  else
    cleanup_case_generated "${case_dir}"
  fi
}

audit_case_codegen() {
  local case_dir=$1 object_dir
  local csv=${case_dir}/codegen.csv
  local -a object_dirs=()
  mapfile -t object_dirs < <(
    find "${case_dir}/artifacts" -type f -name '_mlir_ciface_*.o' \
      ! -name '*-consts-*.o' -printf '%h\n' 2>/dev/null | sort -u
  )
  for object_dir in "${object_dirs[@]}"; do
    "${repo_root}/scripts/audit_hexagon_codegen.sh" "${object_dir}" "${csv}" \
      >>"${case_dir}/codegen.log" 2>&1 || {
        echo "WARN: codegen audit failed for ${object_dir}" >>"${case_dir}/codegen.log"
      }
  done
}

extract_alps_p1_ledger() {
  local case_dir=$1 log=${case_dir}/run.log
  ((alps_p1 || alps_p2a || alps_p2b || alps_p2c || alps_p2d || alps_p3a || alps_p3b || alps_p4a)) || return 0
  grep '^\[ALPS-P1-' "${log}" >"${case_dir}/alps_p1_ledger.log" || true
  "${venv}/bin/python" "${repo_root}/scripts/summarize_alps_movement_ledger.py" \
    "${case_dir}/alps_p1_ledger.log" \
    --csv "${case_dir}/alps_p1_summary.csv" \
    --markdown "${case_dir}/alps_p1_summary.md" \
    --sites-csv "${case_dir}/alps_p1_sites.csv" \
    --sites-markdown "${case_dir}/alps_p1_sites.md"
}

extract_alps_p2g_audit() {
  local case_dir=$1 log=${case_dir}/run.log
  ((alps_p2g)) || return 0
  grep '^\[ALPS-P2G-' "${log}" >"${case_dir}/alps_p2g_continuity.log" || true
}

collect_alps_p1_profile() {
  local case_dir=$1
  ((alps_p1_profile)) || return 0
  local info_dump=${case_dir}/lwp_infodump.txt
  [[ -f /tmp/lwp_infodump.txt ]] || {
    echo "Missing /tmp/lwp_infodump.txt after LWP run" >&2
    return 2
  }
  cp -- /tmp/lwp_infodump.txt "${info_dump}"
  "${venv}/bin/python" "${repo_root}/scripts/summarize_alps_lwp.py" \
    --artifact-root "${case_dir}/artifacts" \
    --info-dump "${info_dump}" \
    --ledger "${case_dir}/alps_p1_ledger.log" \
    --csv "${case_dir}/alps_p1_lwp_regions.csv" \
    --markdown "${case_dir}/alps_p1_lwp_regions.md"
}

collect_runner_intermediates() {
  local case_dir=$1 marker=$2 generated
  local destination=${case_dir}/runner_intermediates
  while IFS= read -r -d '' generated; do
    mkdir -p "${destination}"
    mv -- "${generated}" "${destination}/"
  done < <(
    find "${repo_root}/benchmark_models" -maxdepth 1 -type f \
      -name '*_f16matmul.mlir' -newer "${marker}" -print0
  )
}

extract_admitted_ids() {
  "${venv}/bin/python" - "$1" <<'PY'
import re
import sys
ids, seen = [], set()
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = re.search(r"\[prefetch-kernel-hx\].*?admitted_ids=(.*?)\s*$", line)
    if not match or match.group(1) == "none":
        continue
    for item in (part.strip() for part in match.group(1).split(",")):
        if item and item not in seen:
            seen.add(item)
            ids.append(item)
print(",".join(ids))
PY
}

base_args_for() {
  local model=$1
  if [[ "$(cli_style_for "${model}")" == phase4 ]]; then
    printf '%s\n' --backend-profile hvx-vector
  elif [[ "$(cli_style_for "${model}")" != layered-fp16 ]]; then
    printf '%s\n' --enable-hvx-vector
  fi
  if ((alps_p1_profile)); then
    printf '%s\n' --enable-lwp --lwp-loop-depth 1
  fi
  if ((alps_p1_hexkl_profile)); then
    printf '%s\n' --lwp-hexkl-phases
  fi
  printf '%s\n' --device-iterations "${iterations}"
  extra_args_for "${model}"
}

scheme_args_for() {
  local scheme=$1 candidate_ids=${2:-}
  case "${scheme}" in
    pk-hvx)
      printf '%s\n' --enable-hexkl --prefetch-baseline prefetch-kernel-hx \
        --prefetch-baseline-distance 1
      ;;
    apt-hvx)
      [[ -n "${candidate_ids}" ]] || return 2
      printf '%s\n' --enable-hexkl --prefetch-baseline apt-get-hx \
        --prefetch-baseline-distance 1 --apt-get-hx-manual-candidate-ids "${candidate_ids}"
      ;;
    hmlir-hvx-hexkl-off) ;;
    hmlir-hvx-hexkl-on) printf '%s\n' --enable-hexkl ;;
    item7-only)
      printf '%s\n' --enable-hexkl --enable-omnifetch-kv-cache-prefetch \
        --disable-layout-aware --disable-omnifetch-adaptive
      ;;
    alps-semantic|alps-fusion|alps-elementwise-fusion|alps-multi-use-fusion|alps-split-reduction|alps-slicing|alps-runtime|alps-legacy-all)
      printf '%s\n' --enable-hexkl \
        --alps-p0-mode "${scheme#alps-}" \
        --disable-layout-aware --disable-omnifetch-adaptive
      ;;
    alps-elementwise-zero-copy)
      printf '%s\n' --enable-hexkl --alps-p0-mode elementwise-fusion \
        --disable-layout-aware --disable-omnifetch-adaptive
      ;;
    alps-elementwise-producer-direct)
      printf '%s\n' --enable-hexkl --alps-p0-mode elementwise-fusion \
        --disable-layout-aware --disable-omnifetch-adaptive
      ;;
    alps-elementwise-fused-transfer)
      printf '%s\n' --enable-hexkl --alps-p0-mode elementwise-fusion \
        --disable-layout-aware --disable-omnifetch-adaptive
      ;;
    alps-consumer-driven-layout|alps-consumer-layout-propagation|alps-continuity-audit|alps-loop-interchanged-direct|alps-register-tile-direct|alps-crp-supply-analysis|alps-crp-supply-prefetch|alps-crp-segmented-supply|alps-crp-vtcm-formation|alps-crp-vtcm-window|alps-crp-vtcm-async-window|alps-crp-producer-direct-analysis|alps-crp-producer-direct-vtcm|alps-crp-producer-direct-head-major|alps-crp-producer-head-major-contiguous|alps-attention-destination-formation|alps-patch-conv-formation|alps-hmx-f16-epilogue-formation|alps-hmx-direct-output-formation|alps-hmx-f16-bias-epilogue-formation|alps-hmx-async-drain-analysis|alps-hmx-async-drain|alps-contract-discharge-ledger|alps-representation-supply-analysis|alps-layout-supply-prefetch)
      printf '%s\n' --enable-hexkl --disable-layout-aware --disable-omnifetch-adaptive
      ;;
    alps-elementwise-admission|alps-elementwise-exact-readiness|alps-elementwise-exact-overlap|alps-elementwise-traffic-control)
      printf '%s\n' --enable-hexkl --alps-p0-mode elementwise-fusion \
        --disable-layout-aware --disable-omnifetch-adaptive
      ;;
  esac
}

passing_log() {
  local log=$1 status_file=$2
  [[ -f "${status_file}" && "$(<"${status_file}")" == PASS ]] &&
    grep -q '^[[:space:]]*Perf:' "${log}"
}

upsert_result() {
  local model=$1 scheme=$2 status=$3 log=$4
  local perf_us latency compile_s hints issued issued_bytes kv_pairs kv_sites correctness recorded_log
  # Monolithic runners emit one Perf line. Layered language runners emit one
  # per embedding/block/head stage; their complete-model metric is the sum.
  perf_us=$(awk -F: '/^[[:space:]]*Perf:/{gsub(/[[:space:]]/,"",$2);s+=$2;n++}END{if(n)printf "%.0f",s}' "${log}")
  latency=NA
  [[ -z "${perf_us}" ]] || latency=$(awk -v value="${perf_us}" 'BEGIN{printf "%.2f",value/1000.0}')
  compile_s=$(awk '/Compilation from initial MLIR to .so took/{s+=$(NF-1);n++}END{if(n)printf "%.6f",s}' "${log}")
  hints=$(awk 'match($0,/hints=[0-9]+/){v=substr($0,RSTART+6,RLENGTH-6)}END{print v+0}' "${log}")
  issued=$(awk 'match($0,/issued=[0-9]+/){v=substr($0,RSTART+7,RLENGTH-7)}END{print v+0}' "${log}")
  issued_bytes=$(awk 'match($0,/issued_bytes=[0-9]+/){v=substr($0,RSTART+13,RLENGTH-13)}END{print v+0}' "${log}")
  kv_pairs=$(awk '/\[KVCacheMetadataRoles\]/{for(i=1;i<=NF;i++)if(index($i,"key=")==1){v=$i;sub("key=","",v);s=v}}END{print s+0}' "${log}")
  kv_sites=$(awk '/\[KVCachePrefetch\]/{for(i=1;i<=NF;i++)if(index($i,"sites=")==1){v=$i;sub("sites=","",v);gsub(/[^0-9].*/,"",v);s=v}}END{print s+0}' "${log}")
  correctness=$(awk '/\[Compare\]/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  if [[ -z "${correctness}" ]]; then
    correctness=$(awk '/Hexagon and CPU results matched within the specified tolerance\.|Top-1 class matched \(HexKL numerical tolerance\)/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  fi
  if [[ -z "${correctness}" ]]; then
    correctness=$(awk '/\[(LayeredDeviceFullCompare|CLIPDeviceFullCompare|QwenDeviceFullCompare)\]/{v=$0}END{gsub(/,/,";",v);print v}' "${log}")
  fi
  recorded_log=${log}
  if ((alps_p0)); then
    recorded_log=${remote_dir}/${model}/${scheme}/run.log
  fi
  awk -F, -v m="${model}" -v s="${scheme}" 'NR==1 || !($1==m && $3==s)' "${results}" > "${results}.tmp"
  printf '%s,%s,%s,%s,%s,%s,NA,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "$(domain_for "${model}")" "${scheme}" "${status}" \
    "${perf_us:-NA}" "${latency}" "${compile_s:-NA}" "${hints}" \
    "${issued}" "${issued_bytes}" "${kv_pairs}" "${kv_sites}" \
    "${correctness}" "${recorded_log}" >> "${results}.tmp"
  mv "${results}.tmp" "${results}"
}

run_case() {
  local model=$1 scheme=$2 candidate_ids=${3:-}
  local runner=${repo_root}/$(runner_for "${model}")
  local case_dir=${output_dir}/${model}/${scheme}
  local log=${case_dir}/run.log status_file=${case_dir}/status.txt
  local marker=${case_dir}/.case-started
  local -a args=()
  mkdir -p "${case_dir}/artifacts"
  if ((reuse_valid)) && passing_log "${log}" "${status_file}"; then
    echo "REUSE model=${model} scheme=${scheme}"
    upsert_result "${model}" "${scheme}" PASS "${log}"
    return 0
  fi
  touch "${marker}"
  mapfile -t args < <(base_args_for "${model}"; scheme_args_for "${scheme}" "${candidate_ids}")
  if [[ "$(cli_style_for "${model}")" == layered-fp16 ]]; then
    args+=(--output-dir "${case_dir}/mlirbc")
  fi
  echo "START model=${model} scheme=${scheme} $(date --iso-8601=seconds)"
  # P5n changes only the final HMX drain.  Reuse P5m's cumulative feature
  # identity for every earlier-stage environment gate so control/treatment
  # differ solely by ALPS_ENABLE_HMX_ASYNC_DRAIN.
  result_scheme=${scheme}
  if [[ ${scheme} == alps-hmx-async-drain ]]; then
    scheme=alps-hmx-async-drain-analysis
  fi
  set +e
    ALPS_ENABLE_MOVEMENT_LEDGER="$([[ ${alps_p1} -eq 1 || ${alps_p2a} -eq 1 || ${alps_p2b} -eq 1 || ${alps_p2c} -eq 1 || ${alps_p2d} -eq 1 || ${alps_p2e} -eq 1 || ${alps_p3a} -eq 1 || ${alps_p3b} -eq 1 || ${alps_p4a} -eq 1 ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_ZERO_COPY_ATTENTION="$([[ ${scheme} == alps-elementwise-zero-copy || ${scheme} == alps-elementwise-producer-direct || ${scheme} == alps-elementwise-fused-transfer || ${scheme} == alps-elementwise-admission || ${scheme} == alps-elementwise-exact-readiness || ${scheme} == alps-elementwise-exact-overlap || ${scheme} == alps-elementwise-traffic-control ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_PRODUCER_DIRECT_ATTENTION="$([[ ${scheme} == alps-elementwise-producer-direct ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CONSUMER_DRIVEN_LAYOUT="$([[ ${scheme} == alps-consumer-driven-layout || ${scheme} == alps-consumer-layout-propagation || ${scheme} == alps-continuity-audit || ${scheme} == alps-loop-interchanged-direct || ${scheme} == alps-register-tile-direct || ${scheme} == alps-crp-supply-analysis || ${scheme} == alps-crp-supply-prefetch || ${scheme} == alps-crp-segmented-supply || ${scheme} == alps-crp-vtcm-formation || ${scheme} == alps-crp-vtcm-window || ${scheme} == alps-crp-vtcm-async-window || ${scheme} == alps-crp-producer-direct-analysis || ${scheme} == alps-crp-producer-direct-vtcm || ${scheme} == alps-crp-producer-direct-head-major || ${scheme} == alps-crp-producer-head-major-contiguous || ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis || ${scheme} == alps-hmx-async-drain || ${scheme} == alps-contract-discharge-ledger || ${scheme} == alps-representation-supply-analysis || ${scheme} == alps-layout-supply-prefetch ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CONSUMER_LAYOUT_PROPAGATION="$([[ ${scheme} == alps-consumer-layout-propagation ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CONTINUITY_AUDIT="$([[ ${scheme} == alps-continuity-audit || ${scheme} == alps-loop-interchanged-direct || ${scheme} == alps-register-tile-direct || ${scheme} == alps-crp-supply-analysis || ${scheme} == alps-crp-supply-prefetch || ${scheme} == alps-crp-segmented-supply || ${scheme} == alps-crp-vtcm-formation || ${scheme} == alps-crp-vtcm-window || ${scheme} == alps-crp-vtcm-async-window || ${scheme} == alps-crp-producer-direct-analysis || ${scheme} == alps-crp-producer-direct-vtcm || ${scheme} == alps-crp-producer-direct-head-major || ${scheme} == alps-crp-producer-head-major-contiguous || ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_LOOP_INTERCHANGED_DIRECT="$([[ ${scheme} == alps-loop-interchanged-direct ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_REGISTER_TILE_FORMATION="$([[ ${scheme} == alps-register-tile-direct || ${scheme} == alps-crp-supply-analysis || ${scheme} == alps-crp-supply-prefetch || ${scheme} == alps-crp-segmented-supply || ${scheme} == alps-crp-vtcm-formation || ${scheme} == alps-crp-vtcm-window || ${scheme} == alps-crp-vtcm-async-window || ${scheme} == alps-crp-producer-direct-analysis || ${scheme} == alps-crp-producer-direct-vtcm || ${scheme} == alps-crp-producer-direct-head-major || ${scheme} == alps-crp-producer-head-major-contiguous || ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CONTRACT_DISCHARGE_LEDGER="$([[ ${scheme} == alps-contract-discharge-ledger ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_REPRESENTATION_SUPPLY_ANALYSIS="$([[ ${scheme} == alps-representation-supply-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_LAYOUT_SUPPLY_PREFETCH="$([[ ${scheme} == alps-layout-supply-prefetch ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_SUPPLY_ANALYSIS="$([[ ${scheme} == alps-crp-supply-analysis || ${scheme} == alps-crp-supply-prefetch || ${scheme} == alps-crp-segmented-supply || ${scheme} == alps-crp-vtcm-formation || ${scheme} == alps-crp-vtcm-window || ${scheme} == alps-crp-vtcm-async-window || ${scheme} == alps-crp-producer-direct-analysis || ${scheme} == alps-crp-producer-direct-vtcm || ${scheme} == alps-crp-producer-direct-head-major || ${scheme} == alps-crp-producer-head-major-contiguous || ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_SUPPLY_PREFETCH="$([[ ${scheme} == alps-crp-supply-prefetch || ${scheme} == alps-crp-segmented-supply ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_SEGMENTED_SUPPLY="$([[ ${scheme} == alps-crp-segmented-supply ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_VTCM_FORMATION="$([[ ${scheme} == alps-crp-vtcm-formation ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_VTCM_WINDOW="$([[ ${scheme} == alps-crp-vtcm-window ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_VTCM_ASYNC_WINDOW="$([[ ${scheme} == alps-crp-vtcm-async-window ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_PRODUCER_DIRECT_ANALYSIS="$([[ ${scheme} == alps-crp-producer-direct-analysis || ${scheme} == alps-crp-producer-direct-vtcm ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_PRODUCER_DIRECT_VTCM="$([[ ${scheme} == alps-crp-producer-direct-vtcm ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_PRODUCER_DIRECT_HEAD_MAJOR="$([[ ${scheme} == alps-crp-producer-direct-head-major || ${scheme} == alps-crp-producer-head-major-contiguous || ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_CRP_PRODUCER_LOOP_FORMATION="$([[ ${scheme} == alps-crp-producer-head-major-contiguous || ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_ATTENTION_DESTINATION_FORMATION="$([[ ${scheme} == alps-attention-destination-formation || ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_PATCH_CONV_FORMATION="$([[ ${scheme} == alps-patch-conv-formation || ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_HMX_F16_EPILOGUE_FORMATION="$([[ ${scheme} == alps-hmx-f16-epilogue-formation || ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_HMX_DIRECT_OUTPUT_FORMATION="$([[ ${scheme} == alps-hmx-direct-output-formation || ${scheme} == alps-hmx-f16-bias-epilogue-formation || ${scheme} == alps-hmx-async-drain-analysis || ${scheme} == alps-hmx-async-drain ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_HMX_F16_BIAS_EPILOGUE_FORMATION="$([[ ${scheme} == alps-hmx-f16-bias-epilogue-formation ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_HMX_ASYNC_DRAIN_ANALYSIS="$([[ ${scheme} == alps-hmx-async-drain-analysis || ${scheme} == alps-hmx-async-drain ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_HMX_ASYNC_DRAIN="$([[ ${result_scheme} == alps-hmx-async-drain ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_FUSED_TRANSFORM_TRANSFER="$([[ ${scheme} == alps-elementwise-fused-transfer ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_MINIMAL_STATIC_ADMISSION="$([[ ${scheme} == alps-elementwise-admission || ${scheme} == alps-elementwise-exact-readiness || ${scheme} == alps-elementwise-exact-overlap || ${scheme} == alps-elementwise-traffic-control ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_EXACT_READINESS="$([[ ${scheme} == alps-elementwise-exact-readiness || ${scheme} == alps-elementwise-exact-overlap || ${scheme} == alps-elementwise-traffic-control ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_EXACT_OVERLAP="$([[ ${scheme} == alps-elementwise-exact-overlap || ${scheme} == alps-elementwise-traffic-control ]] && echo 1 || echo 0)" \
    ALPS_ENABLE_TRAFFIC_CONTROL="$([[ ${scheme} == alps-elementwise-traffic-control ]] && echo 1 || echo 0)" \
    HEXAGON_MLIR_DUMP_DIR="${case_dir}/artifacts" \
    "${venv}/bin/python" "${runner}" "${args[@]}" >"${log}" 2>&1
  rc=$?
  scheme=${result_scheme}
  set -e
  collect_runner_intermediates "${case_dir}" "${marker}"
  rm -f -- "${marker}"
  if ((rc == 0)) && grep -q '^[[:space:]]*Perf:' "${log}"; then
    printf '%s\n' PASS > "${status_file}"
    upsert_result "${model}" "${scheme}" PASS "${log}"
  else
    failure_rc=${rc}
    ((failure_rc != 0)) || failure_rc=1
    printf 'FAIL_%s\n' "${rc}" > "${status_file}"
    upsert_result "${model}" "${scheme}" "FAIL_${rc}" "${log}"
    extract_alps_p1_ledger "${case_dir}"
    extract_alps_p2g_audit "${case_dir}"
    collect_alps_p1_profile "${case_dir}" || true
    audit_case_codegen "${case_dir}"
    move_case_to_remote "${model}" "${scheme}" "${case_dir}"
    echo "FAIL model=${model} scheme=${scheme} rc=${rc} log=${log}" >&2
    return "${failure_rc}"
  fi
  extract_alps_p1_ledger "${case_dir}"
  extract_alps_p2g_audit "${case_dir}"
  collect_alps_p1_profile "${case_dir}"
  audit_case_codegen "${case_dir}"
  move_case_to_remote "${model}" "${scheme}" "${case_dir}"
  echo "DONE model=${model} scheme=${scheme}"
}

write_ratios() {
  "${venv}/bin/python" - "${results}" "${output_dir}/summary.md" "${alps_p0}" "${alps_p0b}" "${alps_p1}" "${alps_p2a}" "${alps_p2b}" "${alps_p2c}" "${alps_p2d}" "${alps_p2e}" "${alps_p2f}" "${alps_p2g}" "${alps_p2gb}" "${alps_p2gc}" "${alps_p5a}" "${alps_p5b}" "${alps_p5c}" "${alps_p5fa}" "${alps_p5fb}" "${alps_p5fc}" "${alps_p5g}" "${alps_p5gb}" "${alps_p5gc}" "${alps_p5gd}" "${alps_p5ge}" "${alps_p5gf}" "${alps_p5gg}" "${alps_p5h}" "${alps_p5i}" "${alps_p5j}" "${alps_p5k}" "${alps_p5l}" "${alps_p5m}" "${alps_p5n}" "${alps_p3a}" "${alps_p3b}" "${alps_p4a}" <<'PY'
import csv
import pathlib
import sys

csv_path = pathlib.Path(sys.argv[1])
md_path = pathlib.Path(sys.argv[2])
alps_p0 = bool(int(sys.argv[3]))
alps_p0b = bool(int(sys.argv[4]))
alps_p1 = bool(int(sys.argv[5]))
alps_p2a = bool(int(sys.argv[6]))
alps_p2b = bool(int(sys.argv[7]))
alps_p2c = bool(int(sys.argv[8]))
alps_p2d = bool(int(sys.argv[9]))
alps_p2e = bool(int(sys.argv[10]))
alps_p2f = bool(int(sys.argv[11]))
alps_p2g = bool(int(sys.argv[12]))
alps_p2gb = bool(int(sys.argv[13]))
alps_p2gc = bool(int(sys.argv[14]))
alps_p5a = bool(int(sys.argv[15]))
alps_p5b = bool(int(sys.argv[16]))
alps_p5c = bool(int(sys.argv[17]))
alps_p5fa = bool(int(sys.argv[18]))
alps_p5fb = bool(int(sys.argv[19]))
alps_p5fc = bool(int(sys.argv[20]))
alps_p5g = bool(int(sys.argv[21]))
alps_p5gb = bool(int(sys.argv[22]))
alps_p5gc = bool(int(sys.argv[23]))
alps_p5gd = bool(int(sys.argv[24]))
alps_p5ge = bool(int(sys.argv[25]))
alps_p5gf = bool(int(sys.argv[26]))
alps_p5gg = bool(int(sys.argv[27]))
alps_p5h = bool(int(sys.argv[28]))
alps_p5i = bool(int(sys.argv[29]))
alps_p5j = bool(int(sys.argv[30]))
alps_p5k = bool(int(sys.argv[31]))
alps_p5l = bool(int(sys.argv[32]))
alps_p5m = bool(int(sys.argv[33]))
alps_p5n = bool(int(sys.argv[34]))
alps_p3a = bool(int(sys.argv[35]))
alps_p3b = bool(int(sys.argv[36]))
alps_p4a = bool(int(sys.argv[37]))

def selected_reference():
    if alps_p5n: return "alps-hmx-async-drain"
    if alps_p5m: return "alps-hmx-async-drain-analysis"
    if alps_p5l: return "alps-hmx-f16-bias-epilogue-formation"
    if alps_p5k: return "alps-hmx-direct-output-formation"
    if alps_p5j: return "alps-hmx-f16-epilogue-formation"
    if alps_p5i: return "alps-patch-conv-formation"
    if alps_p5h: return "alps-attention-destination-formation"
    if alps_p5gg: return "alps-crp-producer-head-major-contiguous"
    if alps_p5gf: return "alps-crp-producer-direct-head-major"
    if alps_p5ge: return "alps-crp-producer-direct-vtcm"
    if alps_p5gd: return "alps-crp-producer-direct-analysis"
    if alps_p5gc: return "alps-crp-vtcm-async-window"
    if alps_p5gb: return "alps-crp-vtcm-window"
    if alps_p5g: return "alps-crp-vtcm-formation"
    if alps_p5fc: return "alps-crp-segmented-supply"
    if alps_p5fb: return "alps-crp-supply-prefetch"
    if alps_p5fa: return "alps-crp-supply-analysis"
    if alps_p5c: return "alps-layout-supply-prefetch"
    if alps_p2gc: return "alps-register-tile-direct"
    if alps_p2gb: return "alps-loop-interchanged-direct"
    if alps_p2g: return "alps-continuity-audit"
    if alps_p4a: return "alps-elementwise-traffic-control"
    if alps_p3b: return "alps-elementwise-exact-overlap"
    if alps_p3a: return "alps-elementwise-exact-readiness"
    if alps_p5b: return "alps-representation-supply-analysis"
    if alps_p5a: return "alps-contract-discharge-ledger"
    if alps_p2f: return "alps-consumer-layout-propagation"
    if alps_p2e: return "alps-consumer-driven-layout"
    if alps_p2d: return "alps-elementwise-admission"
    if alps_p2c: return "alps-elementwise-fused-transfer"
    if alps_p2b: return "alps-elementwise-producer-direct"
    if alps_p2a: return "alps-elementwise-zero-copy"
    if alps_p1: return "alps-elementwise-fusion"
    if alps_p0b: return "alps-fusion"
    if alps_p0: return "alps-legacy-all"
    return "item7-only"
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
by_model = {}
for row in rows:
    by_model.setdefault(row["model"], {})[row["scheme"]] = row
for model_rows in by_model.values():
    reference_name = selected_reference()
    item = model_rows.get(reference_name, {})
    try:
        denominator = float(item["latency_ms"])
    except (KeyError, TypeError, ValueError):
        continue
    for row in model_rows.values():
        try:
            row["ratio_over_item7"] = f'{float(row["latency_ms"]) / denominator:.2f}'
        except (TypeError, ValueError):
            row["ratio_over_item7"] = "NA"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
reference_name = selected_reference()
title = "ALPS P5f-b CRP supply prefetch" if alps_p5fb else ("ALPS P5f-a CRP supply analysis" if alps_p5fa else ("ALPS P5c layout-supply prefetch" if alps_p5c else ("ALPS P2g-c register-tile direct formation" if alps_p2gc else ("ALPS P2g-b loop-interchanged direct formation" if alps_p2gb else ("ALPS P2g-a continuity audit" if alps_p2g else ("ALPS P5b representation-aware input supply analysis" if alps_p5b else ("ALPS P5a contract discharge ledger" if alps_p5a else ("ALPS P2f consumer layout codegen propagation" if alps_p2f else ("ALPS P2e consumer-driven layout" if alps_p2e else "ALPS/full-model comparison")))))))))
if alps_p5fc:
    title = "ALPS P5f-c segmented CRP supply"
if alps_p5g:
    title = "ALPS P5g-a contiguous VTCM formation"
if alps_p5gb:
    title = "ALPS P5g-b coalesced VTCM supply window"
if alps_p5gc:
    title = "ALPS P5g-c asynchronous VTCM prefetch window"
if alps_p5gd:
    title = "ALPS P5g-d producer-direct VTCM analysis"
if alps_p5ge:
    title = "ALPS P5g-e producer-direct HVX epoch in VTCM"
if alps_p5gf:
    title = "ALPS P5g-f producer-direct head-major VTCM formation"
if alps_p5gg:
    title = "ALPS P5g-g contiguous head-major producer formation"
if alps_p5h:
    title = "ALPS P5h attention destination formation"
if alps_p5i:
    title = "ALPS P5i consumer-driven patch convolution formation"
if alps_p5j:
    title = "ALPS P5j consumer-driven HMX F16 epilogue formation"
if alps_p5k:
    title = "ALPS P5k consumer-driven HMX direct output formation"
if alps_p5l:
    title = "ALPS P5l consumer-driven HMX F16 bias epilogue formation"
if alps_p5m:
    title = "ALPS P5m HMX asynchronous drain admission analysis"
if alps_p5n:
    title = "ALPS P5n HMX ping-pong VTCM asynchronous result drain"
schemes = (
    ("alps-hmx-async-drain",)
    if alps_p5n else
    ("alps-hmx-async-drain-analysis",)
    if alps_p5m else
    ("alps-hmx-f16-bias-epilogue-formation",)
    if alps_p5l else
    ("alps-hmx-direct-output-formation",)
    if alps_p5k else
    ("alps-hmx-f16-epilogue-formation",)
    if alps_p5j else
    ("alps-patch-conv-formation",)
    if alps_p5i else
    ("alps-attention-destination-formation",)
    if alps_p5h else
    ("alps-crp-producer-head-major-contiguous",)
    if alps_p5gg else
    ("alps-crp-producer-direct-head-major",)
    if alps_p5gf else
    ("alps-crp-producer-direct-vtcm",)
    if alps_p5ge else
    ("alps-crp-producer-direct-analysis",)
    if alps_p5gd else
    ("alps-crp-vtcm-async-window",)
    if alps_p5gc else
    ("alps-crp-vtcm-window",)
    if alps_p5gb else
    ("alps-crp-vtcm-formation",)
    if alps_p5g else
    ("alps-crp-segmented-supply",)
    if alps_p5fc else
    ("alps-crp-supply-prefetch",)
    if alps_p5fb else
    ("alps-crp-supply-analysis",)
    if alps_p5fa else
    ("alps-layout-supply-prefetch",)
    if alps_p5c else
    ("alps-register-tile-direct",)
    if alps_p2gc else
    ("alps-loop-interchanged-direct",)
    if alps_p2gb else
    ("alps-continuity-audit",)
    if alps_p2g else
    ("alps-elementwise-traffic-control",)
    if alps_p4a else
    ("alps-elementwise-exact-overlap",)
    if alps_p3b else
    ("alps-elementwise-exact-readiness",)
    if alps_p3a else
    ("alps-representation-supply-analysis",)
    if alps_p5b else
    ("alps-consumer-driven-layout", "alps-contract-discharge-ledger")
    if alps_p5a else
    ("hmlir-hvx-hexkl-on", "alps-consumer-driven-layout", "alps-consumer-layout-propagation")
    if alps_p2f else
    ("hmlir-hvx-hexkl-on", "alps-consumer-driven-layout")
    if alps_p2e else
    ("alps-elementwise-admission",)
    if alps_p2d else
    ("alps-elementwise-fused-transfer",)
    if alps_p2c else
    ("alps-elementwise-producer-direct",)
    if alps_p2b else
    ("alps-elementwise-zero-copy",)
    if alps_p2a else
    ("hmlir-hvx-hexkl-on", "alps-elementwise-fusion")
    if alps_p1 else
    ("hmlir-hvx-hexkl-on", "alps-elementwise-fusion", "alps-multi-use-fusion", "alps-split-reduction", "alps-fusion")
    if alps_p0b else
    ("hmlir-hvx-hexkl-on", "alps-semantic", "alps-fusion", "alps-slicing", "alps-runtime", "alps-legacy-all")
    if alps_p0 else
    ("pk-hvx", "apt-hvx", "hmlir-hvx-hexkl-off", "hmlir-hvx-hexkl-on", "item7-only")
)
lines = [f"# {title}", "", f"| Model | Scheme | Latency ({reference_name} = 1.00x) |", "|---|---|---:|"]
for model in by_model:
    for scheme in schemes:
        row = by_model[model].get(scheme)
        if not row:
            continue
        ratio = row.get("ratio_over_item7", "NA")
        suffix = f" ({ratio}x)" if ratio != "NA" else ""
        lines.append(f'| {model} | {scheme} | {row["latency_ms"]} ms{suffix} |')
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  rsync -a --partial "${results}" "${output_dir}/summary.md" "nano:${remote_dir}/"
}

for model in "${models[@]}"; do
  if ((alps_p0)); then
    for scheme in "${schemes[@]}"; do
      run_case "${model}" "${scheme}"
    done
  else
    candidate_file=${output_dir}/${model}/apt-candidate-ids.txt
    run_case "${model}" pk-hvx
    candidate_ids=$(extract_admitted_ids "${output_dir}/${model}/pk-hvx/run.log")
    [[ -n "${candidate_ids}" ]] || {
      echo "No PK admitted IDs for ${model}; APT cannot be configured fairly" >&2
      exit 3
    }
    printf '%s\n' "${candidate_ids}" > "${candidate_file}"
    rsync -a --partial "${candidate_file}" "nano:${remote_dir}/${model}/"
    run_case "${model}" apt-hvx "${candidate_ids}"
    run_case "${model}" hmlir-hvx-hexkl-off
    run_case "${model}" hmlir-hvx-hexkl-on
    run_case "${model}" item7-only
  fi
  write_ratios
done

echo "COMPLETE results=${results} summary=${output_dir}/summary.md remote=${remote_dir}"
