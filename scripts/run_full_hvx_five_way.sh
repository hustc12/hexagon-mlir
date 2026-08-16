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
reuse_valid=${REUSE_VALID_LOGS:-1}
list_only=0
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
  --output-dir DIR        Local working/result directory
  --remote-dir DIR        nano working_set destination
  --device-iterations N   Device samples per configuration (default: ${iterations})
  --seq-len N             Full LLM prefill length (default: ${seq_len})
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
    --output-dir) output_dir=$2; shift 2 ;;
    --remote-dir) remote_dir=$2; shift 2 ;;
    --device-iterations) iterations=$2; shift 2 ;;
    --seq-len) seq_len=$2; shift 2 ;;
    --no-reuse) reuse_valid=0; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) models+=("$1"); shift ;;
  esac
done

[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid iterations" >&2; exit 2; }
[[ "${seq_len}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid sequence length" >&2; exit 2; }
if ((${#models[@]} == 0)); then models=("${all_models[@]}"); fi
for model in "${models[@]}"; do
  known_model "${model}" || { echo "Unknown model: ${model}" >&2; exit 2; }
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

mkdir -p "${output_dir}"
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
  esac
}

passing_log() {
  local log=$1 status_file=$2
  [[ -f "${status_file}" && "$(<"${status_file}")" == PASS ]] &&
    grep -q '^[[:space:]]*Perf:' "${log}"
}

upsert_result() {
  local model=$1 scheme=$2 status=$3 log=$4
  local perf_us latency compile_s hints issued issued_bytes kv_pairs kv_sites correctness
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
  awk -F, -v m="${model}" -v s="${scheme}" 'NR==1 || !($1==m && $3==s)' "${results}" > "${results}.tmp"
  printf '%s,%s,%s,%s,%s,%s,NA,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model}" "$(domain_for "${model}")" "${scheme}" "${status}" \
    "${perf_us:-NA}" "${latency}" "${compile_s:-NA}" "${hints}" \
    "${issued}" "${issued_bytes}" "${kv_pairs}" "${kv_sites}" \
    "${correctness}" "${log}" >> "${results}.tmp"
  mv "${results}.tmp" "${results}"
}

run_case() {
  local model=$1 scheme=$2 candidate_ids=${3:-}
  local runner=${repo_root}/$(runner_for "${model}")
  local case_dir=${output_dir}/${model}/${scheme}
  local log=${case_dir}/run.log status_file=${case_dir}/status.txt
  local -a args=()
  mkdir -p "${case_dir}/artifacts"
  if ((reuse_valid)) && passing_log "${log}" "${status_file}"; then
    echo "REUSE model=${model} scheme=${scheme}"
    upsert_result "${model}" "${scheme}" PASS "${log}"
    return 0
  fi
  mapfile -t args < <(base_args_for "${model}"; scheme_args_for "${scheme}" "${candidate_ids}")
  if [[ "$(cli_style_for "${model}")" == layered-fp16 ]]; then
    args+=(--output-dir "${case_dir}/mlirbc")
  fi
  echo "START model=${model} scheme=${scheme} $(date --iso-8601=seconds)"
  set +e
  HEXAGON_MLIR_DUMP_DIR="${case_dir}/artifacts" \
    "${venv}/bin/python" "${runner}" "${args[@]}" >"${log}" 2>&1
  rc=$?
  set -e
  if ((rc == 0)) && grep -q '^[[:space:]]*Perf:' "${log}"; then
    printf '%s\n' PASS > "${status_file}"
    upsert_result "${model}" "${scheme}" PASS "${log}"
  else
    printf 'FAIL_%s\n' "${rc}" > "${status_file}"
    upsert_result "${model}" "${scheme}" "FAIL_${rc}" "${log}"
    sync_case "${model}" "${scheme}" "${case_dir}"
    cleanup_case_generated "${case_dir}"
    echo "FAIL model=${model} scheme=${scheme} rc=${rc} log=${log}" >&2
    return "${rc:-1}"
  fi
  sync_case "${model}" "${scheme}" "${case_dir}"
  cleanup_case_generated "${case_dir}"
  echo "DONE model=${model} scheme=${scheme}"
}

write_ratios() {
  "${venv}/bin/python" - "${results}" "${output_dir}/summary.md" <<'PY'
import csv
import pathlib
import sys

csv_path, md_path = map(pathlib.Path, sys.argv[1:])
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
by_model = {}
for row in rows:
    by_model.setdefault(row["model"], {})[row["scheme"]] = row
for model_rows in by_model.values():
    item = model_rows.get("item7-only", {})
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
lines = ["# Full-model HVX five-way comparison", "", "| Model | Scheme | Latency (item7 = 1.00x) |", "|---|---|---:|"]
for model in by_model:
    for scheme in ("pk-hvx", "apt-hvx", "hmlir-hvx-hexkl-off", "hmlir-hvx-hexkl-on", "item7-only"):
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
  write_ratios
done

echo "COMPLETE results=${results} summary=${output_dir}/summary.md remote=${remote_dir}"
