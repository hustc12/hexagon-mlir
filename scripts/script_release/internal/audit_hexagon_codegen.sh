#!/usr/bin/env bash
# Summarize the static code-generation properties of one Hexagon artifact dir.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 ARTIFACT_DIR [OUTPUT_CSV]" >&2
  exit 2
fi

artifact_dir=$1
output_csv=${2:-}
hexagon_tools=${HEXAGON_TOOLS:-/home/huzq85/2-working/hexagon_npu/HEXAGON_TOOLS/Tools}
objdump="${hexagon_tools}/bin/hexagon-llvm-objdump"

[[ -d "${artifact_dir}" ]] || {
  echo "ERROR: artifact directory not found: ${artifact_dir}" >&2
  exit 2
}
[[ -x "${objdump}" ]] || {
  echo "ERROR: Hexagon objdump not found: ${objdump}" >&2
  exit 2
}

main_object=$(
  find "${artifact_dir}" -maxdepth 1 -type f \
    -name '_mlir_ciface_*.o' ! -name '*-consts-*.o' -print -quit
)
[[ -n "${main_object}" ]] || {
  echo "ERROR: no main _mlir_ciface object found in ${artifact_dir}" >&2
  exit 2
}

disassembly=$(mktemp /tmp/alps-disassembly.XXXXXX)
relocations=$(mktemp /tmp/alps-relocations.XXXXXX)
trap 'rm -f "${disassembly}" "${relocations}"' EXIT
"${objdump}" -d --no-show-raw-insn "${main_object}" >"${disassembly}"
"${objdump}" -r "${main_object}" >"${relocations}"

read -r instruction_count hvx_like_count < <(
  awk '
    /^[[:space:]]*[0-9a-f]+:/ {
      instructions++
      if ($0 ~ /[[:space:]]v[0-9]+\./)
        hvx++
    }
    END { print instructions + 0, hvx + 0 }
  ' "${disassembly}"
)
hvx_percent=$(
  awk -v all="${instruction_count}" -v hvx="${hvx_like_count}" \
    'BEGIN { if (all) printf "%.6f", 100.0 * hvx / all; else print "0.000000" }'
)
hexkl_calls=$(rg -i -c 'hexkl|hmx' "${disassembly}" || true)
dma_calls=$(rg -i -c 'dma|memcpy2d|memcpy3d' "${disassembly}" || true)
vector_load_store=$(rg -c 'vmem|v[0-9]+\.[a-z0-9_]+[[:space:]]*=' "${disassembly}" || true)
hexkl_calls=${hexkl_calls:-0}
dma_calls=${dma_calls:-0}
vector_load_store=${vector_load_store:-0}
extend_hf_sf=$(
  awk '$NF == "__extendhfsf2" {count++} END {print count + 0}' "${relocations}"
)
trunc_sf_hf=$(
  awk '$NF == "__truncsfhf2" {count++} END {print count + 0}' "${relocations}"
)
half_helper_total=$((extend_hf_sf + trunc_sf_hf))

printf '%s\n' \
  "artifact_dir=${artifact_dir}" \
  "main_object=${main_object}" \
  "instruction_count=${instruction_count}" \
  "hvx_like_count=${hvx_like_count}" \
  "hvx_like_percent=${hvx_percent}" \
  "hexkl_hmx_mentions=${hexkl_calls}" \
  "dma_mentions=${dma_calls}" \
  "vector_load_store_mentions=${vector_load_store}" \
  "extendhfsf2_relocations=${extend_hf_sf}" \
  "truncsfhf2_relocations=${trunc_sf_hf}" \
  "half_conversion_helper_relocations=${half_helper_total}"

if [[ -n "${output_csv}" ]]; then
  if [[ ! -f "${output_csv}" ]]; then
    printf '%s\n' \
      'artifact_dir,main_object,instruction_count,hvx_like_count,hvx_like_percent,hexkl_hmx_mentions,dma_mentions,vector_load_store_mentions,extendhfsf2_relocations,truncsfhf2_relocations,half_conversion_helper_relocations' \
      >"${output_csv}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${artifact_dir}" "${main_object}" "${instruction_count}" \
    "${hvx_like_count}" "${hvx_percent}" "${hexkl_calls}" "${dma_calls}" \
    "${vector_load_store}" "${extend_hf_sf}" "${trunc_sf_hf}" \
    "${half_helper_total}" >>"${output_csv}"
fi
