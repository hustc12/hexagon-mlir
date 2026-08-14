#!/usr/bin/env bash
# Re-run the complete DINOv2-small and ViT-Base HVX comparison matrix.
#
# The two child drivers contain no timeout and execute every model/configuration
# strictly serially.  Scalar and HMX configurations are intentionally omitted.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
parent_dir=$(cd -- "${repo_root}/.." && pwd)
result_root=${OUTPUT_DIR:-${parent_dir}/run_artifacts/full_hvx_regression_$(date +%Y%m%d_%H%M%S)}
remote_root=${REMOTE_RESULTS_DIR:-}
iterations=${DEVICE_ITERATIONS:-1}
skip_baselines=${SKIP_BASELINES:-0}

baseline_dir=${result_root}/prefetch-baselines
native_omni_dir=${result_root}/native-and-item7
mkdir -p "${result_root}"

if [[ "${skip_baselines}" == 1 ]]; then
  [[ -f "${baseline_dir}/results.csv" ]] || {
    echo "Missing completed baseline CSV: ${baseline_dir}/results.csv" >&2
    exit 2
  }
  echo "[HVXRegression] reuse phase=external-prefetch-baselines output=${baseline_dir}"
else
  echo "[HVXRegression] phase=external-prefetch-baselines output=${baseline_dir}"
  OUTPUT_DIR="${baseline_dir}" \
  DEVICE_ITERATIONS="${iterations}" \
  REMOTE_RESULTS_DIR="${remote_root:+${remote_root}/prefetch-baselines}" \
    "${script_dir}/run_prefetch_baseline_full_two_models.sh"
fi

echo "[HVXRegression] phase=native-hexkl-item7 output=${native_omni_dir}"
OUTPUT_DIR="${native_omni_dir}" \
DEVICE_ITERATIONS="${iterations}" \
ONLY_SCHEMES="hvx hexkl-control item7-only" \
REUSE_VALID_LOGS=0 \
REMOTE_RESULTS_DIR="${remote_root:+${remote_root}/native-and-item7}" \
  "${script_dir}/run_omnifetch_full_no_item4_ablation.sh"

python_bin=${OMNIFETCH_VENV:-${parent_dir}/mlir-env}/bin/python
"${python_bin}" - \
  "${baseline_dir}/results.csv" \
  "${native_omni_dir}/results.csv" \
  "${result_root}/hvx_regression.csv" \
  "${result_root}/hvx_regression.md" <<'PY'
import csv
import pathlib
import sys

baseline_path, native_path, csv_path, markdown_path = map(pathlib.Path, sys.argv[1:])
values = {}

for path in (baseline_path, native_path):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "").lower() not in {"pass", "passed"}:
                continue
            values[(row["model"], row["scheme"])] = float(row["perf_us"]) / 1000.0

schemes = (
    ("native-hvx", "Native Hexagon-MLIR HVX (HexKL off)", "hvx"),
    ("hexkl-zero-hmx", "Hexagon-MLIR HVX + HexKL pipeline (0 HMX rewrites)", "hexkl-control"),
    ("prefetch-kernel-hx", "Prefetch-Kernel-HX on HVX", "prefetch-kernel-hx"),
    ("apt-get-hx", "APT-GET-HX global-plan MVP on HVX", "apt-get-hx"),
    ("omnifetch-item7", "OmniFetch item7-only on HVX", "item7-only"),
)
models = ("dinov2-small", "vit-base")

missing = [
    (model, source)
    for model in models
    for _, _, source in schemes
    if (model, source) not in values
]
if missing:
    raise SystemExit(f"missing passing rows: {missing}")

with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("model", "configuration", "latency_ms", "latency_over_item7"))
    for model in models:
        item7 = values[(model, "item7-only")]
        for key, _, source in schemes:
            latency = values[(model, source)]
            writer.writerow((model, key, f"{latency:.2f}", f"{latency / item7:.2f}"))

lines = [
    "# Full-model HVX regression",
    "",
    "All latency values and ratios use two decimal places. A ratio is the row's "
    "latency divided by OmniFetch item7 latency, so item7 is 1.00x and larger "
    "values mean a larger item7 speedup.",
    "",
    "| Model | Configuration | Latency (item7 = 1.00x) |",
    "|---|---|---:|",
]
for model in models:
    item7 = values[(model, "item7-only")]
    display_model = "DINOv2-small" if model == "dinov2-small" else "ViT-Base"
    for _, label, source in schemes:
        latency = values[(model, source)]
        lines.append(
            f"| {display_model} | {label} | {latency:.2f} ms ({latency / item7:.2f}x) |"
        )
markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

if [[ -n "${remote_root}" ]]; then
  ssh nano "mkdir -p '${remote_root}'"
  rsync -a --partial "${result_root}/hvx_regression.csv" \
    "${result_root}/hvx_regression.md" "nano:${remote_root}/"
fi

echo "[HVXRegression] complete report=${result_root}/hvx_regression.md"
