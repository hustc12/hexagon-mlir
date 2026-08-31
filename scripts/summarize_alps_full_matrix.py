#!/usr/bin/env python3
"""Build the frozen 15-unique-model ALPS table without rewriting history."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


SCHEMES = (
    "pk-hvx",
    "apt-hvx",
    "hmlir-hvx-hexkl-off",
    "hmlir-hvx-hexkl-on",
    "alps-final",
)

# Retain diagnostic rows in results.csv, but never count a graph-equivalent
# duplicate as an independent model in the paper-facing matrix.
DIAGNOSTIC_ONLY_MODELS = {"unispeech-sat-base"}


def number(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def integer(value: object) -> int | None:
    parsed = number(value)
    return None if parsed is None else int(parsed)


def display_ms(value: object) -> str:
    parsed = number(value)
    return "NA" if parsed is None else f"{parsed:,.2f}"


def display_int(value: object) -> str:
    parsed = integer(value)
    return "NA" if parsed is None else f"{parsed:,}"


def display_ratio(numerator: object, denominator: object) -> str:
    lhs, rhs = number(numerator), number(denominator)
    if lhs is None or rhs is None or rhs == 0:
        return "NA"
    return f"{lhs / rhs:.2f}x"


def load_sysmon(root: Path, model: str, scheme: str) -> dict[str, object]:
    path = root / model / scheme / "sysmon_model_replay" / "kernel_window_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_ledger(root: Path, model: str, scheme: str) -> dict[str, int]:
    path = root / model / scheme / "run.log"
    totals: dict[str, int] = {}
    if not path.exists():
        return totals
    prefixes = (
        "[ALPS-P2E]",
        "[ALPS-P5H]",
        "[ALPS-P5I]",
        "[ALPS-P5J]",
        "[ALPS-P5M-ANALYSIS]",
        "ALPSHMXAsyncDrain:",
        "ALPSP4A:",
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        prefix = next((item for item in prefixes if line.startswith(item)), None)
        if prefix is None:
            continue
        label = {
            "[ALPS-P2E]": "p2e",
            "[ALPS-P5H]": "p5h",
            "[ALPS-P5I]": "p5i",
            "[ALPS-P5J]": "p5j",
            "[ALPS-P5M-ANALYSIS]": "p5m",
            "ALPSHMXAsyncDrain:": "dma",
            "ALPSP4A:": "r",
        }[prefix]
        for key, value in re.findall(r"([a-z0-9_]+)=(-?[0-9]+)", line):
            field = f"{label}_{key}"
            totals[field] = totals.get(field, 0) + int(value)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--long-csv", type=Path, required=True)
    parser.add_argument("--wide-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    with args.results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    all_by_model: dict[str, dict[str, dict[str, str]]] = {}
    domains: dict[str, str] = {}
    for row in rows:
        all_by_model.setdefault(row["model"], {})[row["scheme"]] = row
        domains[row["model"]] = row["domain"]
    by_model = {
        model: cases
        for model, cases in all_by_model.items()
        if model not in DIAGNOSTIC_ONLY_MODELS
    }

    long_rows: list[dict[str, object]] = []
    for model, cases in by_model.items():
        for scheme in SCHEMES:
            row = dict(cases.get(scheme, {}))
            if not row:
                row = {
                    "model": model,
                    "domain": domains.get(model, "NA"),
                    "scheme": scheme,
                    "status": "NA",
                    "latency_ms": "NA",
                }
            sysmon = load_sysmon(args.output_root, model, scheme)
            runtime = load_runtime_ledger(args.output_root, model, scheme)
            row.update(
                {
                    "sysmon_pcycles": sysmon.get("pcycles", "NA"),
                    "sysmon_committed_packets": sysmon.get("committed_packets", "NA"),
                    "sysmon_pcpp": sysmon.get("cycles_per_committed_packet", "NA"),
                    "sysmon_hvx_packets": sysmon.get("hvx_packet_event_count", "NA"),
                    "sysmon_hmx_active": sysmon.get("hmx_active_event_count", "NA"),
                    "sysmon_l2fetch_misses": sysmon.get("l2fetch_misses", "NA"),
                    "sysmon_axi_read_bytes": sysmon.get("axi_read_bytes", "NA"),
                    "sysmon_axi_write_bytes": sysmon.get("axi_write_bytes", "NA"),
                    "sysmon_axi_total_bytes": sysmon.get("axi_total_bytes", "NA"),
                    "sysmon_core_clock_avg_mhz": sysmon.get("npa_core_clock_avg_mhz", "NA"),
                    "sysmon_bus_vote_avg_mhz": sysmon.get("npa_bus_vote_avg_mhz", "NA"),
                    "sysmon_thermal_throttle_max_mhz": sysmon.get(
                        "thermal_throttle_max_mhz", "NA"
                    ),
                    "sysmon_blc_latency_avg_ns": sysmon.get(
                        "blc_latency_weighted_avg_ns", "NA"
                    ),
                }
            )
            row.update(runtime)
            long_rows.append(row)

    args.long_csv.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in long_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with args.long_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(long_rows)

    wide_rows: list[dict[str, object]] = []
    for model, cases in by_model.items():
        final = cases.get("alps-final", {})
        control = cases.get("hmlir-hvx-hexkl-on", {})
        final_sysmon = load_sysmon(args.output_root, model, "alps-final")
        control_sysmon = load_sysmon(args.output_root, model, "hmlir-hvx-hexkl-on")
        final_runtime = load_runtime_ledger(args.output_root, model, "alps-final")
        baseline_mat = integer(control.get("static_materialization_bytes"))
        final_mat = integer(final.get("static_materialization_bytes"))
        reduction = (
            baseline_mat - final_mat
            if baseline_mat is not None and final_mat is not None
            else None
        )
        reduction_percent = (
            reduction / baseline_mat * 100.0
            if reduction is not None and baseline_mat
            else None
        )
        axi_control = integer(control_sysmon.get("axi_total_bytes"))
        axi_final = integer(final_sysmon.get("axi_total_bytes"))
        axi_reduction = (
            axi_control - axi_final
            if axi_control is not None and axi_final is not None
            else None
        )
        wide: dict[str, object] = {
            "model": model,
            "domain": domains.get(model, "NA"),
        }
        for scheme in SCHEMES:
            case = cases.get(scheme, {})
            wide[f"{scheme}_status"] = case.get("status", "NA")
            wide[f"{scheme}_latency_ms"] = case.get("latency_ms", "NA")
            wide[f"{scheme}_over_alps"] = display_ratio(
                case.get("latency_ms"), final.get("latency_ms")
            )
        wide.update(
            {
                "baseline_static_materialization_bytes": (
                    baseline_mat if baseline_mat is not None else "NA"
                ),
                "alps_static_materialization_bytes": (
                    final_mat if final_mat is not None else "NA"
                ),
                "logical_materialization_reduction_bytes": (
                    reduction if reduction is not None else "NA"
                ),
                "logical_materialization_reduction_percent": (
                    f"{reduction_percent:.2f}"
                    if reduction_percent is not None
                    else "NA"
                ),
                "p2e_eliminated_bytes": final.get("p2e_eliminated_bytes", "NA"),
                "p5h_eliminated_copy_bytes": final.get(
                    "p5h_eliminated_copy_bytes", "NA"
                ),
                "p5i_eliminated_transpose_bytes": final.get(
                    "p5i_eliminated_transpose_bytes", "NA"
                ),
                "runtime_dma_issued": final.get("runtime_issued", "NA"),
                "runtime_dma_issued_bytes": final.get("runtime_issued_bytes", "NA"),
                "p2e_demands": final_runtime.get("p2e_demands", "NA"),
                "p2e_producer_direct": final_runtime.get(
                    "p2e_producer_direct", "NA"
                ),
                "p5j_formed_f16_epilogues": final_runtime.get(
                    "p5j_formed_f16_epilogues", "NA"
                ),
                "p5m_admitted_sites": final_runtime.get(
                    "p5m_admitted_sites", "NA"
                ),
                "r_windows": final_runtime.get("r_windows", "NA"),
                "r_hold": final_runtime.get("r_hold", "NA"),
                "r_throttle": final_runtime.get("r_throttle", "NA"),
                "r_dma_suppressed": final_runtime.get(
                    "r_dma_suppressed", "NA"
                ),
                "r_pmu_status": final_runtime.get("r_pmu_status", "NA"),
                "r_pmu_reads": final_runtime.get("r_pmu_reads", "NA"),
                "r_poll_retries": final_runtime.get("r_poll_retries", "NA"),
                "control_sysmon_axi_total_bytes": (
                    axi_control if axi_control is not None else "NA"
                ),
                "alps_sysmon_axi_total_bytes": (
                    axi_final if axi_final is not None else "NA"
                ),
                "external_traffic_reduction_bytes": (
                    axi_reduction if axi_reduction is not None else "NA"
                ),
                "control_sysmon_pcycles": control_sysmon.get("pcycles", "NA"),
                "alps_sysmon_pcycles": final_sysmon.get("pcycles", "NA"),
            }
        )
        wide_rows.append(wide)

    with args.wide_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wide_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wide_rows)

    lines = [
        "# Frozen ALPS 15-unique-model complete-model matrix",
        "",
        "This is a new post-`3b90cd4` table. Historical tables remain unchanged.",
        "All models are complete non-Debug models and all speedups use ALPS final as 1.00x.",
        "UniSpeech-SAT-Base is diagnostic-only: under this runner it is graph-equivalent to UniSpeech-Base and is excluded from the independent-model count.",
        "",
        "## Latency",
        "",
        "| Domain | Model | PK HVX | APT HVX | HMLIR HVX (HexKL Off) | HMLIR HVX (HexKL On) | ALPS C+E+P+R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, cases in by_model.items():
        final = cases.get("alps-final", {})
        cells = []
        for scheme in SCHEMES:
            case = cases.get(scheme, {})
            latency = display_ms(case.get("latency_ms"))
            ratio = display_ratio(case.get("latency_ms"), final.get("latency_ms"))
            cells.append(f"{latency} ms ({ratio})" if latency != "NA" else "NA")
        lines.append(
            f"| {domains.get(model, 'NA')} | {model} | " + " | ".join(cells) + " |"
        )

    lines.extend(
        [
            "",
            "## Materialization and measured external traffic",
            "",
            "| Domain | Model | HMLIR-On materialization | ALPS materialization | Logical reduction | P2e eliminated | P5h eliminated | P5i eliminated | DMA issued bytes | HMLIR-On AXI | ALPS AXI | AXI reduction |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in wide_rows:
        reduction = display_int(row["logical_materialization_reduction_bytes"])
        percent = row["logical_materialization_reduction_percent"]
        reduction_cell = reduction if percent == "NA" else f"{reduction} ({percent}%)"
        lines.append(
            f"| {row['domain']} | {row['model']} | "
            f"{display_int(row['baseline_static_materialization_bytes'])} | "
            f"{display_int(row['alps_static_materialization_bytes'])} | "
            f"{reduction_cell} | {display_int(row['p2e_eliminated_bytes'])} | "
            f"{display_int(row['p5h_eliminated_copy_bytes'])} | "
            f"{display_int(row['p5i_eliminated_transpose_bytes'])} | "
            f"{display_int(row['runtime_dma_issued_bytes'])} | "
            f"{display_int(row['control_sysmon_axi_total_bytes'])} | "
            f"{display_int(row['alps_sysmon_axi_total_bytes'])} | "
            f"{display_int(row['external_traffic_reduction_bytes'])} |"
        )
    lines.extend(
        [
            "",
            "## ALPS admission and runtime audit",
            "",
            "| Domain | Model | P2e direct/demands | P5j formed | P5m admitted | DMA issued/bytes | R windows/hold/throttle/suppressed | PMU status/reads | Poll retries |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in wide_rows:
        lines.append(
            f"| {row['domain']} | {row['model']} | "
            f"{row['p2e_producer_direct']}/{row['p2e_demands']} | "
            f"{row['p5j_formed_f16_epilogues']} | {row['p5m_admitted_sites']} | "
            f"{row['runtime_dma_issued']}/{display_int(row['runtime_dma_issued_bytes'])} | "
            f"{row['r_windows']}/{row['r_hold']}/{row['r_throttle']}/{row['r_dma_suppressed']} | "
            f"{row['r_pmu_status']}/{row['r_pmu_reads']} | {row['r_poll_retries']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
