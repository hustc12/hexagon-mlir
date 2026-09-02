#!/usr/bin/env python3
"""Build unit-explicit ALPS A0--A4 latency and memory-system tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCHEMES = (
    ("A0", "hmlir-hvx-hexkl-on", "HexKL On"),
    ("A1", "alps-hvx-widening-conv", "+C"),
    ("A2", "alps-c-e-hmx-direct-output", "+full E"),
    ("A3", "alps-c-e-hmx-async-drain", "+P"),
    ("A4", "alps-final", "+R / ALPS"),
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def shown(value: object, unit: str = "") -> str:
    if value is None or value == "":
        return "NA"
    suffix = f" {unit}" if unit else ""
    return f"{value}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    with args.results.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    indexed = {(row["model"], row["scheme"]): row for row in source}
    models = list(dict.fromkeys(row["model"] for row in source))
    rows: list[dict[str, object]] = []
    for model in models:
        baseline = number(indexed.get((model, SCHEMES[0][1]), {}).get("latency_ms", ""))
        previous: float | None = None
        for stage, scheme, component in SCHEMES:
            source_row = indexed.get((model, scheme), {})
            latency = number(source_row.get("latency_ms", ""))
            fixed = load_json(
                args.output_root / model / scheme / "sysmon_model_replay" /
                "kernel_window_summary.json"
            )
            memory = load_json(
                args.output_root / model / scheme / "sysmon_memory_replay" /
                "kernel_window_summary.json"
            )
            row: dict[str, object] = {
                "model": model,
                "stage": stage,
                "component": component,
                "scheme": scheme,
                "status": source_row.get("status", "NA"),
                "latency_ms": latency,
                "adjacent_speedup_x": previous / latency if previous and latency else None,
                "cumulative_speedup_x": baseline / latency if baseline and latency else None,
                "runtime_issued_bytes": number(source_row.get("runtime_issued_bytes", "")),
                "static_materialization_bytes": number(
                    source_row.get("static_materialization_bytes", "")
                ),
                "axi_read_bytes": fixed.get("axi_read_bytes"),
                "axi_write_bytes": fixed.get("axi_write_bytes"),
                "axi_total_bytes": fixed.get("axi_total_bytes"),
                "vtcm_peak_usage_bytes": fixed.get("vtcm_peak_usage_bytes"),
                "vtcm_pool_reserved_bytes": fixed.get("vtcm_pool_reserved_bytes"),
                "vtcm_read_access_events_estimated": memory.get(
                    "vtcm_read_access_events_estimated"
                ),
                "vtcm_write_access_events_estimated": memory.get(
                    "vtcm_write_access_events_estimated"
                ),
                "vtcm_read_bytes": memory.get("vtcm_read_bytes"),
                "vtcm_write_bytes": memory.get("vtcm_write_bytes"),
                "dcache_demand_miss_events_estimated": memory.get(
                    "dcache_demand_miss_events_estimated"
                ),
                "l2_du_read_miss_events_estimated": memory.get(
                    "l2_du_read_miss_events_estimated"
                ),
                "l2_du_store_miss_events_estimated": memory.get(
                    "l2_du_store_miss_events_estimated"
                ),
                "hvx_l2_load_miss_events_estimated": memory.get(
                    "hvx_l2_load_miss_events_estimated"
                ),
                "hvx_l2_store_miss_events_estimated": memory.get(
                    "hvx_l2_store_miss_events_estimated"
                ),
                "vtcm_active_cycles_estimated": memory.get(
                    "vtcm_active_cycles_estimated"
                ),
            }
            rows.append(row)
            previous = latency

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ALPS complete-model component ablation",
        "",
        "All byte quantities use bytes (`B`). Cache misses and VTCM accesses are "
        "event counts, not bytes. Fields ending in `_estimated` are normalized "
        "by each event's exposure in sysMon's multiplexed 116-event profile. "
        "sysMon exposes neither a VTCM capacity high-water counter nor the "
        "transferred width of mixed scalar/HVX/HMX VTCM accesses, so VTCM peak "
        "usage and VTCM read/write bytes remain `NA` rather than being inferred.",
        "",
        "## Latency",
        "",
        "| Model | Stage | Component | Latency | Adjacent speedup | A0 cumulative |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        adjacent = row["adjacent_speedup_x"]
        cumulative = row["cumulative_speedup_x"]
        adjacent_text = f"{adjacent:.2f}x" if adjacent is not None else "NA"
        cumulative_text = f"{cumulative:.2f}x" if cumulative is not None else "NA"
        lines.append(
            f"| {row['model']} | {row['stage']} | {row['component']} | "
            f"{shown(row['latency_ms'], 'ms')} | "
            f"{adjacent_text} | {cumulative_text} |"
        )
    lines += [
        "",
        "## Physical-memory metrics",
        "",
        "| Model | Stage | AXI read | AXI write | AXI total | VTCM peak | "
        "VTCM reads | VTCM writes | D-cache miss | L2 DU R/W miss | HVX L2 R/W miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['stage']} | "
            f"{shown(row['axi_read_bytes'], 'B')} | {shown(row['axi_write_bytes'], 'B')} | "
            f"{shown(row['axi_total_bytes'], 'B')} | {shown(row['vtcm_peak_usage_bytes'], 'B')} | "
            f"{shown(row['vtcm_read_access_events_estimated'], 'events')} | "
            f"{shown(row['vtcm_write_access_events_estimated'], 'events')} | "
            f"{shown(row['dcache_demand_miss_events_estimated'], 'events')} | "
            f"{shown(row['l2_du_read_miss_events_estimated'], 'events')}/"
            f"{shown(row['l2_du_store_miss_events_estimated'], 'events')} | "
            f"{shown(row['hvx_l2_load_miss_events_estimated'], 'events')}/"
            f"{shown(row['hvx_l2_store_miss_events_estimated'], 'events')} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
