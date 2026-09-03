#!/usr/bin/env python3
"""Aggregate per-kernel SDK sysMon summaries for one complete model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUM_FIELDS = (
    "kernel_elapsed_seconds",
    "selected_sample_milliseconds",
    "selected_samples",
    "pcycles",
    "committed_packets",
    "hvx_packet_event_count",
    "hmx_active_event_count",
    "l2fetch_misses",
    "axi_read_bytes",
    "axi_write_bytes",
    "axi_total_bytes",
    "bwmon_bytes",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.artifact_root.rglob("kernel_window_summary.json"))
    if not paths:
        raise ValueError("found no sysMon kernel_window_summary.json")
    rows: list[dict[str, object]] = []
    totals: dict[str, float] = {field: 0 for field in SUM_FIELDS}
    activity: dict[str, dict[str, int]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        stage = path.parent.parent.name
        row: dict[str, object] = {"stage": stage}
        for field in SUM_FIELDS:
            value = data.get(field, 0)
            totals[field] += float(value)
            row[field] = value
        rows.append(row)
        for name, values in data.get("activity_windows", {}).items():
            target = activity.setdefault(name, {"samples": 0, "axi_bytes": 0})
            target["samples"] += int(values.get("samples", 0))
            target["axi_bytes"] += int(values.get("axi_bytes", 0))

    elapsed = totals["selected_sample_milliseconds"] / 1000.0
    summary = {
        "interpretation": "complete_model_sum_of_per_stage_sysmon_windows",
        "profiled_stages": len(rows),
        **{
            field: int(value) if field not in ("kernel_elapsed_seconds", "selected_sample_milliseconds") else value
            for field, value in totals.items()
        },
        "axi_read_bandwidth_MBps": (
            totals["axi_read_bytes"] / elapsed / 1_000_000 if elapsed else 0
        ),
        "axi_write_bandwidth_MBps": (
            totals["axi_write_bytes"] / elapsed / 1_000_000 if elapsed else 0
        ),
        "activity_windows": activity,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    columns = ("stage",) + SUM_FIELDS
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda row: float(row["pcycles"]), reverse=True)
    lines = [
        "# Complete-model sysMon aggregate",
        "",
        "System-domain PMU counters are summed across serial stage windows. "
        "They identify hardware behavior and cross-model scale; they do not map "
        "bytes to an individual MLIR operation.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Profiled stages | {len(rows)} |",
        f"| Host kernel windows | {totals['kernel_elapsed_seconds']:.6f} s |",
        f"| Processor cycles | {int(totals['pcycles'])} |",
        f"| HVX packet event | {int(totals['hvx_packet_event_count'])} |",
        f"| HMX active event | {int(totals['hmx_active_event_count'])} |",
        f"| AXI read | {int(totals['axi_read_bytes'])} B |",
        f"| AXI write | {int(totals['axi_write_bytes'])} B |",
        f"| AXI total | {int(totals['axi_total_bytes'])} B |",
        f"| AXI read bandwidth | {summary['axi_read_bandwidth_MBps']:.2f} MB/s |",
        f"| AXI write bandwidth | {summary['axi_write_bandwidth_MBps']:.2f} MB/s |",
        "",
        "## Highest-cycle stage windows",
        "",
        "| Rank | Stage | Processor cycles | AXI total | HVX event | HMX event |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked[:20], 1):
        lines.append(
            f"| {index} | {row['stage']} | {row['pcycles']} | "
            f"{row['axi_total_bytes']} | {row['hvx_packet_event_count']} | "
            f"{row['hmx_active_event_count']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
