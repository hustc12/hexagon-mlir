#!/usr/bin/env python3
"""Summarize the sysMon PMU samples overlapping one Hexagon model run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def normalized(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip(): str(value).strip() for key, value in row.items()}


def parse_number(value: str) -> float:
    return float(value.strip())


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def row_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "0").strip()
    return float(value) if value else 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pmu", type=Path, required=True)
    parser.add_argument("--kernel-window", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    window = json.loads(args.kernel_window.read_text(encoding="utf-8"))
    kernel_seconds = float(window["kernel_elapsed_seconds"])
    with args.raw_pmu.open(newline="", encoding="utf-8") as handle:
        rows = [normalized(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("sysMon raw_pmu.csv contains no samples")

    # The executor starts sysMon, waits one second, runs the model, then stops
    # sysMon immediately. Select samples backwards from the end until their
    # accumulated sampling duration covers the measured adb kernel window.
    selected: list[dict[str, str]] = []
    selected_ms = 0.0
    target_ms = kernel_seconds * 1000.0
    for row in reversed(rows):
        selected.append(row)
        selected_ms += parse_number(row["Sampling period(ms)"])
        if selected_ms >= target_ms:
            break
    selected.reverse()

    events: dict[int, int] = {}
    pcycles = 0
    bwmon_bytes = 0
    axi_bytes_per_sample: list[int] = []
    activity: dict[str, dict[str, int]] = {
        "neither": {"samples": 0, "axi_bytes": 0},
        "hvx_only": {"samples": 0, "axi_bytes": 0},
        "hmx_only": {"samples": 0, "axi_bytes": 0},
        "hvx_and_hmx": {"samples": 0, "axi_bytes": 0},
    }
    for row in selected:
        pcycles += int(row["pcycles"], 0)
        bwmon_bytes += int(row["BWMON Count(Bytes)"], 0)
        sample_events: dict[int, int] = {}
        for index in range(8):
            event = int(row[f"PMU_{index}_Num"], 0)
            value = int(row[f"PMU_{index}_Val"], 0)
            events[event] = events.get(event, 0) + value
            sample_events[event] = sample_events.get(event, 0) + value
        sample_axi = (
            sample_events.get(0x3F, 0) * 128
            + sample_events.get(0xCD, 0) * 256
            + sample_events.get(0x46, 0) * 128
            + sample_events.get(0x55, 0) * 256
        )
        axi_bytes_per_sample.append(sample_axi)
        hvx_active = sample_events.get(0x111, 0) > 0
        hmx_active = sample_events.get(0x200, 0) > 0
        if hvx_active and hmx_active:
            activity_key = "hvx_and_hmx"
        elif hvx_active:
            activity_key = "hvx_only"
        elif hmx_active:
            activity_key = "hmx_only"
        else:
            activity_key = "neither"
        activity[activity_key]["samples"] += 1
        activity[activity_key]["axi_bytes"] += sample_axi

    axi_read_bytes = events.get(0x3F, 0) * 128 + events.get(0xCD, 0) * 256
    axi_write_bytes = events.get(0x46, 0) * 128 + events.get(0x55, 0) * 256
    measured_seconds = selected_ms / 1000.0
    committed_packets = events.get(0x3, 0)
    npa_core_clocks = [row_float(row, "NPA Core Clk(Mhz)") for row in selected]
    npa_bus_votes = [row_float(row, "NPA bus vote(Mhz)") for row in selected]
    dsppm_bus_votes = [row_float(row, "DSPPM bus vote(Mhz)") for row in selected]
    thermal_limits = [
        row_float(row, "Thermal Q6 throttle Freq (MHz)") for row in selected
    ]
    blc_counts = [int(row_float(row, "BLC latency count")) for row in selected]
    blc_latencies = [row_float(row, "BLC latency(ns)") for row in selected]
    blc_weight = sum(blc_counts)
    blc_weighted_latency = (
        sum(count * latency for count, latency in zip(blc_counts, blc_latencies))
        / blc_weight
        if blc_weight
        else math.nan
    )
    effective_core_mhz = pcycles / selected_ms / 1000.0 if selected_ms else math.nan
    npa_core_avg = average(npa_core_clocks)
    result = {
        "interpretation": "sysmon_hardware_pmu_kernel_window",
        "kernel_elapsed_seconds": kernel_seconds,
        "selected_sample_milliseconds": selected_ms,
        "selected_samples": len(selected),
        "pcycles": pcycles,
        "committed_packets": committed_packets,
        "cycles_per_committed_packet": (
            pcycles / committed_packets if committed_packets else math.nan
        ),
        "effective_core_mhz_from_pcycles": effective_core_mhz,
        "effective_core_utilization": (
            effective_core_mhz / npa_core_avg if npa_core_avg else math.nan
        ),
        "hvx_packet_event_count": events.get(0x111, 0),
        "hmx_active_event_count": events.get(0x200, 0),
        "l2fetch_misses": events.get(0x7F, 0),
        "axi_read_bytes": axi_read_bytes,
        "axi_write_bytes": axi_write_bytes,
        "axi_total_bytes": axi_read_bytes + axi_write_bytes,
        "axi_read_bandwidth_MBps": (
            axi_read_bytes / measured_seconds / 1_000_000
            if measured_seconds
            else math.nan
        ),
        "axi_write_bandwidth_MBps": (
            axi_write_bytes / measured_seconds / 1_000_000
            if measured_seconds
            else math.nan
        ),
        "axi_bytes_per_sample_p50": percentile(axi_bytes_per_sample, 0.50),
        "axi_bytes_per_sample_p90": percentile(axi_bytes_per_sample, 0.90),
        "axi_bytes_per_sample_p99": percentile(axi_bytes_per_sample, 0.99),
        "axi_bytes_per_sample_max": max(axi_bytes_per_sample, default=0),
        "activity_windows": activity,
        "bwmon_bytes": bwmon_bytes,
        "packet_count_raw": sum(
            int(row_float(row, "Packet count")) for row in selected
        ),
        "npa_core_clock_avg_mhz": npa_core_avg,
        "npa_core_clock_min_mhz": min(npa_core_clocks, default=math.nan),
        "npa_core_clock_max_mhz": max(npa_core_clocks, default=math.nan),
        "npa_bus_vote_avg_mhz": average(npa_bus_votes),
        "dsppm_bus_vote_avg_mhz": average(dsppm_bus_votes),
        "thermal_throttle_max_mhz": max(thermal_limits, default=math.nan),
        "thermal_throttle_nonzero_samples": sum(value > 0 for value in thermal_limits),
        "blc_transaction_count": sum(
            int(row_float(row, "BLC transaction count")) for row in selected
        ),
        "blc_latency_count": blc_weight,
        "blc_latency_weighted_avg_ns": blc_weighted_latency,
        "raw_event_totals": {hex(key): value for key, value in sorted(events.items())},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ALPS sysMon kernel-window summary",
        "",
        "These are hardware PMU samples collected by the SDK sysMon service. "
        "AXI bytes represent 128/256-byte line requests caused by L2 misses; "
        "they are not the compiler's logical access estimate.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Kernel host window | {kernel_seconds:.6f} s |",
        f"| Selected PMU window | {measured_seconds:.6f} s |",
        f"| AXI read | {axi_read_bytes} B |",
        f"| AXI write | {axi_write_bytes} B |",
        f"| AXI total | {axi_read_bytes + axi_write_bytes} B |",
        f"| AXI read bandwidth | {result['axi_read_bandwidth_MBps']:.2f} MB/s |",
        f"| AXI write bandwidth | {result['axi_write_bandwidth_MBps']:.2f} MB/s |",
        f"| AXI bytes / 1 ms sample, p50 | {result['axi_bytes_per_sample_p50']} B |",
        f"| AXI bytes / 1 ms sample, p90 | {result['axi_bytes_per_sample_p90']} B |",
        f"| AXI bytes / 1 ms sample, p99 | {result['axi_bytes_per_sample_p99']} B |",
        f"| AXI bytes / 1 ms sample, max | {result['axi_bytes_per_sample_max']} B |",
        f"| HVX packet event | {events.get(0x111, 0)} |",
        f"| HMX active event | {events.get(0x200, 0)} |",
        f"| L2fetch miss | {events.get(0x7F, 0)} |",
        f"| Committed packets | {events.get(0x3, 0)} |",
        f"| Processor cycles | {pcycles} |",
        f"| Cycles / committed packet | {result['cycles_per_committed_packet']:.4f} |",
        f"| Effective core frequency from pcycles | {effective_core_mhz:.2f} MHz |",
        f"| NPA core clock, average | {npa_core_avg:.2f} MHz |",
        f"| Effective core utilization | {result['effective_core_utilization']:.4f} |",
        f"| NPA bus vote, average | {result['npa_bus_vote_avg_mhz']:.2f} MHz |",
        f"| DSPPM bus vote, average | {result['dsppm_bus_vote_avg_mhz']:.2f} MHz |",
        f"| Thermal throttle limit, maximum | {result['thermal_throttle_max_mhz']:.2f} MHz |",
        f"| Thermal-throttled samples | {result['thermal_throttle_nonzero_samples']} |",
        f"| BLC weighted latency | {blc_weighted_latency:.2f} ns |",
        "",
        "| 1 ms activity window | Samples | AXI bytes |",
        "|---|---:|---:|",
    ]
    for key in ("neither", "hvx_only", "hmx_only", "hvx_and_hmx"):
        lines.append(
            f"| {key} | {activity[key]['samples']} | "
            f"{activity[key]['axi_bytes']} B |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
