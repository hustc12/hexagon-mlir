#!/usr/bin/env python3
"""Derive a versioned ALPS cross-invocation policy from matched sysMon runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_summary(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("interpretation") != "sysmon_hardware_pmu_kernel_window":
        raise ValueError(f"{path} is not an ALPS sysMon kernel-window summary")
    return value


def positive_number(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def derive_policy(
    baseline: dict,
    alps: dict,
    baseline_latency_ms: float | None = None,
    alps_latency_ms: float | None = None,
) -> dict:
    baseline_axi = positive_number(baseline["axi_total_bytes"], "baseline AXI bytes")
    alps_axi = positive_number(alps["axi_total_bytes"], "ALPS AXI bytes")
    traffic_ratio = alps_axi / baseline_axi
    speedup = None
    if (baseline_latency_ms is None) != (alps_latency_ms is None):
        raise ValueError("latencies must be supplied as a matched pair")
    if baseline_latency_ms is not None and alps_latency_ms is not None:
        speedup = positive_number(baseline_latency_ms, "baseline latency") / positive_number(
            alps_latency_ms, "ALPS latency"
        )

    # sysMon is the slow control loop.  It seeds the next invocation but never
    # overrides compiler legality.  Reject only measured regressions or traffic
    # amplification without a compensating latency gain.
    measured_regression = speedup is not None and speedup < 0.99
    unproductive_amplification = traffic_ratio > 1.10 and (
        speedup is None or speedup < 1.02
    )
    initial_dma_allowed = not (measured_regression or unproductive_amplification)

    p50 = max(1.0, float(alps.get("axi_bytes_per_sample_p50", 0)))
    p99 = max(p50, float(alps.get("axi_bytes_per_sample_p99", p50)))
    burst_ratio = p99 / p50
    window = 32 if burst_ratio >= 4.0 else 64
    probe_interval = 16 if not initial_dma_allowed else 64
    reason = (
        "latency_regression"
        if measured_regression
        else "unproductive_axi_amplification"
        if unproductive_amplification
        else "admit_with_runtime_feedback"
    )
    return {
        "version": 1,
        # This is the slow-loop decision for the next compilation/invocation.
        # A rejected stream must keep the original formation path; merely
        # forcing every exact descriptor to complete synchronously still pays
        # descriptor and queue overhead and is not a native fallback.
        "residual_vdae_admitted": initial_dma_allowed,
        "initial_dma_allowed": initial_dma_allowed,
        "window_completions": window,
        "late_poll_threshold": 4,
        "probe_interval": probe_interval,
        "provenance": "sysmon-cross-invocation-v1",
        "decision": reason,
        "observations": {
            "baseline_axi_bytes": int(baseline_axi),
            "alps_axi_bytes": int(alps_axi),
            "alps_to_baseline_axi_ratio": traffic_ratio,
            "baseline_to_alps_latency_speedup": speedup,
            "alps_axi_p99_to_p50_ratio": burst_ratio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the slow-loop runtime admission policy for a later ALPS "
            "invocation from matched Hexagon-MLIR and ALPS sysMon summaries."
        )
    )
    parser.add_argument("--baseline-sysmon", type=Path, required=True)
    parser.add_argument("--alps-sysmon", type=Path, required=True)
    parser.add_argument("--baseline-latency-ms", type=float)
    parser.add_argument("--alps-latency-ms", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = derive_policy(
        load_summary(args.baseline_sysmon),
        load_summary(args.alps_sysmon),
        args.baseline_latency_ms,
        args.alps_latency_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
