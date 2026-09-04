#!/usr/bin/env python3
"""Host test for sysMon-derived ALPS traffic admission policy."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT / "scripts/script_release/internal/derive_alps_traffic_policy.py"
)
SPEC = importlib.util.spec_from_file_location("alps_traffic_policy", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def summary(axi: int, p50: int = 100, p99: int = 200) -> dict:
    return {
        "axi_total_bytes": axi,
        "axi_bytes_per_sample_p50": p50,
        "axi_bytes_per_sample_p99": p99,
    }


admitted = MODULE.derive_policy(summary(1000), summary(900), 10.0, 8.0)
assert admitted["initial_dma_allowed"] is True
assert admitted["residual_vdae_admitted"] is True
assert admitted["window_completions"] == 64

regressed = MODULE.derive_policy(summary(1000), summary(900), 10.0, 11.0)
assert regressed["initial_dma_allowed"] is False
assert regressed["residual_vdae_admitted"] is False
assert regressed["probe_interval"] == 16
assert regressed["decision"] == "latency_regression"

amplified = MODULE.derive_policy(summary(1000), summary(1200), 10.0, 9.9)
assert amplified["initial_dma_allowed"] is False
assert amplified["residual_vdae_admitted"] is False
assert amplified["decision"] == "unproductive_axi_amplification"

bursty = MODULE.derive_policy(summary(1000), summary(900, 100, 500))
assert bursty["window_completions"] == 32

print("ALPS traffic policy: PASS")
