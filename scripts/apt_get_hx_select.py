#!/usr/bin/env python3
"""Select deterministic APT-GET-HX distance/site plans from cycle profiles.

This is the offline policy half of the Hexagon port. It intentionally does
not discover addresses: APT-GET-HX consumes manually-qualified candidates and
is evaluated independently from Prefetch-Kernel-HX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_DISTANCES = (1, 2, 4, 8, 16)


class ProfileError(ValueError):
    pass


def _require_number(obj: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = obj.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProfileError(f"{key} must be numeric")
    if positive and value <= 0:
        raise ProfileError(f"{key} must be positive")
    return float(value)


def detect_peaks(histogram: list[dict[str, Any]], min_fraction: float = 0.10) -> list[int]:
    """Return stable local maxima without adding a SciPy runtime dependency."""
    if not isinstance(histogram, list) or len(histogram) < 2:
        return []
    bins: dict[int, int] = {}
    for entry in histogram:
        if not isinstance(entry, dict):
            raise ProfileError("histogram entries must be objects")
        cycles = int(_require_number(entry, "cycles", positive=True))
        count = int(_require_number(entry, "count", positive=True))
        bins[cycles] = bins.get(cycles, 0) + count
    points = sorted(bins.items())
    maximum = max(count for _, count in points)
    threshold = maximum * min_fraction
    peaks: list[tuple[int, int]] = []
    for index, (cycles, count) in enumerate(points):
        left = points[index - 1][1] if index else -1
        right = points[index + 1][1] if index + 1 < len(points) else -1
        if count >= threshold and count >= left and count >= right:
            peaks.append((cycles, count))

    # Merge nearby noise maxima, retaining the most populated bin.
    merged: list[tuple[int, int]] = []
    for peak in peaks:
        if merged and peak[0] - merged[-1][0] < max(2, int(merged[-1][0] * 0.15)):
            if peak[1] > merged[-1][1]:
                merged[-1] = peak
        else:
            merged.append(peak)
    return [cycles for cycles, _ in merged]


def _timing(loop: dict[str, Any], min_separation: float) -> tuple[int, int, int] | None:
    peaks = detect_peaks(loop.get("iteration_cycle_histogram", []))
    if len(peaks) < 2:
        return None
    warm, cold = peaks[0], peaks[-1]
    if (cold - warm) / warm < min_separation:
        return None
    return warm, cold, max(1, math.ceil((cold - warm) / warm))


def _nearest_legal(target: int, legal: list[int]) -> int:
    return min(legal, key=lambda value: (abs(value - target), value))


def select_candidate(
    candidate: dict[str, Any],
    *,
    legal_distances: list[int],
    coverage_factor: float,
    min_peak_separation: float,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id", ""))
    result: dict[str, Any] = {"candidate_id": candidate_id, "enabled": False}
    if not candidate_id:
        raise ProfileError("candidate_id must be a non-empty string")
    if candidate.get("memory_bound") is False:
        result["reason"] = "profile_marks_compute_bound"
        return result

    inner = candidate.get("inner_loop")
    if not isinstance(inner, dict):
        raise ProfileError(f"{candidate_id}: inner_loop must be an object")
    timing = _timing(inner, min_peak_separation)
    if timing is None:
        result["reason"] = "no_separable_latency_peaks"
        return result
    warm, cold, modeled_distance = timing
    trip_count = int(_require_number(inner, "trip_count", positive=True))
    site = "inner"
    selected_loop = inner

    if trip_count * coverage_factor < modeled_distance:
        outer = candidate.get("outer_loop")
        if not isinstance(outer, dict):
            result["reason"] = "outer_site_required_but_unprofiled"
            return result
        outer_timing = _timing(outer, min_peak_separation)
        if outer_timing is None:
            result["reason"] = "outer_site_has_no_separable_latency_peaks"
            return result
        warm, cold, modeled_distance = outer_timing
        trip_count = int(_require_number(outer, "trip_count", positive=True))
        site = "outer"
        selected_loop = outer

    row_bytes = int(_require_number(candidate, "row_bytes", positive=True))
    rows = int(_require_number(candidate, "rows", positive=True))
    if row_bytes * rows > int(candidate.get("max_command_bytes", 8191)):
        result["reason"] = "request_exceeds_command_budget"
        return result
    if int(candidate.get("page_split_count", 0)) > int(candidate.get("max_page_splits", 16)):
        result["reason"] = "page_split_budget_exceeded"
        return result

    legal = [distance for distance in legal_distances if distance < trip_count]
    capacity = candidate.get("residency_budget_bytes")
    if capacity is not None:
        capacity = int(capacity)
        legal = [distance for distance in legal if distance * row_bytes * rows <= capacity]
    if not legal:
        result["reason"] = "no_distance_survives_trip_or_capacity_projection"
        return result

    distance = _nearest_legal(modeled_distance, legal)
    result.update(
        enabled=True,
        reason="selected",
        injection_site=site,
        loop_id=str(selected_loop.get("loop_id", site)),
        warm_peak_cycles=warm,
        cold_peak_cycles=cold,
        modeled_distance=modeled_distance,
        distance=distance,
        row_bytes=row_bytes,
        rows=rows,
        stride=int(_require_number(candidate, "stride", positive=True)),
        projected_live_bytes=distance * row_bytes * rows,
        address_source=str(candidate.get("address_source", "manual")),
    )
    return result


def select_plan(profile: dict[str, Any], expected_shape: str | None = None) -> dict[str, Any]:
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ProfileError(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("model", "kernel", "shape"):
        if not isinstance(profile.get(key), str) or not profile[key]:
            raise ProfileError(f"{key} must be a non-empty string")
    if expected_shape is not None and profile["shape"] != expected_shape:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": profile["model"],
            "kernel": profile["kernel"],
            "shape": profile["shape"],
            "status": "no_prefetch",
            "reason": "shape_mismatch",
            "expected_shape": expected_shape,
            "plans": [],
        }

    policy = profile.get("policy", {})
    legal = sorted({int(value) for value in policy.get("legal_distances", DEFAULT_DISTANCES)})
    if not legal or legal[0] <= 0:
        raise ProfileError("legal_distances must contain positive integers")
    candidates = profile.get("candidates")
    if not isinstance(candidates, list):
        raise ProfileError("candidates must be an array")
    plans = [
        select_candidate(
            candidate,
            legal_distances=legal,
            coverage_factor=float(policy.get("coverage_factor", 5.0)),
            min_peak_separation=float(policy.get("min_peak_separation", 0.25)),
        )
        for candidate in candidates
    ]
    enabled = sum(bool(plan["enabled"]) for plan in plans)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": "apt-get-hx",
        "model": profile["model"],
        "kernel": profile["kernel"],
        "shape": profile["shape"],
        "profile_sha256": hashlib.sha256(canonical).hexdigest(),
        "status": "enabled" if enabled else "no_prefetch",
        "enabled_candidates": enabled,
        "plans": plans,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--expected-shape")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        plan = select_plan(profile, args.expected_shape)
    except (OSError, json.JSONDecodeError, ProfileError) as error:
        parser.error(str(error))
    rendered = json.dumps(plan, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
