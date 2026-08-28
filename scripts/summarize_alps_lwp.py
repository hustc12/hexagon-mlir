#!/usr/bin/env python3
"""Rank LWP regions and join them to ALPS movement sites by source line."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_lwp(path: Path) -> dict[int, dict[str, int | None]]:
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    stack: list[tuple[int, int, int | None]] = []
    result: dict[int, dict[str, int | None]] = {}
    for entry in entries:
        region_id, cycle = int(entry["id"]), int(entry["cyc"])
        if stack and stack[-1][0] == region_id:
            _, start, parent = stack.pop()
            row = result.setdefault(
                region_id, {"pcycles": 0, "iterations": 0, "parent": parent}
            )
            row["pcycles"] = int(row["pcycles"] or 0) + cycle - start
            row["iterations"] = int(row["iterations"] or 0) + 1
            if row["parent"] is None:
                row["parent"] = parent
        else:
            if any(open_id == region_id for open_id, _, _ in stack):
                raise ValueError(f"interleaved LWP region {region_id}")
            stack.append((region_id, cycle, stack[-1][0] if stack else None))
    if stack:
        raise ValueError(f"unclosed LWP regions: {[row[0] for row in stack]}")
    return result


def parse_info(path: Path) -> dict[int, dict[str, str | set[int]]]:
    result: dict[int, dict[str, str | set[int]]] = {}
    pattern = re.compile(
        r"Location ([\d,\s]*) corresponds to ID (\d+) \| Collected ops:\s*(.*)"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.fullmatch(line.strip())
        if not match:
            continue
        lines = {int(value) for value in re.findall(r"\d+", match.group(1))}
        result[int(match.group(2))] = {"lines": lines, "ops": match.group(3)}
    return result


def parse_sites(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[ALPS-P1-SITE]"):
            continue
        row: dict[str, object] = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
        source = str(row.get("source_lines", "none"))
        row["_lines"] = {
            int(value) for value in source.split(",") if value.isdigit()
        }
        result.append(row)
    return result


def parse_accesses(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[ALPS-P1-ACCESS]"):
            continue
        row: dict[str, object] = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
        source = str(row.get("source_lines", "none"))
        row["_lines"] = {
            int(value) for value in source.split(",") if value.isdigit()
        }
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--info-dump", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    json_files = sorted(args.artifact_root.rglob("lwp.json"))
    if len(json_files) != 1:
        raise ValueError(
            f"expected one monolithic lwp.json, found {len(json_files)}"
        )
    cycles = parse_lwp(json_files[0])
    info = parse_info(args.info_dump)
    sites = parse_sites(args.ledger)
    accesses = parse_accesses(args.ledger)
    rows = []
    for region_id, timing in cycles.items():
        metadata = info.get(region_id, {"lines": set(), "ops": ""})
        region_lines = metadata["lines"]
        assert isinstance(region_lines, set)
        matched = [site for site in sites if region_lines & site["_lines"]]
        physical = [
            site
            for site in matched
            if site.get("kind")
            in ("physical_copy", "physical_layout_transform")
        ]
        candidates = [
            site
            for site in matched
            if site.get("kind") == "representation_candidate"
        ]
        matched_accesses = [
            access
            for access in accesses
            if access.get("phase") == "post-bufferization"
            and region_lines & access["_lines"]
        ]
        rows.append(
            {
                "id": region_id,
                "pcycles": int(timing["pcycles"] or 0),
                "inclusive_pcycles": int(timing["pcycles"] or 0),
                "exclusive_pcycles": 0,
                "iterations": int(timing["iterations"] or 0),
                "parent": (
                    timing["parent"] if timing["parent"] is not None else "-"
                ),
                "source_lines": ",".join(map(str, sorted(region_lines)))
                or "none",
                "ops": metadata["ops"],
                "matched_sites": len(matched),
                "candidate_sites": len(candidates),
                "physical_sites": len(physical),
                "physical_materialization_bytes": sum(
                    max(0, int(str(site.get("materialization_bytes", "0"))))
                    for site in physical
                ),
                "candidate_static_bytes": sum(
                    max(0, int(str(site.get("static_bytes", "-1"))))
                    for site in candidates
                ),
                "access_sites": len(matched_accesses),
                "logical_read_upper_bytes": sum(
                    max(0, int(str(access.get("logical_read_upper_bytes", "0"))))
                    for access in matched_accesses
                ),
                "logical_write_upper_bytes": sum(
                    max(0, int(str(access.get("logical_write_upper_bytes", "0"))))
                    for access in matched_accesses
                ),
                "unique_operand_bytes": sum(
                    max(0, int(str(access.get("unique_operand_bytes", "0"))))
                    for access in matched_accesses
                ),
            }
        )
    by_id = {int(row["id"]): row for row in rows}
    child_cycles: dict[int, int] = {}
    for row in rows:
        parent = row["parent"]
        if parent is None or parent == "-":
            continue
        parent_id = int(parent)
        child_cycles[parent_id] = child_cycles.get(parent_id, 0) + int(
            row["inclusive_pcycles"]
        )
    for region_id, row in by_id.items():
        row["exclusive_pcycles"] = max(
            0, int(row["inclusive_pcycles"]) - child_cycles.get(region_id, 0)
        )
    rows.sort(key=lambda row: int(row["exclusive_pcycles"]), reverse=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "id",
        "pcycles",
        "inclusive_pcycles",
        "exclusive_pcycles",
        "iterations",
        "parent",
        "source_lines",
        "ops",
        "matched_sites",
        "candidate_sites",
        "physical_sites",
        "physical_materialization_bytes",
        "candidate_static_bytes",
        "access_sites",
        "logical_read_upper_bytes",
        "logical_write_upper_bytes",
        "unique_operand_bytes",
    )
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # Inclusive parent/child regions overlap. Exclusive cycles form a true
    # partition of the instrumented root and are therefore the only valid
    # denominator for hotspot shares.
    total = sum(int(row["exclusive_pcycles"]) for row in rows)
    lines = [
        "# ALPS P1 LWP region ranking",
        "",
        "Instrumented pcycles are for ranking only; they do not replace formal latency.",
        "",
        "| Rank | ID | Exclusive pcycles | Inclusive pcycles | Exclusive share | Iterations | Source lines | Physical bytes | Logical R/W upper bytes | Access sites | Ops |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(rows[:30], 1):
        share = (
            100.0 * int(row["exclusive_pcycles"]) / total if total else 0.0
        )
        ops = str(row["ops"]).replace("|", "/")
        lines.append(
            f"| {index} | {row['id']} | {row['exclusive_pcycles']} | "
            f"{row['inclusive_pcycles']} | {share:.2f}% | "
            f"{row['iterations']} | {row['source_lines']} | "
            f"{row['physical_materialization_bytes']} | "
            f"{row['logical_read_upper_bytes']}/{row['logical_write_upper_bytes']} | "
            f"{row['access_sites']} | {ops} |"
        )
    hexkl_phases: dict[str, dict[str, int]] = {}
    for row in rows:
        op_name = str(row["ops"])
        if not op_name.startswith("hexkl.micro_hmx_") or "," in op_name:
            continue
        phase = hexkl_phases.setdefault(op_name, {"cycles": 0, "iterations": 0})
        phase["cycles"] += int(row["exclusive_pcycles"])
        phase["iterations"] += int(row["iterations"])
    if hexkl_phases:
        lines.extend(
            [
                "",
                "## HexKL/HMX phase aggregate",
                "",
                "Analysis-only per-operation LWP; call overhead is included, so use for ranking rather than formal latency.",
                "",
                "| Phase | Exclusive pcycles | Root share | Dynamic invocations |",
                "|---|---:|---:|---:|",
            ]
        )
        for phase_name, values in sorted(
            hexkl_phases.items(), key=lambda item: item[1]["cycles"], reverse=True
        ):
            share = 100.0 * values["cycles"] / total if total else 0.0
            lines.append(
                f"| {phase_name} | {values['cycles']} | {share:.2f}% | "
                f"{values['iterations']} |"
            )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
