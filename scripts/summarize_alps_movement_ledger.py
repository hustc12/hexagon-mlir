#!/usr/bin/env python3
"""Aggregate ALPS P1 summaries and rank auditable movement/candidate sites."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "candidates",
    "descriptor_sites",
    "physical_transform_sites",
    "copy_sites",
    "alloc_sites",
    "static_read_bytes",
    "static_write_bytes",
    "static_materialization_bytes",
    "dynamic_sites",
)


def parse(path: Path, prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        values = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
        if "phase" in values and "function" in values:
            rows.append(values)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--sites-csv", type=Path)
    parser.add_argument("--sites-markdown", type=Path)
    args = parser.parse_args()
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    functions: dict[str, set[str]] = defaultdict(set)
    for row in parse(args.ledger, "[ALPS-P1-SUMMARY]"):
        phase = row["phase"]
        functions[phase].add(row["function"])
        for field in FIELDS:
            totals[phase][field] += int(row.get(field, 0))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "functions", *FIELDS))
        for phase in sorted(totals):
            writer.writerow(
                (phase, len(functions[phase]), *(totals[phase][f] for f in FIELDS))
            )
    lines = [
        "# ALPS P1 movement ledger",
        "",
        "| Phase | Functions | Candidates | Descriptor views | Physical transforms | Copies | Allocs | Static materialization bytes | Dynamic sites |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in sorted(totals):
        row = totals[phase]
        lines.append(
            f"| {phase} | {len(functions[phase])} | {row['candidates']} | "
            f"{row['descriptor_sites']} | {row['physical_transform_sites']} | "
            f"{row['copy_sites']} | {row['alloc_sites']} | "
            f"{row['static_materialization_bytes']} | {row['dynamic_sites']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sites = parse(args.ledger, "[ALPS-P1-SITE]")
    ranked = []
    for site in sites:
        kind = site.get("kind", "")
        if kind not in (
            "representation_candidate",
            "physical_layout_transform",
            "physical_copy",
        ):
            continue
        static_bytes = max(0, int(site.get("static_bytes", "-1")))
        materialization = max(0, int(site.get("materialization_bytes", "0")))
        uses = max(1, int(site.get("uses", "0")))
        priority = materialization if materialization else static_bytes * uses
        site["priority_bytes"] = str(priority)
        ranked.append(site)
    ranked.sort(key=lambda row: int(row["priority_bytes"]), reverse=True)
    columns = (
        "phase",
        "function",
        "id",
        "kind",
        "op",
        "source_lines",
        "value_version",
        "shape",
        "engine",
        "layout",
        "memory_space",
        "static_bytes",
        "materialization_bytes",
        "uses",
        "pages",
        "first_use_distance",
        "last_use_ordinal",
        "legal_actions",
        "decision",
        "priority_bytes",
    )
    if args.sites_csv:
        args.sites_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.sites_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(ranked)
    if args.sites_markdown:
        site_lines = [
            "# ALPS P1 top movement and representation sites",
            "",
            "Priority is physical materialization bytes for copies/transforms; "
            "candidate priority is static bytes × uses and is not a latency estimate.",
            "",
            "| Rank | Phase | Function | Kind | Op | Source lines | Shape | Bytes | Uses | Priority bytes | Decision |",
            "|---:|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
        for index, row in enumerate(ranked[:30], 1):
            site_lines.append(
                f"| {index} | {row.get('phase', '')} | "
                f"{row.get('function', '')} | {row.get('kind', '')} | "
                f"{row.get('op', '')} | {row.get('source_lines', 'none')} | "
                f"{row.get('shape', '')} | {row.get('static_bytes', '-1')} | "
                f"{row.get('uses', '0')} | {row.get('priority_bytes', '0')} | "
                f"{row.get('decision', '')} |"
            )
        args.sites_markdown.write_text(
            "\n".join(site_lines) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
