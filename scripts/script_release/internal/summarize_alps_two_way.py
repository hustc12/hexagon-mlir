#!/usr/bin/env python3
"""Render the frozen HexKL-On versus ALPS full-E complete-model matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_alps_full_matrix import (
    display_int,
    display_ms,
    display_ratio,
    integer,
    load_runtime_ledger,
    load_sysmon,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    with args.results.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    cases: dict[str, dict[str, dict[str, str]]] = {}
    order: list[str] = []
    for row in input_rows:
        if row["model"] not in cases:
            cases[row["model"]] = {}
            order.append(row["model"])
        cases[row["model"]][row["scheme"]] = row

    output_rows: list[dict[str, object]] = []
    for model in order:
        control = cases[model].get("hmlir-hvx-hexkl-on", {})
        final = cases[model].get("alps-final", {})
        control_mat = integer(control.get("static_materialization_bytes"))
        final_mat = integer(final.get("static_materialization_bytes"))
        mat_reduction = (
            control_mat - final_mat
            if control_mat is not None and final_mat is not None
            else None
        )
        control_sysmon = load_sysmon(
            args.output_root, model, "hmlir-hvx-hexkl-on"
        )
        final_sysmon = load_sysmon(args.output_root, model, "alps-final")
        control_axi = integer(control_sysmon.get("axi_total_bytes"))
        final_axi = integer(final_sysmon.get("axi_total_bytes"))
        runtime = load_runtime_ledger(args.output_root, model, "alps-final")
        reused_control = (
            args.output_root / model / "hmlir-hvx-hexkl-on" / ".reused-from"
        )
        reused_final = args.output_root / model / "alps-final" / ".reused-from"
        output_rows.append(
            {
                "model": model,
                "domain": control.get("domain", final.get("domain", "NA")),
                "hexkl_on_status": control.get("status", "NA"),
                "hexkl_on_latency_ms": control.get("latency_ms", "NA"),
                "alps_status": final.get("status", "NA"),
                "alps_latency_ms": final.get("latency_ms", "NA"),
                "speedup_hexkl_over_alps": display_ratio(
                    control.get("latency_ms"), final.get("latency_ms")
                ).removesuffix("x"),
                "hexkl_on_materialization_bytes": (
                    control_mat if control_mat is not None else "NA"
                ),
                "alps_materialization_bytes": (
                    final_mat if final_mat is not None else "NA"
                ),
                "materialization_reduction_bytes": (
                    mat_reduction if mat_reduction is not None else "NA"
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
                "hexkl_on_axi_total_bytes": (
                    control_axi if control_axi is not None else "NA"
                ),
                "alps_axi_total_bytes": final_axi if final_axi is not None else "NA",
                "axi_reduction_bytes": (
                    control_axi - final_axi
                    if control_axi is not None and final_axi is not None
                    else "NA"
                ),
                "p2e_producer_direct": runtime.get("p2e_producer_direct", "NA"),
                "p2e_demands": runtime.get("p2e_demands", "NA"),
                "p5h_rewritten": runtime.get("p5h_rewritten", "NA"),
                "p5i_formed": runtime.get("p5i_formed", "NA"),
                "control_provenance": "reused" if reused_control.exists() else "new",
                "alps_provenance": "reused" if reused_final.exists() else "new",
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# Frozen ALPS full-E two-way complete-model matrix",
        "",
        "All models are complete non-Debug FP16 models. Speedup is HexKL-On / ALPS.",
        "",
        "| Domain | Model | HMLIR HVX (HexKL On) | ALPS C+full-E+P+R | Speedup | Provenance (control/final) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['domain']} | {row['model']} | "
            f"{display_ms(row['hexkl_on_latency_ms'])} ms | "
            f"{display_ms(row['alps_latency_ms'])} ms | "
            f"{row['speedup_hexkl_over_alps']}x | "
            f"{row['control_provenance']}/{row['alps_provenance']} |"
        )
    lines.extend(
        [
            "",
            "## Movement and runtime audit",
            "",
            "| Model | HMLIR materialization | ALPS materialization | Reduction | P2e/P5h/P5i eliminated | DMA issued/bytes | HMLIR AXI | ALPS AXI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in output_rows:
        lines.append(
            f"| {row['model']} | "
            f"{display_int(row['hexkl_on_materialization_bytes'])} | "
            f"{display_int(row['alps_materialization_bytes'])} | "
            f"{display_int(row['materialization_reduction_bytes'])} | "
            f"{display_int(row['p2e_eliminated_bytes'])}/"
            f"{display_int(row['p5h_eliminated_copy_bytes'])}/"
            f"{display_int(row['p5i_eliminated_transpose_bytes'])} | "
            f"{row['runtime_dma_issued']}/"
            f"{display_int(row['runtime_dma_issued_bytes'])} | "
            f"{display_int(row['hexkl_on_axi_total_bytes'])} | "
            f"{display_int(row['alps_axi_total_bytes'])} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
