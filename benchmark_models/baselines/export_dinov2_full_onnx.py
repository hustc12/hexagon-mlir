#!/usr/bin/env python3
"""Export the exact deterministic DINOv2-small full workload to ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dinov2_full_common import (  # noqa: E402
    create_dinov2_small_full_model_and_input,
    print_dinov2_small_full_identity,
)


def compare_output(
    output_dir: Path, qnn_output: Path, precision: str
) -> None:
    reference_dtype = np.float16 if precision == "fp16" else np.float32
    reference = np.fromfile(
        output_dir / f"reference_logits_{precision}.raw", dtype=reference_dtype
    ).astype(np.float32)
    output = np.fromfile(qnn_output, dtype=np.float32)
    if output.shape != reference.shape:
        raise ValueError(
            f"QNN output shape {output.shape} != reference {reference.shape}"
        )
    max_diff = float(np.max(np.abs(output - reference)))
    reference_top1 = int(reference.argmax())
    output_top1 = int(output.argmax())
    finite = bool(np.isfinite(output).all())
    print(
        f"[QNN Compare] finite={finite} max_abs_diff={max_diff:.6f} "
        f"top1_ref={reference_top1} top1_qnn={output_top1}"
    )
    if not finite or max_diff > 0.02 or reference_top1 != output_top1:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-qnn-output", type=Path)
    parser.add_argument(
        "--precision", choices=("fp16", "fp32"), default="fp16"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_qnn_output is not None:
        compare_output(
            args.output_dir, args.compare_qnn_output, args.precision
        )
        return

    wrapped, pixels = create_dinov2_small_full_model_and_input()
    if args.precision == "fp32":
        # Preserve the already-created FP16-grid parameter values exactly;
        # only expand their representation for the QNN CPU backend.
        wrapped = wrapped.float()
        pixels = pixels.float()
    print_dinov2_small_full_identity(wrapped, pixels)
    model_stem = (
        "dinov2_small_full" if args.precision == "fp16"
        else "dinov2_small_full_cpu"
    )
    onnx_path = args.output_dir / f"{model_stem}.onnx"
    input_path = args.output_dir / "pixels_nhwc_f32.raw"
    reference_path = args.output_dir / f"reference_logits_{args.precision}.raw"
    input_list_path = args.output_dir / "input_list.txt"

    with torch.no_grad():
        reference = wrapped(pixels)
    torch.onnx.export(
        wrapped,
        pixels,
        onnx_path,
        opset_version=17,
        input_names=["pixels"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    np.ascontiguousarray(
        pixels.permute(0, 2, 3, 1).float().numpy()
    ).tofile(input_path)
    np.ascontiguousarray(reference.numpy()).tofile(reference_path)
    input_list_path.write_text(f"pixels:={input_path}\n", encoding="utf-8")
    print(f"ONNX={onnx_path}")
    print(f"INPUT={input_path}")
    print(f"REFERENCE={reference_path}")


if __name__ == "__main__":
    main()
