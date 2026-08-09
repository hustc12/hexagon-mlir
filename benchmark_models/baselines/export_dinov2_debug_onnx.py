#!/usr/bin/env python3
"""Export the deterministic DINOv2 Debug proxy for external baselines."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dinov2_debug_common import (
    create_dinov2_debug_model_and_input,
    print_dinov2_debug_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-qnn-output", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_qnn_output is not None:
        reference = np.fromfile(
            args.output_dir / "reference_logits_f16.raw", dtype=np.float16
        ).astype(np.float32)
        output = np.fromfile(args.compare_qnn_output, dtype=np.float32)
        if output.shape != reference.shape:
            raise ValueError(
                f"QNN output shape {output.shape} != reference {reference.shape}"
            )
        max_diff = float(np.max(np.abs(output - reference)))
        reference_top1 = int(reference.argmax())
        output_top1 = int(output.argmax())
        print(
            f"[QNN Compare] max_abs_diff={max_diff:.6f} "
            f"top1_ref={reference_top1} top1_qnn={output_top1}"
        )
        if max_diff > 0.01 or reference_top1 != output_top1:
            raise SystemExit(1)
        return

    model, pixels = create_dinov2_debug_model_and_input(
        static_position_export=True
    )
    print_dinov2_debug_identity(model, pixels)
    onnx_path = args.output_dir / "dinov2_debug.onnx"
    input_path = args.output_dir / "pixels_nhwc_f32.raw"
    reference_path = args.output_dir / "reference_logits_f16.raw"
    input_list_path = args.output_dir / "input_list.txt"

    with torch.no_grad():
        reference = model(pixels)
    torch.onnx.export(
        model,
        pixels,
        onnx_path,
        opset_version=17,
        input_names=["pixels"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    # QNN's default spatial-first lowering exposes image inputs as NHWC even
    # though the source ONNX input is NCHW.  Serialize the same logical tensor
    # in the physical layout reported by qnn-net-run metadata.
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
