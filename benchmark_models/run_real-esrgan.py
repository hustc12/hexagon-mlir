"""Real-ESRGAN x4 Hexagon Phase-4 harness."""
from __future__ import annotations

from typing import Optional
import argparse
import sys
from pathlib import Path

import torch
import huggingface_hub

# huggingface_hub removed cached_download / hf_hub_url; RealESRGAN still imports them.
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
if not hasattr(huggingface_hub, "hf_hub_url"):
    def _hf_hub_url(repo_id, filename, **kwargs):
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    huggingface_hub.hf_hub_url = _hf_hub_url

from RealESRGAN import RealESRGAN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    patch_dsp_heap_256mb,
    compile_to_linalg,
    hex_execution,
    apply_hexkl_ir_rewrites,
    hexagon_options_phase4,
    add_phase4_args,
)


def compare(hex_outputs, x86_outputs, atol=0.05, fail_on_mismatch: bool = False):
    hexagon_output = hex_outputs[0]
    max_diff = (hexagon_output.float() - x86_outputs.float()).abs().max().item()
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff:.4f}")
    match = torch.allclose(hexagon_output.float(), x86_outputs.float(), atol=atol)
    if match:
        print("Hexagon and CPU results matched within the specified tolerance.")
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: Hexagon vs x86"


def customize_input_size(default: int = 64) -> int:
    """Spatial input size (RRDBNet topology is always full). Debug may shrink.

    Default 64×64 is a practical full-harness smoke size; published demo images
    are often larger. Pass --input-size to override.
    """
    return default


def real_esrgan(
    enable_hexkl: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    seq_len: Optional[int] = None,
    input_size: Optional[int] = None,
):
    patch_dsp_heap_256mb()
    device = torch.device("cpu")
    model_wrapper = RealESRGAN(device, scale=4)
    weights_path = huggingface_hub.hf_hub_download(
        "ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth"
    )
    model_wrapper.load_weights(weights_path, download=False)
    model = model_wrapper.model.eval()
    func_name = model.__class__.__name__

    size = input_size if input_size is not None else customize_input_size(64)
    # Full RRDBNet topology; spatial size is the DSP-capacity knob (not layer count).
    input_tensor = torch.rand(1, 3, size, size)
    print(f"[Config] RealESRGAN x4 RRDBNet input={size}x{size} HexKL={enable_hexkl}")

    module = compile_to_linalg(model, (input_tensor,), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')} conv={ir.count('linalg.conv')}"
    )

    mlir_text = None
    if enable_hexkl:
        ir2, n_bm, n_f16 = apply_hexkl_ir_rewrites(ir)
        mlir_text = ir2
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")

    options = hexagon_options_phase4(
        enable_hexkl,
        enable_omnifetch_vdae,
        enable_omnifetch_layout_aware,
        omnifetch_lookahead,
        enable_omnifetch_adaptive,
        lower_constants_separate=True,
    )
    hex_outputs = hex_execution(
        module, func_name, [input_tensor], options, mlir_text=mlir_text
    )
    print("Successfully ran Real-ESRGAN on Hexagon DSP!")

    with torch.no_grad():
        x86 = model(input_tensor)
    compare(hex_outputs, x86, atol=0.05, fail_on_mismatch=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-ESRGAN Hexagon smoke (optional HexKL/OmniFetch)."
    )
    add_phase4_args(parser)
    parser.add_argument("--input-size", type=int, default=None)
    args = parser.parse_args()
    real_esrgan(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        seq_len=args.seq_len,
        input_size=args.input_size,
    )
