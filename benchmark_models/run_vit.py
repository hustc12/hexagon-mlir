"""ViT-Base Hexagon Phase-4 harness (full published architecture)."""
from __future__ import annotations

from typing import Optional
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageClassification, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    patch_dsp_heap_256mb,
    compile_to_linalg,
    hex_execution,
    apply_hexkl_ir_rewrites,
    hexagon_options_phase4,
    add_phase4_args,
)


def compare(
    hex_outputs,
    x86_tensor,
    atol=0.03,
    fail_on_mismatch: bool = False,
    require_exact_top5: bool = True,
):
    hex_logits = hex_outputs[0]
    max_diff = (hex_logits.float() - x86_tensor.float()).abs().max().item()
    print(f"\nMax difference between Hexagon and x86 outputs: {max_diff:.4f}")

    def top5(logits, tag):
        probs = torch.softmax(logits[0].float(), dim=-1)
        k = min(5, probs.shape[-1])
        vals, idxs = torch.topk(probs, k)
        print(f"\n------- Top-5 class predictions ({tag}) -------")
        for v, i in zip(vals.tolist(), idxs.tolist()):
            print(f"  class {i:4d}: {v:.4f}")
        print("-----------------------------------------------")
        return idxs.tolist(), vals.tolist()

    idxs_hex, vals_hex = top5(hex_logits, "Hexagon")
    idxs_x86, vals_x86 = top5(x86_tensor, "x86")

    if require_exact_top5:
        ok = idxs_hex == idxs_x86 and torch.allclose(
            torch.tensor(vals_hex), torch.tensor(vals_x86), atol=atol
        )
        msg = "Hexagon and CPU top-5 matched"
    else:
        ok = idxs_hex[0] == idxs_x86[0]
        msg = "Top-1 class matched (HexKL numerical tolerance)"
    if ok:
        print(msg)
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: Hexagon vs x86"


def customize_model_config(config):
    """Identity hook except gelu_new (Hexagon lacks math.erf)."""
    config.hidden_act = "gelu_new"
    return config


def load_vit_model(_model_name, config):
    return AutoModelForImageClassification.from_config(config).half()


def vit(
    enable_hexkl: bool = False,
    enable_omnifetch_vdae: bool = False,
    enable_omnifetch_layout_aware: bool = True,
    omnifetch_lookahead: int = 2,
    enable_omnifetch_adaptive: bool = True,
    seq_len: Optional[int] = None,  # unused; kept for CLI parity
    image_size: Optional[int] = None,
):
    patch_dsp_heap_256mb()
    model_name = "google/vit-base-patch16-224"

    config = AutoConfig.from_pretrained(model_name)
    config = customize_model_config(config)
    print(
        f"[Config] layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"patch={config.patch_size} image={config.image_size} "
        f"heads={config.num_attention_heads}"
    )

    model = load_vit_model(model_name, config)
    model.eval()

    class ViTWrapper(torch.nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            self.vit = vit_model

        def forward(self, pixel_values):
            return self.vit(pixel_values=pixel_values).logits

    wrapped = ViTWrapper(model).eval()
    func_name = wrapped.__class__.__name__

    img = image_size or config.image_size
    pixel_values = torch.rand(1, 3, img, img, dtype=torch.float16)
    print(f"[Input] pixel_values={tuple(pixel_values.shape)} HexKL={enable_hexkl}")

    module = compile_to_linalg(wrapped, (pixel_values,), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
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
    )
    hex_outputs = hex_execution(
        module, func_name, [pixel_values], options, mlir_text=mlir_text
    )
    print("Successfully ran ViT on Hexagon DSP!")

    with torch.no_grad():
        x86 = wrapped(pixel_values)
    compare(
        hex_outputs,
        x86,
        fail_on_mismatch=True,
        require_exact_top5=not enable_hexkl,
        atol=0.5 if enable_hexkl else 0.03,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ViT-Base Hexagon smoke (optional HexKL/OmniFetch)."
    )
    add_phase4_args(parser)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()
    vit(
        enable_hexkl=args.enable_hexkl,
        enable_omnifetch_vdae=args.enable_omnifetch_vdae,
        enable_omnifetch_layout_aware=not args.disable_layout_aware,
        omnifetch_lookahead=args.omnifetch_lookahead,
        enable_omnifetch_adaptive=not args.disable_omnifetch_adaptive,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )
