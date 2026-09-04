"""Swin-Tiny Hexagon Phase-4 harness (full published architecture)."""
from __future__ import annotations

from typing import Optional
import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import SwinConfig
from transformers.models.swin.modeling_swin import SwinForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    patch_full_model_dsp_heap,
    compile_to_linalg,
    hex_execution,
    apply_hexkl_ir_rewrites,
    hexagon_options_phase4,
    add_phase4_args,
)


class SwinWrapper(torch.nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.swin = swin_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.swin(pixel_values=pixel_values).logits


def compare(
    hex_outputs,
    x86_tensor,
    atol=0.05,
    fail_on_mismatch: bool = False,
    require_exact_top5: bool = True,
):
    hex_logits = hex_outputs[0]
    max_diff = (hex_logits.float() - x86_tensor.float()).abs().max().item()
    print(f"\nMax logit difference between Hexagon and x86: {max_diff:.4f}")

    def top5(logits, tag):
        probs = torch.softmax(logits[0].float(), dim=-1)
        k = min(5, probs.shape[-1])
        vals, idxs = torch.topk(probs, k)
        print(f"\n------- Top-5 class predictions ({tag}) -------")
        for v, i in zip(vals.tolist(), idxs.tolist()):
            print(f"  class {i:4d}: {v:.4f}")
        print("-----------------------------------------------")
        return idxs.tolist(), vals.tolist()

    idxs_hex, _ = top5(hex_logits, "Hexagon")
    idxs_x86, _ = top5(x86_tensor.to(hex_logits.dtype), "x86")
    if require_exact_top5:
        ok = idxs_hex == idxs_x86 and torch.allclose(
            hex_logits.float(), x86_tensor.float(), atol=atol
        )
        msg = "Hexagon and CPU results matched within the specified tolerance."
    else:
        ok = idxs_hex[0] == idxs_x86[0]
        msg = "Top-1 class matched (HexKL numerical tolerance)"
    if ok:
        print(msg)
    else:
        print("Hexagon and CPU results do not match.")
        assert not fail_on_mismatch, "Correctness issue: Hexagon vs x86"


LOWER_CONSTANTS_SEPARATE = True  # full Swin consts need separate SO; debug sets False


def customize_model_config(config):
    config.hidden_act = "gelu_new"
    return config


def load_swin_model(_model_name, config):
    return SwinForImageClassification(config).half()


def swin_transformer(
    enable_hexkl: bool = False,
    enable_alps_vdae: bool = False,
    enable_alps_layout_aware: bool = True,
    alps_lookahead: int = 2,
    enable_alps_adaptive: bool = True,
    enable_alps_items_1_7: bool = False,
    seq_len: Optional[int] = None,
    device_iterations: int = 1,
    backend_profile: str = "hvx-vector",
    enable_alps_m_pad_hmx: bool = False,
    enable_out_params: bool = False,
    prefetch_baseline: str = "none",
    prefetch_baseline_distance: int = 1,
    apt_get_hx_manual_candidate_ids: str = "",
    enable_lwp: bool = False,
    lwp_loop_depth: int = 1,
    disable_lwp_loop: bool = False,
    alps_items_through: int = 0,
    enable_alps_kv_cache_prefetch: bool = False,
    disable_alps_persistent_wh_cache: bool = False,
    model_layers: Optional[int] = None,
    input_size: int = 224,
):
    # Full 28M-parameter graph exceeds the Debug runner's 256 MiB heap.
    patch_full_model_dsp_heap()
    # Keep random full-structure weights and inputs identical across the five
    # separately launched configurations.
    torch.manual_seed(211)
    model_name = "microsoft/swin-tiny-patch4-window7-224"

    # The benchmark only needs the architecture config and uses deterministic
    # random weights.  Never perform a network metadata probe during a device
    # or simulator experiment; the config is part of the local HF cache.
    config = SwinConfig.from_pretrained(model_name, local_files_only=True)
    config = customize_model_config(config)
    reduced_depths = {
        1: [1],
        3: [1, 1, 1],
        5: [1, 1, 2, 1],
        12: [2, 2, 6, 2],
    }
    if model_layers is not None:
        config.depths = reduced_depths[model_layers]
    config.image_size = input_size
    print(
        f"[Config] depths={config.depths} embed_dim={config.embed_dim} "
        f"num_heads={config.num_heads} window={config.window_size}"
    )

    model = load_swin_model(model_name, config)
    model.eval()
    wrapped = SwinWrapper(model).eval()
    func_name = wrapped.__class__.__name__

    pixel_values = torch.rand(1, 3, input_size, input_size, dtype=torch.float16)

    # relative_position_index buffers are lifted as extra ABI inputs.
    rel_pos_indices = []
    for layer in wrapped.swin.swin.encoder.layers:
        for block in layer.blocks:
            rel_pos_indices.append(
                block.attention.self.relative_position_index.detach()
            )

    module = compile_to_linalg(wrapped, (pixel_values,), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )

    mlir_text = None
    if enable_hexkl:
        ir2, n_bm, n_f16 = apply_hexkl_ir_rewrites(
            ir, enable_m_pad=enable_alps_m_pad_hmx
        )
        mlir_text = ir2
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")

    options = hexagon_options_phase4(
        enable_hexkl,
        enable_alps_vdae,
        enable_alps_layout_aware,
        alps_lookahead,
        enable_alps_adaptive,
        enable_alps_items_1_7,
        lower_constants_separate=LOWER_CONSTANTS_SEPARATE,
        backend_profile=backend_profile,
        enable_alps_m_pad_hmx=enable_alps_m_pad_hmx,
        enable_out_params=enable_out_params,
        prefetch_baseline=prefetch_baseline,
        prefetch_baseline_distance=prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=apt_get_hx_manual_candidate_ids,
        enable_lwp=enable_lwp,
        lwp_loop_depth=lwp_loop_depth,
        disable_lwp_loop=disable_lwp_loop,
        alps_items_through=alps_items_through,
        enable_alps_kv_cache_prefetch=enable_alps_kv_cache_prefetch,
        disable_alps_persistent_wh_cache=(
            disable_alps_persistent_wh_cache
        ),
    )
    inputs = rel_pos_indices + [pixel_values]
    hex_outputs = hex_execution(
        module,
        func_name,
        inputs,
        options,
        mlir_text=mlir_text,
        iterations=device_iterations,
    )
    if os.getenv("HEXAGON_MLIR_COMPILE_ONLY", "0") == "1":
        return
    print("Successfully ran Swin on Hexagon DSP!")

    with torch.no_grad():
        x86 = wrapped(pixel_values)
    compare(
        hex_outputs,
        x86,
        atol=0.5 if enable_hexkl else 0.1,
        fail_on_mismatch=True,
        require_exact_top5=not enable_hexkl,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swin-Tiny Hexagon smoke (optional HexKL/Alps)."
    )
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    parser.add_argument(
        "--model-layers", type=int, choices=(1, 3, 5, 12), default=12,
        help="Total Swin blocks; 12 is the published [2,2,6,2] architecture.",
    )
    args = parser.parse_args()
    swin_transformer(
        enable_hexkl=args.enable_hexkl,
        enable_alps_vdae=args.enable_alps_vdae,
        enable_alps_layout_aware=not args.disable_layout_aware,
        alps_lookahead=args.alps_lookahead,
        enable_alps_adaptive=not args.disable_alps_adaptive,
        enable_alps_items_1_7=args.enable_alps_items_1_7,
        seq_len=args.seq_len,
        device_iterations=args.device_iterations,
        backend_profile=args.backend_profile,
        enable_alps_m_pad_hmx=args.enable_alps_m_pad_hmx,
        enable_out_params=args.enable_out_params,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
        enable_lwp=args.enable_lwp,
        lwp_loop_depth=args.lwp_loop_depth,
        disable_lwp_loop=args.disable_lwp_loop,
        alps_items_through=args.alps_items_through,
        enable_alps_kv_cache_prefetch=(
            args.enable_alps_kv_cache_prefetch
        ),
        disable_alps_persistent_wh_cache=(
            args.disable_alps_persistent_wh_cache
        ),
        model_layers=args.model_layers,
    )
