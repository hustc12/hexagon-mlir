"""Shared full-structure speech-encoder Hexagon benchmark harness."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


class FullAudioEncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=input_values).logits


def run_full_audio_encoder(
    args: argparse.Namespace,
    *,
    display_name: str,
    config,
    model_cls,
    root_name: str,
    seed: int,
) -> None:
    patch_full_model_dsp_heap()
    torch.manual_seed(seed)
    config.apply_spec_augment = False
    # Avoid an unsupported erf lowering while retaining the full architecture.
    config.hidden_act = "gelu_new"
    config.feat_extract_activation = "gelu_new"
    config._attn_implementation = "eager"
    model = model_cls(config).half().eval()

    # Weight normalization is training-time parametrization. Freeze its
    # effective convolution weight before torch-mlir export.
    root = getattr(model, root_name)
    conv = getattr(root.encoder.pos_conv_embed, "conv", None)
    if (
        conv is not None
        and hasattr(conv, "parametrizations")
        and "weight" in conv.parametrizations
    ):
        torch.nn.utils.parametrize.remove_parametrizations(
            conv, "weight", leave_parametrized=True
        )

    wrapped = FullAudioEncoderWrapper(model).eval()
    # The published seven-convolution frontend maps 20,560 samples to exactly
    # 64 encoder frames. At 16 kHz this is a stated 1.285-second workload.
    samples = torch.rand(1, 20560, dtype=torch.float16) * 2 - 1
    inputs = [samples]
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[FullModel] {display_name}: samples=20560 seconds@16k=1.285 "
        f"frames=64 layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} heads={config.num_attention_heads} "
        f"intermediate={config.intermediate_size} params={params} "
        "weights=random_full_structure"
    )

    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )
    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(ir)
        patched = candidate if n_batch or n_f16 else None
        print(
            f"[HexKL] batch_matmul→matmul={n_batch}, "
            f"f16-input rewrite={n_f16}"
        )
    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_omnifetch_vdae,
        not args.disable_layout_aware,
        args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,
        args.enable_omnifetch_items_1_7,
        lower_constants_separate=True,
    )
    output = hex_execution(
        module,
        wrapped.__class__.__name__,
        inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    with torch.no_grad():
        reference = wrapped(*inputs)
    finite = bool(torch.isfinite(output[0]).all())
    diff = (output[0].float() - reference.float()).abs().max().item()
    top1_match = (
        output[0][0, -1].argmax().item()
        == reference[0, -1].argmax().item()
    )
    print(
        f"[Compare] finite={finite} max_abs_diff={diff:.4f} "
        f"last_frame_top1_match={top1_match}"
    )
    if not finite or not top1_match:
        raise AssertionError(f"{display_name} failed correctness gate")


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    return parser
