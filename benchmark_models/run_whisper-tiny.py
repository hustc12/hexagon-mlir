#!/usr/bin/env python3
"""Whisper-tiny full-structure encoder-decoder Hexagon benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_full_model_dsp_heap,
)


class WhisperTinyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits


def run(args: argparse.Namespace) -> None:
    patch_full_model_dsp_heap()

    # Whisper's encoder convs call F.gelu directly. Use the equivalent
    # compiler-supported tanh approximation consistently for all three rows.
    def gelu_tanh(x, approximate="none"):
        del approximate
        return 0.5 * x * (
            1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
        )

    torch.nn.functional.gelu = gelu_tanh
    torch.manual_seed(390)
    config = WhisperConfig()
    config.use_cache = False
    config.activation_function = "gelu_new"
    config._attn_implementation = "eager"
    target_len = args.seq_len or 32
    if target_len > config.max_target_positions:
        raise ValueError(
            f"target length {target_len} exceeds {config.max_target_positions}"
        )
    model = WhisperForConditionalGeneration(config).half().eval()
    wrapped = WhisperTinyWrapper(model).eval()
    inputs = [
        # Official 30-second Whisper feature window: 80 mel bins x 3000 frames.
        torch.rand(1, config.num_mel_bins, 3000, dtype=torch.float16),
        torch.arange(target_len, dtype=torch.long).unsqueeze(0)
        % config.vocab_size,
    ]
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "[FullModel] Whisper-tiny: mel=80x3000 seconds=30 "
        f"encoder_layers={config.encoder_layers} "
        f"decoder_layers={config.decoder_layers} d_model={config.d_model} "
        f"heads={config.encoder_attention_heads}/"
        f"{config.decoder_attention_heads} target_len={target_len} "
        f"params={params} weights=random_full_structure"
    )

    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(
        f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} "
        f"matmul={ir.count('linalg.matmul')}"
    )
    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(
            ir, enable_m_pad=args.enable_omnifetch_m_pad_hmx
        )
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
        backend_profile=args.backend_profile,
        enable_lwp=args.enable_lwp,
        lwp_loop_depth=args.lwp_loop_depth,
        disable_lwp_loop=args.disable_lwp_loop,
        omnifetch_items_through=args.omnifetch_items_through,
        enable_omnifetch_m_pad_hmx=args.enable_omnifetch_m_pad_hmx,
        enable_out_params=args.enable_out_params,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
        enable_omnifetch_kv_cache_prefetch=(
            args.enable_omnifetch_kv_cache_prefetch
        ),
        disable_omnifetch_persistent_wh_cache=(
            args.disable_omnifetch_persistent_wh_cache
        ),
        enable_alps_fp16_hvx_arithmetic=(
            args.enable_alps_fp16_hvx_arithmetic
        ),
        enable_alps_hvx_widening_conv=(
            args.enable_alps_hvx_widening_conv
        ),
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
        f"last_token_top1_match={top1_match}"
    )
    if not finite or not top1_match:
        raise AssertionError("Whisper-tiny failed correctness gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
