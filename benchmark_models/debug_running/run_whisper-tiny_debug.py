#!/usr/bin/env python3
"""Reduced offline Whisper-tiny encoder-decoder candidate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

_BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BENCH))
from hexkl_utils import (  # noqa: E402
    add_phase4_args, apply_hexkl_ir_rewrites, compile_to_linalg, hex_execution,
    hexagon_options_phase4, patch_dsp_heap_256mb,
)


class WhisperDebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits


def run(args):
    patch_dsp_heap_256mb()
    # WhisperEncoder hard-codes F.gelu for its two convolutional layers even
    # when config.activation_function is gelu_new.  Replace it process-locally
    # so torch-mlir does not leave unsupported math.erf in the Hexagon path.
    def gelu_tanh(x, approximate="none"):
        del approximate
        return 0.5 * x * (
            1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
        )

    torch.nn.functional.gelu = gelu_tanh
    torch.manual_seed(390)
    target_len = args.seq_len or 32
    config = WhisperConfig(
        vocab_size=1024,
        num_mel_bins=80,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=16,
        max_target_positions=max(64, target_len),
        activation_function="gelu_new",
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    print(
        "[DebugCandidate] Whisper-tiny proxy: encoder/decoder=1/1 d_model=32 "
        f"source_frames=32 target_len={target_len} (random FP16 weights)"
    )
    wrapped = WhisperDebugWrapper(
        WhisperForConditionalGeneration(config).half().eval()
    ).eval()
    inputs = [
        torch.rand(1, 80, 32, dtype=torch.float16),
        torch.arange(target_len, dtype=torch.long).unsqueeze(0) % config.vocab_size,
    ]
    module = compile_to_linalg(wrapped, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    patched = None
    if args.enable_hexkl:
        patched, n_bm, n_f16 = apply_hexkl_ir_rewrites(ir)
        print(f"[HexKL] batch_matmul→matmul={n_bm}, f16-input rewrite={n_f16}")
        if n_bm == 0 and n_f16 == 0:
            # Preserve the original bytecode so tm_tensor.attention remains
            # registered; reparsing textual IR loses that custom dialect.
            patched = None
    options = hexagon_options_phase4(
        args.enable_hexkl, args.enable_omnifetch_vdae,
        not args.disable_layout_aware, args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive, args.enable_omnifetch_items_1_7,
        lower_constants_separate=False,
        backend_profile=args.backend_profile,
        enable_omnifetch_activation_multicast=(
            args.enable_omnifetch_activation_multicast
        ),
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids,
    )
    outputs = hex_execution(
        module,
        wrapped.__class__.__name__,
        inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    with torch.no_grad():
        reference = wrapped(*inputs)
    diff = (outputs[0].float() - reference.float()).abs().max().item()
    print(f"[Compare] max_abs_diff={diff:.4f} top1_match={outputs[0][0,-1].argmax().item() == reference[0,-1].argmax().item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(parser)
    parser.add_argument("--device-iterations", type=int, default=1)
    run(parser.parse_args())
