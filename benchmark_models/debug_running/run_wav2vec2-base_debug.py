#!/usr/bin/env python3
"""Reduced offline Wav2Vec2-base long-sequence encoder candidate."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (add_phase4_args, apply_hexkl_ir_rewrites, compile_to_linalg,
                         hex_execution, hexagon_options_phase4, patch_dsp_heap_256mb)


class Wav2Vec2DebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, samples):
        return self.model(input_values=samples).logits


def run(args):
    patch_dsp_heap_256mb(); torch.manual_seed(960)
    config=Wav2Vec2Config(vocab_size=32, hidden_size=64,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=128,
        conv_dim=(32,32), conv_kernel=(10,3), conv_stride=(5,2),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        hidden_act="gelu_new", feat_extract_activation="gelu_new",
        apply_spec_augment=False)
    config._attn_implementation = "eager"
    print("[DebugCandidate] Wav2Vec2-base proxy: samples=512 conv=2 layers=2 hidden=64 heads=2")
    model=Wav2Vec2ForCTC(config).half().eval()
    torch.nn.utils.parametrize.remove_parametrizations(
        model.wav2vec2.encoder.pos_conv_embed.conv,
        "weight",
        leave_parametrized=True,
    )
    wrapped=Wav2Vec2DebugWrapper(model).eval()
    inputs=[torch.rand(1,512,dtype=torch.float16)*2-1]
    module=compile_to_linalg(wrapped,tuple(inputs),decomp_pow=False); ir=str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    patched=None
    if args.enable_hexkl:
        candidate,nb,nf=apply_hexkl_ir_rewrites(ir); print(f"[HexKL] batch_matmul→matmul={nb}, f16-input rewrite={nf}")
        patched=candidate if nb or nf else None
    options=hexagon_options_phase4(args.enable_hexkl,args.enable_omnifetch_vdae,
        not args.disable_layout_aware,args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,args.enable_omnifetch_items_1_7,
        lower_constants_separate=False)
    out=hex_execution(module,wrapped.__class__.__name__,inputs,options,mlir_text=patched)
    with torch.no_grad(): ref=wrapped(*inputs)
    diff=(out[0].float()-ref.float()).abs().max().item()
    print(f"[Compare] max_abs_diff={diff:.4f} top1_match={out[0][0,-1].argmax().item()==ref[0,-1].argmax().item()}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); add_phase4_args(p); run(p.parse_args())
