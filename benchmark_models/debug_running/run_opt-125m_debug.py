#!/usr/bin/env python3
"""Reduced offline OPT-125M MHA decoder candidate."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from transformers import OPTConfig, OPTForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (add_phase4_args, apply_hexkl_ir_rewrites, compile_to_linalg,
                         hex_execution, hexagon_options_phase4, patch_dsp_heap_256mb)


class OPTDebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, input_ids, attention_mask, position_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=position_ids, use_cache=False).logits


def run(args):
    patch_dsp_heap_256mb(); torch.manual_seed(125)
    n = args.seq_len or 32
    config = OPTConfig(vocab_size=1024, hidden_size=128, ffn_dim=256,
                       num_hidden_layers=2, num_attention_heads=4,
                       max_position_embeddings=512, word_embed_proj_dim=128,
                       do_layer_norm_before=True, use_cache=False)
    config._attn_implementation = "eager"
    print(f"[DebugCandidate] OPT-125M proxy: layers=2 hidden=128 ffn=256 heads=4 seq={n}")
    wrapped = OPTDebugWrapper(OPTForCausalLM(config).half().eval()).eval()
    inputs = [torch.arange(n).unsqueeze(0) % config.vocab_size,
              torch.ones(1, n, dtype=torch.float16),
              torch.arange(n).unsqueeze(0)]
    module = compile_to_linalg(wrapped, tuple(inputs)); ir = str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    patched = None
    if args.enable_hexkl:
        candidate, nb, nf = apply_hexkl_ir_rewrites(ir)
        print(f"[HexKL] batch_matmul→matmul={nb}, f16-input rewrite={nf}")
        patched = candidate if nb or nf else None
    options = hexagon_options_phase4(args.enable_hexkl, args.enable_alps_vdae,
        not args.disable_layout_aware, args.alps_lookahead,
        not args.disable_alps_adaptive, args.enable_alps_items_1_7,
        lower_constants_separate=False)
    out = hex_execution(module, wrapped.__class__.__name__, inputs, options, mlir_text=patched)
    with torch.no_grad(): ref = wrapped(*inputs)
    diff = (out[0].float()-ref.float()).abs().max().item()
    print(f"[Compare] max_abs_diff={diff:.4f} top1_match={out[0][0,-1].argmax().item()==ref[0,-1].argmax().item()}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); add_phase4_args(p); run(p.parse_args())
