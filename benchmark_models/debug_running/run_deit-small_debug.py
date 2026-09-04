#!/usr/bin/env python3
"""Reduced offline DeiT-Small global-attention vision candidate."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from transformers import DeiTConfig, DeiTForImageClassification
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (add_phase4_args, apply_hexkl_ir_rewrites, compile_to_linalg,
                         hex_execution, hexagon_options_phase4, patch_dsp_heap_256mb)


class DeiTDebugWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, pixels):
        return self.model(pixel_values=pixels).logits


def run(args):
    patch_dsp_heap_256mb(); torch.manual_seed(221)
    config=DeiTConfig(image_size=64, patch_size=8, num_channels=3,
        hidden_size=96, num_hidden_layers=2, num_attention_heads=3,
        intermediate_size=192, hidden_act="gelu_new", num_labels=1000)
    print("[DebugCandidate] DeiT-Small proxy: image=64 patch=8 tokens=66 layers=2 hidden=96 heads=3")
    wrapped=DeiTDebugWrapper(DeiTForImageClassification(config).half().eval()).eval()
    inputs=[torch.rand(1,3,64,64,dtype=torch.float16)]
    module=compile_to_linalg(wrapped,tuple(inputs),decomp_pow=False); ir=str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    patched=None
    if args.enable_hexkl:
        candidate,nb,nf=apply_hexkl_ir_rewrites(ir); print(f"[HexKL] batch_matmul→matmul={nb}, f16-input rewrite={nf}")
        patched=candidate if nb or nf else None
    options=hexagon_options_phase4(args.enable_hexkl,args.enable_alps_vdae,
        not args.disable_layout_aware,args.alps_lookahead,
        not args.disable_alps_adaptive,args.enable_alps_items_1_7,
        lower_constants_separate=False,backend_profile=args.backend_profile,
        prefetch_baseline=args.prefetch_baseline,
        prefetch_baseline_distance=args.prefetch_baseline_distance,
        apt_get_hx_manual_candidate_ids=args.apt_get_hx_manual_candidate_ids)
    out=hex_execution(module,wrapped.__class__.__name__,inputs,options,
        mlir_text=patched,iterations=args.device_iterations)
    with torch.no_grad(): ref=wrapped(*inputs)
    diff=(out[0].float()-ref.float()).abs().max().item()
    print(f"[Compare] max_abs_diff={diff:.4f} top1_match={out[0].argmax().item()==ref.argmax().item()}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); add_phase4_args(p)
    p.add_argument("--device-iterations",type=int,default=1); run(p.parse_args())
