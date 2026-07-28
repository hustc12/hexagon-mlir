#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from transformers import DetrConfig,DetrForObjectDetection,ResNetConfig
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from hexkl_utils import *
class DetrDebugWrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x):return self.m(pixel_values=x).logits
def run(a):
    patch_dsp_heap_256mb();torch.manual_seed(50)
    b=ResNetConfig(num_channels=3,embedding_size=16,hidden_sizes=[16,32,64,128],depths=[1,1,1,1],layer_type="basic")
    c=DetrConfig(backbone_config=b,use_pretrained_backbone=False,use_timm_backbone=False,d_model=64,
        encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
        encoder_ffn_dim=128,decoder_ffn_dim=128,num_queries=32,num_labels=91);c._attn_implementation="eager"
    w=DetrDebugWrapper(DetrForObjectDetection(c).half().eval()).eval();ins=[torch.rand(1,3,64,64,dtype=torch.float16)]
    m=compile_to_linalg(w,tuple(ins),decomp_pow=False);ir=str(m);p=None;print(f"[DebugCandidate] DETR proxy [IR] batch_matmul={ir.count('linalg.batch_matmul')}")
    if a.enable_hexkl:
        q,nb,nf=apply_hexkl_ir_rewrites(ir);p=q if nb or nf else None;print(f"[HexKL] rewrites={nb+nf}")
    o=hex_execution(m,w.__class__.__name__,ins,hexagon_options_phase4(a.enable_hexkl,a.enable_omnifetch_vdae,not a.disable_layout_aware,a.omnifetch_lookahead,not a.disable_omnifetch_adaptive,a.enable_omnifetch_items_1_7,False),mlir_text=p)
    with torch.no_grad():r=w(*ins)
    print(f"[Compare] max_abs_diff={(o[0].float()-r.float()).abs().max().item():.4f} top1_match={o[0][0,-1].argmax().item()==r[0,-1].argmax().item()}")
if __name__=="__main__":
    x=argparse.ArgumentParser();add_phase4_args(x);run(x.parse_args())
