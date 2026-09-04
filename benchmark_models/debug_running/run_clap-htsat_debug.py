#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from transformers import ClapAudioConfig,ClapAudioModelWithProjection
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from hexkl_utils import *
class ClapDebugWrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x,longer):return self.m(input_features=x,is_longer=longer).audio_embeds
def run(a):
    patch_dsp_heap_256mb();torch.manual_seed(527)
    c=ClapAudioConfig(spec_size=32,num_mel_bins=32,patch_size=4,patch_stride=(4,4),
      patch_embeds_hidden_size=32,depths=[1,1],num_attention_heads=[2,4],num_hidden_layers=2,
      hidden_size=64,projection_dim=64,window_size=4,num_classes=527,hidden_act="gelu_new")
    w=ClapDebugWrapper(ClapAudioModelWithProjection(c).half().eval()).eval()
    ins=[torch.rand(1,1,32,32,dtype=torch.float16),torch.tensor([False])]
    m=compile_to_linalg(w,tuple(ins),decomp_pow=False);ir=str(m);p=None;print(f"[DebugCandidate] CLAP-HTSAT batch_matmul={ir.count('linalg.batch_matmul')}")
    if a.enable_hexkl:
      q,nb,nf=apply_hexkl_ir_rewrites(ir);p=q if nb or nf else None
    o=hex_execution(m,w.__class__.__name__,ins,hexagon_options_phase4(a.enable_hexkl,a.enable_alps_vdae,not a.disable_layout_aware,a.alps_lookahead,not a.disable_alps_adaptive,a.enable_alps_items_1_7,False),mlir_text=p)
    with torch.no_grad():r=w(*ins)
    print(f"[Compare] max_abs_diff={(o[0].float()-r.float()).abs().max().item():.4f}")
if __name__=="__main__":
    x=argparse.ArgumentParser();add_phase4_args(x);run(x.parse_args())
