#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from transformers import Speech2TextConfig,Speech2TextForConditionalGeneration
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from hexkl_utils import *
class Speech2TextDebugWrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x,ids):return self.m(input_features=x,decoder_input_ids=ids,use_cache=False).logits
class StaticPositions(torch.nn.Module):
    def __init__(self,weights,length,padding_idx):
        super().__init__()
        ids=torch.arange(length,dtype=torch.long)+padding_idx+1
        self.register_buffer("positions",weights.index_select(0,ids).detach())
    def forward(self,input_ids,past_key_values_length=0):
        del past_key_values_length
        return self.positions.to(input_ids.device).unsqueeze(0).expand(input_ids.shape[0],-1,-1)
def run(a):
    patch_dsp_heap_256mb();torch.manual_seed(200)
    c=Speech2TextConfig(vocab_size=1024,d_model=64,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,
        input_feat_per_channel=80,conv_channels=32,conv_kernel_sizes=(3,3),max_source_positions=128,
        max_target_positions=64,pad_token_id=0,bos_token_id=1,eos_token_id=2,decoder_start_token_id=1,
        activation_function="gelu_new",use_cache=False);c._attn_implementation="eager"
    model=Speech2TextForConditionalGeneration(c).half().eval()
    model.model.encoder.embed_positions=StaticPositions(
        model.model.encoder.embed_positions.weights,16,c.pad_token_id)
    model.model.decoder.embed_positions=StaticPositions(
        model.model.decoder.embed_positions.weights,32,c.pad_token_id)
    w=Speech2TextDebugWrapper(model).eval()
    ins=[torch.rand(1,64,80,dtype=torch.float16),torch.arange(32).unsqueeze(0)%1024]
    m=compile_to_linalg(w,tuple(ins),decomp_pow=False);ir=str(m);p=None;print(f"[DebugCandidate] Speech2Text proxy [IR] batch_matmul={ir.count('linalg.batch_matmul')}")
    if a.enable_hexkl:
        q,nb,nf=apply_hexkl_ir_rewrites(ir);p=q if nb or nf else None;print(f"[HexKL] rewrites={nb+nf}")
    o=hex_execution(m,w.__class__.__name__,ins,hexagon_options_phase4(a.enable_hexkl,a.enable_alps_vdae,not a.disable_layout_aware,a.alps_lookahead,not a.disable_alps_adaptive,a.enable_alps_items_1_7,False),mlir_text=p)
    with torch.no_grad():r=w(*ins)
    print(f"[Compare] max_abs_diff={(o[0].float()-r.float()).abs().max().item():.4f} top1_match={o[0][0,-1].argmax().item()==r[0,-1].argmax().item()}")
if __name__=="__main__":
    x=argparse.ArgumentParser();add_phase4_args(x);run(x.parse_args())
