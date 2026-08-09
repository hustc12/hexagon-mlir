#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from transformers import SpeechT5Config,SpeechT5ForSpeechToText
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from hexkl_utils import *
class SpeechT5DebugWrapper(torch.nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x,ids):return self.m(input_values=x,decoder_input_ids=ids,use_cache=False).logits
class StaticPositions(torch.nn.Module):
    def __init__(self,pos):super().__init__();self.register_buffer("pos",pos)
    def forward(self,input_ids,past_key_values_length=0):
        del past_key_values_length
        return self.pos.to(device=input_ids.device).expand(input_ids.shape[0],-1,-1)
class StaticRelativePosition(torch.nn.Module):
    def __init__(self,pos):super().__init__();self.register_buffer("pos",pos)
    def forward(self,hidden_states):
        return self.pos.to(device=hidden_states.device,dtype=hidden_states.dtype)
def run(a):
    patch_dsp_heap_256mb();torch.manual_seed(55)
    c=SpeechT5Config(vocab_size=128,hidden_size=64,encoder_layers=1,decoder_layers=1,
      encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,
      conv_dim=(32,32),conv_kernel=(10,3),conv_stride=(5,2),num_conv_pos_embeddings=16,
      num_conv_pos_embedding_groups=4,apply_spec_augment=False,use_cache=False,
      pad_token_id=1,bos_token_id=0,eos_token_id=2,decoder_start_token_id=2,
      hidden_act="gelu_new",feat_extract_activation="gelu_new");c._attn_implementation="eager"
    model=SpeechT5ForSpeechToText(c).half().eval()
    conv=model.speecht5.encoder.prenet.pos_conv_embed.conv
    if hasattr(conv,"parametrizations"):torch.nn.utils.parametrize.remove_parametrizations(conv,"weight",leave_parametrized=True)
    # Torch-MLIR currently exports cumsum as the unsupported custom
    # `tm_tensor.scan`.  This is a fixed-shape Debug harness, so materialize the
    # exact sinusoidal positions once and remove both runtime scans without
    # changing the model's numerical result.
    enc_pos=model.speecht5.encoder.prenet.pos_sinusoidal_embed
    dec_pos=model.speecht5.decoder.prenet.embed_positions
    with torch.no_grad():
        enc_static=enc_pos(torch.zeros(1,50,dtype=torch.long)).detach()
        dec_ids=torch.arange(32).unsqueeze(0)%128
        dec_static=dec_pos(dec_ids).detach()
        enc_rel_module=model.speecht5.encoder.wrapped_encoder.embed_positions
        enc_rel=enc_rel_module(torch.zeros(1,50,64,dtype=torch.float16)).detach()
    model.speecht5.encoder.prenet.pos_sinusoidal_embed=StaticPositions(enc_static)
    model.speecht5.decoder.prenet.embed_positions=StaticPositions(dec_static)
    model.speecht5.encoder.wrapped_encoder.embed_positions=StaticRelativePosition(enc_rel)
    w=SpeechT5DebugWrapper(model).eval();ins=[torch.rand(1,512,dtype=torch.float16),torch.arange(32).unsqueeze(0)%128]
    m=compile_to_linalg(w,tuple(ins),decomp_pow=False);ir=str(m);p=None;print(f"[DebugCandidate] SpeechT5-ASR batch_matmul={ir.count('linalg.batch_matmul')}")
    if a.enable_hexkl:
      q,nb,nf=apply_hexkl_ir_rewrites(ir);p=q if nb or nf else None
    o=hex_execution(m,w.__class__.__name__,ins,hexagon_options_phase4(a.enable_hexkl,a.enable_omnifetch_vdae,not a.disable_layout_aware,a.omnifetch_lookahead,not a.disable_omnifetch_adaptive,a.enable_omnifetch_items_1_7,False),mlir_text=p)
    with torch.no_grad():r=w(*ins)
    print(f"[Compare] max_abs_diff={(o[0].float()-r.float()).abs().max().item():.4f} top1_match={o[0][0,-1].argmax().item()==r[0,-1].argmax().item()}")
if __name__=="__main__":
    x=argparse.ArgumentParser();add_phase4_args(x);run(x.parse_args())
