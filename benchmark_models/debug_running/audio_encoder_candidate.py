"""Shared offline speech-encoder Debug harness."""
from __future__ import annotations
import argparse, sys, types
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (add_phase4_args, apply_hexkl_ir_rewrites, compile_to_linalg,
                         hex_execution, hexagon_options_phase4, patch_dsp_heap_256mb)

class AudioEncoderWrapper(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self, samples): return self.model(input_values=samples).logits

def run_candidate(args, name, config, model_cls, root_name, seed):
    patch_dsp_heap_256mb(); torch.manual_seed(seed); config._attn_implementation="eager"
    model=model_cls(config).half().eval()
    root=getattr(model,root_name); pos=root.encoder.pos_conv_embed
    conv=getattr(pos,"conv",None)
    if conv is not None and hasattr(conv,"parametrizations") and "weight" in conv.parametrizations:
        torch.nn.utils.parametrize.remove_parametrizations(conv,"weight",leave_parametrized=True)
    if root_name == "wavlm":
        for layer in root.encoder.layers:
            attention=layer.attention
            bias=attention.compute_bias(50,50).detach().to(torch.float16)
            def constant_bias(self,query_length,key_length,_bias=bias):
                del self,query_length,key_length
                return _bias
            attention.compute_bias=types.MethodType(constant_bias,attention)
    wrapped=AudioEncoderWrapper(model).eval(); inputs=[torch.rand(1,512,dtype=torch.float16)*2-1]
    print(f"[DebugCandidate] {name}: samples=512 layers={config.num_hidden_layers} hidden={config.hidden_size}")
    module=compile_to_linalg(wrapped,tuple(inputs),decomp_pow=False); ir=str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    patched=None
    if args.enable_hexkl:
        candidate,nb,nf=apply_hexkl_ir_rewrites(ir); print(f"[HexKL] batch_matmul→matmul={nb}, f16-input rewrite={nf}")
        patched=candidate if nb or nf else None
    opts=hexagon_options_phase4(args.enable_hexkl,args.enable_omnifetch_vdae,
        not args.disable_layout_aware,args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,args.enable_omnifetch_items_1_7,
        lower_constants_separate=False, backend_profile=args.backend_profile,
        enable_omnifetch_kv_cache_prefetch=args.enable_omnifetch_kv_cache_prefetch,
        enable_omnifetch_kv_vtcm=args.enable_omnifetch_kv_vtcm)
    out=hex_execution(module,wrapped.__class__.__name__,inputs,opts,mlir_text=patched)
    with torch.no_grad(): ref=wrapped(*inputs)
    diff=(out[0].float()-ref.float()).abs().max().item()
    print(f"[Compare] max_abs_diff={diff:.4f} top1_match={out[0][0,-1].argmax().item()==ref[0,-1].argmax().item()}")

def parser():
    p=argparse.ArgumentParser(); add_phase4_args(p); return p
