#!/usr/bin/env python3
"""Full-shape component LWP for Whisper-tiny; not a Debug model."""
from __future__ import annotations
from pathlib import Path
import sys
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmark_models"))
from hexkl_utils import (  # noqa: E402
    apply_hexkl_ir_rewrites, compile_to_linalg, hex_execution,
    hexagon_options_phase4, patch_full_model_dsp_heap,
)


def gelu_tanh(x, approximate="none"):
    del approximate
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x*x*x)))


class EncoderFrontend(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.conv1=encoder.conv1; self.conv2=encoder.conv2
        self.register_buffer("position", encoder.embed_positions.weight.detach(), persistent=False)
    def forward(self, features):
        x=gelu_tanh(self.conv1(features)); x=gelu_tanh(self.conv2(x)).permute(0,2,1)
        return x + self.position


class EncoderLayerStage(torch.nn.Module):
    def __init__(self, layer): super().__init__(); self.layer=layer
    def forward(self, hidden): return self.layer(hidden, None, None, False)[0]


class DecoderLayerStage(torch.nn.Module):
    def __init__(self, layer): super().__init__(); self.layer=layer
    def forward(self, hidden, encoder_hidden):
        return self.layer(hidden, encoder_hidden_states=encoder_hidden,
                          output_attentions=False, use_cache=False)[0]


class VocabularyHead(torch.nn.Module):
    def __init__(self, head): super().__init__(); self.head=head
    def forward(self, hidden): return self.head(hidden)


def options():
    return hexagon_options_phase4(True,False,False,2,False,False,
        lower_constants_separate=True, backend_profile="hvx-vector",
        enable_lwp=True, lwp_loop_depth=1, disable_lwp_loop=False,
        enable_omnifetch_kv_cache_prefetch=True)


def run_stage(label, stage, inputs, device_inputs=None):
    stage=stage.half().eval(); inputs=tuple(inputs)
    with torch.no_grad(): ref=stage(*inputs)
    mod=compile_to_linalg(stage,inputs,decomp_pow=False); ir=str(mod)
    candidate,nb,nf=apply_hexkl_ir_rewrites(ir); patched=candidate if nb or nf else None
    print(f"[WhisperHotspotIR] stage={label} batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')} batch_rewrite={nb} f16_rewrite={nf}")
    out=hex_execution(mod,stage.__class__.__name__,device_inputs or list(inputs),options(),
                      mlir_text=patched,iterations=1)[0]
    finite=bool(torch.isfinite(out).all()); diff=float((out.float()-ref.float()).abs().max())
    tol=max(0.02*max(float(ref.float().abs().max()),1.0),0.05); ok=finite and diff<=tol
    print(f"[WhisperHotspotCompare] stage={label} finite={finite} max_abs={diff:.9g} tolerance={tol:.9g} correct={ok}")
    if not ok: raise AssertionError(label)
    return ref.detach()


def main():
    patch_full_model_dsp_heap(); torch.nn.functional.gelu=gelu_tanh; torch.manual_seed(390)
    cfg=WhisperConfig(); cfg.use_cache=False; cfg.activation_function="gelu_new"; cfg._attn_implementation="eager"
    model=WhisperForConditionalGeneration(cfg).half().eval()
    print(f"[WhisperHotspotFullShape] mel=80x3000 encoder_layers={cfg.encoder_layers} decoder_layers={cfg.decoder_layers} d_model={cfg.d_model} target=32 params={sum(p.numel() for p in model.parameters())}")
    features=torch.rand(1,cfg.num_mel_bins,3000,dtype=torch.float16)
    frontend=EncoderFrontend(model.model.encoder)
    enc=run_stage("encoder_frontend",frontend,[features],
                  [frontend.position,features])
    enc=run_stage("encoder_layer_0",EncoderLayerStage(model.model.encoder.layers[0]),[enc])
    dec=torch.rand(1,32,cfg.d_model,dtype=torch.float16)
    dec=run_stage("decoder_layer_0",DecoderLayerStage(model.model.decoder.layers[0]),[dec,enc])
    run_stage("vocabulary_head",VocabularyHead(model.proj_out),[dec])


if __name__ == "__main__": main()
