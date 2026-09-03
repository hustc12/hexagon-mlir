#!/usr/bin/env python3
"""Progressive full-model subgraph bisection for the Wav2Vec2 crash.

conv[0]+GroupNorm PASSES in isolation, so the crash is deeper. This probe
runs real sub-modules of the full-size model at the real 20560-sample input:
  feat        -> feature_extractor (all 7 conv layers)          (1,512,64)
  proj        -> + feature_projection                           (1,64,768)
  enc1        -> + encoder with num_hidden_layers forced to 1
  full        -> whole model logits
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_dsp_heap_256mb,
)
from full_audio_encoder import _reformulate_pos_conv  # noqa: E402


class Sub(torch.nn.Module):
    def __init__(self, model, stage):
        super().__init__()
        self.m = model
        self.stage = stage

    def forward(self, x):
        w2v = self.m.wav2vec2
        feats = w2v.feature_extractor(x)  # (1,512,64)
        if self.stage == "feat":
            return feats
        feats = feats.transpose(1, 2)  # (1,64,512)
        hidden, _ = w2v.feature_projection(feats)  # (1,64,768)
        if self.stage == "proj":
            return hidden
        enc = w2v.encoder
        if self.stage == "pos_only":
            return enc.pos_conv_embed(hidden)  # grouped conv over seq
        pos = enc.pos_conv_embed(hidden)
        hidden = enc.layer_norm(hidden + pos)
        if self.stage == "pos":
            return hidden
        if self.stage == "layer0":
            return enc.layers[0](hidden, attention_mask=None)[0]
        if self.stage == "enc":
            return w2v.encoder(
                w2v.feature_projection(w2v.feature_extractor(x).transpose(1, 2))[0]
            )[0]
        raise ValueError(self.stage)


def run(args):
    patch_dsp_heap_256mb()
    torch.manual_seed(960)
    config = Wav2Vec2Config()
    config.apply_spec_augment = False
    config.hidden_act = "gelu_new"
    config.feat_extract_activation = "gelu_new"
    config._attn_implementation = "eager"
    if args.stage in ("enc", "layer0", "pos", "pos_only"):
        config.num_hidden_layers = 1
    model = Wav2Vec2ForCTC(config).half().eval()
    conv = model.wav2vec2.encoder.pos_conv_embed.conv
    if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
        torch.nn.utils.parametrize.remove_parametrizations(
            conv, "weight", leave_parametrized=True
        )
    _reformulate_pos_conv(model.wav2vec2.encoder.pos_conv_embed)
    sub = Sub(model, args.stage).eval()
    samples = torch.rand(1, 20560, dtype=torch.float16) * 2 - 1
    inputs = [samples]
    module = compile_to_linalg(sub, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(f"[Stage={args.stage}] has_f64={'f64' in ir} "
          f"batch_matmul={ir.count('linalg.batch_matmul')} "
          f"matmul={ir.count('linalg.matmul')}")
    options = hexagon_options_phase4(
        False, args.enable_omnifetch_vdae,
        not args.disable_layout_aware, args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive, args.enable_omnifetch_items_1_7,
        lower_constants_separate=True,
        backend_profile=args.backend_profile,
        enable_out_params=args.enable_out_params,
    )
    out = hex_execution(module, sub.__class__.__name__, inputs, options)
    with torch.no_grad():
        ref = sub(*inputs)
    finite = bool(torch.isfinite(out[0]).all())
    diff = (out[0].float() - ref.float()).abs().max().item()
    print(f"[Stage={args.stage}] shape={tuple(out[0].shape)} finite={finite} "
          f"max_abs_diff={diff:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(p)
    p.add_argument("--stage", required=True,
                   choices=["feat", "proj", "pos_only", "pos", "layer0", "enc"])
    run(p.parse_args())
