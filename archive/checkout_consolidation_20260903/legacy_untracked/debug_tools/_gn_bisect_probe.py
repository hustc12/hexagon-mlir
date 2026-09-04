#!/usr/bin/env python3
"""Staged GroupNorm bisection probe for the Wav2Vec2 conv-frontend crash.

conv-only PASSES, conv+mean PASSES. This narrows the remaining GroupNorm
decomposition steps (broadcast subtract / variance / normalize / affine) to
find the exact op whose async-tiled lowering faults on device.
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

EPS = 1e-5


class Stage(torch.nn.Module):
    """Wrap the first conv layer + a selectable slice of GroupNorm math."""

    def __init__(self, conv, stage, gn=None, act=None):
        super().__init__()
        self.c = conv
        self.stage = stage
        self.gn = gn
        self.act = act

    def forward(self, x):
        y = self.c(x[:, None])  # (1,512,L)
        if self.stage == "real_gn":
            return self.gn(y)
        if self.stage == "real_gn_gelu":
            return self.act(self.gn(y))
        y = y.float()
        mean = y.mean(dim=2, keepdim=True)  # (1,512,1)
        if self.stage == "sub":
            return (y - mean).to(torch.float16)  # broadcast subtract over L
        var = ((y - mean) ** 2).mean(dim=2, keepdim=True)  # (1,512,1)
        if self.stage == "var":
            return var.to(torch.float16)
        rstd = torch.rsqrt(var + EPS)
        if self.stage == "rstd":
            return rstd.to(torch.float16)
        norm = (y - mean) * rstd  # (1,512,L)
        if self.stage == "norm":
            return norm.to(torch.float16)
        raise ValueError(self.stage)


def run(args):
    patch_dsp_heap_256mb()
    torch.manual_seed(960)
    config = Wav2Vec2Config()
    config.apply_spec_augment = False
    config._attn_implementation = "eager"
    model = Wav2Vec2ForCTC(config).half().eval()
    conv_layer = model.wav2vec2.feature_extractor.conv_layers[0]
    conv = conv_layer.conv
    gn = getattr(conv_layer, "layer_norm", None)
    act = getattr(conv_layer, "activation", None)
    stage = Stage(conv, args.stage, gn=gn, act=act).eval()
    samples = torch.rand(1, 20560, dtype=torch.float16) * 2 - 1
    inputs = [samples]
    module = compile_to_linalg(stage, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(f"[Stage={args.stage}] has_f64={'f64' in ir}")
    options = hexagon_options_phase4(
        False, args.enable_alps_vdae,
        not args.disable_layout_aware, args.alps_lookahead,
        not args.disable_alps_adaptive, args.enable_alps_items_1_7,
        lower_constants_separate=False,
    )
    out = hex_execution(module, stage.__class__.__name__, inputs, options)
    with torch.no_grad():
        ref = stage(*inputs)
    finite = bool(torch.isfinite(out[0]).all())
    diff = (out[0].float() - ref.float()).abs().max().item()
    print(f"[Stage={args.stage}] shape={tuple(out[0].shape)} finite={finite} "
          f"max_abs_diff={diff:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(p)
    p.add_argument("--stage", required=True,
                   choices=["sub", "var", "rstd", "norm",
                            "real_gn", "real_gn_gelu"])
    run(p.parse_args())
