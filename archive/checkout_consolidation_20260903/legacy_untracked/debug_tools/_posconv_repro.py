#!/usr/bin/env python3
"""Minimal repro + fix search for the Wav2Vec2 pos_conv_embed crash.

pos_conv_embed is a grouped Conv1d(768,768, kernel=128, groups=16, pad=64)
applied over a 64-frame sequence (kernel > seq). Isolated (feat+proj+
pos_conv_embed) it faults on device with exit 13. This drives a synthetic
(1,768,64) input straight into variants of that conv so we can iterate fast.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hexkl_utils import (  # noqa: E402
    add_phase4_args,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
    patch_dsp_heap_256mb,
)


class PosConv(torch.nn.Module):
    """Mirror Wav2Vec2PositionalConvEmbedding without weight-norm."""

    def __init__(self, groups, kernel, remove_pad, mode="conv"):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            768, 768, kernel_size=kernel, padding=kernel // 2, groups=groups
        )
        self.groups = groups
        self.kernel = kernel
        self.remove = remove_pad  # SamePadLayer removes 1 if kernel even
        self.mode = mode

    def _split(self, y):  # per-group dense conv1d
        cin = 768 // self.groups
        w, b, pad = self.conv.weight, self.conv.bias, self.kernel // 2
        parts = []
        for g in range(self.groups):
            xs = y[:, g * cin:(g + 1) * cin, :]
            ws = w[g * cin:(g + 1) * cin, :, :]
            bs = b[g * cin:(g + 1) * cin]
            parts.append(torch.nn.functional.conv1d(xs, ws, bs, padding=pad))
        return torch.cat(parts, dim=1)

    def _matmul(self, y):  # im2col + matmul, avoids linalg.conv
        cin = 768 // self.groups
        k, pad = self.kernel, self.kernel // 2
        w, b = self.conv.weight, self.conv.bias  # w:(768,cin,k)
        yp = torch.nn.functional.pad(y, (pad, pad))  # (1,768,L+2p)
        parts = []
        for g in range(self.groups):
            xs = yp[:, g * cin:(g + 1) * cin, :]  # (1,cin,L+2p)
            cols = xs.unfold(2, k, 1)  # (1,cin,out_len,k)
            out_len = cols.shape[2]
            cols = cols.permute(0, 2, 1, 3).reshape(1, out_len, cin * k)
            ws = w[g * cin:(g + 1) * cin].reshape(cin, cin * k)  # (cout=cin,cin*k)
            og = torch.matmul(cols, ws.t()) + b[g * cin:(g + 1) * cin]
            parts.append(og.transpose(1, 2))  # (1,cout,out_len)
        return torch.cat(parts, dim=1)

    def _prepad(self, x):  # x:(1,seq,hidden); pad seq on non-innermost axis
        pad = self.kernel // 2
        xp = torch.nn.functional.pad(x, (0, 0, pad, pad))  # (1,seq+2p,768)
        y = xp.transpose(1, 2)  # (1,768,seq+2p)
        return torch.nn.functional.conv1d(
            y, self.conv.weight, self.conv.bias, padding=0, groups=self.groups
        )  # innermost shrinks, no implicit pad

    def forward(self, x):  # x: (1,64,768)
        if self.mode == "prepad":
            y = self._prepad(x)  # already (1,768,out)
        else:
            y = x.transpose(1, 2)  # (1,768,64)
            if self.mode == "conv" or self.groups == 1:
                y = self.conv(y)
            elif self.mode == "split":
                y = self._split(y)
            else:
                y = self._matmul(y)
        if self.remove:
            y = y[:, :, :-1]
        return y.transpose(1, 2)  # (1,64,768)


def run(args):
    patch_dsp_heap_256mb()
    torch.manual_seed(0)
    model = PosConv(args.groups, args.kernel, args.kernel % 2 == 0,
                    mode=args.mode).half().eval()
    x = torch.rand(1, 64, 768, dtype=torch.float16) * 2 - 1
    inputs = [x]
    module = compile_to_linalg(model, tuple(inputs), decomp_pow=False)
    ir = str(module)
    print(f"[groups={args.groups} kernel={args.kernel}] has_f64={'f64' in ir} "
          f"conv_ops={ir.count('linalg.conv') + ir.count('linalg.depthwise')}")
    options = hexagon_options_phase4(
        False, args.enable_omnifetch_vdae,
        not args.disable_layout_aware, args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive, args.enable_omnifetch_items_1_7,
        lower_constants_separate=False,
    )
    out = hex_execution(module, model.__class__.__name__, inputs, options)
    with torch.no_grad():
        ref = model(*inputs)
    finite = bool(torch.isfinite(out[0]).all())
    diff = (out[0].float() - ref.float()).abs().max().item()
    print(f"[groups={args.groups} kernel={args.kernel}] "
          f"shape={tuple(out[0].shape)} finite={finite} max_abs_diff={diff:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_phase4_args(p)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--kernel", type=int, default=128)
    p.add_argument("--mode", choices=["conv", "split", "matmul", "prepad"],
                   default="conv")
    run(p.parse_args())
