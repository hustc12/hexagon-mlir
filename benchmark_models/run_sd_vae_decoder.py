# ===- test_sd_vae_decoder.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# Stable Diffusion — VAE Decoder benchmark for Hexagon NPU.
#
# Model: CompVis/stable-diffusion-v1-4  →  vae subfolder
# Architecture: AutoencoderKL (decode path only)
# Input:  latents  (1, 4, 64, 64)  float32
# Output: sample   (1, 3, 512, 512) float32
#
# The VAE Decoder is the most compute-intensive part of the VAE used in SD.
# We wrap vae.decode() so torch-mlir sees a plain tensor→tensor function.
#
# Usage:
#   python benchmark_models/test_sd_vae_decoder.py [--lwp]
#
# ===------------------------------------------------------------------------===

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

from sd_utils import (
    SD_MODEL_ID,
    compile_to_linalg,
    hex_execution,
    x86_execution,
    compare,
    add_phase4_cli,
    options_from_args,
    process_lwp,
)


class GroupNormFP16(torch.nn.Module):
    """Drop-in replacement for torch.nn.GroupNorm that stays in f16.

    torch-mlir's built-in GroupNorm lowering hard-codes a f16→f64 promotion
    for the variance reduction, regardless of the model dtype.  On Hexagon DSP
    there is no f64 hardware unit — every f64 op is software-emulated and
    ~100× slower than f16.  At 512×512 resolution a single GroupNorm reduction
    operates on a 1×32×1048576 f64 tensor (256 MB); with ~12 such reductions
    the total f64 data movement exceeds 3 GB, making execution infeasibly slow.

    This replacement implements the identical mathematical operation using only
    f16 arithmetic so torch-mlir emits pure f16 linalg.generic ops instead of
    the f64-promoting GroupNorm lowering.  The model structure (number of
    groups, affine parameters, eps) is preserved exactly.
    """

    def __init__(self, orig: torch.nn.GroupNorm):
        super().__init__()
        self.num_groups = orig.num_groups
        self.eps = orig.eps
        self.weight = orig.weight  # shared reference — no copy
        self.bias = orig.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, *spatial)  — all f16
        N, C = x.shape[0], x.shape[1]
        G = self.num_groups
        # Reshape to (N, G, -1) so the reduction is over the last dim only.
        # This avoids the large-tensor f64 path that torch.nn.GroupNorm triggers.
        x_grouped = x.reshape(N, G, -1)                          # (N, G, L)
        mean = x_grouped.mean(dim=-1, keepdim=True)              # (N, G, 1)
        var  = ((x_grouped - mean) ** 2).mean(dim=-1, keepdim=True)  # (N, G, 1)
        x_norm = (x_grouped - mean) / (var + self.eps).sqrt()   # (N, G, L)
        x_norm = x_norm.reshape(x.shape)                         # (N, C, *spatial)
        if self.weight is not None:
            # weight/bias are (C,) — broadcast over N and spatial dims
            shape = (1, C) + (1,) * (x.dim() - 2)
            x_norm = x_norm * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x_norm


class Conv2dAsMatmul(nn.Module):
    """Replace nn.Conv2d with unfold + matmul so torch-mlir emits linalg.matmul.

    linalg.conv_2d_nchw_fchw is lowered to scalar loops by the Hexagon backend
    (no HVX vectorization).  linalg.matmul is vectorized via HexKL or the HVX
    tiling path.  This wrapper implements the identical computation:
        output = unfold(input) @ weight.reshape(C_out, -1).T + bias
    which is mathematically equivalent to a strided conv2d.

    Only supports stride=1 or stride=2, dilation=1 (covers all VAE conv layers).
    """

    def __init__(self, orig: nn.Conv2d):
        super().__init__()
        self.in_channels  = orig.in_channels
        self.out_channels = orig.out_channels
        self.kernel_size  = orig.kernel_size   # (kH, kW)
        self.stride       = orig.stride        # (sH, sW)
        self.padding      = orig.padding       # (pH, pW)
        self.dilation     = orig.dilation
        # weight: (C_out, C_in, kH, kW) → flatten to (C_out, C_in*kH*kW)
        self.weight = orig.weight              # keep as-is; reshape in forward
        self.bias   = orig.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        # unfold: (N, C_in*kH*kW, L)  where L = H_out * W_out
        cols = F.unfold(x, kernel_size=(kH, kW), dilation=self.dilation,
                        padding=(pH, pW), stride=(sH, sW))
        # cols: (N, C_in*kH*kW, L)
        L = cols.shape[2]
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        # weight_mat: (C_out, C_in*kH*kW)
        weight_mat = self.weight.reshape(self.out_channels, -1)

        # matmul: (N, C_out, L)
        # cols.transpose(1,2): (N, L, C_in*kH*kW)
        # weight_mat.T:        (C_in*kH*kW, C_out)
        out = cols.transpose(1, 2).reshape(N * L, -1) @ weight_mat.t()
        # out: (N*L, C_out) → (N, C_out, H_out, W_out)
        out = out.reshape(N, L, self.out_channels).transpose(1, 2)
        out = out.reshape(N, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out


def replace_conv2d_with_matmul(model: nn.Module) -> nn.Module:
    """Recursively replace all nn.Conv2d with Conv2dAsMatmul.

    Conv2dAsMatmul uses unfold+matmul which torch-mlir lowers to linalg.matmul.
    The Hexagon backend vectorizes linalg.matmul via HexKL/HVX but has no HVX
    path for linalg.conv_2d_nchw_fchw (scalar loops only).
    Requires sufficient DSP heap: largest intermediate is C_in*kH*kW * H*W * 2 bytes
    (e.g. 512ch 3x3 at 64x64 = 36 MB, at 128x128 = 150 MB).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            setattr(model, name, Conv2dAsMatmul(module))
        else:
            replace_conv2d_with_matmul(module)
    return model


class VAEDecodeWrapper(torch.nn.Module):
    """Wrap AutoencoderKL.decode() to return the sample tensor directly.

    torch_mlir's fx.export_and_import cannot handle dataclass outputs
    (DecoderOutput).  This wrapper returns a plain tensor so the C++
    MemRefDescriptor interface gets a single concrete output.
    """

    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents).sample


def test_vae_decoder(
    enablelwp: bool = False,
    disable_mid_attn: bool = False,
    enable_hexkl: bool = False,
    enable_alps_vdae: bool = False,
    enable_alps_layout_aware: bool = True,
    alps_lookahead: int = 2,
    enable_alps_adaptive: bool = True,
    enable_alps_items_1_7: bool = False,
):
    print("\n=== Stable Diffusion — VAE Decoder ===")

    config = AutoencoderKL.load_config(SD_MODEL_ID, subfolder="vae")
    config = customize_vae_config(config)
    print(f"[Config] VAE sample_size={config.get('sample_size', '?')} HexKL={enable_hexkl}")
    vae = load_vae(config)

    vae = vae.half()
    for name, module in list(vae.named_modules()):
        if isinstance(module, torch.nn.GroupNorm):
            parts = name.rsplit(".", 1)
            parent = vae
            if len(parts) == 2:
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], GroupNormFP16(module))

    if disable_mid_attn:
        print("  [diag] mid-block self-attention DISABLED")
        vae.decoder.mid_block.attentions = torch.nn.ModuleList()

    replace_conv2d_with_matmul(vae)
    print("  Replaced Conv2d layers with unfold+matmul")

    model = VAEDecodeWrapper(vae)
    model.eval()

    latents = customize_vae_latents(config)
    print(f"Input latents shape: {latents.shape}  dtype: {latents.dtype}")

    print("\nCompiling VAE Decoder to linalg …")
    module = compile_to_linalg(model, latents)

    class _Args:
        pass

    args = _Args()
    args.lwp = enablelwp
    args.enable_hexkl = enable_hexkl
    args.enable_alps_vdae = enable_alps_vdae
    args.disable_layout_aware = not enable_alps_layout_aware
    args.alps_lookahead = alps_lookahead
    args.disable_alps_adaptive = not enable_alps_adaptive
    args.enable_alps_items_1_7 = enable_alps_items_1_7
    options = options_from_args(args)

    print("Running VAE Decoder on Hexagon NPU …")
    hex_out = hex_execution(
        module, "VAEDecodeWrapper", [latents], options, heap_size_mb=256
    )

    print("Running reference on x86 …")
    x86_out = x86_execution(model, latents)
    compare(hex_out, x86_out, atol=0.05, fail_on_mismatch=True)
    if enablelwp:
        process_lwp()
    print("\nVAE Decoder test PASSED.")


def customize_vae_config(config):
    return config


def load_vae(config):
    return AutoencoderKL.from_config(config)


def customize_vae_latents(config):
    # Published latent grid 64×64 (→ 512×512 image). Debug may shrink.
    return torch.rand(1, 4, 64, 64, dtype=torch.float16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD VAE Decoder Hexagon benchmark")
    add_phase4_cli(parser)
    parser.add_argument(
        "--no-mid-attn",
        action="store_true",
        help="[diagnostic] Disable mid-block self-attention",
    )
    args = parser.parse_args()
    test_vae_decoder(
        enablelwp=args.lwp,
        disable_mid_attn=args.no_mid_attn,
        enable_hexkl=args.enable_hexkl,
        enable_alps_vdae=args.enable_alps_vdae,
        enable_alps_layout_aware=not args.disable_layout_aware,
        alps_lookahead=args.alps_lookahead,
        enable_alps_adaptive=not args.disable_alps_adaptive,
        enable_alps_items_1_7=args.enable_alps_items_1_7,
    )
