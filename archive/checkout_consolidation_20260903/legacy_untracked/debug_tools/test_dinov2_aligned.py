#!/usr/bin/env python3
"""DINOv2-like debug model with M=64 (aligned) for clean HMX testing."""
import argparse
import sys
from pathlib import Path
import torch
from transformers import Dinov2Config, Dinov2ForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (
    add_phase4_args,
    apply_hexkl_ir_rewrites,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
)

class Dinov2AlignedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixels):
        # Get original output
        out = self.model(pixel_values=pixels).logits
        return out

def main():
    parser = argparse.ArgumentParser()
    add_phase4_args(parser)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--device-iterations", type=int, default=3)
    args = parser.parse_args()
    
    torch.manual_seed(142)
    
    # Key trick: Use patch_size=4 and image_size=28
    # This gives (28/4)^2 = 7^2 = 49 patches + 1 CLS = 50 tokens
    # Round to M=64 for alignment
    # Actually, let's use a different approach: use image=56, patch=8 -> 7×7=49+1=50
    # Then manually pad in the model to M=64
    
    # Simpler: just use image=248, patch=8 -> 31×31=961+1=962 -> round to 992 (31×32)
    # Or use batch dimension trick: batch_size=2, each with M=32
    
    # Easiest: use smaller hidden_size and adjust config
    config = Dinov2Config(
        image_size=56,  # 56/8 = 7, 7×7 = 49 patches
        patch_size=8,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=args.num_layers,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_act="gelu_new",
        num_labels=10,
        use_mask_token=False,
    )
    
    model = Dinov2AlignedWrapper(
        Dinov2ForImageClassification(config).half().eval()
    ).eval()
    
    pixels = torch.randn(1, 3, 56, 56, dtype=torch.float16)
    
    # Actual M = 49 + 1 = 50, will be padded to 64
    print(f"[Model] DINOv2-aligned: image=56, patch=8, M=50->64, layers={args.num_layers}")
    
    # Compile
    module = compile_to_linalg(model, (pixels,))
    ir = str(module)
    print(f"[IR] batch_matmul={ir.count('linalg.batch_matmul')} matmul={ir.count('linalg.matmul')}")
    
    # Reference
    with torch.no_grad():
        reference = model(pixels)
    
    # Apply HexKL rewrites if enabled
    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(
            ir, enable_m_pad=args.enable_omnifetch_m_pad_hmx
        )
        patched = candidate if n_batch or n_f16 else None
        print(f"[HexKL] batch_matmul→matmul={n_batch}, f16-input rewrite={n_f16}")
    
    # Hexagon options
    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_omnifetch_vdae,
        not args.disable_layout_aware,
        args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,
        args.enable_omnifetch_items_1_7,
        backend_profile=args.backend_profile or "legacy-scalar",
        enable_omnifetch_m_pad_hmx=args.enable_omnifetch_m_pad_hmx,
    )
    
    # Run
    outputs = hex_execution(
        patched if patched else module,
        "Dinov2AlignedWrapper",
        [pixels],
        options,
        iterations=args.device_iterations,
    )
    
    # Compare
    finite = bool(torch.isfinite(outputs[0]).all())
    diff = (outputs[0].float() - reference.float()).abs().max().item()
    top1_match = outputs[0].argmax().item() == reference.argmax().item()
    print(f"[Compare] finite={finite} max_abs_diff={diff:.4f} top1_match={top1_match}")

if __name__ == "__main__":
    main()
