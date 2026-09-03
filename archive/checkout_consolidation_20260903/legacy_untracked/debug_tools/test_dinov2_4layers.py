#!/usr/bin/env python3
"""Test DINOv2 with 4 layers to find the scaling limit."""
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
    patch_full_model_dsp_heap,
)

class Dinov2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, position_embeddings, pixels):
        return self.model(pixel_values=pixels).logits

def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_phase4_args(parser)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--device-iterations", type=int, default=1)
    args = parser.parse_args()
    
    patch_full_model_dsp_heap()
    torch.manual_seed(142)
    
    config = Dinov2Config(
        image_size=224,
        patch_size=14,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=args.num_layers,  # Variable!
        num_attention_heads=6,
        intermediate_size=1536,
        hidden_act="gelu_new",
        num_labels=1000,
        use_mask_token=False,
    )
    
    model = Dinov2Wrapper(
        Dinov2ForImageClassification(config).half().eval()
    ).eval()
    
    # Store position embeddings
    model.model.dinov2.embeddings.omnifetch_fixed_position_embeddings = (
        model.model.dinov2.embeddings.position_embeddings.detach().clone()
    )
    
    pixels = torch.rand(1, 3, 224, 224, dtype=torch.float16)
    device_inputs = [
        model.model.dinov2.embeddings.omnifetch_fixed_position_embeddings,
        pixels,
    ]
    
    module = compile_to_linalg(model, [pixels], decomp_pow=False)
    ir = str(module)
    print(f"[Model] layers={args.num_layers} batch_matmul={ir.count('linalg.batch_matmul')}")
    
    patched = None
    if args.enable_hexkl:
        candidate, n_batch, n_f16 = apply_hexkl_ir_rewrites(
            ir, enable_m_pad=args.enable_omnifetch_m_pad_hmx
        )
        patched = candidate if n_batch or n_f16 else None
        print(f"[HexKL] rewrites={n_batch}")
    
    options = hexagon_options_phase4(
        args.enable_hexkl,
        args.enable_omnifetch_vdae,
        not args.disable_layout_aware,
        args.omnifetch_lookahead,
        not args.disable_omnifetch_adaptive,
        args.enable_omnifetch_items_1_7,
        lower_constants_separate=True,
        backend_profile=args.backend_profile,
        enable_omnifetch_m_pad_hmx=args.enable_omnifetch_m_pad_hmx,
        enable_out_params=args.enable_out_params,
    )
    
    with torch.no_grad():
        reference = model(None, pixels)  # Pass both args
    
    output = hex_execution(
        module,
        model.__class__.__name__,
        device_inputs,
        options,
        mlir_text=patched,
        iterations=args.device_iterations,
    )
    
    finite = bool(torch.isfinite(output[0]).all())
    diff = (output[0].float() - reference.float()).abs().max().item()
    top1_match = output[0].argmax().item() == reference.argmax().item()
    print(f"[Compare] finite={finite} max_abs_diff={diff:.4f} top1_match={top1_match}")
    
    if not finite or not top1_match:
        raise AssertionError(f"DINOv2-{args.num_layers}L failed correctness")

if __name__ == "__main__":
    main()
