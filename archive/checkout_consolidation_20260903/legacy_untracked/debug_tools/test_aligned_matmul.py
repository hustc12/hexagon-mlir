#!/usr/bin/env python3
"""Test aligned matmuls (M%32==0) to verify HMX works without padding."""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexkl_utils import (
    add_phase4_args,
    compile_to_linalg,
    hex_execution,
    hexagon_options_phase4,
)

class AlignedMatmulModel(nn.Module):
    """Simple model with aligned matmuls: M=64, K=64, N=64."""
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(64, 64, bias=False))
    
    def forward(self, x):
        # x: [64, 64] - M and K both aligned
        for layer in self.layers:
            x = layer(x)
            # Keep in f16 to match HexKL requirements
            if x.dtype != torch.float16:
                x = x.half()
        return x

def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_phase4_args(parser)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device-iterations", type=int, default=3)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    model = AlignedMatmulModel(num_layers=args.num_layers).half().eval()
    
    # Input: [64, 64] - M=64 (aligned), K=64 (aligned)
    x = torch.randn(64, 64, dtype=torch.float16)
    
    print(f"[Model] {args.num_layers} layers, M=64 (aligned), K=64, N=64")
    
    # Compile
    module = compile_to_linalg(model, (x,))
    ir = str(module)
    print(f"[IR] matmul={ir.count('linalg.matmul')}")
    
    # Reference
    with torch.no_grad():
        reference = model(x)
    
    # Hexagon options
    options = hexagon_options_phase4(
        enable_hexkl=args.enable_hexkl,
        backend_profile=args.backend_profile or "legacy-scalar",
    )
    
    print(f"[Compile] enable_hexkl={args.enable_hexkl}")
    
    # Run on device
    outputs = hex_execution(
        module,
        "AlignedMatmulModel",
        [x],
        options,
        iterations=args.device_iterations,
    )
    
    # Compare
    finite = bool(torch.isfinite(outputs[0]).all())
    diff = (outputs[0].float() - reference.float()).abs().max().item()
    print(f"[Compare] finite={finite} max_abs_diff={diff:.6f}")
    
    if not finite or diff > 0.01:
        raise AssertionError(f"Result mismatch: finite={finite}, diff={diff}")

if __name__ == "__main__":
    main()
