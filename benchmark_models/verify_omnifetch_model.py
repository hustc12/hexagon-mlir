# ===- verify_omnifetch_model.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
    TorchMLIRHexagonLauncher,
)
import argparse
from pathlib import Path
import os

class TransformerBlock(nn.Module):
    """A standard Transformer block with large MatMuls."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        # x shape: [1, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # --- Large MatMul: [seq_len, head_dim] x [head_dim, seq_len] -> [seq_len, seq_len] ---
        # With seq_len=512, this is a 512x512 MatMul (per head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # --- Large MatMul: [seq_len, seq_len] x [seq_len, head_dim] -> [seq_len, head_dim] ---
        attn_out = torch.matmul(attn_weights, v)
        
        out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        return self.proj(out) + x

class OmniFetchVerifModel(nn.Module):
    """
    A large-scale model containing multiple Conv2d and Transformer blocks.
    Designed to stress-test Omni-Fetch optimizations.
    """
    def __init__(self, hidden_dim=1024, seq_len=512, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 1. Deeper Conv front-end
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        
        # 2. Large projection to Transformer dim
        self.fc_in = nn.Linear(128 * 8 * 8, hidden_dim, bias=False)
        
        # 3. Multiple Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # 4. Final output
        self.proj_out = nn.Linear(hidden_dim, 10, bias=False)

    def forward(self, x):
        # x: [1, 3, 32, 32]
        
        # --- Conv layers ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # --- Slice and Flatten ---
        x_slice = x[:, :, 12:20, 12:20] # [1, 128, 8, 8]
        flat = x_slice.reshape(1, -1)   # [1, 8192]
        
        # --- Large MatMul: [1, 8192] x [8192, 1024] ---
        hidden = self.fc_in(flat)       # [1, hidden_dim]
        
        # Expand to sequence
        seq = hidden.repeat(1, self.seq_len, 1).view(1, self.seq_len, self.hidden_dim)
        
        # --- Multiple Transformer Layers ---
        for layer in self.layers:
            seq = layer(seq)
            
        # --- Final Slice and MatMul ---
        out = seq[:, -1, :] # [1, hidden_dim]
        logits = self.proj_out(out) # [1, 10]
        
        return logits

def compile_and_run(
    enable_omnifetch: bool = False,
    enable_hexkl: bool = True,
    lookahead: int = 2,
    adaptive: bool = True
):
    print(f"\n{'='*60}")
    print(f"  Omni-Fetch Verification Model")
    print(f"  Omni-Fetch: {'ON' if enable_omnifetch else 'OFF'}")
    print(f"  HexKL: {'ON' if enable_hexkl else 'OFF'}")
    print(f"{'='*60}\n")

    # 1. Initialize model and data in FP16 (No FP32)
    device = "cpu"
    model = OmniFetchVerifModel().to(device).half()
    model.eval()
    
    input_shape = (1, 3, 32, 32)
    dummy_input = torch.randn(input_shape).half().to(device)
    
    print(f"[Model] Initialized in float16")
    print(f"[Input] Shape: {input_shape}, Dtype: {dummy_input.dtype}")

    # 2. Export to Linalg-on-Tensors MLIR
    print("[Compile] Exporting to Linalg MLIR...")
    linalg_module = fx.export_and_import(
        model,
        dummy_input,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name="forward"
    )

    # 3. Configure Hexagon Options
    options = HexagonOptions().__dict__
    options["enableHexKL"] = enable_hexkl
    options["enableOmniFetchVDAE"] = enable_omnifetch
    options["enableOmniFetchLayoutAware"] = True
    options["omniFetchLookahead"] = lookahead
    options["enableOmniFetchAdaptive"] = adaptive
    
    # Optional: Enable VTCM Tiling if helpful for prefetching
    options["enableVTCMTiling"] = enable_omnifetch 

    # 4. Execute on Hexagon NPU
    print("[Hexagon] Launching on NPU...")
    launcher = TorchMLIRHexagonLauncher()
    
    # Save MLIR for inspection if needed
    mlir_path = "/tmp/verify_omnifetch.mlirbc"
    with open(mlir_path, "wb") as f:
        f.write(linalg_module.operation.get_asm(binary=True))
        
    hex_outputs = launcher.run_torch_mlir(
        mlir_path,
        [dummy_input],
        "forward",
        options=options
    )
    
    # 5. CPU Reference for Verification
    print("[CPU] Running reference...")
    with torch.no_grad():
        expected_output = model(dummy_input)
    
    # 6. Compare
    actual_output = hex_outputs[0]
    
    # Convert to torch for comparison if it's a numpy array
    if not isinstance(actual_output, torch.Tensor):
        actual_output = torch.from_numpy(actual_output)
        
    print("\n[Compare] Hexagon vs CPU Reference:")
    allclose = torch.allclose(actual_output.float(), expected_output.float(), atol=1e-2)
    if allclose:
        print("✓ SUCCESS: Results match!")
    else:
        print("✗ FAILURE: Results mismatch!")
        diff = (actual_output.float() - expected_output.float()).abs().max()
        print(f"Max absolute difference: {diff}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-omnifetch", action="store_true", help="Enable Omni-Fetch optimization")
    parser.add_argument("--disable-hexkl", action="store_true", help="Disable HexKL HMX lowering")
    parser.add_argument("--lookahead", type=int, default=2, help="Omni-Fetch lookahead distance")
    parser.add_argument("--disable-adaptive", action="store_true", help="Disable adaptive prefetch")
    
    args = parser.parse_args()
    
    compile_and_run(
        enable_omnifetch=args.enable_omnifetch,
        enable_hexkl=not args.disable_hexkl,
        lookahead=args.lookahead,
        adaptive=not args.disable_adaptive
    )

# # 1. 基础运行（仅开启 HexKL HMX 加速）
# python3 verify_omnifetch_model.py

# # 2. 开启 Omni-Fetch 优化
# python3 verify_omnifetch_model.py --enable-omnifetch

# # 3. 开启 Omni-Fetch 并调整 lookahead 距离
# python3 verify_omnifetch_model.py --enable-omnifetch --lookahead 4

