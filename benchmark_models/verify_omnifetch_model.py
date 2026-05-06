# ===- verify_omnifetch_model.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
#
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
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
    Minimal model: a single TransformerBlock for Omni-Fetch verification.
    Input: [1, seq_len, dim] float16
    Output: [1, seq_len, dim] float16
    """
    def __init__(self, dim=256, seq_len=64, num_heads=8):
        super().__init__()
        self.block = TransformerBlock(dim, num_heads)

    def forward(self, x):
        # x: [1, seq_len, dim]
        return self.block(x)

def compile_and_run(
    enable_omnifetch: bool = True,
    enable_hexkl: bool = True,
    lookahead: int = 2,
    adaptive: bool = True
):
    print(f"\n{'='*60}")
    print(f"  Omni-Fetch Verification Model")
    print(f"  Omni-Fetch: {'ON' if enable_omnifetch else 'OFF'}")
    print(f"  HexKL: DISABLED (HMX path causes AEE_EBADSTATE on device)")
    print(f"{'='*60}\n")

    # 1. Initialize model and data in FP16 (No FP32)
    device = "cpu"
    model = OmniFetchVerifModel().to(device).half()
    model.eval()
    
    # Input: [1, seq_len, dim] matching OmniFetchVerifModel defaults
    input_shape = (1, 64, 256)
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
    options["enableVectorization"] = enable_hexkl, # Seems like this option is the prerequisite for setting enableHexKL=True
    options["enableHexKL"] = enable_hexkl
    options["enableOmniFetchVDAE"] = enable_omnifetch
    options["enableOmniFetchLayoutAware"] = True
    options["omniFetchLookahead"] = lookahead
    options["enableOmniFetchAdaptive"] = adaptive
    # Model is small enough to fit in a single .so — no need to split constants.
    options["lowerConstantsInSeparateSharedObjects"] = False
    options["enableVTCMTiling"] = enable_omnifetch
    # Optional: Enable VTCM Tiling if helpful for prefetching (only with OmniFetch)
    # options["enableVTCMTiling"] = enable_omnifetch

    # 4. Execute on Hexagon NPU
    print("[Hexagon] Launching on NPU...")
    launcher = TorchMLIRHexagonLauncher()
    
    # Save MLIR for inspection if needed
    mlir_path = "/tmp/verify_omnifetch.mlirbc"
    with open(mlir_path, "wb") as f:
        f.write(linalg_module.operation.get_asm(binary=True))

    # FIX: Reduce _QURT_MAX_HEAP_SIZE from 1 GB to 256 MB.
    # 1 GB causes DSP TLB miss (heap addresses outside mapped region).
    from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb
    _ORIG = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
    _NEW  = "unsigned int _QURT_MAX_HEAP_SIZE = 268435456;  // 256 MB Max Heap Size"
    _orig_init = _hlb.WrapperGeneratorStrings.__init__
    def _patched_init(self):
        _orig_init(self)
        self.code_string = self.code_string.replace(_ORIG, _NEW)
    _hlb.WrapperGeneratorStrings.__init__ = _patched_init
    try:
        hex_outputs = launcher.run_torch_mlir(
            mlir_path,
            [dummy_input],
            "forward",
            options=options
        )
    finally:
        _hlb.WrapperGeneratorStrings.__init__ = _orig_init
    
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
    parser.add_argument("--enable-hexkl", action="store_true", help="Enable HexKL HMX lowering")
    parser.add_argument("--lookahead", type=int, default=2, help="Omni-Fetch lookahead distance")
    parser.add_argument("--enable-adaptive", action="store_true", help="Enable adaptive prefetch")

    args = parser.parse_args()

    compile_and_run(
        enable_omnifetch=args.enable_omnifetch,
        enable_hexkl=args.enable_hexkl,
        lookahead=args.lookahead,
        adaptive=args.enable_adaptive,
    )

# # 1. 基础运行（无 HexKL，无 Omni-Fetch）
# python3 verify_omnifetch_model.py

# # 2. 开启 HexKL HMX 加速
# python3 verify_omnifetch_model.py --enable-hexkl

# # 3. 开启 Omni-Fetch 优化（需同时开启 HexKL）
# python3 verify_omnifetch_model.py --enable-hexkl --enable-omnifetch

# # 4. 开启 Omni-Fetch 并调整 lookahead 距离
# python3 verify_omnifetch_model.py --enable-hexkl --enable-omnifetch --lookahead 4

# # 5. 开启全部选项
# python3 verify_omnifetch_model.py --enable-hexkl --enable-omnifetch --enable-adaptive --lookahead 4

