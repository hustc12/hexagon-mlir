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

class PureAttentionModel(nn.Module):
    """
    Pure Attention model without Linear layers - HexKL compatible.
    
    This model takes pre-computed Q and K tensors and performs attention.
    It avoids the Linear+Attention batch_matmul combination that causes
    HexKL error 13 on Hexagon DSP.
    
    Input: q, k - [1, num_heads, seq_len, head_dim] float16
    Output: attention weights - [1, num_heads, seq_len, seq_len] float16
    """
    def __init__(self, num_heads=8, seq_len=64, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim

    def forward(self, q, k):
        # q, k: [1, num_heads, seq_len, head_dim]
        # Attention: Q @ K^T -> [1, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply softmax (optional, but makes it more realistic)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        return attn_weights

def compile_and_run(
    enable_omnifetch: bool = True,
    enable_hexkl: bool = True,
    lookahead: int = 2,
    adaptive: bool = True,
    enable_vtcm_tiling: bool = True,
    enable_layout_aware: bool = True,
    verbose: bool = True
):
    """
    Compile and run the model on Hexagon NPU.
    
    Returns:
        tuple: (success: bool, npu_time_us: float or None)
               - success: Whether the execution succeeded and results match
               - npu_time_us: NPU kernel execution time in microseconds (from Test_Info)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Pure Attention Model (HexKL Compatible)")
        print(f"  Omni-Fetch: {'ON' if enable_omnifetch else 'OFF'}")
        print(f"  HexKL: {'ON' if enable_hexkl else 'OFF'}")
        print(f"  Lookahead: {lookahead}")
        print(f"  Adaptive: {'ON' if adaptive else 'OFF'}")
        print(f"  VTCM Tiling: {'ON' if enable_vtcm_tiling else 'OFF'}")
        print(f"  Layout Aware: {'ON' if enable_layout_aware else 'OFF'}")
        print(f"{'='*60}\n")

    # 1. Initialize model and data in FP16
    device = "cpu"
    
    # Use PureAttentionModel - compatible with HexKL
    # No Linear layers, only attention batch_matmul
    num_heads = 8
    seq_len = 64
    head_dim = 32
    
    model = PureAttentionModel(num_heads=num_heads, seq_len=seq_len, head_dim=head_dim).to(device).half()
    model.eval()
    
    # Prepare inputs: Q and K tensors in multi-head format
    # Shape: [1, num_heads, seq_len, head_dim]
    q = torch.randn(1, num_heads, seq_len, head_dim).half().to(device)
    k = torch.randn(1, num_heads, seq_len, head_dim).half().to(device)
    
    if verbose:
        print(f"[Model] PureAttentionModel (no Linear layers)")
        print(f"[Input] Q shape: {q.shape}, K shape: {k.shape}, Dtype: {q.dtype}")

    # 2. Export to Linalg-on-Tensors MLIR
    if verbose:
        print("[Compile] Exporting to Linalg MLIR...")
    linalg_module = fx.export_and_import(
        model,
        q, k,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name="forward"
    )

    # 3. Configure Hexagon Options
    # IMPORTANT: Pass options to HexagonOptions constructor for proper validation
    # Do NOT manually assign to __dict__ as it bypasses type checking
    options = HexagonOptions(
        enableVectorization=enable_hexkl,  # Required prerequisite for enableHexKL
        enableHexKL=enable_hexkl,
        enableOmniFetchVDAE=enable_omnifetch,
        enableOmniFetchLayoutAware=enable_layout_aware,
        omniFetchLookahead=lookahead,
        enableOmniFetchAdaptive=adaptive,
        lowerConstantsInSeparateSharedObjects=False,
        enableVTCMTiling=enable_vtcm_tiling
    ).__dict__

    # 4. Execute on Hexagon NPU
    if verbose:
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
    
    # Capture stdout to extract Test_Info performance data
    import io
    import sys
    import re
    
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to capture launcher output
        sys.stdout = stdout_capture
        
        hex_outputs = launcher.run_torch_mlir(
            mlir_path,
            [q, k],  # Two inputs: Q and K
            "forward",
            options=options
        )
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Get captured output
        launcher_output = stdout_capture.getvalue()
        
        # Print output if verbose
        if verbose:
            print(launcher_output, end='')
        
        # Extract NPU kernel time from Test_Info
        # Format: "Perf:90084.000000" in microseconds
        npu_time_us = None
        perf_match = re.search(r'Perf:\s*([0-9]+(?:\.[0-9]+)?)', launcher_output)
        if perf_match:
            npu_time_us = float(perf_match.group(1))
            if verbose:
                print(f"\n[NPU Performance] Kernel execution time: {npu_time_us:.2f} us ({npu_time_us/1000:.2f} ms)")
        
    except Exception as e:
        sys.stdout = original_stdout
        raise e
    finally:
        _hlb.WrapperGeneratorStrings.__init__ = _orig_init
    
    # 5. CPU Reference for Verification
    if verbose:
        print("[CPU] Running reference...")
    with torch.no_grad():
        expected_output = model(q, k)
    
    # 6. Compare
    actual_output = hex_outputs[0]
    
    # Convert to torch for comparison if it's a numpy array
    if not isinstance(actual_output, torch.Tensor):
        actual_output = torch.from_numpy(actual_output)
        
    if verbose:
        print("\n[Compare] Hexagon vs CPU Reference:")
    allclose = torch.allclose(actual_output.float(), expected_output.float(), atol=1e-2)
    if verbose:
        if allclose:
            print("✓ SUCCESS: Results match!")
        else:
            print("✗ FAILURE: Results mismatch!")
            diff = (actual_output.float() - expected_output.float()).abs().max()
            print(f"Max absolute difference: {diff}")
    
    return allclose, npu_time_us

def run_ablation_study():
    """
    Run comprehensive ablation studies to isolate performance factors.
    
    Experiment 1: OmniFetch lookahead sweep with adaptive on/off (HexKL=ON)
    Experiment 2: HexKL + OmniFetch + VTCMTiling 2x2 ablation
    Experiment 3: Layout-aware on/off comparison
    
    Performance is measured using NPU kernel execution time from Test_Info (in microseconds).
    """
    import time
    
    print("\n" + "="*80)
    print("ABLATION STUDY: Omni-Fetch Performance Analysis")
    print("="*80 + "\n")
    
    results = []
    
    # Experiment 1: Lookahead sweep with adaptive on/off (HexKL=ON, OmniFetch=ON)
    print("\n" + "-"*80)
    print("Experiment 1: OmniFetch Lookahead Sweep (HexKL=ON)")
    print("-"*80)
    
    for lookahead in [0, 1, 2, 4]:
        for adaptive in [False, True]:
            config_name = f"Lookahead={lookahead}, Adaptive={'ON' if adaptive else 'OFF'}"
            print(f"\n[Config] {config_name}")
            
            start_time = time.time()
            try:
                success, npu_time_us = compile_and_run(
                    enable_omnifetch=True,
                    enable_hexkl=True,
                    lookahead=lookahead,
                    adaptive=adaptive,
                    enable_vtcm_tiling=True,
                    enable_layout_aware=True,
                    verbose=False
                )
                e2e_time = time.time() - start_time
                results.append({
                    'experiment': 'Exp1: Lookahead Sweep',
                    'config': config_name,
                    'hexkl': True,
                    'omnifetch': True,
                    'lookahead': lookahead,
                    'adaptive': adaptive,
                    'vtcm_tiling': True,
                    'layout_aware': True,
                    'npu_time_us': npu_time_us,
                    'npu_time_ms': npu_time_us / 1000 if npu_time_us else None,
                    'e2e_time_s': e2e_time,
                    'success': success
                })
                if npu_time_us:
                    print(f"  NPU Time: {npu_time_us/1000:.2f} ms, E2E Time: {e2e_time:.2f}s, Success: {success}")
                else:
                    print(f"  E2E Time: {e2e_time:.2f}s, Success: {success}")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    'experiment': 'Exp1: Lookahead Sweep',
                    'config': config_name,
                    'error': str(e)
                })
    
    # Experiment 2: HexKL + OmniFetch + VTCMTiling 2x2 ablation
    print("\n" + "-"*80)
    print("Experiment 2: OmniFetch × VTCMTiling Ablation (HexKL=ON)")
    print("-"*80)
    
    for omnifetch in [False, True]:
        for vtcm_tiling in [False, True]:
            config_name = f"OmniFetch={'ON' if omnifetch else 'OFF'}, VTCMTiling={'ON' if vtcm_tiling else 'OFF'}"
            print(f"\n[Config] {config_name}")
            
            start_time = time.time()
            try:
                success, npu_time_us = compile_and_run(
                    enable_omnifetch=omnifetch,
                    enable_hexkl=True,
                    lookahead=2,
                    adaptive=True,
                    enable_vtcm_tiling=vtcm_tiling,
                    enable_layout_aware=True,
                    verbose=False
                )
                e2e_time = time.time() - start_time
                results.append({
                    'experiment': 'Exp2: OmniFetch×VTCMTiling',
                    'config': config_name,
                    'hexkl': True,
                    'omnifetch': omnifetch,
                    'lookahead': 2,
                    'adaptive': True,
                    'vtcm_tiling': vtcm_tiling,
                    'layout_aware': True,
                    'npu_time_us': npu_time_us,
                    'npu_time_ms': npu_time_us / 1000 if npu_time_us else None,
                    'e2e_time_s': e2e_time,
                    'success': success
                })
                if npu_time_us:
                    print(f"  NPU Time: {npu_time_us/1000:.2f} ms, E2E Time: {e2e_time:.2f}s, Success: {success}")
                else:
                    print(f"  E2E Time: {e2e_time:.2f}s, Success: {success}")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    'experiment': 'Exp2: OmniFetch×VTCMTiling',
                    'config': config_name,
                    'error': str(e)
                })
    
    # Experiment 3: Layout-aware on/off (HexKL=ON, OmniFetch=ON)
    print("\n" + "-"*80)
    print("Experiment 3: Layout-Aware Comparison (HexKL=ON, OmniFetch=ON)")
    print("-"*80)
    
    for layout_aware in [False, True]:
        config_name = f"Layout-Aware={'ON' if layout_aware else 'OFF'}"
        print(f"\n[Config] {config_name}")
        
        start_time = time.time()
        try:
            success, npu_time_us = compile_and_run(
                enable_omnifetch=True,
                enable_hexkl=True,
                lookahead=2,
                adaptive=True,
                enable_vtcm_tiling=True,
                enable_layout_aware=layout_aware,
                verbose=False
            )
            e2e_time = time.time() - start_time
            results.append({
                'experiment': 'Exp3: Layout-Aware',
                'config': config_name,
                'hexkl': True,
                'omnifetch': True,
                'lookahead': 2,
                'adaptive': True,
                'vtcm_tiling': True,
                'layout_aware': layout_aware,
                'npu_time_us': npu_time_us,
                'npu_time_ms': npu_time_us / 1000 if npu_time_us else None,
                'e2e_time_s': e2e_time,
                'success': success
            })
            if npu_time_us:
                print(f"  NPU Time: {npu_time_us/1000:.2f} ms, E2E Time: {e2e_time:.2f}s, Success: {success}")
            else:
                print(f"  E2E Time: {e2e_time:.2f}s, Success: {success}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'experiment': 'Exp3: Layout-Aware',
                'config': config_name,
                'error': str(e)
            })
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80 + "\n")
    
    # Group by experiment
    for exp_name in ['Exp1: Lookahead Sweep', 'Exp2: OmniFetch×VTCMTiling', 'Exp3: Layout-Aware']:
        exp_results = [r for r in results if r.get('experiment') == exp_name]
        if not exp_results:
            continue
            
        print(f"\n{exp_name}")
        print("-" * 80)
        
        # Print header
        print(f"{'Configuration':<50} {'NPU Time (ms)':<15} {'E2E Time (s)':<15} {'Success':<10}")
        print("-" * 80)
        
        # Print rows
        for r in exp_results:
            if 'error' in r:
                print(f"{r['config']:<50} {'ERROR':<15} {'N/A':<15} {'False':<10}")
            else:
                npu_time_str = f"{r['npu_time_ms']:.2f}" if r.get('npu_time_ms') else "N/A"
                e2e_time_str = f"{r['e2e_time_s']:.2f}" if r.get('e2e_time_s') else "N/A"
                print(f"{r['config']:<50} {npu_time_str:<15} {e2e_time_str:<15} {str(r['success']):<10}")
    
    print("\n" + "="*80)
    print("Ablation study complete!")
    print("\nNote: NPU Time is the actual kernel execution time on Hexagon DSP (from Test_Info).")
    print("      E2E Time includes compilation, data transfer, and NPU execution.")
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-omnifetch", action="store_true", help="Enable Omni-Fetch optimization")
    parser.add_argument("--enable-hexkl", action="store_true", help="Enable HexKL HMX lowering")
    parser.add_argument("--lookahead", type=int, default=2, help="Omni-Fetch lookahead distance")
    parser.add_argument("--enable-adaptive", action="store_true", help="Enable adaptive prefetch")
    parser.add_argument("--enable-vtcm-tiling", action="store_true", help="Enable VTCM tiling")
    parser.add_argument("--enable-layout-aware", action="store_true", help="Enable layout-aware optimization")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")

    args = parser.parse_args()

    if args.ablation:
        # Run comprehensive ablation study
        run_ablation_study()
    else:
        # Run single configuration
        compile_and_run(
            enable_omnifetch=args.enable_omnifetch,
            enable_hexkl=args.enable_hexkl,
            lookahead=args.lookahead,
            adaptive=args.enable_adaptive,
            enable_vtcm_tiling=args.enable_vtcm_tiling,
            enable_layout_aware=args.enable_layout_aware,
            verbose=True
        )

# Usage Examples:
# ================

# 1. Basic run (all optimizations OFF by default)
# python3 verify_omnifetch_model.py

# 2. Enable HexKL HMX acceleration (now works with Pure Attention model!)
# python3 verify_omnifetch_model.py --enable-hexkl

# 3. Enable Omni-Fetch with HexKL
# python3 verify_omnifetch_model.py --enable-hexkl --enable-omnifetch

# 4. Full optimization stack
# python3 verify_omnifetch_model.py --enable-hexkl --enable-omnifetch --enable-adaptive --enable-vtcm-tiling --enable-layout-aware

# 5. Adjust lookahead distance
# python3 verify_omnifetch_model.py --enable-hexkl --enable-omnifetch --lookahead 4

# 6. Run comprehensive ablation study (automatic sweep of all configurations)
# python3 verify_omnifetch_model.py --ablation

# Note: This version uses PureAttentionModel (no Linear layers) which is compatible with HexKL.
# The model performs attention computation: Q @ K^T with softmax, avoiding the Linear+Attention
# batch_matmul combination that causes error 13 on HexKL.
