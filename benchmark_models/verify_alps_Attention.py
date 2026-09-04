# ===- verify_alps_model.py ---------------------------------------------===
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
        # NOTE: this is linalg.batch_matmul after export. MatmulToHexKL only
        # matches 2D linalg.matmul — use verify_alps_Gemm.py for HexKL
        # MicroHMX + Alps experiments.
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights

def compile_and_run(
    enable_alps: bool = True,
    enable_hexkl: bool = True,
    lookahead: int = 2,
    adaptive: bool = True,
    enable_vtcm_tiling: bool = True,
    enable_layout_aware: bool = True,
    verbose: bool = True,
    num_heads: int = 8,
    seq_len: int = 128,
    head_dim: int = 64,
):
    """
    Compile and run the model on Hexagon NPU.

    Default Q/K shapes are sized so HexKL tiles are more likely ≥4KB
    (PrefetchInsert sync-copy gate), unlike the old 64×32 microbench.

    Returns:
        tuple: (success: bool, npu_time_us: float or None)
               - success: Whether the execution succeeded and results match
               - npu_time_us: NPU kernel execution time in microseconds (from Test_Info)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Pure Attention Model (HexKL Compatible)")
        print(f"  Shape: Q/K=[1,{num_heads},{seq_len},{head_dim}] f16")
        print(f"  ALPS: {'ON' if enable_alps else 'OFF'}")
        print(f"  HexKL: {'ON' if enable_hexkl else 'OFF'}")
        print(f"  Lookahead: {lookahead}")
        print(f"  Adaptive: {'ON' if adaptive else 'OFF'}")
        print(f"  VTCM Tiling: {'ON' if enable_vtcm_tiling else 'OFF'}")
        print(f"  Layout Aware: {'ON' if enable_layout_aware else 'OFF'}")
        print(f"{'='*60}\n")

    # --- DEBUG prints below are intentionally kept under verbose=True only ---

    # 1. Initialize model and data in FP16
    device = "cpu"

    # Use PureAttentionModel - compatible with HexKL
    # No Linear layers, only attention batch_matmul
    assert seq_len % 32 == 0 and head_dim % 32 == 0, (
        f"HexKL alignment: seq_len={seq_len} and head_dim={head_dim} must be multiples of 32"
    )

    model = PureAttentionModel(num_heads=num_heads, seq_len=seq_len, head_dim=head_dim).to(device).half()
    model.eval()

    # Prepare inputs: Q and K tensors in multi-head format
    # Shape: [1, num_heads, seq_len, head_dim]
    q = torch.randn(1, num_heads, seq_len, head_dim).half().to(device)
    k = torch.randn(1, num_heads, seq_len, head_dim).half().to(device)
    
    # DEBUG: model/input info (disabled)
    # if verbose:
    #     print(f"[Model] PureAttentionModel (no Linear layers)")
    #     print(f"[Input] Q shape: {q.shape}, K shape: {k.shape}, Dtype: {q.dtype}")

    # 2. Export to Linalg-on-Tensors MLIR
    # DEBUG: compile step info (disabled)
    # if verbose:
    #     print("[Compile] Exporting to Linalg MLIR...")
    linalg_module = fx.export_and_import(
        model,
        q, k,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name="forward"
    )

    # 3. Configure Hexagon Options
    # IMPORTANT: Pass options to HexagonOptions constructor for proper validation
    # Do NOT manually assign to __dict__ as it bypasses type checking
    # PrefetchInsert is gated by enablePrefetch.  enableAlpsVDAE alone is
    # a no-op for insertion (pipeline warns and skips).  Mirror GPT2 runner:
    # when Alps is requested, enable both Prefetch and V-DAE together.
    # Note: LinalgToLLVM skips VTCMTiling whenever Alps is active.
    options = HexagonOptions(
        enableVectorization=enable_hexkl,  # Required prerequisite for enableHexKL
        enableHexKL=enable_hexkl,
        enablePrefetch=enable_alps,
        enableAlpsVDAE=enable_alps,
        enableAlpsLayoutAware=enable_layout_aware and enable_alps,
        alpsLookahead=lookahead,
        enableAlpsAdaptive=adaptive and enable_alps,
        lowerConstantsInSeparateSharedObjects=False,
        enableVTCMTiling=enable_vtcm_tiling,
        enableConvertToHexagonmem=True,
    ).__dict__

    # 4. Execute on Hexagon NPU
    # DEBUG: launcher info (disabled)
    # if verbose:
    #     print("[Hexagon] Launching on NPU...")
    launcher = TorchMLIRHexagonLauncher()
    
    # Save MLIR for inspection if needed
    mlir_path = "/tmp/verify_alps.mlirbc"
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
            # DEBUG: per-run NPU time print (disabled)
            # if verbose:
            #     print(f"\n[NPU Performance] Kernel execution time: {npu_time_us:.2f} us ({npu_time_us/1000:.2f} ms)")
        
    except Exception as e:
        sys.stdout = original_stdout
        raise e
    finally:
        _hlb.WrapperGeneratorStrings.__init__ = _orig_init
    
    # 5. CPU Reference for Verification
    # DEBUG: CPU reference info (disabled)
    # if verbose:
    #     print("[CPU] Running reference...")
    with torch.no_grad():
        expected_output = model(q, k)
    
    # 6. Compare
    actual_output = hex_outputs[0]
    
    # Convert to torch for comparison if it's a numpy array
    if not isinstance(actual_output, torch.Tensor):
        actual_output = torch.from_numpy(actual_output)
        
    # DEBUG: comparison header (disabled)
    # if verbose:
    #     print("\n[Compare] Hexagon vs CPU Reference:")
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

    Experiment 0: Baseline (all optimizations OFF)
    Experiment 1: Alps lookahead sweep with adaptive on/off (HexKL=ON)
    Experiment 2: HexKL + Alps + VTCMTiling 2x2 ablation
    Experiment 3: Layout-aware on/off comparison
    Experiment 4: Full stack vs. individual passes (single-pass contribution)

    Performance is measured using NPU kernel execution time from Test_Info (in microseconds).
    Speedup is computed relative to the Baseline (Exp0) NPU time.
    """
    import time

    print("\n" + "="*80)
    print("ABLATION STUDY: ALPS Performance Analysis")
    print("Model: PureAttentionModel  Q/K=[1,8,128,64] fp16  -> attn_weights=[1,8,128,128]")
    print("="*80 + "\n")

    results = []
    # Shared geometry for Phase-0/ablation (tiles large enough for ≥4KB gate).
    shape_kwargs = dict(num_heads=8, seq_len=128, head_dim=64)

    def _run(exp_name, config_name, **kwargs):
        """Helper: run one config, record result, print one-liner."""
        print(f"  {config_name:<55}", end="", flush=True)
        start = time.time()
        try:
            success, npu_time_us = compile_and_run(
                **shape_kwargs, **kwargs, verbose=False
            )
            e2e = time.time() - start
            rec = {
                'experiment': exp_name,
                'config': config_name,
                'npu_time_us': npu_time_us,
                'npu_time_ms': npu_time_us / 1000 if npu_time_us else None,
                'e2e_time_s': e2e,
                'success': success,
                **kwargs,
            }
            npu_str = f"{npu_time_us/1000:.2f} ms" if npu_time_us else "N/A"
            print(f"  NPU={npu_str}  E2E={e2e:.1f}s  {'OK' if success else 'MISMATCH'}")
        except Exception as exc:
            e2e = time.time() - start
            rec = {'experiment': exp_name, 'config': config_name, 'error': str(exc), 'e2e_time_s': e2e}
            print(f"  FAILED ({exc})")
        results.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Experiment 0: Baseline – everything OFF
    # ------------------------------------------------------------------
    print("-"*80)
    print("Experiment 0: Baseline (all optimizations OFF)")
    print("-"*80)
    baseline_rec = _run(
        'Exp0: Baseline', 'HexKL=OFF, Alps=OFF, VTCM=OFF, Layout=OFF',
        enable_alps=False, enable_hexkl=False,
        lookahead=0, adaptive=False,
        enable_vtcm_tiling=False, enable_layout_aware=False,
    )
    baseline_npu = baseline_rec.get('npu_time_us')

    # ------------------------------------------------------------------
    # Experiment 1: Lookahead sweep (HexKL=ON, Alps=ON, VTCM=ON, Layout=ON)
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 1: Alps Lookahead Sweep  (HexKL=ON, VTCM=ON, Layout=ON)")
    print("-"*80)
    for lookahead in [0, 1, 2, 4]:
        for adaptive in [False, True]:
            _run(
                'Exp1: Lookahead Sweep',
                f"Lookahead={lookahead}, Adaptive={'ON' if adaptive else 'OFF'}",
                enable_alps=True, enable_hexkl=True,
                lookahead=lookahead, adaptive=adaptive,
                enable_vtcm_tiling=True, enable_layout_aware=True,
            )

    # ------------------------------------------------------------------
    # Experiment 2: Alps × VTCMTiling 2×2 ablation (HexKL=ON)
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 2: Alps × VTCMTiling Ablation  (HexKL=ON, Lookahead=2, Adaptive=ON)")
    print("-"*80)
    for alps in [False, True]:
        for vtcm_tiling in [False, True]:
            _run(
                'Exp2: Alps×VTCMTiling',
                f"Alps={'ON' if alps else 'OFF'}, VTCMTiling={'ON' if vtcm_tiling else 'OFF'}",
                enable_alps=alps, enable_hexkl=True,
                lookahead=2, adaptive=True,
                enable_vtcm_tiling=vtcm_tiling, enable_layout_aware=True,
            )

    # ------------------------------------------------------------------
    # Experiment 3: Layout-aware on/off (HexKL=ON, Alps=ON, VTCM=ON)
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 3: Layout-Aware Comparison  (HexKL=ON, Alps=ON, VTCM=ON)")
    print("-"*80)
    for layout_aware in [False, True]:
        _run(
            'Exp3: Layout-Aware',
            f"Layout-Aware={'ON' if layout_aware else 'OFF'}",
            enable_alps=True, enable_hexkl=True,
            lookahead=2, adaptive=True,
            enable_vtcm_tiling=True, enable_layout_aware=layout_aware,
        )

    # ------------------------------------------------------------------
    # Experiment 4: Single-pass contribution (add one pass at a time)
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Experiment 4: Single-Pass Contribution  (incremental stack over Baseline)")
    print("-"*80)
    single_pass_configs = [
        ("HexKL only",
         dict(enable_alps=False, enable_hexkl=True,  lookahead=0, adaptive=False, enable_vtcm_tiling=False, enable_layout_aware=False)),
        ("HexKL + VTCM",
         dict(enable_alps=False, enable_hexkl=True,  lookahead=0, adaptive=False, enable_vtcm_tiling=True,  enable_layout_aware=False)),
        ("HexKL + Alps (no adaptive)",
         dict(enable_alps=True,  enable_hexkl=True,  lookahead=2, adaptive=False, enable_vtcm_tiling=False, enable_layout_aware=False)),
        ("HexKL + Alps + Adaptive",
         dict(enable_alps=True,  enable_hexkl=True,  lookahead=2, adaptive=True,  enable_vtcm_tiling=False, enable_layout_aware=False)),
        ("HexKL + Alps + Adaptive + VTCM",
         dict(enable_alps=True,  enable_hexkl=True,  lookahead=2, adaptive=True,  enable_vtcm_tiling=True,  enable_layout_aware=False)),
        ("Full Stack (all ON)",
         dict(enable_alps=True,  enable_hexkl=True,  lookahead=2, adaptive=True,  enable_vtcm_tiling=True,  enable_layout_aware=True)),
    ]
    for label, kwargs in single_pass_configs:
        _run('Exp4: Single-Pass', label, **kwargs)

    # ------------------------------------------------------------------
    # Summary table with speedup
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY  (Speedup = Baseline NPU time / Config NPU time)")
    print("="*80)

    exp_order = [
        'Exp0: Baseline',
        'Exp1: Lookahead Sweep',
        'Exp2: Alps×VTCMTiling',
        'Exp3: Layout-Aware',
        'Exp4: Single-Pass',
    ]

    for exp_name in exp_order:
        exp_results = [r for r in results if r.get('experiment') == exp_name]
        if not exp_results:
            continue

        print(f"\n{exp_name}")
        print("-" * 90)
        hdr = f"{'Configuration':<55} {'NPU (ms)':<12} {'Speedup':<10} {'E2E (s)':<10} {'OK?'}"
        print(hdr)
        print("-" * 90)

        for r in exp_results:
            if 'error' in r:
                print(f"  {r['config']:<53} {'ERROR':<12} {'N/A':<10} {r.get('e2e_time_s', 0):<10.1f} False")
                continue
            npu_ms  = r.get('npu_time_ms')
            npu_str = f"{npu_ms:.2f}" if npu_ms is not None else "N/A"
            if baseline_npu and npu_ms is not None:
                speedup = baseline_npu / (npu_ms * 1000)
                spd_str = f"{speedup:.2f}x"
            else:
                spd_str = "N/A"
            e2e_str = f"{r['e2e_time_s']:.1f}" if r.get('e2e_time_s') is not None else "N/A"
            ok_str  = "✓" if r.get('success') else "✗"
            print(f"  {r['config']:<53} {npu_str:<12} {spd_str:<10} {e2e_str:<10} {ok_str}")

    # Best config per experiment
    print("\n" + "="*80)
    print("BEST CONFIGURATION PER EXPERIMENT")
    print("="*80)
    for exp_name in exp_order[1:]:   # skip baseline itself
        exp_results = [r for r in results
                       if r.get('experiment') == exp_name
                       and r.get('npu_time_us') is not None
                       and r.get('success')]
        if not exp_results:
            continue
        best = min(exp_results, key=lambda r: r['npu_time_us'])
        speedup_str = ""
        if baseline_npu:
            spd = baseline_npu / best['npu_time_us']
            speedup_str = f"  →  {spd:.2f}x speedup over baseline"
        print(f"  {exp_name}: {best['config']}  ({best['npu_time_ms']:.2f} ms){speedup_str}")

    print("\n" + "="*80)
    print("Notes:")
    print("  • NPU Time  = actual kernel execution time on Hexagon DSP (from Test_Info Perf field)")
    print("  • Speedup   = Baseline NPU time / Config NPU time  (>1 means faster than baseline)")
    print("  • E2E Time  = wall-clock time including compile, data transfer, and NPU execution")
    print("="*80 + "\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-alps", action="store_true", help="Enable ALPS optimization")
    parser.add_argument("--enable-hexkl", action="store_true", help="Enable HexKL HMX lowering")
    parser.add_argument("--lookahead", type=int, default=2, help="ALPS lookahead distance")
    parser.add_argument("--enable-adaptive", action="store_true", help="Enable adaptive prefetch")
    parser.add_argument("--enable-vtcm-tiling", action="store_true", help="Enable VTCM tiling")
    parser.add_argument("--enable-layout-aware", action="store_true", help="Enable layout-aware optimization")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--seq-len", type=int, default=128, help="Attention sequence length (multiple of 32)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (multiple of 32)")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--phase0", action="store_true",
                        help="Run Phase-0 smoke: HexKL / HexKL+VTCM / HexKL+Alps(layout OFF)")

    args = parser.parse_args()

    if args.ablation:
        # Run comprehensive ablation study
        run_ablation_study()
    elif args.phase0:
        import time
        configs = [
            ("HexKL only",
             dict(enable_alps=False, enable_hexkl=True, lookahead=0, adaptive=False,
                  enable_vtcm_tiling=False, enable_layout_aware=False)),
            ("HexKL + VTCM",
             dict(enable_alps=False, enable_hexkl=True, lookahead=0, adaptive=False,
                  enable_vtcm_tiling=True, enable_layout_aware=False)),
            ("HexKL + Alps (sync, layout OFF)",
             dict(enable_alps=True, enable_hexkl=True, lookahead=2, adaptive=True,
                  enable_vtcm_tiling=False, enable_layout_aware=False)),
        ]
        print("\nPhase 0 smoke (Attention)")
        print(f"  Q/K=[1,{args.num_heads},{args.seq_len},{args.head_dim}] f16")
        print("  Note: pipeline skips VTCMTiling when Alps is active.\n")
        for name, kw in configs:
            print(f"=== {name} ===", flush=True)
            t0 = time.time()
            ok, npu = compile_and_run(
                verbose=True,
                num_heads=args.num_heads,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                **kw,
            )
            npu_str = f"{npu/1000:.2f} ms" if npu is not None else "N/A"
            print(f">>> RESULT {name}: Pass={ok} NPU={npu_str} E2E={time.time()-t0:.1f}s\n",
                  flush=True)
    else:
        # Run single configuration
        compile_and_run(
            enable_alps=args.enable_alps,
            enable_hexkl=args.enable_hexkl,
            lookahead=args.lookahead,
            adaptive=args.enable_adaptive,
            enable_vtcm_tiling=args.enable_vtcm_tiling,
            enable_layout_aware=args.enable_layout_aware,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            verbose=True
        )

# Usage Examples:
# ================

# 1. Basic run (all optimizations OFF by default)
# python3 verify_alps_model.py

# 2. Enable HexKL HMX acceleration (now works with Pure Attention model!)
# python3 verify_alps_model.py --enable-hexkl

# 3. Enable ALPS with HexKL
# python3 verify_alps_model.py --enable-hexkl --enable-alps

# 4. Full optimization stack
# python3 verify_alps_model.py --enable-hexkl --enable-alps --enable-adaptive --enable-vtcm-tiling --enable-layout-aware

# 5. Adjust lookahead distance
# python3 verify_alps_model.py --enable-hexkl --enable-alps --lookahead 4

# 6. Run comprehensive ablation study (automatic sweep of all configurations)
# python3 verify_alps_model.py --ablation

# Note: This version uses PureAttentionModel (no Linear layers) which is compatible with HexKL.
# The model performs attention computation: Q @ K^T with softmax, avoiding the Linear+Attention
# batch_matmul combination that causes error 13 on HexKL.
