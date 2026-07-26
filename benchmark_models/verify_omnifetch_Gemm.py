# ===- verify_omnifetch_Gemm.py ----------------------------------------------===
#
# Phase-0 / Phase-1 vehicle for OmniFetch on a real HexKL MicroHMX path.
# MatmulToHexKL only lowers 2D linalg.matmul (not batch_matmul), so Attention
# alone cannot exercise HexKL DDR→VTCM copies.
#
# ===------------------------------------------------------------------------===

import argparse
import io
import re
import sys
import time

import torch
import torch.nn as nn
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
    TorchMLIRHexagonLauncher,
)


class GemmModel(nn.Module):
    """C = A @ B with f16 inputs / f32 accum (torch-mlir → linalg.matmul)."""

    def forward(self, a, b):
        return torch.matmul(a, b)


def _patch_dsp_heap_256mb():
    from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as _hlb

    _ORIG = "unsigned int _QURT_MAX_HEAP_SIZE = 1073741824; // 1 GB Max Heap Size"
    _NEW = "unsigned int _QURT_MAX_HEAP_SIZE = 268435456;  // 256 MB Max Heap Size"
    _orig_init = _hlb.WrapperGeneratorStrings.__init__

    def _patched_init(self):
        _orig_init(self)
        self.code_string = self.code_string.replace(_ORIG, _NEW)

    _hlb.WrapperGeneratorStrings.__init__ = _patched_init
    return _hlb, _orig_init


def compile_and_run(
    enable_omnifetch: bool = False,
    enable_hexkl: bool = True,
    enable_layout_aware: bool = False,
    enable_weight_prepack: bool = False,
    lookahead: int = 2,
    adaptive: bool = True,
    m: int = 256,
    k: int = 256,
    n: int = 256,
    verbose: bool = True,
):
    assert m % 32 == 0 and k % 32 == 0 and n % 32 == 0
    if verbose:
        print(f"\n{'='*60}")
        print(f"  GEMM [{m}x{k}] @ [{k}x{n}] f16")
        print(f"  HexKL={enable_hexkl} OmniFetch={enable_omnifetch} "
              f"LayoutAware={enable_layout_aware} Prepack={enable_weight_prepack}")
        print(f"{'='*60}\n")

    model = GemmModel().half().eval()
    a = torch.randn(m, k).half()
    b = torch.randn(k, n).half()

    linalg_module = fx.export_and_import(
        model, a, b, output_type=OutputType.LINALG_ON_TENSORS, func_name="forward"
    )
    asm = linalg_module.operation.get_asm(binary=False)
    if verbose:
        print(f"[IR] linalg.matmul count={asm.count('linalg.matmul')} "
              f"batch_matmul={asm.count('batch_matmul')}")

    options = HexagonOptions(
        enableVectorization=enable_hexkl,
        enableHexKL=enable_hexkl,
        enablePrefetch=enable_omnifetch,
        enableOmniFetchVDAE=enable_omnifetch,
        enableOmniFetchLayoutAware=enable_layout_aware and enable_omnifetch,
        enableOmniFetchWeightPrepack=enable_weight_prepack,
        omniFetchLookahead=lookahead,
        enableOmniFetchAdaptive=adaptive and enable_omnifetch,
        lowerConstantsInSeparateSharedObjects=False,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=True,
    ).__dict__

    mlir_path = "/tmp/verify_omnifetch_gemm.mlirbc"
    with open(mlir_path, "wb") as f:
        f.write(linalg_module.operation.get_asm(binary=True))

    _hlb, _orig_init = _patch_dsp_heap_256mb()
    launcher = TorchMLIRHexagonLauncher()
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    npu_time_us = None
    try:
        sys.stdout = stdout_capture
        hex_outputs = launcher.run_torch_mlir(
            mlir_path, [a, b], "forward", options=options
        )
        sys.stdout = original_stdout
        launcher_output = stdout_capture.getvalue()
        if verbose:
            print(launcher_output, end="")
        perf_match = re.search(r"Perf:\s*([0-9]+(?:\.[0-9]+)?)", launcher_output)
        if perf_match:
            npu_time_us = float(perf_match.group(1))
    except Exception:
        sys.stdout = original_stdout
        raise
    finally:
        _hlb.WrapperGeneratorStrings.__init__ = _orig_init

    with torch.no_grad():
        expected = model(a, b)
    actual = hex_outputs[0]
    if not isinstance(actual, torch.Tensor):
        actual = torch.from_numpy(actual)
    # HexKL f16 path: looser atol
    ok = torch.allclose(actual.float(), expected.float(), atol=0.5, rtol=0.05)
    if verbose:
        print("✓ SUCCESS: Results match!" if ok else "✗ FAILURE: mismatch")
        if not ok:
            print("max abs diff", (actual.float() - expected.float()).abs().max())
    return ok, npu_time_us


def run_phase0(m=256, k=256, n=256):
    configs = [
        ("HexKL only",
         dict(enable_omnifetch=False, enable_hexkl=True, enable_layout_aware=False,
              lookahead=0, adaptive=False)),
        ("HexKL + OmniFetch (layout OFF)",
         dict(enable_omnifetch=True, enable_hexkl=True, enable_layout_aware=False,
              lookahead=2, adaptive=True)),
        ("HexKL + OmniFetch (layout ON)",
         dict(enable_omnifetch=True, enable_hexkl=True, enable_layout_aware=True,
              lookahead=2, adaptive=True)),
    ]
    print(f"\nPhase 0/1 GEMM smoke  [{m}x{k}]@[{k}x{n}] f16\n")
    results = []
    for name, kw in configs:
        print(f"=== {name} ===", flush=True)
        t0 = time.time()
        ok, npu = compile_and_run(verbose=True, m=m, k=k, n=n, **kw)
        npu_ms = npu / 1000 if npu is not None else None
        print(f">>> RESULT {name}: Pass={ok} NPU={npu_ms} ms E2E={time.time()-t0:.1f}s\n",
              flush=True)
        results.append((name, ok, npu_ms))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0", action="store_true")
    parser.add_argument("--enable-hexkl", action="store_true")
    parser.add_argument("--enable-omnifetch", action="store_true")
    parser.add_argument("--enable-layout-aware", action="store_true")
    parser.add_argument("--enable-omnifetch-weight-prepack", action="store_true")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    if args.phase0:
        run_phase0(args.m, args.k, args.n)
    else:
        compile_and_run(
            enable_omnifetch=args.enable_omnifetch,
            enable_hexkl=args.enable_hexkl,
            enable_layout_aware=args.enable_layout_aware,
            enable_weight_prepack=args.enable_omnifetch_weight_prepack,
            m=args.m, k=args.k, n=args.n,
        )
