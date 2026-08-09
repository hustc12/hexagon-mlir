#!/usr/bin/env python3
# ===- test_quick_validation.py ---------------------------------------------===
#
# Quick validation test to verify the benchmark environment is set up correctly
# Runs a small test case for each operation type
#
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import torch_mlir.fx as fx_mlir
from pathlib import Path
import sys
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher


def test_matmul():
    """Quick test of matrix multiplication"""
    print("\n" + "="*60)
    print("Testing Matrix Multiplication...")
    print("="*60)
    
    class MatMul(nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)
    
    model = MatMul()
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    
    # Create MLIR module
    func_name = "MatMul"
    mlir_module = fx_mlir.export_and_import(
        model, a, b,
        output_type="linalg-on-tensors",
        func_name=func_name,
    )
    
    # Save bytecode
    filename = Path(__file__).parent / "test_matmul_quick.mlirbc"
    bytecode = mlir_module.operation.get_asm(binary=True)
    with open(filename, "wb") as f:
        f.write(bytecode)
    
    # Test scalar mode
    print("\n  Testing Scalar mode...")
    options = HexagonOptions(
        enableVectorization=False,
        enableHexKL=False,
    ).__dict__
    
    try:
        output = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(filename), [a, b], func_name, iterations=1, options=options
        )
        reference = model(a, b)
        is_correct = torch.allclose(output[0], reference, rtol=1e-03, atol=1e-03)
        
        if is_correct:
            print("  ✓ Scalar mode: PASS")
            return True
        else:
            print("  ✗ Scalar mode: FAIL (output mismatch)")
            return False
    except Exception as e:
        print(f"  ✗ Scalar mode: FAIL ({e})")
        return False


def test_conv():
    """Quick test of convolution"""
    print("\n" + "="*60)
    print("Testing Convolution...")
    print("="*60)
    
    class Conv2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        
        def forward(self, x):
            return self.conv(x)
    
    model = Conv2D()
    model.eval()
    x = torch.randn(1, 3, 8, 8)
    
    # Create MLIR module
    func_name = "Conv2D"
    mlir_module = fx_mlir.export_and_import(
        model, x,
        output_type="linalg-on-tensors",
        func_name=func_name,
    )
    
    # Save bytecode
    filename = Path(__file__).parent / "test_conv_quick.mlirbc"
    bytecode = mlir_module.operation.get_asm(binary=True)
    with open(filename, "wb") as f:
        f.write(bytecode)
    
    # Test scalar mode
    print("\n  Testing Scalar mode...")
    options = HexagonOptions(
        enableVectorization=False,
        enableHexKL=False,
    ).__dict__
    
    try:
        output = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(filename), [x], func_name, iterations=1, options=options
        )
        with torch.no_grad():
            reference = model(x)
        is_correct = torch.allclose(output[0], reference, rtol=1e-03, atol=1e-03)
        
        if is_correct:
            print("  ✓ Scalar mode: PASS")
            return True
        else:
            print("  ✗ Scalar mode: FAIL (output mismatch)")
            return False
    except Exception as e:
        print(f"  ✗ Scalar mode: FAIL ({e})")
        return False


def test_linear():
    """Quick test of linear layer"""
    print("\n" + "="*60)
    print("Testing Linear Layer...")
    print("="*60)
    
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 16, bias=False)
        
        def forward(self, x):
            return self.fc(x)
    
    model = Linear()
    model.eval()
    x = torch.randn(4, 32)
    
    # Create MLIR module
    func_name = "Linear"
    mlir_module = fx_mlir.export_and_import(
        model, x,
        output_type="linalg-on-tensors",
        func_name=func_name,
    )
    
    # Save bytecode
    filename = Path(__file__).parent / "test_linear_quick.mlirbc"
    bytecode = mlir_module.operation.get_asm(binary=True)
    with open(filename, "wb") as f:
        f.write(bytecode)
    
    # Test scalar mode
    print("\n  Testing Scalar mode...")
    options = HexagonOptions(
        enableVectorization=False,
        enableHexKL=False,
    ).__dict__
    
    try:
        output = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(filename), [x], func_name, iterations=1, options=options
        )
        with torch.no_grad():
            reference = model(x)
        is_correct = torch.allclose(output[0], reference, rtol=1e-03, atol=1e-03)
        
        if is_correct:
            print("  ✓ Scalar mode: PASS")
            return True
        else:
            print("  ✗ Scalar mode: FAIL (output mismatch)")
            return False
    except Exception as e:
        print(f"  ✗ Scalar mode: FAIL ({e})")
        return False


def check_environment():
    """Check if required environment variables are set"""
    print("\n" + "="*60)
    print("Checking Environment...")
    print("="*60)
    
    import os
    
    checks = {
        'HEXAGON_MLIR_ROOT': os.environ.get('HEXAGON_MLIR_ROOT'),
        'TRITON_SHARED_OPT_PATH': os.environ.get('TRITON_SHARED_OPT_PATH'),
    }
    
    all_ok = True
    for key, value in checks.items():
        if value:
            print(f"  ✓ {key}: {value}")
        else:
            print(f"  ⚠ {key}: Not set (optional for some tests)")
    
    # Check Python packages
    print("\nChecking Python Packages...")
    required_packages = ['torch', 'torch_mlir', 'triton']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}: Available")
        except ImportError:
            print(f"  ✗ {package}: Not found")
            all_ok = False
    
    return all_ok


def main():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("HEXAGON NPU - QUICK VALIDATION TEST")
    print("="*60)
    print("\nThis will verify your environment is correctly set up")
    print("for running the full benchmark suite.")
    
    # Check environment first
    env_ok = check_environment()
    
    if not env_ok:
        print("\n" + "="*60)
        print("⚠ WARNING: Some environment checks failed")
        print("="*60)
        print("\nYou may encounter issues running the benchmarks.")
        print("Please ensure all required packages are installed.")
    
    # Run tests
    results = {}
    
    try:
        results['matmul'] = test_matmul()
    except Exception as e:
        print(f"\n✗ MatMul test failed with exception: {e}")
        results['matmul'] = False
    
    try:
        results['conv'] = test_conv()
    except Exception as e:
        print(f"\n✗ Conv test failed with exception: {e}")
        results['conv'] = False
    
    try:
        results['linear'] = test_linear()
    except Exception as e:
        print(f"\n✗ Linear test failed with exception: {e}")
        results['linear'] = False
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.upper():<15} {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All validation tests passed!")
        print("You can now run the full benchmark suite:")
        print("  python3 run_all_benchmarks.py")
        return 0
    else:
        print("\n✗ Some validation tests failed.")
        print("Please check the error messages above and verify your setup.")
        print("\nCommon issues:")
        print("  - Missing environment variables")
        print("  - Device not connected (check adb devices)")
        print("  - Missing Python packages")
        return 1


if __name__ == "__main__":
    sys.exit(main())
