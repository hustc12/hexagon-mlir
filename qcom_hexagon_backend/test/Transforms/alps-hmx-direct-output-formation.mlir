// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-direct-output-formation=true}))' | FileCheck %s --check-prefix=DIRECT
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul))' | FileCheck %s --check-prefix=CONTROL

// P5k keeps the M-padded activation required by HMX, but lets the HexKL
// epilogue clip its stores to 33 valid rows in the caller-provided output.
// It must not materialize a padded output followed by subview+memref.copy.
func.func @unaligned_m_f16(%lhs: memref<33x64xf16>,
                           %rhs: memref<64x128xf16>,
                           %out: memref<33x128xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<33x64xf16>, memref<64x128xf16>)
               outs(%out : memref<33x128xf16>)
  return
}

// DIRECT-LABEL: func.func @unaligned_m_f16(
// DIRECT-SAME:  {{.*}}, %[[OUT:.+]]: memref<33x128xf16>)
// DIRECT:       %[[MVALID:.+]] = arith.constant 33 : i32
// DIRECT:       hexkl.micro_hmx_copy_f16_to_submatrix({{.*}}%[[OUT]]{{.*}}%[[MVALID]]{{.*}})
// DIRECT-NOT:   memref.copy {{.*}}, %[[OUT]]
// DIRECT:       return

// CONTROL-LABEL: func.func @unaligned_m_f16(
// CONTROL-SAME:  {{.*}}, %[[OUT:.+]]: memref<33x128xf16>)
// CONTROL:       %[[PADDED_OUT:.+]] = memref.alloc() : memref<64x128xf16>
// CONTROL:       hexkl.micro_hmx_copy_f16_to_submatrix({{.*}}%[[PADDED_OUT]],
// CONTROL:       %[[VALID:.+]] = memref.subview %[[PADDED_OUT]]
// CONTROL:       memref.copy %[[VALID]], %[[OUT]]
// CONTROL:       return
