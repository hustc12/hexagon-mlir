// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-insert{lookahead=0 enable-layout-aware=true enable-alps-fused-transform-transfer=true}))' \
// RUN:   | FileCheck %s

// P2c is deliberately synchronous. It replaces only proven HexKL micro
// transfer/layout chains; readiness and asynchronous lookahead belong to P3.
func.func @p2c_hmx_micro(
    %hmx: memref<?xi8, 1>,
    %weight: memref<64x64xf16>,
    %activation: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0 : i32
  %cols = arith.constant 64 : i32
  scf.for %i = %c0 to %c2 step %c1 {
    %kt = arith.index_cast %i : index to i32
    hexkl.micro_hmx_copy_submatrix_to_f16(
        %hmx, %zero, %activation, %zero, %kt, %cols, %cols)
        : memref<?xi8, 1>, i32, memref<64x64xf16>, i32, i32, i32, i32
    hexkl.micro_hmx_rm_to_ah_f16(%hmx, %zero, %zero)
        : memref<?xi8, 1>, i32, i32
    hexkl.micro_hmx_rm_to_wh_f16(
        %hmx, %zero, %weight, %kt, %zero, %cols)
        : memref<?xi8, 1>, i32, memref<64x64xf16>, i32, i32, i32
    hexkl.micro_hmx_mm_f16(%hmx, %zero, %zero)
        : memref<?xi8, 1>, i32, i32
  }
  return
}

// CHECK-LABEL: func.func @p2c_hmx_micro
// CHECK-SAME: alps.p2c.activation_sites = 1
// CHECK-SAME: alps.p2c.proven_eliminated_physical_bytes = 0
// CHECK-SAME: alps.p2c.replaced_ir_ops = 3
// CHECK-SAME: alps.p2c.weight_sites = 1
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.p2c.fused_transform_transfer
// CHECK-SAME: alps.p2c.kind = "hmx_activation"
// CHECK-SAME: alps.p2c.synchronous
// CHECK-SAME: layout_transform = 2 : i32
// CHECK-SAME: lookahead = 0
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.p2c.fused_transform_transfer
// CHECK-SAME: alps.p2c.kind = "hmx_weight"
// CHECK-SAME: alps.p2c.synchronous
// CHECK-SAME: layout_transform = 1 : i32
// CHECK-SAME: lookahead = 0
// CHECK-NOT: hexkl.micro_hmx_copy_submatrix_to_f16
// CHECK-NOT: hexkl.micro_hmx_rm_to_ah_f16
// CHECK-NOT: hexkl.micro_hmx_rm_to_wh_f16
// CHECK: hexkl.micro_hmx_mm_f16
