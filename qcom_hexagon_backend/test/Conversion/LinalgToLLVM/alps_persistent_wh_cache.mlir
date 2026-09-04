// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-insert{lookahead=2 enable-layout-aware=true enable-persistent-wh-cache=true}))' \
// RUN:   | FileCheck %s

func.func @persistent_weight(
    %hmx: memref<?xi8, 1>,
    %weight: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %cols = arith.constant 64 : i32
  scf.for %outer = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c4 step %c1 {
      %kt = arith.index_cast %i : index to i32
      hexkl.micro_hmx_rm_to_wh_f16(
          %hmx, %zero, %weight, %kt, %zero, %cols)
          : memref<?xi8, 1>, i32, memref<64x64xf16>, i32, i32, i32
      hexkl.micro_hmx_mm_f16(%hmx, %zero, %zero)
          : memref<?xi8, 1>, i32, i32
    }
  }
  return
}

// CHECK-LABEL: func.func @persistent_weight
// CHECK-SAME: alps.cost_persistent_sites = 1
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.transform_mode = "persistent"
// CHECK-SAME: lookahead = -1
// CHECK-NOT: hexkl.micro_hmx_rm_to_wh_f16
