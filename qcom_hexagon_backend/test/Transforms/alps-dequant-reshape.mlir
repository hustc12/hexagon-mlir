// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-insert{lookahead=2 enable-layout-aware=true enable-dequant-reshape=true}))' \
// RUN:   | FileCheck %s

func.func @w8_prefetch_dequant_wh(
    %hmx: memref<?xi8, 1>,
    %weight: memref<256x256xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0 : i32
  %cols = arith.constant 256 : i32
  scf.for %outer = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      %kt = arith.index_cast %i : index to i32
      hexkl.micro_hmx_rm_to_wh_f16(
          %hmx, %zero, %weight, %kt, %zero, %cols)
          : memref<?xi8, 1>, i32, memref<256x256xf16>, i32, i32, i32
      hexkl.micro_hmx_mm_f16(%hmx, %zero, %zero)
          : memref<?xi8, 1>, i32, i32
    }
  }
  return
}

// CHECK-LABEL: func.func @w8_prefetch_dequant_wh
// CHECK-SAME: alps.dequant_reshape_enabled = true
// CHECK-SAME: alps.dequant_reshape_sites = 1
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: tile_params({{.*}} : i32, i32, i32, i32, i32, i32)
// CHECK-SAME: alps.transform_mode = "sync"
// CHECK-SAME: layout_transform = 5 : i32
// CHECK-SAME: lookahead = 0
// CHECK-NOT: hexkl.micro_hmx_rm_to_wh_f16
