// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-insert{lookahead=2 enable-layout-aware=true}))' \
// RUN:   | FileCheck %s

func.func @short_weight_and_activation(
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

// CHECK-LABEL: func.func @short_weight_and_activation
// CHECK-SAME: alps.cost_async_sites = 0
// CHECK-SAME: alps.cost_native_sites = 1
// CHECK-SAME: alps.cost_sync_sites = 1
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.transform_mode = "sync"
// CHECK-SAME: layout_transform = 2 : i32
// CHECK: hexkl.micro_hmx_rm_to_wh_f16
// CHECK-SAME: alps.transform_mode = "native"

func.func @long_weight_persistent_candidate(
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

// CHECK-LABEL: func.func @long_weight_persistent_candidate
// CHECK-SAME: alps.cost_async_sites = 1
// CHECK-SAME: alps.cost_persistent_candidates = 1
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.persistent_candidate
// CHECK-SAME: alps.transform_mode = "async"
