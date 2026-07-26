// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-insert{lookahead=2 enable-layout-aware=true enable-persistent-wh-cache=true enable-two-dim-pipeline=true}))' \
// RUN:   | FileCheck %s

func.func @load_reshape_compute_pipeline(
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

// The current tile is synchronously bootstrapped only under the first-tile
// guard and uses the item-4 persistent cache (lookahead=-1).
// CHECK-LABEL: func.func @load_reshape_compute_pipeline
// CHECK-SAME: omni_fetch.cost_async_sites = 1
// CHECK-SAME: omni_fetch.cost_persistent_sites = 1
// CHECK: scf.if
// CHECK: omni_fetch.prefetch_in_situ
// CHECK-SAME: lookahead = -1
// CHECK-SAME: omni_fetch.transform_mode = "async"

// Every non-final iteration schedules the next tile. Six tile parameters
// carry row, column, source width, destination slot, stage slot, and stable
// cache site ID into the LOAD_PENDING -> LOAD_READY -> TRANSFORM_READY runtime.
// CHECK: scf.if
// CHECK: omni_fetch.prefetch_in_situ {{.*}} tile_params({{.*}} : i32, i32, i32, i32, i32, i32)
// CHECK-SAME: lookahead = 1
// CHECK-SAME: omni_fetch.transform_mode = "async"
// CHECK-NOT: hexkl.micro_hmx_rm_to_wh_f16
