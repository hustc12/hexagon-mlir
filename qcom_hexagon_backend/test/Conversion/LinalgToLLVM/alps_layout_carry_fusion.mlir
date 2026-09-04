// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(layout-ops-elimination))' \
// RUN:   | FileCheck %s

#activation = #alps.layout_transform<hmx_activation>

func.func @carry_unit_batch_activation(
    %producer: memref<1x128x64xf16>,
    %hmx: memref<?xi8, 1>) {
  %view = memref.collapse_shape %producer [[0, 1], [2]]
      : memref<1x128x64xf16> into memref<128x64xf16>
  %zero = arith.constant 0 : i32
  %rows = arith.constant 128 : i32
  %cols = arith.constant 64 : i32
  alps.prefetch_in_situ %view, %hmx
      tile_params(%zero, %zero, %cols, %zero, %zero, %rows
          : i32, i32, i32, i32, i32, i32)
      {layout_transform = #activation, lookahead = 0 : i32}
      : memref<128x64xf16>, memref<?xi8, 1>
  return
}

// CHECK-LABEL: func.func @carry_unit_batch_activation
// CHECK-SAME: alps.layout_carried_sites = 1
// CHECK-SAME: alps.layout_carried_views = 1
// CHECK-NOT: memref.collapse_shape
// CHECK: alps.prefetch_in_situ %arg0, %arg1
// CHECK-SAME: alps.layout_carried_from_producer
// CHECK-SAME: alps.layout_carried_view_depth = 1
// CHECK-SAME: : memref<1x128x64xf16>, memref<?xi8, 1>

func.func @keep_dynamic_source(
    %producer: memref<?x128x64xf16>,
    %hmx: memref<?xi8, 1>) {
  %view = memref.collapse_shape %producer [[0, 1], [2]]
      : memref<?x128x64xf16> into memref<?x64xf16>
  %zero = arith.constant 0 : i32
  %rows = arith.constant 256 : i32
  %cols = arith.constant 64 : i32
  alps.prefetch_in_situ %view, %hmx
      tile_params(%zero, %zero, %cols, %zero, %zero, %rows
          : i32, i32, i32, i32, i32, i32)
      {layout_transform = #activation, lookahead = 0 : i32}
      : memref<?x64xf16>, memref<?xi8, 1>
  return
}

// CHECK-LABEL: func.func @keep_dynamic_source
// CHECK-SAME: alps.layout_carried_sites = 0
// CHECK: memref.collapse_shape
// CHECK: alps.prefetch_in_situ %{{.*}}, %arg1
// CHECK-NOT: alps.layout_carried_from_producer
