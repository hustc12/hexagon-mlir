// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(layout-ops-elimination))' \
// RUN:   | FileCheck %s

#weight = #alps.layout_transform<hmx_weight>

func.func @layout_value_identity(
    %src: memref<64x64xf16>,
    %dst0: memref<32x32xf16, 1>,
    %dst1: memref<32x32xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %tile = memref.subview %src[%c0, %c0] [32, 32] [1, 1]
      : memref<64x64xf16> to memref<32x32xf16, strided<[64, 1], offset: ?>>
  scf.for %i = %c0 to %c4 step %c1 {
    alps.prefetch_in_situ %tile, %dst0
        {layout_transform = #weight, lookahead = 0 : i32}
        : memref<32x32xf16, strided<[64, 1], offset: ?>>, memref<32x32xf16, 1>
  }
  alps.prefetch_in_situ %tile, %dst1
      {layout_transform = #weight, lookahead = 0 : i32}
      : memref<32x32xf16, strided<[64, 1], offset: ?>>, memref<32x32xf16, 1>
  return
}

// CHECK-LABEL: func.func @layout_value_identity
// CHECK-SAME: alps.layout_prefetch_instances = 2
// CHECK-SAME: alps.layout_reusable_sites = 1
// CHECK-SAME: alps.layout_value_sites = 1
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.layout_estimated_executions = 5
// CHECK-SAME: alps.layout_site_occurrences = 2
// CHECK-SAME: alps.layout_source_kind = "argument"
// CHECK-SAME: alps.layout_value_id = 0
// CHECK: alps.prefetch_in_situ
// CHECK-SAME: alps.layout_estimated_executions = 5
// CHECK-SAME: alps.layout_site_occurrences = 2
// CHECK-SAME: alps.layout_source_kind = "argument"
// CHECK-SAME: alps.layout_value_id = 0
