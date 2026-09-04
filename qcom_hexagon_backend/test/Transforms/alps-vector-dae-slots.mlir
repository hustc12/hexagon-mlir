// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-dma-to-vtcm=true exact-weight-lookahead=2}))' \
// RUN:   | FileCheck %s

// A depth-two exact V-DAE schedule needs three non-aliasing HMX weight slots.
// Decomposition expresses the selected slot as kt % 3; PrefetchInsert later
// derives the future destination from this same ring mapping.
func.func @depth_two_wh_ring(%lhs: memref<32x64xf16>,
                             %rhs: memref<64x64xf16>,
                             %out: memref<32x64xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<32x64xf16>, memref<64x64xf16>)
               outs(%out : memref<32x64xf16>)
  return
}

// CHECK-LABEL: func.func @depth_two_wh_ring
// CHECK: %[[THREE:.*]] = arith.constant 3 : i32
// CHECK: %[[SLOT:.*]] = arith.remui {{.*}}, %[[THREE]] : i32
// CHECK: %[[SLOT_BYTES:.*]] = arith.muli %[[SLOT]], {{.*}} : i32
// CHECK: %[[WEIGHT_OFF:.*]] = arith.addi %[[SLOT_BYTES]], {{.*}} : i32
// CHECK: hexkl.micro_hmx_rm_to_wh_f16({{.*}}, %[[WEIGHT_OFF]],
// CHECK: hexkl.micro_hmx_mm_f16({{.*}}, {{.*}}, %[[WEIGHT_OFF]])
