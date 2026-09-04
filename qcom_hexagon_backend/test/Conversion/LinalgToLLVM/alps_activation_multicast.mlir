// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(schedule-matmul-for-hvx{enable-activation-multicast=true}))' | FileCheck %s

// CHECK-LABEL: func.func @sibling_projections
// CHECK-SAME: alps.n2_admitted = 1
// CHECK-SAME: alps.n2_candidates = 1
// CHECK-SAME: alps.n2_estimated_vector_activation_bytes_saved = 16384
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%arg0, %arg1, %arg2 : tensor<128x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>)
// CHECK-SAME: outs(%arg3, %arg4 : tensor<128x64xf32>, tensor<128x64xf32>)
// CHECK-SAME: alps.n2_activation_multicast
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.yield {{.*}}, {{.*}} : f32, f32
func.func @sibling_projections(
    %a: tensor<128x64xf16>,
    %b0: tensor<64x64xf16>, %b1: tensor<64x64xf16>,
    %init0: tensor<128x64xf32>,
    %init1: tensor<128x64xf32>)
    -> (tensor<128x64xf32>, tensor<128x64xf32>) {
  %0 = linalg.matmul
      ins(%a, %b0 : tensor<128x64xf16>, tensor<64x64xf16>)
      outs(%init0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %1 = linalg.matmul
      ins(%a, %b1 : tensor<128x64xf16>, tensor<64x64xf16>)
      outs(%init1 : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0, %1 : tensor<128x64xf32>, tensor<128x64xf32>
}

// Moving the first projection past an intervening consumer is illegal.
// CHECK-LABEL: func.func @intervening_use
// CHECK-SAME: alps.n2_admitted = 0
// CHECK-SAME: alps.n2_candidates = 1
// CHECK-NOT: alps.n2_activation_multicast
// CHECK: linalg.generic
// CHECK: linalg.add
// CHECK: linalg.generic
func.func @intervening_use(
    %a: tensor<128x64xf16>,
    %b0: tensor<64x64xf16>, %b1: tensor<64x64xf16>,
    %init0: tensor<128x64xf32>,
    %init1: tensor<128x64xf32>)
    -> (tensor<128x64xf32>, tensor<128x64xf32>) {
  %0 = linalg.matmul
      ins(%a, %b0 : tensor<128x64xf16>, tensor<64x64xf16>)
      outs(%init0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %used = linalg.add
      ins(%0, %0 : tensor<128x64xf32>, tensor<128x64xf32>)
      outs(%init0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %1 = linalg.matmul
      ins(%a, %b1 : tensor<128x64xf16>, tensor<64x64xf16>)
      outs(%init1 : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %used, %1 : tensor<128x64xf32>, tensor<128x64xf32>
}
