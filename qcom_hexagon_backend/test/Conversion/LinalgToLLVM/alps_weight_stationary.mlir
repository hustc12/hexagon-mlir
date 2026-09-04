// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(schedule-matmul-for-hvx{enable-weight-stationary=true}))' | FileCheck %s

// N1 admits a projection whose 128 positions can amortize each weight load.
// The transpose-equivalent matmul is scheduled as N x K x M, leaving M as
// the contiguous innermost HVX dimension.
// CHECK-LABEL: func.func @projection
// CHECK-SAME: attributes {
// CHECK-SAME: alps.n1_admitted = 1
// CHECK-SAME: alps.n1_candidates = 1
// CHECK-SAME: alps.n1_predicted_saved_bytes = 2646016
// CHECK: linalg.transpose
// CHECK-SAME: tensor<64x192xf16>
// CHECK-SAME: tensor<192x64xf16>
// CHECK: linalg.transpose
// CHECK-SAME: tensor<128x64xf16>
// CHECK-SAME: tensor<64x128xf16>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins({{.*}} : tensor<192x64xf16>, tensor<64x128xf16>)
// CHECK-SAME: outs({{.*}} : tensor<192x128xf32>)
// CHECK-SAME: alps.n1_mkn = array<i64: 128, 64, 192>
// CHECK-SAME: alps.n1_weight_stationary
// CHECK: linalg.transpose
// CHECK-SAME: tensor<192x128xf32>
// CHECK-SAME: tensor<128x192xf32>
func.func @projection(
    %a: tensor<128x64xf16>, %b: tensor<64x192xf16>,
    %init: tensor<128x192xf32>) -> tensor<128x192xf32> {
  %0 = linalg.matmul
      ins(%a, %b : tensor<128x64xf16>, tensor<64x192xf16>)
      outs(%init : tensor<128x192xf32>) -> tensor<128x192xf32>
  return %0 : tensor<128x192xf32>
}

// An attention-shaped K == M contraction belongs to N3 and must not be
// rewritten by N1.
// CHECK-LABEL: func.func @attention
// CHECK-SAME: attributes {
// CHECK-SAME: alps.n1_admitted = 0
// CHECK-SAME: alps.n1_candidates = 0
// CHECK-NOT: linalg.transpose
// CHECK-NOT: alps.n1_weight_stationary
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0, %arg1 : tensor<128x128xf16>, tensor<128x64xf16>)
func.func @attention(
    %a: tensor<128x128xf16>, %b: tensor<128x64xf16>,
    %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %0 = linalg.matmul
      ins(%a, %b : tensor<128x128xf16>, tensor<128x64xf16>)
      outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
