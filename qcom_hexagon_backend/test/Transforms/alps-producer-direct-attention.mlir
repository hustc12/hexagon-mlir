// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(alps-producer-direct-attention))' | FileCheck %s

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#bias = affine_map<(d0, d1, d2) -> (d2)>

func.func @direct_q(%activation: tensor<1x3x8xf16>,
                    %bias: tensor<8xf16>) -> tensor<1x2x3x4xf16> {
  %flat_init = tensor.empty() : tensor<1x3x8xf16>
  %added = linalg.generic {
      indexing_maps = [#id3, #bias, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%activation, %bias : tensor<1x3x8xf16>, tensor<8xf16>)
      outs(%flat_init : tensor<1x3x8xf16>) {
    ^bb0(%a: f16, %b: f16, %out: f16):
      %sum = arith.addf %a, %b : f16
      linalg.yield %sum : f16
  } -> tensor<1x3x8xf16>
  %expanded = tensor.expand_shape %added [[0], [1], [2, 3]]
      output_shape [1, 3, 2, 4]
      : tensor<1x3x8xf16> into tensor<1x3x2x4xf16>
  %target = tensor.empty() : tensor<1x2x3x4xf16>
  %result = linalg.transpose ins(%expanded : tensor<1x3x2x4xf16>)
      outs(%target : tensor<1x2x3x4xf16>) permutation = [0, 2, 1, 3]
  return %result : tensor<1x2x3x4xf16>
}

// CHECK: #[[SRC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1 * 4 + d3)>
// CHECK: #[[BIAS:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 * 4 + d3)>
// CHECK-LABEL: func.func @direct_q
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: alps.p2b.eliminated_canonical_materialization_bytes = 48
// CHECK-SAME: alps.p2b.producer_direct_attention
// CHECK-SAME: alps.p2b.target_layout = "BHMD"
// CHECK: arith.addf
