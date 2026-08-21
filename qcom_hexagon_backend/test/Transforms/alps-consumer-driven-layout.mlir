// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-consumer-driven-layout))' \
// RUN:   | FileCheck %s

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// The terminal linalg consumer explicitly demands [N,M,K].  Because K stays
// innermost, P2e can make the elementwise producer write that representation
// directly without creating [M,N,K] and physically transposing it.
func.func @direct_hvx_consumer(%input: tensor<8x16x32xf16>,
                               %out: tensor<16x8x32xf16>)
    -> tensor<16x8x32xf16> {
  %tmp = tensor.empty() : tensor<8x16x32xf16>
  %produced = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<8x16x32xf16>)
      outs(%tmp : tensor<8x16x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<8x16x32xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<8x16x32xf16>)
      outs(%out : tensor<16x8x32xf16>)
      permutation = [1, 0, 2]
  %consumer_out = tensor.empty() : tensor<16x8x32xf16>
  %result = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<16x8x32xf16>)
      outs(%consumer_out : tensor<16x8x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<16x8x32xf16>
  return %result : tensor<16x8x32xf16>
}

// CHECK: affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: func.func @direct_hvx_consumer
// CHECK-SAME: alps.p2e.demands = 1
// CHECK-SAME: alps.p2e.eliminated_materialization_bytes = 8192
// CHECK-SAME: alps.p2e.hvx_consumers = 1
// CHECK-SAME: alps.p2e.producer_direct = 1
// CHECK-NOT: linalg.transpose

// Moving the innermost dimension would turn a unit-stride stream into a
// strided one.  The contract remains auditable, but native materialization is
// retained.
func.func @reject_strided_consumer(%input: tensor<8x16x32xf16>,
                                   %out: tensor<8x32x16xf16>)
    -> tensor<8x32x16xf16> {
  %tmp = tensor.empty() : tensor<8x16x32xf16>
  %produced = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<8x16x32xf16>)
      outs(%tmp : tensor<8x16x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<8x16x32xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<8x16x32xf16>)
      outs(%out : tensor<8x32x16xf16>)
      permutation = [0, 2, 1]
  %consumer_out = tensor.empty() : tensor<8x32x16xf16>
  %result = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<8x32x16xf16>)
      outs(%consumer_out : tensor<8x32x16xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<8x32x16xf16>
  return %result : tensor<8x32x16xf16>
}

// CHECK-LABEL: func.func @reject_strided_consumer
// CHECK-SAME: alps.p2e.native = 1
// CHECK: linalg.transpose

func.func @direct_expanded_hvx_consumer(%input: tensor<1x4x12xf16>,
                                        %out: tensor<1x3x4x4xf16>)
    -> tensor<1x3x4x4xf16> {
  %tmp = tensor.empty() : tensor<1x4x12xf16>
  %produced = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x4x12xf16>)
      outs(%tmp : tensor<1x4x12xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<1x4x12xf16>
  %expanded = tensor.expand_shape %produced [[0], [1], [2, 3]]
      output_shape [1, 4, 3, 4]
      : tensor<1x4x12xf16> into tensor<1x4x3x4xf16>
  %transposed = linalg.transpose
      ins(%expanded : tensor<1x4x3x4xf16>)
      outs(%out : tensor<1x3x4x4xf16>)
      permutation = [0, 2, 1, 3]
  %consumer_out = tensor.empty() : tensor<1x3x4x4xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<1x3x4x4xf16>)
      outs(%consumer_out : tensor<1x3x4x4xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<1x3x4x4xf16>
  return %result : tensor<1x3x4x4xf16>
}

// CHECK-LABEL: func.func @direct_expanded_hvx_consumer
// CHECK-SAME: alps.p2e.eliminated_materialization_bytes = 96
// CHECK-SAME: alps.p2e.producer_direct = 1
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: linalg.transpose
