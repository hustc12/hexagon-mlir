// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion))' | FileCheck %s

#map = affine_map<(d0) -> (d0)>

module {
  // Semantic identity alone must preserve the native fusion decision.
  func.func @semantic_only(%x: tensor<16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
    %empty = tensor.empty() : tensor<16xf32>
    %producer = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel"],
        omni_fetch.kv_cache_role = "key"
      } ins(%x : tensor<16xf32>) outs(%empty : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %square = arith.mulf %in, %in : f32
      linalg.yield %square : f32
    } -> tensor<16xf32>
    %consumer = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel"]
      } ins(%producer : tensor<16xf32>) outs(%out : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %sum = arith.addf %in, %in : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    return %consumer : tensor<16xf32>
  }

  // The independent policy marker, rather than K/V semantics, protects the
  // producer/consumer topology.
  func.func @fusion_boundary(%x: tensor<16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
    %empty = tensor.empty() : tensor<16xf32>
    %producer = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel"],
        omni_fetch.kv_cache_role = "key",
        alps.kv_elementwise_fusion_boundary
      } ins(%x : tensor<16xf32>) outs(%empty : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %square = arith.mulf %in, %in : f32
      linalg.yield %square : f32
    } -> tensor<16xf32>
    %consumer = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel"]
      } ins(%producer : tensor<16xf32>) outs(%out : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %sum = arith.addf %in, %in : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    return %consumer : tensor<16xf32>
  }

  // Multi-use fusion normally folds the producer into the first dominating
  // consumer after the ordinary one-use fusion driver declines it.  The P0b
  // multi-use marker keeps that second driver disabled for this function.
  func.func @multi_use_boundary(%x: tensor<16xf32>, %out0: tensor<16xf32>,
                                %out1: tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>) {
    %empty = tensor.empty() : tensor<16xf32>
    %producer = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel"],
        alps.kv_multi_use_fusion_boundary
      } ins(%x : tensor<16xf32>) outs(%empty : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %square = arith.mulf %in, %in : f32
      linalg.yield %square : f32
    } -> tensor<16xf32>
    %consumer0 = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel"]
      } ins(%producer : tensor<16xf32>) outs(%out0 : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %sum = arith.addf %in, %in : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    %consumer1 = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel"]
      } ins(%producer : tensor<16xf32>) outs(%out1 : tensor<16xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %difference = arith.subf %in, %in : f32
      linalg.yield %difference : f32
    } -> tensor<16xf32>
    return %consumer0, %consumer1 : tensor<16xf32>, tensor<16xf32>
  }
}

// CHECK-LABEL: func.func @semantic_only
// CHECK-COUNT-1: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: return

// CHECK-LABEL: func.func @fusion_boundary
// CHECK: linalg.generic
// CHECK-SAME: alps.kv_elementwise_fusion_boundary
// CHECK: linalg.generic
// CHECK: return

// CHECK-LABEL: func.func @multi_use_boundary
// CHECK-COUNT-3: linalg.generic
// CHECK: return
