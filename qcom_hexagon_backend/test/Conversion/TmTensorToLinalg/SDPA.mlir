// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor{emit-kv-cache-metadata=true}))' %s | FileCheck %s
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor{emit-kv-cache-metadata=true emit-kv-fusion-boundary=true}))' %s | FileCheck %s --check-prefix=BOUNDARY
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor{emit-kv-cache-metadata=true emit-kv-elementwise-fusion-boundary=true}))' %s | FileCheck %s --check-prefix=P0B-ELEMENTWISE
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor{emit-kv-cache-metadata=true emit-kv-multi-use-fusion-boundary=true}))' %s | FileCheck %s --check-prefix=P0B-MULTI
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor{emit-kv-cache-metadata=true emit-kv-split-reduction-boundary=true}))' %s | FileCheck %s --check-prefix=P0B-SPLIT
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(hexagon-lower-tm-tensor))' %s | FileCheck %s --check-prefix=NO-KV

func.func @SDPA(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x4x8xf32>,
                            %arg2: tensor<2x4x8xf32>, %arg3: tensor<2x4x4xf32>) -> tensor<2x4x8xf32> {
  %0 = tensor.empty() : tensor<2x4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  %2 = tm_tensor.attention
         ins(%arg0, %arg1, %arg2, %arg3
             : tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x4xf32>)
         outs(%1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %2 : tensor<2x4x8xf32>
}

// A complete, model-independent consumer-formation proof subsumes the K/V
// topology contract. The attention still lowers, but item-7 must not attach
// fusion/tiling metadata to it.
func.func @SDPA_covered(
    %arg0: tensor<2x4x8xf32>, %arg1: tensor<2x4x8xf32>,
    %arg2: tensor<2x4x8xf32>, %arg3: tensor<2x4x4xf32>)
    -> tensor<2x4x8xf32> attributes {
      alps.p2e.demands = 4 : i64,
      alps.p2e.native = 0 : i64,
      alps.p2e.producer_direct = 4 : i64} {
  %0 = tensor.empty() : tensor<2x4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x8xf32>)
      -> tensor<2x4x8xf32>
  %2 = tm_tensor.attention
         ins(%arg0, %arg1, %arg2, %arg3
             : tensor<2x4x8xf32>, tensor<2x4x8xf32>,
               tensor<2x4x8xf32>, tensor<2x4x4xf32>)
         outs(%1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %2 : tensor<2x4x8xf32>
}

// CHECK-LABEL: func.func @SDPA
// CHECK-NOT: alps.kv_fusion_boundary
// CHECK: %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.353553385 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 0.000000e+00 : f32

// CHECK: %[[TRANSPOSED:.*]] = linalg.transpose ins(%arg1 : tensor<2x4x8xf32>) outs(%{{.*}} : tensor<2x8x4xf32>)
// CHECK-SAME:                  permutation = [0, 2, 1]

// CHECK: %[[QK:.*]] = linalg.batch_matmul
// CHECK-SAME:                 {omni_fetch.kv_cache_operand = 1 : i64, omni_fetch.kv_cache_role = "key"}
// CHECK-SAME:                 ins(%arg0, %[[TRANSPOSED]] : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

// CHECK: %[[SCALED:.*]] = linalg.generic {{.*}} ins(%[[QK]] : tensor<2x4x4xf32>) outs(%{{.*}} : tensor<2x4x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %{{.*}} = arith.mulf %[[IN]], %[[CST_0]] : f32

// CHECK: %[[QK_BIAS:.*]] = linalg.add ins(%[[SCALED]], %arg3 : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

// CHECK: %[[REDUCED:.*]] = linalg.reduce ins(%[[QK_BIAS]] : tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4xf32>) dimensions = [2]
// CHECK-NEXT: (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK-NEXT: %{{.*}} = arith.maximumf %[[IN]], %[[INIT]] : f32

// CHECK: %[[SUBBED:.*]] = linalg.generic {{.*}} ins(%[[QK_BIAS]], %[[REDUCED]] : tensor<2x4x4xf32>, tensor<2x4xf32>)
// CHECK-SAME:                  outs(%{{.*}} : tensor<2x4x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[IN_3:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %{{.*}} = arith.subf %[[IN]], %[[IN_3]] : f32

// CHECK: %[[EXP:.*]] = linalg.exp ins(%[[SUBBED]] : tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

// CHECK: %[[REDUCED_2:.*]] = linalg.reduce ins(%[[EXP]] : tensor<2x4x4xf32>)
// CHECK-SAME:                 outs(%{{.*}} : tensor<2x4xf32>) dimensions = [2]
// CHECK-NEXT: (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK-NEXT: %{{.*}} = arith.addf %[[IN]], %[[INIT]] : f32

// CHECK: %[[SOFTMAX:.*]] = linalg.generic {{.*}} ins(%[[EXP]], %[[REDUCED_2]] : tensor<2x4x4xf32>, tensor<2x4xf32>)
// CHECK-SAME:                  outs(%{{.*}} : tensor<2x4x4xf32>) {
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[IN_3:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT: %{{.*}} = arith.divf %[[IN]], %[[IN_3]] : f32

// CHECK: %[[RESULT:.*]] = linalg.batch_matmul
// CHECK-SAME:                {omni_fetch.kv_cache_operand = 1 : i64, omni_fetch.kv_cache_role = "value"}
// CHECK-SAME:                ins(%[[SOFTMAX]], %arg2 : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
// CHECK-SAME:                outs(%{{.*}} : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK: return %[[RESULT]] : tensor<2x4x8xf32>

// BOUNDARY-LABEL: func.func @SDPA
// BOUNDARY-COUNT-2: alps.kv_fusion_boundary
// BOUNDARY: return

// P0B-ELEMENTWISE-LABEL: func.func @SDPA
// P0B-ELEMENTWISE-COUNT-2: alps.kv_elementwise_fusion_boundary
// P0B-ELEMENTWISE-NOT: alps.kv_multi_use_fusion_boundary
// P0B-ELEMENTWISE-NOT: alps.kv_split_reduction_boundary
// P0B-ELEMENTWISE: return

// P0B-MULTI-LABEL: func.func @SDPA
// P0B-MULTI-COUNT-2: alps.kv_multi_use_fusion_boundary
// P0B-MULTI-NOT: alps.kv_elementwise_fusion_boundary
// P0B-MULTI-NOT: alps.kv_split_reduction_boundary
// P0B-MULTI: return

// P0B-SPLIT-LABEL: func.func @SDPA
// P0B-SPLIT-COUNT-2: alps.kv_split_reduction_boundary
// P0B-SPLIT-NOT: alps.kv_elementwise_fusion_boundary
// P0B-SPLIT-NOT: alps.kv_multi_use_fusion_boundary
// P0B-SPLIT: return

// NO-KV-LABEL: func.func @SDPA
// NO-KV-NOT: omni_fetch.kv_cache_role
// NO-KV: return

// CHECK-LABEL: func.func @SDPA_covered
// CHECK-SAME: alps.kv_topology_admission = "covered_by_consumer_formation"
// CHECK-NOT: omni_fetch.kv_cache_role
// CHECK: return
