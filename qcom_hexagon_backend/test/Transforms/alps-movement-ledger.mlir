// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(alps-movement-ledger{phase=test page-bytes=64 vtcm-budget-bytes=256}))' 2>&1 | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>

func.func @ledger(%src: memref<4x8xf32>, %dst: memref<8x4xf32>,
                  %tensor: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %view = memref.subview %src[0, 0] [4, 8] [1, 1]
    : memref<4x8xf32> to memref<4x8xf32, strided<[8, 1]>>
  %flat = memref.collapse_shape %view [[0, 1]]
    : memref<4x8xf32, strided<[8, 1]>> into memref<32xf32, strided<[1]>>
  %scratch = memref.alloc() : memref<4x8xf32>
  memref.copy %src, %scratch : memref<4x8xf32> to memref<4x8xf32>
  linalg.transpose ins(%src : memref<4x8xf32>)
                   outs(%dst : memref<8x4xf32>) permutation = [1, 0]
  %empty = tensor.empty() : tensor<4x8xf32>
  %candidate = linalg.generic {
      indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"],
      omni_fetch.kv_cache_role = "key",
      omni_fetch.kv_cache_operand = 0 : i64,
      alps.kv_elementwise_fusion_boundary
    } ins(%tensor : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  return %candidate : tensor<4x8xf32>
}

// CHECK: [ALPS-P1-SITE] phase=test function=ledger
// CHECK-SAME: kind=descriptor_view
// CHECK-SAME: op=memref.subview
// CHECK: [ALPS-P1-SITE] phase=test function=ledger
// CHECK-SAME: kind=descriptor_view
// CHECK-SAME: op=memref.collapse_shape
// CHECK-SAME: materialization_bytes=0 descriptor_only=1
// CHECK: [ALPS-P1-SITE] phase=test function=ledger
// CHECK-SAME: kind=allocation
// CHECK-SAME: op=memref.alloc
// CHECK-SAME: decision=capacity_only_not_movement
// CHECK: [ALPS-P1-SITE] phase=test function=ledger
// CHECK-SAME: kind=physical_copy
// CHECK-SAME: op=memref.copy
// CHECK-SAME: static_bytes=128 read_bytes=128 write_bytes=128 materialization_bytes=128
// CHECK: [ALPS-P1-SITE] phase=test function=ledger
// CHECK-SAME: kind=physical_layout_transform
// CHECK-SAME: op=linalg.transpose
// CHECK-SAME: pages=2
// CHECK: [ALPS-P1-SITE] phase=test function=ledger
// CHECK-SAME: kind=representation_candidate
// CHECK-SAME: op=linalg.generic kv_role=key
// CHECK-SAME: legal_actions=native+l2_hint+in_situ_sync+dma_vtcm_async
// CHECK-SAME: decision=candidate_static_vtcm_fit
// CHECK: [ALPS-P1-SUMMARY] phase=test function=ledger candidates=1 descriptor_sites=2 physical_transform_sites=1 copy_sites=1 alloc_sites=1
// CHECK-SAME: static_read_bytes=256 static_write_bytes=256 static_materialization_bytes=256 dynamic_sites=0
