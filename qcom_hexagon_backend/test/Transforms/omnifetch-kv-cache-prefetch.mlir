// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(prefetch-insert{enable-kv-cache-prefetch=true kv-cache-page-tokens=32}))' \
// RUN:   | FileCheck %s

#q = affine_map<(b, m, n, k) -> (b, m, k)>
#k = affine_map<(b, m, n, k) -> (b, n, k)>
#v = affine_map<(b, m, n, k) -> (b, k, n)>
#o = affine_map<(b, m, n, k) -> (b, m, n)>

func.func @paged_kv(
    %q: memref<2x128x32xf16>,
    %k: memref<2x128x32xf16>,
    %prob: memref<2x128x128xf16>,
    %v: memref<2x128x32xf16>,
    %score: memref<2x128x128xf16>,
    %out: memref<2x128x32xf16>) {
  linalg.generic {
      indexing_maps = [#q, #k, #o],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      omni_fetch.kv_cache_operand = 1 : i64,
      omni_fetch.kv_cache_role = "key"}
      ins(%q, %k : memref<2x128x32xf16>, memref<2x128x32xf16>)
      outs(%score : memref<2x128x128xf16>) {
    ^bb0(%a: f16, %b: f16, %acc: f16):
      %mul = arith.mulf %a, %b : f16
      %sum = arith.addf %acc, %mul : f16
      linalg.yield %sum : f16
  }
  linalg.generic {
      indexing_maps = [#q, #v, #o],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      omni_fetch.kv_cache_operand = 1 : i64,
      omni_fetch.kv_cache_role = "value"}
      ins(%prob, %v : memref<2x128x128xf16>, memref<2x128x32xf16>)
      outs(%out : memref<2x128x32xf16>) {
    ^bb0(%a: f16, %b: f16, %acc: f16):
      %mul = arith.mulf %a, %b : f16
      %sum = arith.addf %acc, %mul : f16
      linalg.yield %sum : f16
  }
  return
}

// Two K streams and two V streams. Each 128-token stream has four logical
// 32-token pages, coalesced into one contiguous L2 hint per stream.
// CHECK-LABEL: func.func @paged_kv
// CHECK-SAME: omni_fetch.kv_direct_layout_sites = 2
// CHECK-SAME: omni_fetch.kv_prefetch_bytes = 32768
// CHECK-SAME: omni_fetch.kv_prefetch_hints = 4
// CHECK-SAME: omni_fetch.kv_prefetch_pages = 16
// CHECK-SAME: omni_fetch.kv_prefetch_sites = 2
// CHECK: %[[K0:.+]] = memref.subview %arg1[0, 0, 0] [1, 128, 32] [1, 1, 1]
// CHECK: omni_fetch.prefetch_in_situ %[[K0]], %[[K0]]
// CHECK-SAME: layout_transform = 4 : i32
// CHECK: %[[K1:.+]] = memref.subview %arg1[1, 0, 0] [1, 128, 32] [1, 1, 1]
// CHECK: omni_fetch.prefetch_in_situ %[[K1]], %[[K1]]
// CHECK: %[[V0:.+]] = memref.subview %arg3[0, 0, 0] [1, 128, 32] [1, 1, 1]
// CHECK: omni_fetch.prefetch_in_situ %[[V0]], %[[V0]]
// CHECK: %[[V1:.+]] = memref.subview %arg3[1, 0, 0] [1, 128, 32] [1, 1, 1]
// CHECK: omni_fetch.prefetch_in_situ %[[V1]], %[[V1]]
