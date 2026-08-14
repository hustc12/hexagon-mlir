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
// CHECK-SAME: omni_fetch.kv_prefetch_bytes = 256
// CHECK-SAME: omni_fetch.kv_prefetch_hints = 4
// CHECK-SAME: omni_fetch.kv_prefetch_pages = 16
// CHECK-SAME: omni_fetch.kv_prefetch_sites = 2
// CHECK-SAME: omni_fetch.kv_rejected_produced_sites = 0
// CHECK: %[[K0:.+]] = memref.subview %arg1[0, 0, 0] [1, 1, 32] [1, 1, 1]
// CHECK: omni_fetch.l2_hint %[[K0]]
// CHECK: %[[K1:.+]] = memref.subview %arg1[1, 0, 0] [1, 1, 32] [1, 1, 1]
// CHECK: omni_fetch.l2_hint %[[K1]]
// CHECK: %[[V0:.+]] = memref.subview %arg3[0, 0, 0] [1, 1, 32] [1, 1, 1]
// CHECK: omni_fetch.l2_hint %[[V0]]
// CHECK: %[[V1:.+]] = memref.subview %arg3[1, 0, 0] [1, 1, 32] [1, 1, 1]
// CHECK: omni_fetch.l2_hint %[[V1]]

func.func @loop_hoist(
    %q: memref<1x32x32xf16>,
    %k: memref<1x32x32xf16>,
    %out: memref<1x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    linalg.generic {
        indexing_maps = [#q, #k, #o],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"],
        omni_fetch.kv_cache_operand = 1 : i64,
        omni_fetch.kv_cache_role = "key"}
        ins(%q, %k : memref<1x32x32xf16>, memref<1x32x32xf16>)
        outs(%out : memref<1x32x32xf16>) {
      ^bb0(%a: f16, %b: f16, %acc: f16):
        %mul = arith.mulf %a, %b : f16
        %sum = arith.addf %acc, %mul : f16
        linalg.yield %sum : f16
    }
  }
  return
}

// The K buffer is loop-invariant, so its single logical hint must be in the
// preheader rather than dynamically reissued four times.
// CHECK-LABEL: func.func @loop_hoist
// CHECK-SAME: omni_fetch.kv_hoisted_sites = 1
// CHECK-SAME: omni_fetch.kv_prefetch_bytes = 64
// CHECK: %[[HK:.+]] = memref.subview %arg1[0, 0, 0] [1, 1, 32]
// CHECK: omni_fetch.l2_hint %[[HK]]
// CHECK: scf.for
// CHECK-NOT: omni_fetch.l2_hint
