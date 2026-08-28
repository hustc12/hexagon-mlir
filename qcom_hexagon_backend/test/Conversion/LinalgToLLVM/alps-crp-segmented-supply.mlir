// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(omni-fetch-to-llvm)' | FileCheck %s

func.func @contiguous(
    %src: memref<1x4x16xf16, strided<[64, 16, 1], offset: ?>>) {
  omni_fetch.l2_hint %src {
    alps.p5f_b.requested_bytes = 128 : i64,
    alps.p5f_c.page_safe_segmented,
    alps.p5f_c.site_id = 0 : i64
  } : memref<1x4x16xf16, strided<[64, 16, 1], offset: ?>>
  return
}

func.func @segmented(
    %src: memref<1x4x16xf16, strided<[256, 32, 1], offset: ?>>) {
  omni_fetch.l2_hint %src {
    alps.p5f_b.requested_bytes = 128 : i64,
    alps.p5f_c.page_safe_segmented,
    alps.p5f_c.site_id = 1 : i64
  } : memref<1x4x16xf16, strided<[256, 32, 1], offset: ?>>
  return
}

// CHECK-LABEL: func.func @contiguous
// CHECK-COUNT-1: llvm.call @__omni_fetch_l2_hint_2d
// CHECK-LABEL: func.func @segmented
// CHECK-COUNT-1: llvm.call @__omni_fetch_l2_hint_segmented
