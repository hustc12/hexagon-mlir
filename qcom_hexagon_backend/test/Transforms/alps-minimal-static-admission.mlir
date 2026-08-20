// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-minimal-static-admission{page-bytes=4096 vtcm-budget-bytes=8192 min-l2-bytes=64 min-dma-bytes=4096 min-overlap-ops=0}))' \
// RUN:   2>&1 | FileCheck %s --check-prefix=ADMIT
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-minimal-static-admission{min-l2-bytes=64 min-overlap-ops=0},prefetch-insert{enable-kv-cache-prefetch=true kv-cache-only=true require-alps-admission=true}))' \
// RUN:   | FileCheck %s --check-prefix=MATERIALIZE
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-minimal-static-admission{min-dma-bytes=2048 min-overlap-ops=2 enable-p3-exact-readiness=true}))' \
// RUN:   2>&1 | FileCheck %s --check-prefix=P3B

#q = affine_map<(b, m, n, k) -> (b, m, k)>
#k = affine_map<(b, m, n, k) -> (b, n, k)>
#o = affine_map<(b, m, n, k) -> (b, m, n)>

func.func @p2d_kv(
    %q: memref<1x32x32xf16>,
    %persistent_k: memref<1x32x32xf16>,
    %score: memref<1x32x32xf16>) {
  linalg.generic {
      indexing_maps = [#q, #k, #o],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      omni_fetch.kv_cache_operand = 1 : i64,
      omni_fetch.kv_cache_role = "key",
      omni_fetch.kv_cache_layout = "bshd"}
      ins(%q, %persistent_k : memref<1x32x32xf16>, memref<1x32x32xf16>)
      outs(%score : memref<1x32x32xf16>) {
    ^bb0(%a: f16, %b: f16, %acc: f16):
      %mul = arith.mulf %a, %b : f16
      %sum = arith.addf %acc, %mul : f16
      linalg.yield %sum : f16
  }

  %produced_k = memref.alloc() : memref<1x32x32xf16>
  linalg.generic {
      indexing_maps = [#q, #k, #o],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      omni_fetch.kv_cache_operand = 1 : i64,
      omni_fetch.kv_cache_role = "value",
      omni_fetch.kv_cache_layout = "bshd"}
      ins(%q, %produced_k : memref<1x32x32xf16>, memref<1x32x32xf16>)
      outs(%score : memref<1x32x32xf16>) {
    ^bb0(%a: f16, %b: f16, %acc: f16):
      %mul = arith.mulf %a, %b : f16
      %sum = arith.addf %acc, %mul : f16
      linalg.yield %sum : f16
  }
  memref.dealloc %produced_k : memref<1x32x32xf16>
  return
}

// ADMIT: [ALPS-P2D-SITE] function=p2d_kv
// ADMIT-SAME: kind=attention_kv_stream action=l2_hint
// ADMIT-SAME: reason=persistent_page_safe_stream
// ADMIT-SAME: materialize=1
// ADMIT: [ALPS-P2D-SITE] function=p2d_kv
// ADMIT-SAME: kind=attention_kv_stream action=native
// ADMIT-SAME: reason=source_not_entry_persistent
// ADMIT: [ALPS-P2D-SUMMARY] function=p2d_kv candidates=2
// ADMIT-SAME: native=1 l2_hint=1
// ADMIT-SAME: rejected=1 materialized=1

// MATERIALIZE-LABEL: func.func @p2d_kv
// MATERIALIZE-SAME: alps.p2d.l2_hint = 1
// MATERIALIZE-SAME: alps.p2d.materialized = 1
// MATERIALIZE-SAME: alps.p2d.native = 1
// MATERIALIZE-COUNT-1: omni_fetch.l2_hint

func.func @p2d_hexkl(
    %hmx: memref<?xi8, 1>,
    %weight: memref<256x256xf16>,
    %activation: memref<256x256xf16>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %cols = arith.constant 256 : i32
  scf.for %i = %c0 to %c8 step %c1 {
    %kt = arith.index_cast %i : index to i32
    hexkl.micro_hmx_copy_submatrix_to_f16(
        %hmx, %zero, %activation, %zero, %kt, %cols, %cols)
        : memref<?xi8, 1>, i32, memref<256x256xf16>, i32, i32, i32, i32
    hexkl.micro_hmx_rm_to_ah_f16(%hmx, %zero, %zero)
        : memref<?xi8, 1>, i32, i32
    hexkl.micro_hmx_rm_to_wh_f16(
        %hmx, %zero, %weight, %kt, %zero, %cols)
        : memref<?xi8, 1>, i32, memref<256x256xf16>, i32, i32, i32
    hexkl.micro_hmx_mm_f16(%hmx, %zero, %zero)
        : memref<?xi8, 1>, i32, i32
  }
  return
}

// ADMIT: [ALPS-P2D-SITE] function=p2d_hexkl
// ADMIT-SAME: kind=hmx_activation_transform action=native
// ADMIT-SAME: reason=sync_has_zero_proven_byte_reduction
// ADMIT: [ALPS-P2D-SITE] function=p2d_hexkl
// ADMIT-SAME: kind=hmx_weight_transform action=native
// ADMIT-SAME: reason=below_dma_byte_threshold
// ADMIT: [ALPS-P2D-SUMMARY] function=p2d_hexkl candidates=2
// ADMIT-SAME: native=2

// P3B: [ALPS-P2D-SITE] function=p2d_hexkl ordinal=8
// P3B-SAME: kind=hmx_weight_transform action=dma_vtcm_async
// P3B-SAME: reason=p3_exact_weight_pipeline
// P3B-SAME: materialize=1
// P3B: [ALPS-P2D-SUMMARY] function=p2d_hexkl candidates=2
// P3B-SAME: native=1
// P3B-SAME: dma_vtcm_async=1
