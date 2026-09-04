// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-minimal-static-admission{min-dma-bytes=2048 min-overlap-ops=2 enable-p3-exact-readiness=true},prefetch-insert{lookahead=1 enable-layout-aware=true enable-two-dim-pipeline=true enable-alps-exact-overlap=true},alps-exact-readiness))' \
// RUN:   2>&1 | FileCheck %s

func.func @p3b_weight(
    %hmx: memref<?xi8, 1>, %weight: memref<256x256xf16>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %cols = arith.constant 256 : i32
  scf.for %i = %c0 to %c8 step %c1 {
    %kt = arith.index_cast %i : index to i32
    %phase = arith.remui %kt, %cols : i32
    hexkl.micro_hmx_rm_to_wh_f16(
        %hmx, %zero, %weight, %kt, %zero, %cols)
        : memref<?xi8, 1>, i32, memref<256x256xf16>, i32, i32, i32
    hexkl.micro_hmx_mm_f16(%hmx, %zero, %zero)
        : memref<?xi8, 1>, i32, i32
  }
  return
}

// CHECK: [ALPS-P2D-SITE] function=p3b_weight
// CHECK-SAME: action=dma_vtcm_async reason=p3_exact_weight_pipeline
// CHECK: [ALPS-P3A-SUMMARY] function=p3b_weight async_candidates=1 exact_contracts=1 rejected=0
// CHECK-LABEL: func.func @p3b_weight
// CHECK-SAME: alps.p3a.exact_contracts = 1
// CHECK: %[[CTX:.*]] = alps.invocation_begin
// CHECK: alps.exact_weight_consume %[[CTX]]
// CHECK: alps.exact_weight_kick %[[CTX]]
// CHECK: hexkl.micro_hmx_mm_f16
// CHECK: alps.exact_weight_release %[[CTX]]
// CHECK: alps.invocation_end %[[CTX]]
// CHECK-NOT: alps.create_sem
// CHECK-NOT: alps.wait
// CHECK-NOT: alps.signal
