// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(alps-exact-readiness))' 2>&1 | FileCheck %s

#weight = #alps.layout_transform<hmx_weight>

func.func @p3a_contract(%src: memref<64x64xf16>, %dst: memref<?xi8, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %cols = arith.constant 64 : i32
  scf.for %tile = %c0 to %c4 step %c1 {
    %tile_i32 = arith.index_cast %tile : index to i32
    alps.prefetch_in_situ %src, %dst
        tile_params(%tile_i32, %zero, %cols, %zero : i32, i32, i32, i32)
        {alps.p2d.action = "dma_vtcm_async",
         layout_transform = #weight,
         lookahead = 1 : i32}
        : memref<64x64xf16>, memref<?xi8, 1>
    alps.prefetch_in_situ %src, %dst
        tile_params(%tile_i32, %zero, %cols, %zero : i32, i32, i32, i32)
        {layout_transform = #weight,
         lookahead = 1 : i32}
        : memref<64x64xf16>, memref<?xi8, 1>
  }
  return
}

// CHECK: [ALPS-P3A-SITE] function=p3a_contract layout=1 lookahead=1 accepted=1 reason=exact_descriptor_contract
// CHECK: [ALPS-P3A-SITE] function=p3a_contract layout=1 lookahead=1 accepted=0 reason=not_p2d_dma_admitted
// CHECK: [ALPS-P3A-SUMMARY] function=p3a_contract async_candidates=2 exact_contracts=1 rejected=1
// CHECK-LABEL: func.func @p3a_contract
// CHECK-SAME: alps.p3a.async_candidates = 2
// CHECK-SAME: alps.p3a.exact_contracts = 1
// CHECK: alps.p3a.exact_readiness = true
// CHECK: alps.p3a.exact_readiness = false
