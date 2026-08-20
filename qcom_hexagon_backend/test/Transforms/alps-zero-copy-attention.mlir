// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(alps-zero-copy-attention))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(alps-zero-copy-attention,alps-minimal-static-admission))' 2>&1 | FileCheck %s --check-prefix=ADMISSION

func.func @qk_layout(%q: tensor<1x3x2x4xf16>,
                     %k: tensor<1x5x2x4xf16>,
                     %init: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
  %qe = tensor.empty() : tensor<1x2x3x4xf16>
  %qt = linalg.transpose ins(%q : tensor<1x3x2x4xf16>)
      outs(%qe : tensor<1x2x3x4xf16>) permutation = [0, 2, 1, 3]
  %ke = tensor.empty() : tensor<1x2x4x5xf16>
  %kt = linalg.transpose ins(%k : tensor<1x5x2x4xf16>)
      outs(%ke : tensor<1x2x4x5xf16>) permutation = [0, 2, 3, 1]
  %qc = tensor.collapse_shape %qt [[0, 1], [2], [3]]
      : tensor<1x2x3x4xf16> into tensor<2x3x4xf16>
  %kc = tensor.collapse_shape %kt [[0, 1], [2], [3]]
      : tensor<1x2x4x5xf16> into tensor<2x4x5xf16>
  %result = linalg.batch_matmul
      ins(%qc, %kc : tensor<2x3x4xf16>, tensor<2x4x5xf16>)
      outs(%init : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
  return %result : tensor<2x3x5xf32>
}

// CHECK: #[[LHS:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d1, d0, d3)>
// CHECK: #[[RHS:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d2, d0, d3)>
// CHECK-LABEL: func.func @qk_layout
// CHECK-SAME: alps.p2a.eliminated_transpose_materialization_bytes = 128
// CHECK-SAME: alps.p2a.zero_copy_sites = 1
// CHECK-NOT: linalg.transpose
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.batch_matmul
// CHECK: linalg.generic {{.*}}alps.p2a.eliminated_transpose_materialization_bytes = 128
// CHECK-SAME: alps.p2a.zero_copy_attention
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: arith.addf

// ADMISSION: [ALPS-P2D-SITE] function=qk_layout ordinal=-1
// ADMISSION-SAME: kind=zero_copy_representation action=no_op
// ADMISSION-SAME: reason=p2a_eliminated_transfer
// ADMISSION-SAME: count=1
// ADMISSION: [ALPS-P2D-SUMMARY] function=qk_layout candidates=1 no_op=1 native=0
