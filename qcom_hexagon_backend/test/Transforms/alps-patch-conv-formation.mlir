// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-patch-conv-formation,fold-resource-transpose,canonicalize,cse))' \
// RUN:   2>/dev/null | FileCheck %s

#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// A non-overlapping patch convolution is a matrix-like reduction whose next
// consumer requires NHWC tokens.  P5i forms output channels contiguously at the
// producer, rather than materializing NCHW and transposing it afterwards.
func.func @direct_patch_tokens(%input: tensor<1x3x4x4xf16>)
    -> tensor<1x4x64xf16> {
  %filter = arith.constant dense<1.0> : tensor<64x3x2x2xf16>
  %bias = arith.constant dense<0.0> : tensor<64xf32>
  %bias_init = tensor.empty() : tensor<1x64x2x2xf32>
  %broadcast = linalg.broadcast
      ins(%bias : tensor<64xf32>)
      outs(%bias_init : tensor<1x64x2x2xf32>)
      dimensions = [0, 2, 3]
  %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<2> : vector<2xi64>}
      ins(%input, %filter : tensor<1x3x4x4xf16>, tensor<64x3x2x2xf16>)
      outs(%broadcast : tensor<1x64x2x2xf32>) -> tensor<1x64x2x2xf32>
  %trunc_init = tensor.empty() : tensor<1x64x2x2xf16>
  %trunc = linalg.generic {
      indexing_maps = [#id4, #id4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv : tensor<1x64x2x2xf32>)
      outs(%trunc_init : tensor<1x64x2x2xf16>) {
    ^bb0(%in: f32, %old: f16):
      %value = arith.truncf %in : f32 to f16
      linalg.yield %value : f16
  } -> tensor<1x64x2x2xf16>
  %collapsed = tensor.collapse_shape %trunc [[0], [1], [2, 3]] :
      tensor<1x64x2x2xf16> into tensor<1x64x4xf16>
  %tokens_init = tensor.empty() : tensor<1x4x64xf16>
  %tokens = linalg.transpose
      ins(%collapsed : tensor<1x64x4xf16>)
      outs(%tokens_init : tensor<1x4x64xf16>)
      permutation = [0, 2, 1]
  return %tokens : tensor<1x4x64xf16>
}

// CHECK-LABEL: func.func @direct_patch_tokens
// CHECK-SAME: alps.p5i.eliminated_output_transpose_bytes = 512
// CHECK-SAME: alps.p5i.patch_conv_formed = 1
// CHECK: tensor<3x2x2x64xf16>
// CHECK: linalg.generic
// CHECK-SAME: alps.p5i.contiguous_output_channel = 64
// CHECK-SAME: alps.p5i.patch_conv_formation
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK-NOT: linalg.transpose

// Overlapping windows are intentionally rejected: they require a different
// reuse/supply contract and cannot use the one-patch-per-output proof.
func.func @reject_overlapping_patch(%input: tensor<1x3x4x4xf16>)
    -> tensor<1x9x64xf16> {
  %filter = arith.constant dense<1.0> : tensor<64x3x2x2xf16>
  %bias = arith.constant dense<0.0> : tensor<64xf32>
  %bias_init = tensor.empty() : tensor<1x64x3x3xf32>
  %broadcast = linalg.broadcast
      ins(%bias : tensor<64xf32>)
      outs(%bias_init : tensor<1x64x3x3xf32>)
      dimensions = [0, 2, 3]
  %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>,
       strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x3x4x4xf16>, tensor<64x3x2x2xf16>)
      outs(%broadcast : tensor<1x64x3x3xf32>) -> tensor<1x64x3x3xf32>
  %trunc_init = tensor.empty() : tensor<1x64x3x3xf16>
  %trunc = linalg.generic {
      indexing_maps = [#id4, #id4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv : tensor<1x64x3x3xf32>)
      outs(%trunc_init : tensor<1x64x3x3xf16>) {
    ^bb0(%in: f32, %old: f16):
      %value = arith.truncf %in : f32 to f16
      linalg.yield %value : f16
  } -> tensor<1x64x3x3xf16>
  %collapsed = tensor.collapse_shape %trunc [[0], [1], [2, 3]] :
      tensor<1x64x3x3xf16> into tensor<1x64x9xf16>
  %tokens_init = tensor.empty() : tensor<1x9x64xf16>
  %tokens = linalg.transpose
      ins(%collapsed : tensor<1x64x9xf16>)
      outs(%tokens_init : tensor<1x9x64xf16>)
      permutation = [0, 2, 1]
  return %tokens : tensor<1x9x64xf16>
}

// CHECK-LABEL: func.func @reject_overlapping_patch
// CHECK-SAME: alps.p5i.eliminated_output_transpose_bytes = 0
// CHECK-SAME: alps.p5i.patch_conv_formed = 0
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: linalg.transpose
