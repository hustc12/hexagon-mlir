// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hexkl))' | FileCheck %s --check-prefix=CHECK
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hexkl{consumer-f16-epilogue=true}))' | FileCheck %s --check-prefix=P5J
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hexkl{consumer-f16-epilogue=true consumer-f16-bias-epilogue=true}))' | FileCheck %s --check-prefix=P5L

func.func @matmul_hexkl(%arg0: memref<256x128xf16>,
                  %arg1: memref<128x64xf16>,
                  %output: memref<256x64xf32>) {
  %arg0_tensor = bufferization.to_tensor %arg0 restrict : memref<256x128xf16> to tensor<256x128xf16>
  %arg1_tensor = bufferization.to_tensor %arg1 restrict : memref<128x64xf16> to tensor<128x64xf16>
  %empty = tensor.empty() : tensor<256x64xf32>
  %res = linalg.matmul ins (%arg0_tensor, %arg1_tensor: tensor<256x128xf16>, tensor<128x64xf16>)
                     outs (%empty: tensor<256x64xf32>) -> tensor<256x64xf32>
  bufferization.materialize_in_destination %res in restrict writable %output
                                : (tensor<256x64xf32>, memref<256x64xf32>) -> ()
  return
}

// CHECK-LABEL: @matmul_hexkl
// CHECK:  %[[Input1:.*]] = bufferization.to_tensor %arg0 restrict : memref<256x128xf16> to tensor<256x128xf16>
// CHECK:  %[[Input2:.*]] = bufferization.to_tensor %arg1 restrict : memref<128x64xf16> to tensor<128x64xf16>
// CHECK:  %[[Empty:.*]] = tensor.empty() : tensor<256x64xf32>
// CHECK:  %[[Res:.*]] = hexkl.matmul ins(%[[Input1]], %[[Input2]] : tensor<256x128xf16>, tensor<128x64xf16>) outs(%[[Empty]] : tensor<256x64xf32>) -> tensor<256x64xf32>
// CHECK:  bufferization.materialize_in_destination %[[Res]] in restrict writable %arg2 : (tensor<256x64xf32>, memref<256x64xf32>) -> ()

func.func @matmul_then_identity_trunc(%arg0: tensor<32x64xf16>,
                                      %arg1: tensor<64x128xf16>) -> tensor<32x128xf16> {
  %f32_empty = tensor.empty() : tensor<32x128xf32>
  %mm = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf16>, tensor<64x128xf16>)
      outs(%f32_empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %f16_empty = tensor.empty() : tensor<32x128xf16>
  %trunc = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<32x128xf32>) outs(%f16_empty : tensor<32x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %v = arith.truncf %in : f32 to f16
      linalg.yield %v : f16
  } -> tensor<32x128xf16>
  return %trunc : tensor<32x128xf16>
}

// P5J-LABEL: func.func @matmul_then_identity_trunc
// P5J-NOT: linalg.matmul
// P5J-NOT: arith.truncf
// P5J: hexkl.matmul {{.*}} {alps.p5j.consumer_f16_epilogue}
// P5J-SAME: tensor<32x128xf16>

func.func @matmul_trunc_then_rank2_bias(%arg0: tensor<32x64xf16>,
                                        %arg1: tensor<64x128xf16>,
                                        %bias: tensor<128xf16>)
    -> tensor<32x128xf16> {
  %f32_empty = tensor.empty() : tensor<32x128xf32>
  %mm = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf16>, tensor<64x128xf16>)
      outs(%f32_empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %f16_empty = tensor.empty() : tensor<32x128xf16>
  %trunc = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<32x128xf32>) outs(%f16_empty : tensor<32x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %v = arith.truncf %in : f32 to f16
      linalg.yield %v : f16
  } -> tensor<32x128xf16>
  %out_empty = tensor.empty() : tensor<32x128xf16>
  %add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%trunc, %bias : tensor<32x128xf16>, tensor<128xf16>)
      outs(%out_empty : tensor<32x128xf16>) {
    ^bb0(%in: f16, %b: f16, %out: f16):
      %v = arith.addf %in, %b : f16
      linalg.yield %v : f16
  } -> tensor<32x128xf16>
  return %add : tensor<32x128xf16>
}

// P5L-LABEL: func.func @matmul_trunc_then_rank2_bias
// P5L-NOT: arith.truncf
// P5L-NOT: linalg.generic
// P5L: %[[MM:.+]] = hexkl.matmul {{.*}} {alps.p5j.consumer_f16_epilogue}
// P5L: hexkl.f16_bias_epilogue ins(%[[MM]], %arg2
// P5L-SAME: {alps.p5l.consumer_bias_formation}
