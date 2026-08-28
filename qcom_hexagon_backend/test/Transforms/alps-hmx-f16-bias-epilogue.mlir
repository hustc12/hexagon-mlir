// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-direct-output-formation=true enable-f16-bias-epilogue-formation=true}))' | FileCheck %s --check-prefix=FUSED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-direct-output-formation=true}))' | FileCheck %s --check-prefix=CONTROL

func.func @rank2_bias(%lhs: memref<33x64xf16>,
                      %rhs: memref<64x128xf16>,
                      %intermediate: memref<33x128xf16>,
                      %bias: memref<128xf16>,
                      %final: memref<33x128xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<33x64xf16>, memref<64x128xf16>)
               outs(%intermediate : memref<33x128xf16>)
  hexkl.f16_bias_epilogue ins(%intermediate, %bias : memref<33x128xf16>, memref<128xf16>)
               outs(%final : memref<33x128xf16>)
  return
}

// FUSED-LABEL: func.func @rank2_bias
// FUSED-NOT: hexkl.f16_bias_epilogue
// FUSED: hexkl.micro_hmx_copy_f16_bias_to_submatrix({{.*}}%arg3, %arg4
// FUSED-NOT: hexkl.micro_hmx_copy_f16_to_submatrix

// CONTROL-LABEL: func.func @rank2_bias
// CONTROL: hexkl.micro_hmx_copy_f16_to_submatrix
// CONTROL: hexkl.f16_bias_epilogue

// Buffer deallocation may reuse one allocation for sequential value versions.
// A future matmul/user pair must not reject fusion of the current pair.
func.func @sequential_reuse(%lhs: memref<32x64xf16>,
                            %rhs0: memref<64x128xf16>,
                            %rhs1: memref<64x128xf16>,
                            %shared: memref<32x128xf16>,
                            %bias0: memref<128xf16>,
                            %bias1: memref<128xf16>,
                            %final0: memref<32x128xf16>,
                            %final1: memref<32x128xf16>) {
  hexkl.matmul ins(%lhs, %rhs0 : memref<32x64xf16>, memref<64x128xf16>)
               outs(%shared : memref<32x128xf16>)
  hexkl.f16_bias_epilogue ins(%shared, %bias0 : memref<32x128xf16>, memref<128xf16>)
               outs(%final0 : memref<32x128xf16>)
  hexkl.matmul ins(%lhs, %rhs1 : memref<32x64xf16>, memref<64x128xf16>)
               outs(%shared : memref<32x128xf16>)
  hexkl.f16_bias_epilogue ins(%shared, %bias1 : memref<32x128xf16>, memref<128xf16>)
               outs(%final1 : memref<32x128xf16>)
  return
}

// FUSED-LABEL: func.func @sequential_reuse
// FUSED-COUNT-2: hexkl.micro_hmx_copy_f16_bias_to_submatrix
// FUSED-NOT: hexkl.f16_bias_epilogue

// The consumer-selected destination and bias view dominate the producer, as
// guaranteed by P5l tensor formation before one-shot bufferization.
func.func @contract_operands_dominate(%lhs: memref<32x64xf16>,
                                      %rhs: memref<64x128xf16>,
                                      %shared: memref<32x128xf16>,
                                      %bias_storage: memref<256xf16>) {
  %final = memref.alloc() : memref<32x128xf16>
  %bias = memref.subview %bias_storage[64] [128] [1]
      : memref<256xf16> to memref<128xf16, strided<[1], offset: 64>>
  hexkl.matmul ins(%lhs, %rhs : memref<32x64xf16>, memref<64x128xf16>)
               outs(%shared : memref<32x128xf16>)
  hexkl.f16_bias_epilogue ins(%shared, %bias : memref<32x128xf16>, memref<128xf16, strided<[1], offset: 64>>)
               outs(%final : memref<32x128xf16>)
  memref.dealloc %final : memref<32x128xf16>
  return
}

// FUSED-LABEL: func.func @contract_operands_dominate
// FUSED: %[[FINAL:.*]] = memref.alloc()
// FUSED: %[[BIAS:.*]] = memref.subview
// FUSED: hexkl.micro_hmx_copy_f16_bias_to_submatrix({{.*}}%[[BIAS]], %[[FINAL]]
// FUSED-NOT: hexkl.f16_bias_epilogue

// A late destination is not legal for producer-side direct formation.  It
// falls back to ordinary elementwise add instead of moving HMX computation.
func.func @late_destination_falls_back(%lhs: memref<32x64xf16>,
                                       %rhs: memref<64x128xf16>,
                                       %shared: memref<32x128xf16>,
                                       %bias: memref<128xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<32x64xf16>, memref<64x128xf16>)
               outs(%shared : memref<32x128xf16>)
  %final = memref.alloc() : memref<32x128xf16>
  hexkl.f16_bias_epilogue ins(%shared, %bias : memref<32x128xf16>, memref<128xf16>)
               outs(%final : memref<32x128xf16>)
  memref.dealloc %final : memref<32x128xf16>
  return
}

// FUSED-LABEL: func.func @late_destination_falls_back
// FUSED: hexkl.micro_hmx_copy_f16_to_submatrix
// FUSED: arith.addf
// FUSED-NOT: hexkl.f16_bias_epilogue
