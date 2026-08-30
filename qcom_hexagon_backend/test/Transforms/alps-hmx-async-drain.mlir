// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-async-drain=true enable-async-drain-analysis=true enable-direct-output-formation=true}))' | FileCheck %s

// Two full tiles are drained through the ping-pong VTCM slots.  The short-M
// boundary remains on the synchronous clipped-copy path, then both descriptors
// are retired before the result can escape.
func.func @p5n_static_f16(%lhs: memref<33x64xf16>,
                          %rhs: memref<64x64xf16>,
                          %out: memref<33x64xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<33x64xf16>, memref<64x64xf16>)
               outs(%out : memref<33x64xf16>)
  return
}

// CHECK-LABEL: func.func @p5n_static_f16
// CHECK: hexkl.micro_hmx_async_drain_wait_slot
// CHECK: hexkl.micro_hmx_async_drain_start_f16
// CHECK: hexkl.micro_hmx_copy_f16_to_submatrix
// CHECK: hexkl.micro_hmx_async_drain_flush

// An unrepresentable 2-D destination stride must stay on the synchronous
// drain path rather than being silently truncated by UserDMA.
func.func @p5n_stride_overflow(%lhs: memref<32x32xf16>,
                               %rhs: memref<32x32768xf16>,
                               %out: memref<32x32768xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<32x32xf16>, memref<32x32768xf16>)
               outs(%out : memref<32x32768xf16>)
  return
}

// CHECK-LABEL: func.func @p5n_stride_overflow
// CHECK-NOT: hexkl.micro_hmx_async_drain_start_f16
// CHECK: hexkl.micro_hmx_copy_f16_to_submatrix
