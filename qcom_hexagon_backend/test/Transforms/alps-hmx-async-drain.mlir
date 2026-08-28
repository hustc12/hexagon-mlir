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
