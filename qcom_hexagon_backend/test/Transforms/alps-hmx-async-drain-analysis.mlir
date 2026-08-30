// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-async-drain-analysis=true enable-direct-output-formation=true}))' 2>&1 | FileCheck %s

// P5m must be observational: it records exact static admission/overlap facts,
// while the existing P5k synchronous copy remains in the generated program.
func.func @p5m_static_f16(%lhs: memref<33x64xf16>,
                          %rhs: memref<64x64xf16>,
                          %out: memref<33x64xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<33x64xf16>, memref<64x64xf16>)
               outs(%out : memref<33x64xf16>)
  return
}

// Function passes may emit their atomic ledger records in any order.
// CHECK-DAG: [ALPS-P5M-ANALYSIS] function=p5m_static_f16 sites=1 admitted_sites=1 rejected_sites=0 drain_bytes=4224 descriptors=4 full_tiles=2 boundary_tiles=2 admitted_descriptors=2 admitted_bytes=4096 boundary_bytes=128 overlap_descriptors=1 overlap_bytes=2048 overlap_hmx_calls=2 extra_vtcm_bytes=8192 max_dst_stride_bytes=128 dma2d_legal=1 decision=admit
// CHECK-DAG: [ALPS-P5M-ANALYSIS] function=p5m_too_small sites=1 admitted_sites=0 rejected_sites=1 drain_bytes=512 descriptors=1 full_tiles=0 boundary_tiles=1 admitted_descriptors=0 admitted_bytes=0 boundary_bytes=512 overlap_descriptors=0 overlap_bytes=0 overlap_hmx_calls=0 extra_vtcm_bytes=8192 max_dst_stride_bytes=32 dma2d_legal=1 decision=reject
// CHECK-DAG: [ALPS-P5M-ANALYSIS] function=p5m_stride_overflow sites=1 admitted_sites=0 rejected_sites=1 drain_bytes=2097152 descriptors=1024 full_tiles=1024 boundary_tiles=0 admitted_descriptors=0 admitted_bytes=0 boundary_bytes=2097152 overlap_descriptors=0 overlap_bytes=0 overlap_hmx_calls=0 extra_vtcm_bytes=8192 max_dst_stride_bytes=65536 dma2d_legal=0 decision=reject
// CHECK: func.func @p5m_static_f16
// CHECK-SAME: alps.p5m.admitted_bytes = 4096
// CHECK: hexkl.micro_hmx_copy_f16_to_submatrix

// A one-tile output cannot hide its only descriptor behind later HMX work.
func.func @p5m_too_small(%lhs: memref<16x32xf16>,
                         %rhs: memref<32x16xf16>,
                         %out: memref<16x16xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<16x32xf16>, memref<32x16xf16>)
               outs(%out : memref<16x16xf16>)
  return
}

// A 32768-column f16 result needs a 65536-byte destination stride, one byte
// wider than the V73 UserDMA 2-D descriptor can encode.
func.func @p5m_stride_overflow(%lhs: memref<32x32xf16>,
                               %rhs: memref<32x32768xf16>,
                               %out: memref<32x32768xf16>) {
  hexkl.matmul ins(%lhs, %rhs : memref<32x32xf16>, memref<32x32768xf16>)
               outs(%out : memref<32x32768xf16>)
  return
}
