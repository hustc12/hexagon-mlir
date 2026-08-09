// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(decompose-hexkl-matmul{enable-vtcm-lifetime-coloring=true}))' \
// RUN:   | FileCheck %s

func.func @colored_k2(
    %lhs: memref<32x64xf16>,
    %rhs: memref<64x128xf16>,
    %out: memref<32x128xf32>) {
  hexkl.matmul
      ins(%lhs, %rhs : memref<32x64xf16>, memref<64x128xf16>)
      outs(%out : memref<32x128xf32>)
  return
}

// K=64 gives two K tiles. The legacy arena is (2*Ktiles+7)*4096 = 45056
// bytes. Coloring retains two AH tiles plus two phase-shared scratch/WH
// colors, for 4*4096 = 16384 bytes.
// CHECK-LABEL: func.func @colored_k2
// CHECK-SAME: omni_fetch.vtcm_colored_peak_bytes = 16384
// CHECK-SAME: omni_fetch.vtcm_colored_sites = 1
// CHECK-SAME: omni_fetch.vtcm_coloring_enabled
// CHECK-SAME: omni_fetch.vtcm_legacy_peak_bytes = 45056
// CHECK-SAME: omni_fetch.vtcm_saved_peak_bytes = 28672
// CHECK: %[[BYTES:.+]] = arith.constant 16384 : index
// CHECK: hexagonmem.alloc(%[[BYTES]])
// CHECK: hexkl.micro_hmx_setup_acc_read_f16
// CHECK: hexkl.micro_hmx_copy_submatrix_to_f16(%{{[^,]+}}, %[[SHARED:[^, ]+]],
// CHECK: hexkl.micro_hmx_rm_to_ah_f16({{.*}}, {{.*}}, %[[SHARED]])
// CHECK: arith.select {{.*}}, {{.*}}, %[[SHARED]]
// CHECK: hexkl.micro_hmx_rm_to_wh_f16
// CHECK: hexkl.micro_hmx_mm_f16
// CHECK: hexkl.micro_hmx_acc_read_f16({{.*}}, %[[ACC:.+]])
// CHECK: hexkl.micro_hmx_ah_to_rm_f16({{.*}}, %{{.*}}, %[[ACC]])
// CHECK: hexagonmem.dealloc
