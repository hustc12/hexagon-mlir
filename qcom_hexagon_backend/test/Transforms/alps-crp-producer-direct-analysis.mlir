// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(alps-crp-producer-direct-analysis))' %s | FileCheck %s --check-prefix=P5GD
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(alps-crp-producer-direct-analysis{rewrite-epoch-vtcm=true}))' %s | FileCheck %s --check-prefix=P5GE
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(alps-crp-producer-direct-analysis{rewrite-epoch-head-major-vtcm=true}))' %s | FileCheck %s --check-prefix=P5GF
// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(alps-crp-producer-direct-analysis{rewrite-epoch-head-major-vtcm=true rewrite-producer-loop-order=true}))' %s | FileCheck %s --check-prefix=P5GG

func.func @producer_direct_ready(
    %source: memref<1x256x6x64xf16>,
    %output: memref<256x6x64xf16>) {
  %root = memref.alloc() : memref<1x256x6x64xf16>
  memref.copy %source, %root : memref<1x256x6x64xf16> to memref<1x256x6x64xf16>
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  scf.for %channel = %c0 to %c64 step %c4 {
    scf.for %token = %c0 to %c256 step %c16 {
      %tile = memref.subview %root[0, %token, 0, %channel]
          [1, 16, 1, 4] [1, 1, 1, 1]
        : memref<1x256x6x64xf16> to
          memref<16x4xf16, strided<[384, 1], offset: ?>>
      %value = vector.transfer_read %tile[%c0, %c0], %c0f
          {alps.p2g.register_tile}
        : memref<16x4xf16, strided<[384, 1], offset: ?>>,
          vector<16x4xf16>
      %out = memref.subview %output[%token, 0, %channel]
          [16, 1, 4] [1, 1, 1]
        : memref<256x6x64xf16> to
          memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
      %expanded = vector.shape_cast %value
          : vector<16x4xf16> to vector<16x1x4xf16>
      vector.transfer_write %expanded, %out[%c0, %c0, %c0]
          : vector<16x1x4xf16>,
            memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
    }
  }
  memref.dealloc %root : memref<1x256x6x64xf16>
  return
}

// P5GD-LABEL: func.func @producer_direct_ready
// P5GD-SAME: alps.p5g_d.rewrite_ready = 1

// P5GE-LABEL: func.func @tiled_vector_overwrite_ready
// P5GE-SAME: alps.p5g_e.rewritten_epochs = 1
// P5GE: %[[VTCM:.*]] = memref.alloc() {bufferization.manual_deallocation} : memref<257x6x64xf16, 1 : i32>
// P5GE: vector.transfer_write {{.*}} {alps.p5g_e.producer_direct_vtcm}
// P5GE: vector.transfer_read {{.*}} {alps.p2g.register_tile, alps.p5g_e.consumer_direct_vtcm}
// P5GE: memref.dealloc %[[VTCM]] {bufferization.manual_deallocation} : memref<257x6x64xf16, 1 : i32>

// P5GF-LABEL: func.func @tiled_vector_overwrite_ready
// P5GF-SAME: alps.p5g_f.head_major_rewritten_epochs = 1
// P5GF: %[[HEAD:.*]] = memref.alloc() {bufferization.manual_deallocation} : memref<257x6x64xf16, strided<[64, 16448, 1]>, 1 : i32>
// P5GF: vector.transfer_write {{.*}} {alps.p5g_f.producer_direct_head_major_vtcm}
// P5GF: vector.transfer_read {{.*}} {alps.p2g.register_tile, alps.p5g_f.consumer_head_major_vtcm}
// P5GF: memref.dealloc %[[HEAD]] {bufferization.manual_deallocation}
// P5GG-LABEL: func.func @tiled_vector_overwrite_ready
// P5GG-SAME: alps.p5g_g.interchanged_producer_epochs = 1
// P5GG: %[[VTCM:.*]] = memref.alloc() {bufferization.manual_deallocation} : memref<257x6x64xf16, strided<[64, 16448, 1]>, 1 : i32>
// P5GG: scf.for %[[HEAD_IV:.*]] = %{{.*}} to %c6 step %{{.*}}
// P5GG-NEXT: scf.for %[[TOKEN:.*]] = %{{.*}} to %c257 step %{{.*}}
// P5GG: memref.subview %[[VTCM]][%[[TOKEN]], %[[HEAD_IV]], 0]
// P5GG: vector.transfer_write {{.*}} {alps.p5g_f.producer_direct_head_major_vtcm}
// P5GG: vector.transfer_read {{.*}} {alps.p2g.register_tile, alps.p5g_f.consumer_head_major_vtcm}
// P5GD-SAME: alps.p5g_d.unique_roots = 1
// P5GD: memref.copy
// P5GD-NOT: memref<{{.*}}, 1>

func.func @tiled_vector_overwrite_ready(
    %input: memref<257x6x64xf16>,
    %output: memref<257x6x64xf16>) {
  %root = memref.alloc() : memref<257x6x64xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c257 = arith.constant 257 : index
  %zero = arith.constant 0.0 : f16
  scf.for %token = %c0 to %c257 step %c1 {
    scf.for %head = %c0 to %c6 step %c1 {
      %source = memref.subview %input[%token, %head, 0] [1, 1, 64]
          [1, 1, 1] : memref<257x6x64xf16> to
          memref<64xf16, strided<[1], offset: ?>>
      %value = vector.transfer_read %source[%c0], %zero
          : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
      %target = memref.subview %root[%token, %head, 0] [1, 1, 64]
          [1, 1, 1] : memref<257x6x64xf16> to
          memref<64xf16, strided<[1], offset: ?>>
      vector.transfer_write %value, %target[%c0]
          : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>>
    }
  }
  scf.for %channel = %c0 to %c64 step %c4 {
    scf.for %token = %c0 to %c257 step %c16 {
      %tile = memref.subview %root[%token, 0, %channel] [16, 1, 4]
          [1, 1, 1] : memref<257x6x64xf16> to
          memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
      %value = vector.transfer_read %tile[%c0, %c0, %c0], %zero
          {alps.p2g.register_tile} :
          memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>,
          vector<16x1x4xf16>
      %target = memref.subview %output[%token, 0, %channel] [16, 1, 4]
          [1, 1, 1] : memref<257x6x64xf16> to
          memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
      vector.transfer_write %value, %target[%c0, %c0, %c0]
          : vector<16x1x4xf16>,
            memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
    }
  }
  memref.dealloc %root : memref<257x6x64xf16>
  return
}

// P5GD-LABEL: func.func @tiled_vector_overwrite_ready
// P5GD-SAME: alps.p5g_d.coverage_proven_epochs = 1
// P5GD-SAME: alps.p5g_d.epoch_redirect_candidates = 1
// P5GD-SAME: alps.p5g_d.rewrite_ready = 1
