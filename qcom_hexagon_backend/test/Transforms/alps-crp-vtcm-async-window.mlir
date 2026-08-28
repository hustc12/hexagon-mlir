// RUN: linalg-hexagon-opt -pass-pipeline='builtin.module(func.func(alps-crp-vtcm-window))' %s | FileCheck %s --check-prefix=P5GC

module attributes {alps.p5g_c.vtcm_async_window = true} {
  func.func @p5gc_async_vtcm_window(
      %input: memref<1x256x6x64xf16, strided<[98304, 384, 64, 1]>>,
      %output: memref<256x6x64xf16>) {
    %c0 = arith.constant 0 : index
    %c0f = arith.constant 0.0 : f16
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    scf.for %channel = %c0 to %c64 step %c4 {
      scf.for %token = %c0 to %c256 step %c16 {
        %input_tile = memref.subview %input[0, %token, 0, %channel]
            [1, 16, 1, 4] [1, 1, 1, 1]
          : memref<1x256x6x64xf16, strided<[98304, 384, 64, 1]>> to
            memref<16x4xf16, strided<[384, 1], offset: ?>>
        %input_tile4 = memref.expand_shape %input_tile [[0, 1], [2, 3]]
            output_shape [1, 16, 1, 4]
          : memref<16x4xf16, strided<[384, 1], offset: ?>> into
            memref<1x16x1x4xf16, strided<[6144, 384, 4, 1], offset: ?>>
        %tile = vector.transfer_read %input_tile4[%c0, %c0, %c0, %c0], %c0f
            {alps.p2g.register_tile}
          : memref<1x16x1x4xf16, strided<[6144, 384, 4, 1], offset: ?>>,
            vector<1x16x1x4xf16>
        %output_tile = memref.subview %output[%token, 0, %channel]
            [16, 1, 4] [1, 1, 1]
          : memref<256x6x64xf16> to
            memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
        %tile3 = vector.shape_cast %tile
            : vector<1x16x1x4xf16> to vector<16x1x4xf16>
        vector.transfer_write %tile3, %output_tile[%c0, %c0, %c0]
          : vector<16x1x4xf16>,
            memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
      }
    }
    return
  }
}

// P5GC-LABEL: func.func @p5gc_async_vtcm_window
// P5GC-SAME: alps.p5g_c.vtcm_async_windows = 1
// P5GC-SAME: alps.p5g_c.window_bytes = 32768
// P5GC: %[[PING_ALLOC:.*]] = memref.alloc() : memref<256x32xf16, 1>
// P5GC: %[[PONG_ALLOC:.*]] = memref.alloc() : memref<256x32xf16, 1>
// P5GC: %{{.*}}:2 = memref.distinct_objects %[[PING_ALLOC]], %[[PONG_ALLOC]]
// P5GC: %[[TAGS:.*]] = memref.alloca() : memref<2xi32>
// P5GC: memref.dma_start {{.*}}, %{{.*}}#0{{.*}}, %[[TAGS]]
// P5GC: scf.for
// P5GC: %[[CURRENT:.*]] = arith.select
// P5GC: %[[NEXT:.*]] = arith.select
// P5GC: %[[SLOTS:.*]]:2 = memref.distinct_objects %[[CURRENT]], %[[NEXT]]
// P5GC: scf.if
// P5GC: memref.dma_wait %[[TAGS]]
// P5GC: scf.if
// P5GC: memref.dma_start {{.*}}, %[[SLOTS]]#1{{.*}}, %[[TAGS]]
// P5GC: vector.transfer_read {{.*}}alps.p5g_c.vtcm_async_window
// P5GC-NOT: memref.copy
