// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-consumer-driven-layout{propagate-codegen-contract=true}))' \
// RUN:   | FileCheck %s
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-layout-supply-prefetch))' \
// RUN:   | FileCheck %s --check-prefix=P5C
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-continuity-audit))' \
// RUN:   2>&1 | FileCheck %s --check-prefix=P2G
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-consumer-driven-layout{allow-innermost-loop-interchange=true}))' \
// RUN:   | FileCheck %s --check-prefix=P2GB
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-consumer-driven-layout{allow-register-tile-formation=true}))' \
// RUN:   | FileCheck %s --check-prefix=P2GC
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-consumer-driven-layout{allow-register-tile-formation=true register-tile-demand-end=0}))' \
// RUN:   | FileCheck %s --check-prefix=P2GC-NONE
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-crp-supply-analysis))' \
// RUN:   | FileCheck %s --check-prefix=P5FA
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-crp-supply-prefetch))' \
// RUN:   | FileCheck %s --check-prefix=P5FB
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-crp-supply-prefetch{page-safe-segmented=true}))' \
// RUN:   | FileCheck %s --check-prefix=P5FC
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(alps-crp-vtcm-formation))' \
// RUN:   | FileCheck %s --check-prefix=P5GA

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// The terminal linalg consumer explicitly demands [N,M,K].  Because K stays
// innermost, P2e can make the elementwise producer write that representation
// directly without creating [M,N,K] and physically transposing it.
func.func @direct_hvx_consumer(%input: tensor<8x16x32xf16>,
                               %out: tensor<16x8x32xf16>)
    -> tensor<16x8x32xf16> {
  %tmp = tensor.empty() : tensor<8x16x32xf16>
  %produced = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<8x16x32xf16>)
      outs(%tmp : tensor<8x16x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<8x16x32xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<8x16x32xf16>)
      outs(%out : tensor<16x8x32xf16>)
      permutation = [1, 0, 2]
  %consumer_out = tensor.empty() : tensor<16x8x32xf16>
  %result = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<16x8x32xf16>)
      outs(%consumer_out : tensor<16x8x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<16x8x32xf16>
  return %result : tensor<16x8x32xf16>
}

// CHECK: affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: func.func @direct_hvx_consumer
// CHECK-SAME: alps.p2e.demands = 1
// CHECK-SAME: alps.p2e.eliminated_materialization_bytes = 8192
// CHECK-SAME: alps.p2e.hvx_consumers = 1
// CHECK-SAME: alps.p2e.producer_direct = 1
// CHECK: alps.p2f.consumer_layout_contract = "hvx_innermost_unit_stride"
// CHECK-NOT: linalg.transpose

func.func @p5c_next_tile(%input: memref<4x128xf16>,
                         %output: memref<4x128xf16>) attributes {
    alps.p5a.contracts = [{id = "p5c:0", origin = "p5c_supply"}]
  } {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %in_tile = memref.subview %input[%iv, 0] [1, 128] [1, 1]
      : memref<4x128xf16> to memref<1x128xf16, strided<[128, 1], offset: ?>>
      loc("p5c_supply")
    %out_tile = memref.subview %output[%iv, 0] [1, 128] [1, 1]
      : memref<4x128xf16> to memref<1x128xf16, strided<[128, 1], offset: ?>>
    %value = vector.transfer_read %in_tile[%c0, %c0], %c0f
      : memref<1x128xf16, strided<[128, 1], offset: ?>>, vector<1x128xf16>
      loc("p5c_supply")
    vector.transfer_write %value, %out_tile[%c0, %c0]
      : vector<1x128xf16>, memref<1x128xf16, strided<[128, 1], offset: ?>>
  }
  return
}

func.func @p2g_contiguous(%input: memref<4x128xf16>,
                          %formed: memref<4x128xf16>,
                          %output: memref<4x128xf16>) attributes {
    alps.p5a.contracts = [{id = "p2g:0", origin = "p2g_producer",
      consumer_origins = ["p2g_consumer"], moves_innermost = false}]
  } {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %tile = vector.transfer_read %input[%c0, %c0], %c0f
      : memref<4x128xf16>, vector<128xf16> loc("p2g_producer")
  vector.transfer_write %tile, %formed[%c0, %c0]
      : vector<128xf16>, memref<4x128xf16> loc("p2g_producer")
  %ready = vector.transfer_read %formed[%c0, %c0], %c0f
      : memref<4x128xf16>, vector<128xf16> loc("p2g_consumer")
  vector.transfer_write %ready, %output[%c0, %c0]
      : vector<128xf16>, memref<4x128xf16>
  return
}

// P2G: [ALPS-P2G-CONTRACT] function=p2g_contiguous kind=p2e_direct id=p2g:0 moves_innermost=0 producer_read=unit_stride producer_write=unit_stride consumer_read=unit_stride
// P2G: [ALPS-P2G-SUMMARY] function=p2g_contiguous contracts=1 observed=1 moves_innermost=0 producer_reads=1 producer_unit_reads=1 producer_writes=1 producer_unit_writes=1 consumer_reads=1 consumer_unit_reads=1 vmemu_risk=0 static_tile_bytes=768

// P5f-a must only admit an explicitly marked P2g-c register tile whose
// producer advances through an exact, read-only, loop-carried unit-stride
// subview.  This pass records supply eligibility without changing the IR.
func.func @p5fa_register_tile_supply(%input: memref<8x4x16xf16>,
                                     %output: memref<8x4x16xf16>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %input_storage = memref.alloc() : memref<8x4x16xf16>
  scf.for %iv = %c0 to %c8 step %c1 {
    %input_tile = memref.subview %input_storage[%iv, 0, 0] [1, 4, 16] [1, 1, 1]
      : memref<8x4x16xf16> to memref<1x4x16xf16, strided<[64, 16, 1], offset: ?>>
    %input_tile_cast = memref.cast %input_tile
      : memref<1x4x16xf16, strided<[64, 16, 1], offset: ?>>
        to memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>>
    %output_tile = memref.subview %output[%iv, 0, 0] [1, 4, 16] [1, 1, 1]
      : memref<8x4x16xf16> to memref<1x4x16xf16, strided<[64, 16, 1], offset: ?>>
    %tile = vector.transfer_read %input_tile_cast[%c0, %c0, %c0], %c0f
        {alps.p2g.register_tile}
      : memref<?x?x?xf16, strided<[?, ?, 1], offset: ?>>,
        vector<4x16xf16>
    vector.transfer_write %tile, %output_tile[%c0, %c0, %c0]
      : vector<4x16xf16>,
        memref<1x4x16xf16, strided<[64, 16, 1], offset: ?>>
  }
  return
}

// P5FA-LABEL: func.func @p5fa_register_tile_supply
// P5FA-SAME: alps.p5f_a.admitted = 1
// P5FA-SAME: alps.p5f_a.admitted_bytes = 128
// P5FA-SAME: alps.p5f_a.matched = 1

// P5FB-LABEL: func.func @p5fa_register_tile_supply
// P5FB-SAME: alps.p5f_b.admitted = 1
// P5FB-SAME: alps.p5f_b.matched = 1
// P5FB-SAME: alps.p5f_b.requested_bytes = 128
// P5FB: scf.if
// P5FB: omni_fetch.l2_hint
// P5FB-SAME: alps.p5f_b.crp_supply

// P5FC-LABEL: func.func @p5fa_register_tile_supply
// P5FC-SAME: alps.p5f_c.contiguous_hints = 1
// P5FC-SAME: alps.p5f_c.physical_rows = 4
// P5FC-SAME: alps.p5f_c.rejected_segment_utilization = 0
// P5FC-SAME: alps.p5f_c.segmented_hints = 0
// P5FC: omni_fetch.l2_hint
// P5FC-SAME: alps.p5f_c.page_safe_segmented
// P5FC-SAME: alps.p5f_c.physical_rows = 4
// P5FC-SAME: alps.p5f_c.physically_contiguous = true
// P5FC-SAME: alps.p5f_c.segment_utilization_percent = 100

// Sparse physical rows must be rejected before an L2 hint is created.  The
// logical tile has 128 useful bytes, but 8 useful bytes in each of sixteen
// 128-byte cache lines is only 6.25% line utilization.
func.func @p5fc_sparse_segment_rejected(
    %input: memref<8x16x4xf16, strided<[6144, 384, 1]>>,
    %output: memref<8x16x4xf16>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %input_tile = memref.subview %input[%iv, 0, 0] [1, 16, 4] [1, 1, 1]
      : memref<8x16x4xf16, strided<[6144, 384, 1]>> to
        memref<1x16x4xf16, strided<[6144, 384, 1], offset: ?>>
    %output_tile = memref.subview %output[%iv, 0, 0] [1, 16, 4] [1, 1, 1]
      : memref<8x16x4xf16> to
        memref<1x16x4xf16, strided<[64, 4, 1], offset: ?>>
    %tile = vector.transfer_read %input_tile[%c0, %c0, %c0], %c0f
        {alps.p2g.register_tile}
      : memref<1x16x4xf16, strided<[6144, 384, 1], offset: ?>>,
        vector<16x4xf16>
    vector.transfer_write %tile, %output_tile[%c0, %c0, %c0]
      : vector<16x4xf16>,
        memref<1x16x4xf16, strided<[64, 4, 1], offset: ?>>
  }
  return
}

// P5FC-LABEL: func.func @p5fc_sparse_segment_rejected
// P5FC-SAME: alps.p5f_b.admitted = 0
// P5FC-SAME: alps.p5f_c.rejected_segment_utilization = 1
// P5FC-NOT: omni_fetch.l2_hint

func.func @p5ga_sparse_vtcm_formation(
    %input: memref<8x16x1x32xf16, strided<[6144, 384, 32, 1]>>,
    %output: memref<8x16x1x32xf16>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f16
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c8 step %c1 {
    %input_tile = memref.subview %input[%iv, 0, 0, 0] [1, 16, 1, 32] [1, 1, 1, 1]
      : memref<8x16x1x32xf16, strided<[6144, 384, 32, 1]>> to
        memref<1x16x1x32xf16, strided<[6144, 384, 32, 1], offset: ?>>
    %output_tile = memref.subview %output[%iv, 0, 0, 0] [1, 16, 1, 32] [1, 1, 1, 1]
      : memref<8x16x1x32xf16> to
        memref<1x16x1x32xf16, strided<[512, 32, 32, 1], offset: ?>>
    %tile = vector.transfer_read %input_tile[%c0, %c0, %c0, %c0], %c0f
        {alps.p2g.register_tile}
      : memref<1x16x1x32xf16, strided<[6144, 384, 32, 1], offset: ?>>,
        vector<1x16x1x32xf16>
    vector.transfer_write %tile, %output_tile[%c0, %c0, %c0, %c0]
      : vector<1x16x1x32xf16>,
        memref<1x16x1x32xf16, strided<[512, 32, 32, 1], offset: ?>>
  }
  return
}

// P5GA-LABEL: func.func @p5ga_sparse_vtcm_formation
// P5GA-SAME: alps.p5g_a.vtcm_formed = 1
// P5GA-SAME: alps.p5g_a.vtcm_formed_bytes = 1024
// P5GA: %[[TILE:.*]] = memref.alloc() : memref<16x32xf16, 1>
// P5GA: %[[SOURCE2D:.*]] = memref.collapse_shape
// P5GA-SAME: into memref<16x32xf16, strided<[384, 1], offset: ?>>
// P5GA: memref.copy %[[SOURCE2D]], %[[TILE]]
// P5GA: %[[TILE4D:.*]] = memref.expand_shape %[[TILE]]
// P5GA: vector.transfer_read %[[TILE4D]]
// P5GA-SAME: alps.p5g_a.vtcm_contiguous

func.func @p5gb_coalesced_vtcm_window(
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
      %tile4 = vector.shape_cast %tile : vector<1x16x1x4xf16> to vector<16x1x4xf16>
      vector.transfer_write %tile4, %output_tile[%c0, %c0, %c0]
        : vector<16x1x4xf16>,
          memref<16x1x4xf16, strided<[384, 64, 1], offset: ?>>
    }
  }
  return
}

// P5GB-LABEL: func.func @p5gb_coalesced_vtcm_window
// P5GB-SAME: alps.p5g_b.vtcm_windows = 1
// P5GB-SAME: alps.p5g_b.window_bytes = 16384
// P5GB: %[[WINDOW:.*]] = memref.alloc() : memref<256x32xf16, 1>
// P5GB: scf.for
// P5GB: scf.if
// P5GB: memref.copy {{.*}}, %[[WINDOW]]
// P5GB: %[[WINDOW_TILE:.*]] = memref.subview %[[WINDOW]]
// P5GB: vector.transfer_read %[[WINDOW_TILE]]{{.*}}alps.p5g_b.vtcm_window{{.*}} : memref<16x4xf16, strided<[32, 1], offset: ?>, 1>

// P5C-LABEL: func.func @p5c_next_tile
// P5C-SAME: alps.p5c.admitted = 1
// P5C-SAME: alps.p5c.matched = 1
// P5C-SAME: alps.p5c.requested_bytes = 256
// P5C: scf.if
// P5C: omni_fetch.l2_hint

// Moving the innermost dimension would turn a unit-stride stream into a
// strided one.  The contract remains auditable, but native materialization is
// retained.
func.func @reject_strided_consumer(%input: tensor<8x16x64xf16>,
                                   %out: tensor<8x64x16xf16>)
    -> tensor<8x64x16xf16> {
  %tmp = tensor.empty() : tensor<8x16x64xf16>
  %produced = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<8x16x64xf16>)
      outs(%tmp : tensor<8x16x64xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<8x16x64xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<8x16x64xf16>)
      outs(%out : tensor<8x64x16xf16>)
      permutation = [0, 2, 1]
  %consumer_out = tensor.empty() : tensor<8x64x16xf16>
  %result = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<8x64x16xf16>)
      outs(%consumer_out : tensor<8x64x16xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<8x64x16xf16>
  return %result : tensor<8x64x16xf16>
}

// CHECK-LABEL: func.func @reject_strided_consumer
// CHECK-SAME: alps.p2e.native = 1
// CHECK: linalg.transpose

// Moving the innermost dimension is legal only when composing the target loop
// order into every producer input recovers either unit stride or invariance.
func.func @direct_interchanged_hvx_consumer(
    %input: tensor<8x32x16xf16>, %out: tensor<8x32x16xf16>)
    -> tensor<8x32x16xf16> {
  %tmp = tensor.empty() : tensor<8x16x32xf16>
  %produced = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<8x32x16xf16>)
      outs(%tmp : tensor<8x16x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<8x16x32xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<8x16x32xf16>)
      outs(%out : tensor<8x32x16xf16>) permutation = [0, 2, 1]
  %consumer_out = tensor.empty() : tensor<8x32x16xf16>
  %result = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<8x32x16xf16>)
      outs(%consumer_out : tensor<8x32x16xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<8x32x16xf16>
  return %result : tensor<8x32x16xf16>
}

// Native P2g-c source loads are 128 B.  A 96xf16 physical row has a partial
// final vector, so retain the transpose until masked/padded tail lowering is
// available rather than allowing an out-of-row vmemu.
func.func @reject_register_tile_source_tail(
    %input: tensor<1x16x96xf16>, %out: tensor<1x96x16xf16>)
    -> tensor<1x96x16xf16> {
  %tmp = tensor.empty() : tensor<1x16x96xf16>
  %produced = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x16x96xf16>)
      outs(%tmp : tensor<1x16x96xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<1x16x96xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<1x16x96xf16>)
      outs(%out : tensor<1x96x16xf16>) permutation = [0, 2, 1]
  return %transposed : tensor<1x96x16xf16>
}

// A native 64xf16 vector contains exactly two complete 32xf16 physical rows.
// This is row-safe without masked loads and restores the Swin head-layout
// formation that was over-rejected by the initial tail guard.
func.func @direct_register_tile_subvector_rows(
    %input: tensor<1x16x32xf16>, %out: tensor<1x32x16xf16>)
    -> tensor<1x32x16xf16> {
  %tmp = tensor.empty() : tensor<1x16x32xf16>
  %produced = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x16x32xf16>)
      outs(%tmp : tensor<1x16x32xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<1x16x32xf16>
  %transposed = linalg.transpose
      ins(%produced : tensor<1x16x32xf16>)
      outs(%out : tensor<1x32x16xf16>) permutation = [0, 2, 1]
  %consumer_out = tensor.empty() : tensor<1x32x16xf16>
  %result = linalg.generic {
      indexing_maps = [#id3, #id3],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<1x32x16xf16>)
      outs(%consumer_out : tensor<1x32x16xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<1x32x16xf16>
  return %result : tensor<1x32x16xf16>
}

// P2GB-LABEL: func.func @reject_strided_consumer
// P2GB-SAME: alps.p2g.loop_interchanged_direct = 0
// P2GB: linalg.transpose
// P2GB-LABEL: func.func @direct_interchanged_hvx_consumer
// P2GB-SAME: alps.p2e.producer_direct = 1
// P2GB-SAME: alps.p2g.loop_interchanged_direct = 1
// P2GB-NOT: linalg.transpose
// P2GB-LABEL: func.func @reject_register_tile_source_tail
// P2GB: linalg.transpose
// P2GB-LABEL: func.func @direct_register_tile_subvector_rows
// P2GB: linalg.transpose

// P2GC-LABEL: func.func @reject_strided_consumer
// P2GC-SAME: alps.p2e.producer_direct = 1
// P2GC-SAME: alps.p2g.register_tile_direct = 1
// P2GC: alps.p2g.register_tile_contract
// P2GC: alps.p2g.register_tile_sizes = array<i64: 4, 16>
// P2GC-NOT: linalg.transpose
// P2GC-LABEL: func.func @reject_register_tile_source_tail
// P2GC-SAME: alps.p2g.register_tile_direct = 0
// P2GC: linalg.transpose
// P2GC-LABEL: func.func @direct_register_tile_subvector_rows
// P2GC-SAME: alps.p2e.producer_direct = 1
// P2GC-SAME: alps.p2g.register_tile_direct = 1
// P2GC: alps.p2g.register_tile_contract
// P2GC-NOT: linalg.transpose
// P2GC-NONE-LABEL: func.func @reject_strided_consumer
// P2GC-NONE-SAME: alps.p2g.register_tile_direct = 0
// P2GC-NONE: linalg.transpose

func.func @direct_expanded_hvx_consumer(%input: tensor<1x4x12xf16>,
                                        %out: tensor<1x3x4x4xf16>)
    -> tensor<1x3x4x4xf16> {
  %tmp = tensor.empty() : tensor<1x4x12xf16>
  %produced = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x4x12xf16>)
      outs(%tmp : tensor<1x4x12xf16>) {
    ^bb0(%in: f16, %old: f16):
      %v = arith.addf %in, %in : f16
      linalg.yield %v : f16
  } -> tensor<1x4x12xf16>
  %expanded = tensor.expand_shape %produced [[0], [1], [2, 3]]
      output_shape [1, 4, 3, 4]
      : tensor<1x4x12xf16> into tensor<1x4x3x4xf16>
  %transposed = linalg.transpose
      ins(%expanded : tensor<1x4x3x4xf16>)
      outs(%out : tensor<1x3x4x4xf16>)
      permutation = [0, 2, 1, 3]
  %consumer_out = tensor.empty() : tensor<1x3x4x4xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<1x3x4x4xf16>)
      outs(%consumer_out : tensor<1x3x4x4xf16>) {
    ^bb0(%in: f16, %old: f16):
      linalg.yield %in : f16
  } -> tensor<1x3x4x4xf16>
  return %result : tensor<1x3x4x4xf16>
}

// CHECK-LABEL: func.func @direct_expanded_hvx_consumer
// CHECK-SAME: alps.p2e.eliminated_materialization_bytes = 96
// CHECK-SAME: alps.p2e.producer_direct = 1
// CHECK: alps.p2f.consumer_layout_contract = "hvx_innermost_unit_stride"
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: linalg.transpose
