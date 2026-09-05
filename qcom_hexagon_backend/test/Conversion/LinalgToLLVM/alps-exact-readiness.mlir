// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(alps-to-llvm)' | FileCheck %s

func.func @exact_readiness(%version: index, %tile: index) -> i1 {
  %layout = arith.constant 1 : i32
  %ddr = arith.constant 0 : i32
  %vtcm = arith.constant 1 : i32
  %load_pending = arith.constant 1 : i32
  %layout_pending = arith.constant 2 : i32
  %ready = arith.constant 3 : i32
  %context = alps.invocation_begin : index
  %descriptor = alps.descriptor_acquire
      %context, %version, %tile, %layout, %ddr, %vtcm
      : index, index, index, i32, i32, i32 -> index
  %loaded = alps.descriptor_transition
      %descriptor, %load_pending, %layout_pending
      : index, i32, i32 -> i1
  %transformed = alps.descriptor_transition
      %descriptor, %layout_pending, %ready
      : index, i32, i32 -> i1
  %consumed = alps.descriptor_consume
      %descriptor, %version, %tile, %layout, %ddr, %vtcm
      : index, index, index, i32, i32, i32 -> i1
  %released = alps.descriptor_release %descriptor : index -> i1
  %ended = alps.invocation_end %context : index -> i1
  return %ended : i1
}

func.func @exact_weight_pipeline(
    %src: memref<64x64xf16>, %dst: memref<?xi8, 1>,
    %context: index, %version: index, %tile: i32) {
  %zero = arith.constant 0 : i32
  %cols = arith.constant 64 : i32
  %no_stage = arith.constant -1 : i32
  %one = arith.constant 1 : i32
  %scheduled = alps.exact_weight_kick
      %context, %version, %src, %dst, %tile, %zero, %cols, %zero, %no_stage,
      %one
      : index, index, memref<64x64xf16>, memref<?xi8, 1>,
        i32, i32, i32, i32, i32, i32 -> i1
  %consumed = alps.exact_weight_consume
      %context, %version, %tile, %zero : index, index, i32, i32 -> i1
  %released = alps.exact_weight_release
      %context, %version, %tile, %zero : index, index, i32, i32 -> i1
  return
}

// CHECK-LABEL: func.func @exact_readiness
// CHECK: llvm.call @__alps_invocation_begin
// CHECK: llvm.call @__alps_descriptor_acquire
// CHECK-COUNT-2: llvm.call @__alps_descriptor_transition
// CHECK: llvm.call @__alps_descriptor_consume
// CHECK: llvm.call @__alps_descriptor_release
// CHECK: llvm.call @__alps_invocation_end
// CHECK-LABEL: func.func @exact_weight_pipeline
// CHECK: llvm.call @__alps_exact_weight_kick
// CHECK: llvm.call @__alps_exact_weight_consume
// CHECK: llvm.call @__alps_exact_weight_release
