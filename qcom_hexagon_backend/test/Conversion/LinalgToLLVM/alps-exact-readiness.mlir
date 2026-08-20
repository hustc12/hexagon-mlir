// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(omni-fetch-to-llvm)' | FileCheck %s

func.func @exact_readiness(%version: index, %tile: index) -> i1 {
  %layout = arith.constant 1 : i32
  %ddr = arith.constant 0 : i32
  %vtcm = arith.constant 1 : i32
  %load_pending = arith.constant 1 : i32
  %layout_pending = arith.constant 2 : i32
  %ready = arith.constant 3 : i32
  %context = omni_fetch.invocation_begin : index
  %descriptor = omni_fetch.descriptor_acquire
      %context, %version, %tile, %layout, %ddr, %vtcm
      : index, index, index, i32, i32, i32 -> index
  %loaded = omni_fetch.descriptor_transition
      %descriptor, %load_pending, %layout_pending
      : index, i32, i32 -> i1
  %transformed = omni_fetch.descriptor_transition
      %descriptor, %layout_pending, %ready
      : index, i32, i32 -> i1
  %consumed = omni_fetch.descriptor_consume
      %descriptor, %version, %tile, %layout, %ddr, %vtcm
      : index, index, index, i32, i32, i32 -> i1
  %released = omni_fetch.descriptor_release %descriptor : index -> i1
  %ended = omni_fetch.invocation_end %context : index -> i1
  return %ended : i1
}

func.func @exact_weight_pipeline(
    %src: memref<64x64xf16>, %dst: memref<?xi8, 1>,
    %context: index, %version: index, %tile: i32) {
  %zero = arith.constant 0 : i32
  %cols = arith.constant 64 : i32
  %no_stage = arith.constant -1 : i32
  %scheduled = omni_fetch.exact_weight_kick
      %context, %version, %src, %dst, %tile, %zero, %cols, %zero, %no_stage
      : index, index, memref<64x64xf16>, memref<?xi8, 1>,
        i32, i32, i32, i32, i32 -> i1
  %consumed = omni_fetch.exact_weight_consume
      %context, %version, %tile, %zero : index, index, i32, i32 -> i1
  %released = omni_fetch.exact_weight_release
      %context, %version, %tile, %zero : index, index, i32, i32 -> i1
  return
}

// CHECK-LABEL: func.func @exact_readiness
// CHECK: llvm.call @__omni_fetch_invocation_begin
// CHECK: llvm.call @__omni_fetch_descriptor_acquire
// CHECK-COUNT-2: llvm.call @__omni_fetch_descriptor_transition
// CHECK: llvm.call @__omni_fetch_descriptor_consume
// CHECK: llvm.call @__omni_fetch_descriptor_release
// CHECK: llvm.call @__omni_fetch_invocation_end
// CHECK-LABEL: func.func @exact_weight_pipeline
// CHECK: llvm.call @__omni_fetch_exact_weight_kick
// CHECK: llvm.call @__omni_fetch_exact_weight_consume
// CHECK: llvm.call @__omni_fetch_exact_weight_release
