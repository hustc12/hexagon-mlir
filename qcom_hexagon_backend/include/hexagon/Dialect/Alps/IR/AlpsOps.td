//===- AlpsOps.td - Alps Ops ----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Defines the ops of the alps dialect.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_ALPS_IR_ALPS_OPS_TD
#define HEXAGON_DIALECT_ALPS_IR_ALPS_OPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "hexagon/Dialect/Alps/IR/AlpsDialect.td"

// ---------------------------------------------------------------------------
// Layout transform enum
// ---------------------------------------------------------------------------
def Alps_LayoutNone        : I32EnumAttrCase<"None",        0>;
def Alps_LayoutHMXWeight   : I32EnumAttrCase<"HMXWeight",   1,
    "hmx_weight">;
def Alps_LayoutHMXActivation : I32EnumAttrCase<"HMXActivation", 2,
    "hmx_activation">;
def Alps_LayoutCustom      : I32EnumAttrCase<"Custom",      3>;
def Alps_LayoutL2Hint      : I32EnumAttrCase<"L2Hint",      4,
    "l2_hint">;
def Alps_LayoutHMXWeightDequantI8 :
    I32EnumAttrCase<"HMXWeightDequantI8", 5, "hmx_weight_dequant_i8">;

def Alps_LayoutTransformAttr : I32EnumAttr<"LayoutTransform",
    "Describes the in-situ layout reshaping required during prefetch",
    [Alps_LayoutNone, Alps_LayoutHMXWeight,
     Alps_LayoutHMXActivation, Alps_LayoutCustom,
     Alps_LayoutL2Hint, Alps_LayoutHMXWeightDequantI8]> {
  let cppNamespace = "::mlir::alps";
  let genSpecializedAttr = 1;
}

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
class Alps_Op<string mnemonic, list<Trait> traits = []>
    : Op<Alps_Dialect, mnemonic, traits> {}

// ---------------------------------------------------------------------------
// Semaphore ops
// ---------------------------------------------------------------------------

def Alps_CreateSemOp : Alps_Op<"create_sem", [Pure]> {
  let summary = "Create a Hexagon hardware semaphore handle";
  let description = [{
    Allocates one Hexagon hardware semaphore slot and returns an opaque
    handle (represented as `index`).  The semaphore is initialised to zero.
    Each `create_sem` must be paired with exactly one `wait` and one
    `signal` per loop iteration in the enclosing V-DAE region.

    Example:
    ```mlir
    %sem = alps.create_sem : index
    ```
  }];

  let results = (outs Index:$sem_handle);
  let assemblyFormat = "attr-dict `:` type($sem_handle)";
}

def Alps_SignalOp : Alps_Op<"signal",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Signal (post) a hardware semaphore (Access Thread side)";
  let description = [{
    Posts the semaphore identified by `sem_handle`, indicating that the
    prefetch for the current iteration is complete and the corresponding
    VTCM tile is ready for consumption by the Execute Thread.

    Maps to the Hexagon `sem_signal` instruction (< 10 cycles overhead).

    Example:
    ```mlir
    alps.signal %sem : index
    ```
  }];

  let arguments = (ins Index:$sem_handle);
  let assemblyFormat = "$sem_handle attr-dict `:` type($sem_handle)";
}

def Alps_WaitOp : Alps_Op<"wait",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Wait on a hardware semaphore (Execute Thread side)";
  let description = [{
    Stalls the Execute Thread until the semaphore `sem_handle` is posted
    by the Access Thread, guaranteeing that the VTCM tile is valid before
    the HMX compute begins.

    Maps to a polling `sem_wait` / `p-wait` pattern on Hexagon hardware.

    Example:
    ```mlir
    alps.wait %sem : index
    ```
  }];

  let arguments = (ins Index:$sem_handle);
  let assemblyFormat = "$sem_handle attr-dict `:` type($sem_handle)";
}

// ---------------------------------------------------------------------------
// ALPS P3 exact-readiness descriptor ops
// ---------------------------------------------------------------------------

def Alps_InvocationBeginOp : Alps_Op<"invocation_begin",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Begin an invocation-local exact-readiness context";
  let description = [{
    Acquires a generation-tagged runtime context.  The opaque index result is
    the only authority under which P3 descriptors may be acquired; it avoids
    the implicit process-global ownership of the legacy async ring.
  }];
  let results = (outs Index:$context);
  let assemblyFormat = "attr-dict `:` type($context)";
}

def Alps_InvocationEndOp : Alps_Op<"invocation_end",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "End an invocation-local exact-readiness context";
  let arguments = (ins Index:$context);
  let results = (outs I1:$success);
  let assemblyFormat = "$context attr-dict `:` type($context) `->` type($success)";
}

def Alps_DescriptorAcquireOp : Alps_Op<"descriptor_acquire",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Acquire one exact tile descriptor in LOAD_PENDING state";
  let description = [{
    Binds one runtime slot to the exact tuple (invocation generation, value
    version, tile, layout, source tier, destination tier).  A negative opaque
    handle means that bounded descriptor capacity was unavailable and P3b
    must take its synchronous fallback.
  }];
  let arguments = (ins Index:$context, Index:$value_version, Index:$tile,
                       I32:$layout, I32:$source_tier, I32:$destination_tier);
  let results = (outs Index:$descriptor);
  let assemblyFormat = [{
    $context `,` $value_version `,` $tile `,` $layout `,` $source_tier `,`
    $destination_tier attr-dict `:` type($context) `,` type($value_version)
    `,` type($tile) `,` type($layout) `,` type($source_tier) `,`
    type($destination_tier) `->` type($descriptor)
  }];
}

def Alps_DescriptorTransitionOp : Alps_Op<"descriptor_transition",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Perform one checked exact-readiness state transition";
  let arguments = (ins Index:$descriptor, I32:$expected_state,
                       I32:$next_state);
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $descriptor `,` $expected_state `,` $next_state attr-dict `:`
    type($descriptor) `,` type($expected_state) `,` type($next_state)
    `->` type($success)
  }];
}

def Alps_DescriptorConsumeOp : Alps_Op<"descriptor_consume",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Consume READY only when the complete tile identity matches";
  let arguments = (ins Index:$descriptor, Index:$value_version, Index:$tile,
                       I32:$layout, I32:$source_tier, I32:$destination_tier);
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $descriptor `,` $value_version `,` $tile `,` $layout `,` $source_tier `,`
    $destination_tier attr-dict `:` type($descriptor) `,` type($value_version)
    `,` type($tile) `,` type($layout) `,` type($source_tier) `,`
    type($destination_tier) `->` type($success)
  }];
}

def Alps_DescriptorReleaseOp : Alps_Op<"descriptor_release",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Release a CONSUMING descriptor back to FREE";
  let arguments = (ins Index:$descriptor);
  let results = (outs I1:$success);
  let assemblyFormat = "$descriptor attr-dict `:` type($descriptor) `->` type($success)";
}

def Alps_ExactWeightKickOp : Alps_Op<"exact_weight_kick",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Start descriptor-bound DMA for one future HMX weight tile";
  let arguments = (ins Index:$context, Index:$value_version,
                       AnyMemRef:$src, AnyMemRef:$dest,
                       I32:$tile_row, I32:$tile_col, I32:$source_cols,
                       I32:$weight_offset, I32:$stage_offset);
  let results = (outs I1:$scheduled);
  let assemblyFormat = [{
    $context `,` $value_version `,` $src `,` $dest `,` $tile_row `,`
    $tile_col `,` $source_cols `,` $weight_offset `,` $stage_offset
    attr-dict `:` type($context) `,` type($value_version) `,`
    qualified(type($src)) `,` qualified(type($dest)) `,` type($tile_row) `,`
    type($tile_col) `,` type($source_cols) `,` type($weight_offset) `,`
    type($stage_offset) `->` type($scheduled)
  }];
}

def Alps_ExactWeightConsumeOp : Alps_Op<"exact_weight_consume",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for and consume the exact READY HMX weight tile";
  let arguments = (ins Index:$context, Index:$value_version,
                       I32:$tile_row, I32:$tile_col);
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $context `,` $value_version `,` $tile_row `,` $tile_col attr-dict `:`
    type($context) `,` type($value_version) `,` type($tile_row) `,`
    type($tile_col) `->` type($success)
  }];
}

def Alps_ExactWeightReleaseOp : Alps_Op<"exact_weight_release",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Release the exact weight tile after its HMX consumer";
  let arguments = (ins Index:$context, Index:$value_version,
                       I32:$tile_row, I32:$tile_col);
  let results = (outs I1:$success);
  let assemblyFormat = [{
    $context `,` $value_version `,` $tile_row `,` $tile_col attr-dict `:`
    type($context) `,` type($value_version) `,` type($tile_row) `,`
    type($tile_col) `->` type($success)
  }];
}

// ---------------------------------------------------------------------------
// Prefetch op
// ---------------------------------------------------------------------------

def Alps_L2HintOp : Alps_Op<"l2_hint",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Read-only L2 prefetch hint for one memref region";
  let description = [{
    Requests an L2 prefetch for the complete static source region.  Unlike
    `prefetch_in_situ`, this operation has no destination operand, so the live
    K/V source does not become a destination-style buffer.  The conservative
    global write effect deliberately makes this performance-affecting hardware
    request non-DCE-able; it does not describe a write to the source memref.
  }];

  let arguments = (ins
    AnyMemRef:$src,
    DefaultValuedAttr<I32Attr, "1">:$lookahead
  );

  let assemblyFormat =
      "$src attr-dict `:` qualified(type($src))";
}

def Alps_PrefetchInSituOp : Alps_Op<"prefetch_in_situ",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Layout-aware in-situ prefetch from DDR to VTCM";
  let description = [{
    Initiates a prefetch of tensor data from the DDR `src` memref into
    the VTCM `dest` memref, performing an optional in-situ layout
    transformation during the transfer.

    When `layout_transform` is `None`, the op performs a straight
    `l2fetch`-style linear copy.

    When `layout_transform` is `HMXWeight`, `HMXWeightDequantI8`, or
    `HMXActivation`, the
    runtime uses `vgather`-based scatter-gather to place each element
    directly in the HMX-preferred interleaved format, eliminating a
    separate reshape step (the "Layout Wall" problem).

    When `layout_transform` is `Custom`, the optional `index_map`
    attribute supplies an explicit element-wise offset table (a flat
    array of i32 offsets).  This enables arbitrary strided or tiled
    access patterns computed statically by the compiler's
    `LayoutAwareMapping` analysis.

    The `lookahead` attribute records the number of loop iterations by
    which this prefetch leads the corresponding compute — used by the
    runtime's adaptive scheduler to tune aggressiveness.

    Example (weight tile, HMX layout):
    ```mlir
    alps.prefetch_in_situ %weight_ddr, %weight_vtcm
        {layout_transform = #alps<layout_transform HMXWeight>,
         lookahead = 2 : i32}
        : memref<?x?xf16>, memref<32x64xf16, 1>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dest,
    Alps_LayoutTransformAttr:$layout_transform,
    DefaultValuedAttr<I32Attr, "2">:$lookahead,
    OptionalAttr<DenseI32ArrayAttr>:$index_map,
    // Optional HexKL tile metadata:
    //   Weight (3): [tile_row, tile_col, src_cols]
    //   Activation (6): [tile_row, tile_col, src_cols, act_off, scr_off, src_rows]
    // When present with HMXWeight, runtime calls hexkl_micro_hmx_rm_to_wh_f16
    // on the full src matrix (handles strided DDR tiles correctly).
    // HMXWeightDequantI8 retains an int8+scale tile in the runtime cache and
    // fuses compressed load, dequantization, and RM->WH production.
    // When present with HMXActivation (6), runtime runs CopySubmatrix+RmToAh.
    Variadic<I32>:$tile_params
  );

  let assemblyFormat = [{
    $src `,` $dest
    (`tile_params` `(` $tile_params^ `:` type($tile_params) `)`)?
    attr-dict
    `:` qualified(type($src)) `,` qualified(type($dest))
  }];

  let hasVerifier = 1;
}

// ---------------------------------------------------------------------------
// Adaptive control op
// ---------------------------------------------------------------------------

def Alps_AdaptiveControlOp : Alps_Op<"adaptive_control",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Query PMU counters and update prefetch look-ahead distance";
  let description = [{
    Reads the Hexagon PMU AXI-stall counter and returns an updated
    prefetch look-ahead distance.  Implements the control policy:

      - If AXI stall delta > threshold → decrease distance (back off)
      - Otherwise → increase distance (be more aggressive)

    This op is emitted by the V-DAE pass at the loop-back edge and is
    lowered to a call to `__alps_update_distance()` in the runtime.

    Example:
    ```mlir
    %new_dist = alps.adaptive_control(%current_dist : i32) -> i32
    ```
  }];

  let arguments = (ins I32:$current_distance);
  let results   = (outs I32:$new_distance);
  let assemblyFormat = [{
    `(` $current_distance `:` type($current_distance) `)`
    attr-dict `->` type($new_distance)
  }];
}

#endif // HEXAGON_DIALECT_ALPS_IR_ALPS_OPS_TD
