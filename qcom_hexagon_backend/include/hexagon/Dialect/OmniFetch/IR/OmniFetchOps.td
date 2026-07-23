//===- OmniFetchOps.td - OmniFetch Ops ----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Defines the ops of the omni_fetch dialect.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_OPS_TD
#define HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_OPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.td"

// ---------------------------------------------------------------------------
// Layout transform enum
// ---------------------------------------------------------------------------
def OmniFetch_LayoutNone        : I32EnumAttrCase<"None",        0>;
def OmniFetch_LayoutHMXWeight   : I32EnumAttrCase<"HMXWeight",   1,
    "hmx_weight">;
def OmniFetch_LayoutHMXActivation : I32EnumAttrCase<"HMXActivation", 2,
    "hmx_activation">;
def OmniFetch_LayoutCustom      : I32EnumAttrCase<"Custom",      3>;
def OmniFetch_LayoutL2Hint      : I32EnumAttrCase<"L2Hint",      4,
    "l2_hint">;

def OmniFetch_LayoutTransformAttr : I32EnumAttr<"LayoutTransform",
    "Describes the in-situ layout reshaping required during prefetch",
    [OmniFetch_LayoutNone, OmniFetch_LayoutHMXWeight,
     OmniFetch_LayoutHMXActivation, OmniFetch_LayoutCustom,
     OmniFetch_LayoutL2Hint]> {
  let cppNamespace = "::mlir::omni_fetch";
  let genSpecializedAttr = 1;
}

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
class OmniFetch_Op<string mnemonic, list<Trait> traits = []>
    : Op<OmniFetch_Dialect, mnemonic, traits> {}

// ---------------------------------------------------------------------------
// Semaphore ops
// ---------------------------------------------------------------------------

def OmniFetch_CreateSemOp : OmniFetch_Op<"create_sem", [Pure]> {
  let summary = "Create a Hexagon hardware semaphore handle";
  let description = [{
    Allocates one Hexagon hardware semaphore slot and returns an opaque
    handle (represented as `index`).  The semaphore is initialised to zero.
    Each `create_sem` must be paired with exactly one `wait` and one
    `signal` per loop iteration in the enclosing V-DAE region.

    Example:
    ```mlir
    %sem = omni_fetch.create_sem : index
    ```
  }];

  let results = (outs Index:$sem_handle);
  let assemblyFormat = "attr-dict `:` type($sem_handle)";
}

def OmniFetch_SignalOp : OmniFetch_Op<"signal",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Signal (post) a hardware semaphore (Access Thread side)";
  let description = [{
    Posts the semaphore identified by `sem_handle`, indicating that the
    prefetch for the current iteration is complete and the corresponding
    VTCM tile is ready for consumption by the Execute Thread.

    Maps to the Hexagon `sem_signal` instruction (< 10 cycles overhead).

    Example:
    ```mlir
    omni_fetch.signal %sem : index
    ```
  }];

  let arguments = (ins Index:$sem_handle);
  let assemblyFormat = "$sem_handle attr-dict `:` type($sem_handle)";
}

def OmniFetch_WaitOp : OmniFetch_Op<"wait",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Wait on a hardware semaphore (Execute Thread side)";
  let description = [{
    Stalls the Execute Thread until the semaphore `sem_handle` is posted
    by the Access Thread, guaranteeing that the VTCM tile is valid before
    the HMX compute begins.

    Maps to a polling `sem_wait` / `p-wait` pattern on Hexagon hardware.

    Example:
    ```mlir
    omni_fetch.wait %sem : index
    ```
  }];

  let arguments = (ins Index:$sem_handle);
  let assemblyFormat = "$sem_handle attr-dict `:` type($sem_handle)";
}

// ---------------------------------------------------------------------------
// Prefetch op
// ---------------------------------------------------------------------------

def OmniFetch_PrefetchInSituOp : OmniFetch_Op<"prefetch_in_situ",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Layout-aware in-situ prefetch from DDR to VTCM";
  let description = [{
    Initiates a prefetch of tensor data from the DDR `src` memref into
    the VTCM `dest` memref, performing an optional in-situ layout
    transformation during the transfer.

    When `layout_transform` is `None`, the op performs a straight
    `l2fetch`-style linear copy.

    When `layout_transform` is `HMXWeight` or `HMXActivation`, the
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
    omni_fetch.prefetch_in_situ %weight_ddr, %weight_vtcm
        {layout_transform = #omni_fetch<layout_transform HMXWeight>,
         lookahead = 2 : i32}
        : memref<?x?xf16>, memref<32x64xf16, 1>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dest,
    OmniFetch_LayoutTransformAttr:$layout_transform,
    DefaultValuedAttr<I32Attr, "2">:$lookahead,
    OptionalAttr<DenseI32ArrayAttr>:$index_map
  );

  let assemblyFormat = [{
    $src `,` $dest
    attr-dict
    `:` qualified(type($src)) `,` qualified(type($dest))
  }];

  let hasVerifier = 1;
}

// ---------------------------------------------------------------------------
// Adaptive control op
// ---------------------------------------------------------------------------

def OmniFetch_AdaptiveControlOp : OmniFetch_Op<"adaptive_control",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Query PMU counters and update prefetch look-ahead distance";
  let description = [{
    Reads the Hexagon PMU AXI-stall counter and returns an updated
    prefetch look-ahead distance.  Implements the control policy:

      - If AXI stall delta > threshold → decrease distance (back off)
      - Otherwise → increase distance (be more aggressive)

    This op is emitted by the V-DAE pass at the loop-back edge and is
    lowered to a call to `__omni_fetch_update_distance()` in the runtime.

    Example:
    ```mlir
    %new_dist = omni_fetch.adaptive_control(%current_dist : i32) -> i32
    ```
  }];

  let arguments = (ins I32:$current_distance);
  let results   = (outs I32:$new_distance);
  let assemblyFormat = [{
    `(` $current_distance `:` type($current_distance) `)`
    attr-dict `->` type($new_distance)
  }];
}

#endif // HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_OPS_TD
