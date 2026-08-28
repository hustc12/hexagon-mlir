//===- OmniFetchExternalFnNames.h - Runtime symbol names  -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
// Canonical names of the Hexagon device-side Omni-Fetch runtime functions.
// These must match exactly what OmniFetchRuntime.c exports.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHEXTERNALFNNAMES_H
#define HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHEXTERNALFNNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace omni_fetch {

/// Creates a Hexagon hardware semaphore slot; returns its integer index.
inline llvm::StringRef getCreateSemFnName() {
  return "__omni_fetch_create_sem";
}

/// Posts (signals) the semaphore with the given index.
inline llvm::StringRef getSignalFnName() { return "__omni_fetch_signal"; }

/// Waits (polls) until the semaphore is posted.
inline llvm::StringRef getWaitFnName() { return "__omni_fetch_wait"; }

inline llvm::StringRef getInvocationBeginFnName() {
  return "__omni_fetch_invocation_begin";
}
inline llvm::StringRef getInvocationEndFnName() {
  return "__omni_fetch_invocation_end";
}
inline llvm::StringRef getDescriptorAcquireFnName() {
  return "__omni_fetch_descriptor_acquire";
}
inline llvm::StringRef getDescriptorTransitionFnName() {
  return "__omni_fetch_descriptor_transition";
}
inline llvm::StringRef getDescriptorConsumeFnName() {
  return "__omni_fetch_descriptor_consume";
}
inline llvm::StringRef getDescriptorReleaseFnName() {
  return "__omni_fetch_descriptor_release";
}
inline llvm::StringRef getExactWeightKickFnName() {
  return "__omni_fetch_exact_weight_kick";
}
inline llvm::StringRef getExactWeightConsumeFnName() {
  return "__omni_fetch_exact_weight_consume";
}
inline llvm::StringRef getExactWeightReleaseFnName() {
  return "__omni_fetch_exact_weight_release";
}

/// Performs a layout-aware prefetch: DDR → VTCM with optional gather-reshape.
/// Signature (C):
///   void __omni_fetch_prefetch_insitu(
///       const void *src, void *dest,
///       int32_t elem_bytes, int32_t num_elems,
///       int32_t layout_kind, int32_t lookahead,
///       const int32_t *index_map,   // NULL for non-Custom layouts
///       int32_t tile_row, int32_t tile_col, int32_t src_cols,
///       int32_t act_off, int32_t scr_off, int32_t src_rows);
/// tile_row/col/src_cols are -1 when unused; for HMXWeight+HexKL they select
/// a tile in the full src matrix (src_cols = matrix width).
/// act_off/scr_off/src_rows are -1 when unused; for HMXActivation+HexKL they
/// select VTCM byte offsets and activation matrix height.
inline llvm::StringRef getPrefetchInSituFnName() {
  return "__omni_fetch_prefetch_insitu";
}

/// Issues one page-safe, strided 2-D L2 prefetch request.
/// Signature (C):
///   void __omni_fetch_l2_hint_2d(
///       const void *src, int32_t width_bytes, int32_t height,
///       int32_t stride_bytes);
inline llvm::StringRef getL2Hint2DFnName() { return "__omni_fetch_l2_hint_2d"; }

/// Issues one temporally selected physical row for a repeated segmented CRP
/// site. Signature (C):
///   void __omni_fetch_l2_hint_segmented(
///       const void *src, int32_t width_bytes, int32_t rows,
///       int32_t stride_bytes, int32_t site_id);
inline llvm::StringRef getL2HintSegmentedFnName() {
  return "__omni_fetch_l2_hint_segmented";
}

/// Rank-2 (possibly strided) tile copy used for LAYOUT_NONE prefetches.
/// Signature (C):
///   void __omni_fetch_copy2d(
///       const void *src, void *dest,
///       int32_t elem_bytes, int32_t rows, int32_t cols,
///       int32_t src_row_stride_elems, int32_t dst_row_stride_elems);
inline llvm::StringRef getCopy2DFnName() { return "__omni_fetch_copy2d"; }

/// Reads PMU AXI-stall counter and returns adjusted prefetch distance.
inline llvm::StringRef getUpdateDistanceFnName() {
  return "__omni_fetch_update_distance";
}

/// Enable/disable dual-thread DAE scout (default off). Signature:
///   void __omni_fetch_set_dual_thread_dae(int32_t enable);
inline llvm::StringRef getSetDualThreadDaeFnName() {
  return "__omni_fetch_set_dual_thread_dae";
}

/// Select the model/invocation generation used by the cross-token WH cache.
/// The embedding runtime must increment generation whenever weights belonging
/// to the same context ID may have changed.
inline llvm::StringRef getWhCacheSetContextFnName() {
  return "__omni_fetch_wh_cache_set_context";
}

inline llvm::StringRef getWhCacheInvalidateFnName() {
  return "__omni_fetch_wh_cache_invalidate";
}

inline llvm::StringRef getWhCacheStatsFnName() {
  return "__omni_fetch_wh_cache_stats";
}

} // namespace omni_fetch
} // namespace mlir

#endif // HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHEXTERNALFNNAMES_H
