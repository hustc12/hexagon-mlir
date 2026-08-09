//===- OmniFetchOps.cpp - OmniFetch Op Implementations  -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::omni_fetch;

//===----------------------------------------------------------------------===//
// OmniFetchDialect::registerOperations
//   Must be defined here (next to the ops) so the linker finds it.
//===----------------------------------------------------------------------===//
void OmniFetchDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Auto-generated op table-gen boilerplate
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Auto-generated enum definitions
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// PrefetchInSituOp::verify
//===----------------------------------------------------------------------===//
LogicalResult PrefetchInSituOp::verify() {
  auto srcType = cast<MemRefType>(getSrc().getType());
  auto dstType = cast<MemRefType>(getDest().getType());

  // dest must reside in address space 0 (DDR/heap via malloc) or 1 (VTCM).
  // Address space 0 is used when true VTCM allocation is not yet available;
  // the prefetch still performs a useful DDR→DDR copy for software pipelining.
  int destAS = dstType.getMemorySpaceAsInt();
  if (destAS != 0 && destAS != 1)
    return emitOpError("dest memref must be in DDR (address space 0) or VTCM (address space 1)");

  // Element types must match, except HexKL HMX* fusion which writes into the
  // i8 VTCM slab via hexkl_micro (src is f16 DDR, dest is i8 VTCM).
  bool hexklSlabFusion =
      (getLayoutTransform() == LayoutTransform::HMXActivation &&
       getTileParams().size() == 6) ||
      ((getLayoutTransform() == LayoutTransform::HMXWeight ||
        getLayoutTransform() == LayoutTransform::HMXWeightDequantI8) &&
       getTileParams().size() >= 4);
  if (!hexklSlabFusion &&
      srcType.getElementType() != dstType.getElementType())
    return emitOpError(
        "src and dest element types must match; got ")
        << srcType.getElementType() << " vs " << dstType.getElementType();

  // Custom index_map must cover every dest element.
  if (auto idxMap = getIndexMap()) {
    if (getLayoutTransform() == LayoutTransform::Custom) {
      int64_t numElems = 1;
      for (auto d : dstType.getShape())
        numElems *= d;
      if ((int64_t)idxMap->size() != numElems)
        return emitOpError("custom index_map size (")
               << idxMap->size()
               << ") must equal total dest element count (" << numElems << ")";
    }
  }

  // HexKL tile_params: empty, weight (3/4/5), or activation (6).
  auto tileParams = getTileParams();
  unsigned n = tileParams.size();
  if (!tileParams.empty() && n != 3 && n != 4 && n != 5 && n != 6)
    return emitOpError("tile_params must be empty, 3-5 i32 values "
                       "(tile_row, tile_col, src_cols[, weight_off[, stage_off]]), "
                       "or 6 i32 values (…, act_off, scr_off, src_rows)");

  return success();
}
