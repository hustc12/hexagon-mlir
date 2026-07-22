//===- OmniFetchDialect.h - OmniFetch Dialect  ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_DIALECT_H
#define HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// OmniFetch Dialect – auto-generated dialect declarations
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h.inc"

//===----------------------------------------------------------------------===//
// OmniFetch Enums / Attributes – auto-generated
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchEnums.h.inc"
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchAttrs.h.inc"

//===----------------------------------------------------------------------===//
// OmniFetch Ops – auto-generated
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchOps.h.inc"

//===----------------------------------------------------------------------===//
// LayoutAwareMapping helpers (defined in LayoutAwareMapping.cpp)
//===----------------------------------------------------------------------===//
namespace mlir {
namespace omni_fetch {

/// Compute a flat i32 offset table mapping logical element indices in the
/// `src` layout to their target positions in the HMX-preferred `dest` layout.
/// Returns the index_map as a vector of int32 values that can be embedded as
/// a DenseI32ArrayAttr on a PrefetchInSituOp.
///
/// The map is computed statically at compile-time from the memref shapes and
/// the requested LayoutTransform kind.
SmallVector<int32_t> computeHMXIndexMap(MLIRContext *ctx, MemRefType srcType,
                                        MemRefType destType,
                                        LayoutTransform transform);

} // namespace omni_fetch
} // namespace mlir

#endif // HEXAGON_DIALECT_OMNIFETCH_IR_OMNIFETCH_DIALECT_H
