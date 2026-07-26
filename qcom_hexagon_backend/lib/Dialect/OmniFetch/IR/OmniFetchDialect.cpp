//===- OmniFetchDialect.cpp - OmniFetch Dialect Implementation  -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::omni_fetch;

/// Dialect initialization.  The instance is owned by the MLIRContext.
void OmniFetchDialect::initialize() {
  registerOperations();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchAttrs.cpp.inc"
      >();
}

Attribute OmniFetchDialect::parseAttribute(DialectAsmParser &parser,
                                           Type type) const {
  StringRef tag;
  if (failed(parser.parseKeyword(&tag)))
    return {};
  if (tag != "layout_transform") {
    parser.emitError(parser.getNameLoc(), "unknown omni_fetch attribute: ")
        << tag;
    return {};
  }
  if (failed(parser.parseLess()))
    return {};
  StringRef value;
  if (failed(parser.parseKeyword(&value)))
    return {};
  auto transform = symbolizeLayoutTransform(value);
  if (!transform) {
    parser.emitError(parser.getNameLoc(), "unknown layout transform: ") << value;
    return {};
  }
  if (failed(parser.parseGreater()))
    return {};
  return LayoutTransformAttr::get(getContext(), *transform);
}

void OmniFetchDialect::printAttribute(Attribute attr,
                                      DialectAsmPrinter &printer) const {
  auto transform = cast<LayoutTransformAttr>(attr);
  printer << "layout_transform<"
          << stringifyLayoutTransform(transform.getValue()) << ">";
}

//===----------------------------------------------------------------------===//
// Auto-generated dialect / attr implementations (must come after initialize)
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchAttrs.cpp.inc"
