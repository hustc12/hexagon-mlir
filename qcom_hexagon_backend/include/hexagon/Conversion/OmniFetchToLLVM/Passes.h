//===- Passes.h - Convert OmniFetch to LLVM ops ----------------*- C++ -*-===//

#ifndef HEXAGON_CONVERSION_OMNIFETCHTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_OMNIFETCHTOLLVM_PASSES_H

#include "hexagon/Conversion/OmniFetchToLLVM/OmniFetchToLLVM.h"

namespace mlir {
namespace omni_fetch {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/OmniFetchToLLVM/Passes.h.inc"

} // namespace omni_fetch
} // namespace mlir

#endif // HEXAGON_CONVERSION_OMNIFETCHTOLLVM_PASSES_H
