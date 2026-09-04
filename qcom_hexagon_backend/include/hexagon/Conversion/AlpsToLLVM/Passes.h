//===- Passes.h - Convert Alps to LLVM ops ----------------*- C++ -*-===//

#ifndef HEXAGON_CONVERSION_ALPSTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_ALPSTOLLVM_PASSES_H

#include "hexagon/Conversion/AlpsToLLVM/AlpsToLLVM.h"

namespace mlir {
namespace alps {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/AlpsToLLVM/Passes.h.inc"

} // namespace alps
} // namespace mlir

#endif // HEXAGON_CONVERSION_ALPSTOLLVM_PASSES_H
