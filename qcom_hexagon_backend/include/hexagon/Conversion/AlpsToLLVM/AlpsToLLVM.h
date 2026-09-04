//===- AlpsToLLVM.h - Alps to LLVM Conversion  ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_ALPSTOLLVM_ALPSTOLLVM_H
#define HEXAGON_CONVERSION_ALPSTOLLVM_ALPSTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace alps {

#define GEN_PASS_DECL
#include "hexagon/Conversion/AlpsToLLVM/Passes.h.inc"

/// Create the pass that lowers alps dialect ops to LLVM runtime calls.
std::unique_ptr<Pass> createAlpsToLLVMPass(
    const AlpsToLLVMOptions &options = AlpsToLLVMOptions());

} // namespace alps
} // namespace mlir

#endif // HEXAGON_CONVERSION_ALPSTOLLVM_ALPSTOLLVM_H
