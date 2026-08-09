//===- OmniFetchToLLVM.h - OmniFetch to LLVM Conversion  ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHTOLLVM_H
#define HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace omni_fetch {

#define GEN_PASS_DECL
#include "hexagon/Conversion/OmniFetchToLLVM/Passes.h.inc"

/// Create the pass that lowers omni_fetch dialect ops to LLVM runtime calls.
std::unique_ptr<Pass> createOmniFetchToLLVMPass(
    const OmniFetchToLLVMOptions &options = OmniFetchToLLVMOptions());

} // namespace omni_fetch
} // namespace mlir

#endif // HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHTOLLVM_H
