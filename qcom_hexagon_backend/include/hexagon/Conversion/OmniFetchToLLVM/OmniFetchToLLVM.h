//===- OmniFetchToLLVM.h - OmniFetch to LLVM Conversion  ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHTOLLVM_H
#define HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHTOLLVM_H

#include <memory>

namespace mlir {
class Pass;
namespace omni_fetch {

/// Create the pass that lowers omni_fetch dialect ops to LLVM runtime calls.
std::unique_ptr<Pass> createOmniFetchToLLVMPass();

} // namespace omni_fetch
} // namespace mlir

#endif // HEXAGON_CONVERSION_OMNIFETCHTOLLVM_OMNIFETCHTOLLVM_H
