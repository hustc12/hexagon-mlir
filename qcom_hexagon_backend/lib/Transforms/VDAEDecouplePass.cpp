//===- VDAEDecouplePass_v2.cpp - V-DAE Access-Execute decoupling ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Implements the "Virtual Decoupled Access-Execute" (V-DAE) transformation
// for DNN inference on Hexagon NPU.
//
// Motivation
// ----------
// VLIW (Very Long Instruction Word) architecture of Hexagon means a single
// memory load stall can halt an entire instruction packet, freezing all
// compute units in the same cycle.
//
// This pass addresses this problem by decoupling Memory Access and Compute
// Execution using hardware semaphores, allowing Access Thread and Execute
// Thread to run in parallel.
//
// Prerequisites
// -------------
// This pass requires that prefetch operations have already been inserted
// by the PrefetchInsertPass. It detects existing `omni_fetch.prefetch_in_situ`
// operations and adds synchronization around them.
//
// Transformation overview
// -----------------------
// For each loop containing prefetch operations, the pass:
//
//   A. CREATES a hardware semaphore for synchronization.
//
//   B. EMITS PROLOGUE before the loop:
//        %sem = omni_fetch.create_sem
//        <existing prefetch operations>
//        omni_fetch.signal %sem
//
//   C. REWRITES THE LOOP BODY:
//        omni_fetch.wait %sem           // ensure tile[i] ready
//        <original HMX compute on vtcm_tile[i]>
//        <existing prefetch for tile[i+K]>
//        omni_fetch.signal %sem         // notify tile[i+K] ready
//        %dist = omni_fetch.adaptive_control(%dist)  // optional
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "omni-fetch-vdae-decouple"

using namespace mlir;
using namespace mlir::omni_fetch;
using namespace hexagon;

#define GEN_PASS_DEF_OMNIFETCHVDAEINSERT
#include "hexagon/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Collect all prefetch operations before the loop (prologue prefetches).
/// These are prefetch operations that are defined before the loop starts.
static SmallVector<PrefetchInSituOp> collectProloguePrefetches(scf::ForOp loop) {
  SmallVector<PrefetchInSituOp> prefetches;
  
  // Walk backwards from the loop to find prefetch operations
  Operation *op = loop.getOperation();
  while (op) {
    // Check previous operation
    op = op->getPrevNode();
    if (!op)
      break;
    
    // If we find a prefetch, add it
    if (auto prefetch = dyn_cast<PrefetchInSituOp>(op)) {
      prefetches.push_back(prefetch);
    }
    
    // Stop if we hit another loop or a different control flow structure
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op))
      break;
  }
  
  // Reverse to get them in execution order
  std::reverse(prefetches.begin(), prefetches.end());
  return prefetches;
}

/// Collect all prefetch operations inside the loop body.
static SmallVector<PrefetchInSituOp> collectLoopBodyPrefetches(scf::ForOp loop) {
  SmallVector<PrefetchInSituOp> prefetches;
  loop.getBody()->walk([&](PrefetchInSituOp op) {
    prefetches.push_back(op);
  });
  return prefetches;
}

//===----------------------------------------------------------------------===//
// Core transformation: add synchronization to a loop with prefetch
//===----------------------------------------------------------------------===//

static void addSynchronizationToLoop(scf::ForOp loop, bool enableAdaptive) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();

  // Check if there are prefetch operations associated with this loop
  SmallVector<PrefetchInSituOp> prologuePrefetches = collectProloguePrefetches(loop);
  SmallVector<PrefetchInSituOp> loopBodyPrefetches = collectLoopBodyPrefetches(loop);

  if (prologuePrefetches.empty() && loopBodyPrefetches.empty()) {
    // No prefetch operations, nothing to synchronize
    return;
  }

  // Create semaphore before the loop
  builder.setInsertionPoint(loop);
  Value sem = builder.create<CreateSemOp>(loc, builder.getIndexType());

  // Signal after prologue prefetches (which are before the loop)
  // Insert signal right before the loop starts
  builder.create<SignalOp>(loc, sem);

  // Insert wait at the top of loop body
  Block *body = loop.getBody();
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(body);
    builder.create<WaitOp>(loc, sem);
  }

  // Insert signal + adaptive control before yield
  {
    OpBuilder::InsertionGuard g(builder);
    Operation *yieldOp = body->getTerminator();
    builder.setInsertionPoint(yieldOp);

    // Signal after issuing next-iteration prefetch
    builder.create<SignalOp>(loc, sem);

    // Adaptive control (optional)
    if (enableAdaptive) {
      // Get lookahead from first prefetch operation
      int lookahead = 2;  // default
      if (!prologuePrefetches.empty()) {
        lookahead = prologuePrefetches[0].getLookahead();
      } else if (!loopBodyPrefetches.empty()) {
        lookahead = loopBodyPrefetches[0].getLookahead();
      }
      
      Value initDist =
          builder.create<arith::ConstantIntOp>(loc, (int64_t)lookahead, 32u);
      builder.create<AdaptiveControlOp>(loc, builder.getI32Type(), initDist);
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct OmniFetchVDAEInsertPass
    : public ::impl::OmniFetchVDAEInsertBase<OmniFetchVDAEInsertPass> {

  explicit OmniFetchVDAEInsertPass() = default;
  explicit OmniFetchVDAEInsertPass(const OmniFetchVDAEInsertOptions &options)
      : OmniFetchVDAEInsertBase(options) {}

  void runOnOperation() override {
    auto func = cast<func::FuncOp>(getOperation());

    // Find all loops and add synchronization if they have prefetch operations
    func.walk([&](scf::ForOp loop) {
      addSynchronizationToLoop(loop, enableAdaptive);
    });
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public factory
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createOmniFetchVDAEInsertPass(
    const OmniFetchVDAEInsertOptions &options) {
  return std::make_unique<OmniFetchVDAEInsertPass>(options);
}
