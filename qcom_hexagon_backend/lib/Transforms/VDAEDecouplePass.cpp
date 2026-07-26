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

/// Collect prefetch ops whose nearest enclosing scf.for is `loop`.
/// Walking nested loops would otherwise attach wait/signal to parents and
/// complete async DMA too early (Phase 2b corruption).
static SmallVector<PrefetchInSituOp> collectLoopBodyPrefetches(scf::ForOp loop) {
  SmallVector<PrefetchInSituOp> prefetches;
  loop.getBody()->walk([&](PrefetchInSituOp op) {
    if (op->getParentOfType<scf::ForOp>() == loop)
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

  // L2-hint prefetches are fire-and-forget cache warmups; wait/signal around
  // them only adds overhead and must not gate compute.
  auto isRealPrefetch = [](PrefetchInSituOp op) {
    return op.getLayoutTransform() != LayoutTransform::L2Hint;
  };
  prologuePrefetches.erase(
      llvm::remove_if(prologuePrefetches, [&](PrefetchInSituOp op) {
        return !isRealPrefetch(op);
      }),
      prologuePrefetches.end());
  loopBodyPrefetches.erase(
      llvm::remove_if(loopBodyPrefetches, [&](PrefetchInSituOp op) {
        return !isRealPrefetch(op);
      }),
      loopBodyPrefetches.end());

  llvm::dbgs() << "[VDAEDecouple]   Found " << prologuePrefetches.size() 
               << " prologue prefetches\n";
  llvm::dbgs() << "[VDAEDecouple]   Found " << loopBodyPrefetches.size() 
               << " loop body prefetches\n";

  if (prologuePrefetches.empty() && loopBodyPrefetches.empty()) {
    // No prefetch operations, nothing to synchronize
    llvm::dbgs() << "[VDAEDecouple]   No prefetch operations, skipping synchronization\n";
    return;
  }

  // Sync-only in-situ prefetches (lookahead==0) complete before signal would
  // fire; wrapping them in wait/signal is pure overhead and — worse — wait()
  // drains OmniFetchRuntime async DMA jobs belonging to sibling loops
  // (e.g. activation loop wait completing a weight-loop deferred WH).
  auto hasAsyncPrefetch = [](ArrayRef<PrefetchInSituOp> ops) {
    return llvm::any_of(ops,
                        [](PrefetchInSituOp p) { return p.getLookahead() > 0; });
  };
  if (!hasAsyncPrefetch(prologuePrefetches) &&
      !hasAsyncPrefetch(loopBodyPrefetches)) {
    llvm::dbgs() << "[VDAEDecouple]   Sync-only prefetches, skipping wait/signal\n";
    return;
  }

  llvm::dbgs() << "[VDAEDecouple]   Adding synchronization (semaphore + wait/signal)\n";

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

    llvm::dbgs() << "\n[VDAEDecouple] ========== PASS STARTING ==========\n";
    llvm::dbgs() << "[VDAEDecouple] Function: " << func.getName() << "\n";
    llvm::dbgs() << "[VDAEDecouple] Options: enableAdaptive=" << enableAdaptive << "\n";

    int loopIdx = 0;
    // Find all loops and add synchronization if they have prefetch operations
    func.walk([&](scf::ForOp loop) {
      llvm::dbgs() << "\n[VDAEDecouple] --- Processing loop " << loopIdx++ 
                   << " at " << loop.getLoc() << " ---\n";
      addSynchronizationToLoop(loop, enableAdaptive);
    });
    
    llvm::dbgs() << "[VDAEDecouple] ========== PASS COMPLETE ==========\n\n";
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
