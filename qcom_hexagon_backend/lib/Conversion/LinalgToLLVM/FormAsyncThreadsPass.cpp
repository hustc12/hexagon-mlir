//===- FormAsyncThreadsPass.cpp :  Lower scf::forall to scf.for -----------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Lower scf.forall to sequential scf.for.  (Async.execute was abandoned for
// the DSP User PD: AsyncToken heap allocs fault with Bad VA 0x18 / exit 13.)
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "form-async-threads"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_FORMASYNCTHREADS
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {
/// Lower scf.forall to sequential scf.for.
///
/// Historical path emitted async.execute per iteration.  On the DSP User PD
/// that allocates AsyncToken via `new` and routinely faults (Bad VA 0x18 /
/// exit 13) when HexagonTiling introduces forall under enableVectorization
/// (see run_swin_transformer.py).  Sequential execution is correct and avoids
/// the async runtime heap; intentional multi-threading remains behind
/// enableMultiThreading / enableSCFThreading once async heap is fixed.
LogicalResult formSequentialFor(RewriterBase &rewriter, scf::ForallOp forallOp) {
  if (!forallOp.getOutputs().empty())
    return rewriter.notifyMatchFailure(
        forallOp, "only fully bufferized scf.forall ops can be lowered");

  if (forallOp->getNumRegions() != 1 ||
      forallOp->getRegions().front().getBlocks().size() != 1 ||
      forallOp->getNumResults() != 0)
    return rewriter.notifyMatchFailure(
        forallOp, "scf::forall region/block not matching expectation");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  Location loc = forallOp.getLoc();

  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);

  scf::LoopNest loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps);
  SmallVector<Value> ivs = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  Block *forBody = loopNest.loops.back().getBody();

  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(forallOp.getBody(), forBody->getTerminator(), ivs);
  rewriter.eraseOp(forallOp);
  return success();
}

struct FormAsyncThreadsPass
    : public ::impl::FormAsyncThreadsBase<FormAsyncThreadsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    funcOp.walk([&](scf::ForallOp op) {
      if (failed(formSequentialFor(rewriter, op)))
        return signalPassFailure();
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createFormAsyncThreadsPass() {
  return std::make_unique<FormAsyncThreadsPass>();
}
