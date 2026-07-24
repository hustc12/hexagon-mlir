//===-- RewriteUBPoisonToZero.cpp -  rewrite ub.poison        -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Rewrite ub.poison to zero constants.  The Hexagon LLVM translation path has
// no LLVMTranslationDialectInterface for ub.poison; leaving any residual poison
// (from vectorization padding, unreachable cf.assert paths, etc.) fails
// translation.  Zero is a safe stand-in for padding / dead values on this
// backend.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_REWRITEUBPOISONTOZERO
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// Build a typed zero attribute for scalar / vector / ranked-tensor element
/// types that arith.constant can materialize.
static FailureOr<TypedAttr> getZeroAttr(Type ty, Builder &b) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    TypedAttr attr = b.getIntegerAttr(intTy, 0);
    return attr;
  }
  if (auto floatTy = dyn_cast<FloatType>(ty)) {
    TypedAttr attr = b.getFloatAttr(floatTy, 0.0);
    return attr;
  }
  if (auto vecTy = dyn_cast<VectorType>(ty)) {
    auto elemZero = getZeroAttr(vecTy.getElementType(), b);
    if (failed(elemZero))
      return failure();
    TypedAttr attr = DenseElementsAttr::get(vecTy, *elemZero);
    return attr;
  }
  if (auto ranked = dyn_cast<RankedTensorType>(ty)) {
    if (!ranked.hasStaticShape())
      return failure();
    auto elemZero = getZeroAttr(ranked.getElementType(), b);
    if (failed(elemZero))
      return failure();
    TypedAttr attr = DenseElementsAttr::get(ranked, *elemZero);
    return attr;
  }
  return failure();
}

/// Replace every ub.poison with a zero constant of the same type.
struct UBPoisonToZeroPattern : public OpRewritePattern<ub::PoisonOp> {
  using OpRewritePattern<ub::PoisonOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ub::PoisonOp op,
                                PatternRewriter &rewriter) const override {
    auto zeroAttr = getZeroAttr(op.getType(), rewriter);
    if (failed(zeroAttr))
      return rewriter.notifyMatchFailure(op, "unsupported poison type");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(), *zeroAttr);
    return success();
  }
};

/// Legacy targeted rewrite: transfer_read padding that is ub.poison → zero.
struct UBPoisonPaddingToZeroPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    Value padding = op.getPadding();
    if (!padding)
      return failure();

    auto poison = padding.getDefiningOp<ub::PoisonOp>();
    if (!poison)
      return failure();

    auto zeroAttr = getZeroAttr(padding.getType(), rewriter);
    if (failed(zeroAttr))
      return rewriter.notifyMatchFailure(op, "unsupported padding type");

    auto zeroConst =
        rewriter.create<arith::ConstantOp>(op.getLoc(), padding.getType(),
                                           *zeroAttr);
    rewriter.modifyOpInPlace(
        op, [&]() { op.getPaddingMutable().assign(zeroConst); });
    return success();
  }
};

struct RewriteUBPoisonToZeroPass
    : public ::impl::RewriteUBPoisonToZeroBase<RewriteUBPoisonToZeroPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ub::UBDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
  }

  StringRef getArgument() const final {
    return "hexagon-rewrite-ub-poison-to-zero";
  }
  StringRef getDescription() const final {
    return "Rewrite ub.poison operations to zero constants "
           "(all uses, plus vector.transfer_read padding)";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<UBPoisonToZeroPattern, UBPoisonPaddingToZeroPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createRewriteUBPoisonToZeroPass() {
  return std::make_unique<RewriteUBPoisonToZeroPass>();
}
