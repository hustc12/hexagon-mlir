//===- AlpsZeroCopyAttentionPass.cpp - absorb attention layouts -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSZEROCOPYATTENTION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static bool hasPermutation(linalg::TransposeOp op,
                           ArrayRef<int64_t> expected) {
  return llvm::equal(op.getPermutation(), expected);
}

static int64_t staticTensorBytes(RankedTensorType type) {
  if (!type.hasStaticShape())
    return -1;
  auto elementType = dyn_cast<IntegerType>(type.getElementType());
  unsigned bits = elementType ? elementType.getWidth() : 0;
  if (auto floatType = dyn_cast<FloatType>(type.getElementType()))
    bits = floatType.getWidth();
  if (bits == 0 || bits % 8 != 0)
    return -1;
  return type.getNumElements() * (bits / 8);
}

static Value widenTo(OpBuilder &builder, Location loc, Value value,
                     Type targetType) {
  if (value.getType() == targetType)
    return value;
  auto source = dyn_cast<FloatType>(value.getType());
  auto target = dyn_cast<FloatType>(targetType);
  if (!source || !target)
    return {};
  if (source.getWidth() < target.getWidth())
    return arith::ExtFOp::create(builder, loc, targetType, value);
  return arith::TruncFOp::create(builder, loc, targetType, value);
}

/// Absorb the two physical head-layout transposes in eager QK^T attention:
///
///   [1,M,H,K] --transpose[0,2,1,3]--collapse--> [H,M,K]
///   [1,N,H,K] --transpose[0,2,3,1]--collapse--> [H,K,N]
///   linalg.batch_matmul -> [H,M,N]
///
/// into one contraction whose input indexing maps directly address the two
/// producer tensors. Batch must be exactly one, making the leading constant-0
/// index a strict equivalence proof rather than a flattening assumption.
struct AbsorbQKHeadLayout final
    : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics() || op->getNumResults() != 1)
      return failure();

    auto lhsCollapse =
        op.getInputs()[0].getDefiningOp<tensor::CollapseShapeOp>();
    auto rhsCollapse =
        op.getInputs()[1].getDefiningOp<tensor::CollapseShapeOp>();
    if (!lhsCollapse || !rhsCollapse)
      return failure();
    auto lhsTranspose =
        lhsCollapse.getSrc().getDefiningOp<linalg::TransposeOp>();
    auto rhsTranspose =
        rhsCollapse.getSrc().getDefiningOp<linalg::TransposeOp>();
    if (!lhsTranspose || !rhsTranspose ||
        !hasPermutation(lhsTranspose, {0, 2, 1, 3}) ||
        !hasPermutation(rhsTranspose, {0, 2, 3, 1}))
      return failure();

    auto lhsType = dyn_cast<RankedTensorType>(lhsTranspose.getInput().getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsTranspose.getInput().getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!lhsType || !rhsType || !outType || lhsType.getRank() != 4 ||
        rhsType.getRank() != 4 || outType.getRank() != 3 ||
        !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outType.hasStaticShape() || lhsType.getDimSize(0) != 1 ||
        rhsType.getDimSize(0) != 1 ||
        lhsType.getElementType() != rhsType.getElementType())
      return failure();

    int64_t m = lhsType.getDimSize(1);
    int64_t h = lhsType.getDimSize(2);
    int64_t k = lhsType.getDimSize(3);
    int64_t n = rhsType.getDimSize(1);
    if (rhsType.getDimSize(2) != h || rhsType.getDimSize(3) != k ||
        outType.getDimSize(0) != h || outType.getDimSize(1) != m ||
        outType.getDimSize(2) != n)
      return failure();

    auto inputElement = dyn_cast<FloatType>(lhsType.getElementType());
    auto accumulatorElement = dyn_cast<FloatType>(outType.getElementType());
    if (!inputElement || !accumulatorElement)
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    AffineExpr dh = getAffineDimExpr(0, ctx);
    AffineExpr dm = getAffineDimExpr(1, ctx);
    AffineExpr dn = getAffineDimExpr(2, ctx);
    AffineExpr dk = getAffineDimExpr(3, ctx);
    AffineExpr zero = getAffineConstantExpr(0, ctx);
    SmallVector<AffineMap> maps{
        AffineMap::get(4, 0, {zero, dm, dh, dk}, ctx),
        AffineMap::get(4, 0, {zero, dn, dh, dk}, ctx),
        AffineMap::get(4, 0, {dh, dm, dn}, ctx)};
    SmallVector<utils::IteratorType> iterators{
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction};

    auto replacement = rewriter.create<linalg::GenericOp>(
        op.getLoc(), TypeRange{outType},
        ValueRange{lhsTranspose.getInput(), rhsTranspose.getInput()},
        op.getOutputs(), maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value lhs = widenTo(builder, loc, args[0], accumulatorElement);
          Value rhs = widenTo(builder, loc, args[1], accumulatorElement);
          Value product = arith::MulFOp::create(builder, loc, lhs, rhs);
          Value sum = arith::AddFOp::create(builder, loc, args[2], product);
          linalg::YieldOp::create(builder, loc, sum);
        });

    int64_t lhsBytes = staticTensorBytes(
        cast<RankedTensorType>(lhsTranspose.getResult()[0].getType()));
    int64_t rhsBytes = staticTensorBytes(
        cast<RankedTensorType>(rhsTranspose.getResult()[0].getType()));
    replacement->setAttr("alps.p2a.zero_copy_attention",
                         rewriter.getUnitAttr());
    replacement->setAttr(
        "alps.p2a.eliminated_transpose_materialization_bytes",
        rewriter.getI64IntegerAttr(lhsBytes + rhsBytes));
    llvm::errs() << "[ALPS-P2A] function="
                 << op->getParentOfType<FunctionOpInterface>().getName()
                 << " kind=qk_head_layout_absorption"
                 << " eliminated_transposes=2"
                 << " materialization_bytes=" << lhsBytes + rhsBytes << "\n";
    rewriter.replaceOp(op, replacement.getResults());
    return success();
  }
};

struct AlpsZeroCopyAttentionPass final
    : ::impl::AlpsZeroCopyAttentionBase<AlpsZeroCopyAttentionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AbsorbQKHeadLayout>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsZeroCopyAttentionPass() {
  return std::make_unique<AlpsZeroCopyAttentionPass>();
}
