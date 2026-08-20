//===- AlpsProducerDirectAttentionPass.cpp - direct attention layout ------===//
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

#define GEN_PASS_DEF_ALPSPRODUCERDIRECTATTENTION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static bool hasPermutation(linalg::TransposeOp op,
                           ArrayRef<int64_t> expected) {
  return llvm::equal(op.getPermutation(), expected);
}

static bool isStrictBiasAdd(linalg::GenericOp op) {
  if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1 || op->getNumResults() != 1)
    return false;
  auto activation = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto bias = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto output = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!activation || !bias || !output || activation.getRank() != 3 ||
      bias.getRank() != 1 || output != activation)
    return false;

  Block *body = op.getBody();
  if (!body || body->getNumArguments() != 3 ||
      body->getOperations().size() != 2)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto add = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add || &body->front() != add.getOperation())
    return false;
  Value lhs = add.getLhs(), rhs = add.getRhs();
  return (lhs == body->getArgument(0) && rhs == body->getArgument(1)) ||
         (lhs == body->getArgument(1) && rhs == body->getArgument(0));
}

static int64_t tensorBytes(RankedTensorType type) {
  if (!type.hasStaticShape())
    return -1;
  auto floatType = dyn_cast<FloatType>(type.getElementType());
  if (!floatType || floatType.getWidth() % 8 != 0)
    return -1;
  return type.getNumElements() * (floatType.getWidth() / 8);
}

/// Fuse a projection bias-add with its sole expand/transpose consumer. The
/// producer writes the final contiguous head-major representation directly:
///
///   add [B,M,H*D] -> expand [B,M,H,D] -> transpose
///
/// becomes one add generic writing either [B,H,M,D] (Q/V) or [B,H,D,M]
/// (K^T). This removes the canonical add result without replacing it by
/// strided consumer reads.
struct DirectBiasAddHeadLayout final
    : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    bool qvLayout = hasPermutation(transpose, {0, 2, 1, 3});
    bool ktLayout = hasPermutation(transpose, {0, 2, 3, 1});
    if (!qvLayout && !ktLayout)
      return failure();
    auto expand =
        transpose.getInput().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand || !expand.getResult().hasOneUse())
      return failure();
    auto add = expand.getSrc().getDefiningOp<linalg::GenericOp>();
    if (!add || !add.getResult(0).hasOneUse() || !isStrictBiasAdd(add))
      return failure();

    auto flatType = cast<RankedTensorType>(add.getResult(0).getType());
    auto expandedType = dyn_cast<RankedTensorType>(expand.getResult().getType());
    auto targetType = dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
    auto biasType = cast<RankedTensorType>(add.getInputs()[1].getType());
    if (!expandedType || !targetType || !flatType.hasStaticShape() ||
        !expandedType.hasStaticShape() || !targetType.hasStaticShape() ||
        expandedType.getRank() != 4 || targetType.getRank() != 4 ||
        flatType.getDimSize(0) != expandedType.getDimSize(0) ||
        flatType.getDimSize(1) != expandedType.getDimSize(1))
      return failure();
    int64_t heads = expandedType.getDimSize(2);
    int64_t depth = expandedType.getDimSize(3);
    if (heads <= 0 || depth <= 0 || flatType.getDimSize(2) != heads * depth ||
        biasType.getDimSize(0) != heads * depth)
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx);
    AffineExpr d1 = getAffineDimExpr(1, ctx);
    AffineExpr d2 = getAffineDimExpr(2, ctx);
    AffineExpr d3 = getAffineDimExpr(3, ctx);
    AffineExpr channel;
    AffineMap sourceMap;
    if (qvLayout) {
      // target loops [B,H,M,D]
      channel = d1 * depth + d3;
      sourceMap = AffineMap::get(4, 0, {d0, d2, channel}, ctx);
    } else {
      // target loops [B,H,D,M]
      channel = d1 * depth + d2;
      sourceMap = AffineMap::get(4, 0, {d0, d3, channel}, ctx);
    }
    SmallVector<AffineMap> maps{
        sourceMap, AffineMap::get(4, 0, {channel}, ctx),
        AffineMap::getMultiDimIdentityMap(4, ctx)};
    SmallVector<utils::IteratorType> iterators(
        4, utils::IteratorType::parallel);

    auto direct = rewriter.create<linalg::GenericOp>(
        transpose.getLoc(), TypeRange{targetType}, add.getInputs(),
        transpose.getDpsInits(), maps, iterators,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value sum = arith::AddFOp::create(builder, loc, args[0], args[1]);
          linalg::YieldOp::create(builder, loc, sum);
        });
    int64_t eliminatedBytes = tensorBytes(flatType);
    direct->setAttr("alps.p2b.producer_direct_attention",
                    rewriter.getUnitAttr());
    direct->setAttr("alps.p2b.eliminated_canonical_materialization_bytes",
                    rewriter.getI64IntegerAttr(eliminatedBytes));
    direct->setAttr("alps.p2b.target_layout",
                    rewriter.getStringAttr(qvLayout ? "BHMD" : "BHDM"));
    llvm::errs() << "[ALPS-P2B] function="
                 << transpose->getParentOfType<FunctionOpInterface>().getName()
                 << " kind=producer_direct_bias_head_layout"
                 << " target=" << (qvLayout ? "BHMD" : "BHDM")
                 << " materialization_bytes=" << eliminatedBytes << "\n";
    rewriter.replaceOp(transpose, direct.getResults());
    return success();
  }
};

struct AlpsProducerDirectAttentionPass final
    : ::impl::AlpsProducerDirectAttentionBase<
          AlpsProducerDirectAttentionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DirectBiasAddHeadLayout>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsProducerDirectAttentionPass() {
  return std::make_unique<AlpsProducerDirectAttentionPass>();
}
