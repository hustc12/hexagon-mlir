//===- AlpsPatchConvFormationPass.cpp - patch consumer formation ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSPATCHCONVFORMATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static std::mutex reportMutex;

static bool hasOneUse(Value value) {
  return value && llvm::hasSingleElement(value.getUsers());
}

static SmallVector<int64_t> values(DenseIntElementsAttr attr) {
  SmallVector<int64_t> result;
  for (APInt value : attr.getValues<APInt>())
    result.push_back(value.getSExtValue());
  return result;
}

static bool isF32ToF16Identity(linalg::GenericOp op) {
  if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumLoops() != 4)
    return false;
  auto inputType =
      dyn_cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType());
  auto outputType =
      dyn_cast<RankedTensorType>(op.getDpsInitOperand(0)->get().getType());
  if (!inputType || !outputType || !inputType.getElementType().isF32() ||
      !outputType.getElementType().isF16() ||
      inputType.getShape() != outputType.getShape())
    return false;
  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::parallel;
      }))
    return false;
  for (AffineMap map : op.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  auto trunc = dyn_cast<arith::TruncFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  return trunc && yield && yield.getValues().size() == 1 &&
         yield.getValues().front() == trunc.getResult();
}

static LogicalResult formPatchConv(linalg::Conv2DNchwFchwOp conv,
                                   PatternRewriter &rewriter,
                                   int64_t &eliminatedBytes) {
  auto inputType = dyn_cast<RankedTensorType>(conv.getInputs()[0].getType());
  auto filterType = dyn_cast<RankedTensorType>(conv.getInputs()[1].getType());
  auto outputType = dyn_cast<RankedTensorType>(conv.getOutputs()[0].getType());
  if (!inputType || !filterType || !outputType || !inputType.hasStaticShape() ||
      !filterType.hasStaticShape() || !outputType.hasStaticShape() ||
      inputType.getRank() != 4 || filterType.getRank() != 4 ||
      outputType.getRank() != 4 || !inputType.getElementType().isF16() ||
      !filterType.getElementType().isF16() ||
      !outputType.getElementType().isF32())
    return failure();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> filterShape = filterType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  int64_t n = inputShape[0], ic = inputShape[1], ih = inputShape[2];
  int64_t iw = inputShape[3], oc = filterShape[0], kh = filterShape[2];
  int64_t kw = filterShape[3], oh = outputShape[2], ow = outputShape[3];
  if (n != outputShape[0] || ic != filterShape[1] || oc != outputShape[1] ||
      oc % 64 != 0)
    return failure();

  SmallVector<int64_t> strides = values(conv.getStrides());
  SmallVector<int64_t> dilations = values(conv.getDilations());
  if (strides != SmallVector<int64_t>{kh, kw} ||
      dilations != SmallVector<int64_t>{1, 1} || ih != oh * kh || iw != ow * kw)
    return failure();

  Value filter = conv.getInputs()[1];
  auto filterConstant = filter.getDefiningOp<arith::ConstantOp>();
  if (!filterConstant || !isa<DenseElementsAttr, DenseResourceElementsAttr>(
                             filterConstant.getValue()))
    return failure();

  if (!hasOneUse(conv.getResult(0)))
    return failure();
  auto trunc =
      dyn_cast<linalg::GenericOp>(*conv.getResult(0).getUsers().begin());
  if (!isF32ToF16Identity(trunc) || !hasOneUse(trunc.getResult(0)))
    return failure();
  auto collapse =
      dyn_cast<tensor::CollapseShapeOp>(*trunc.getResult(0).getUsers().begin());
  if (!collapse || !hasOneUse(collapse.getResult()))
    return failure();
  auto collapsedType = dyn_cast<RankedTensorType>(collapse.getType());
  if (!collapsedType ||
      collapsedType.getShape() != ArrayRef<int64_t>({n, oc, oh * ow}))
    return failure();
  auto transpose =
      dyn_cast<linalg::TransposeOp>(*collapse.getResult().getUsers().begin());
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return failure();
  auto tokenType =
      dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
  if (!tokenType ||
      tokenType.getShape() != ArrayRef<int64_t>({n, oh * ow, oc}) ||
      !tokenType.getElementType().isF16())
    return failure();

  auto biasBroadcast =
      conv.getOutputs()[0].getDefiningOp<linalg::BroadcastOp>();
  if (!biasBroadcast || biasBroadcast.getDpsInputs().size() != 1)
    return failure();
  Value bias = biasBroadcast.getDpsInputOperand(0)->get();
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || biasType.getShape() != ArrayRef<int64_t>({oc}) ||
      !biasType.getElementType().isF32())
    return failure();

  Location loc = conv.getLoc();
  rewriter.setInsertionPoint(conv);

  SmallVector<ReassociationIndices> flattenFilter = {{0}, {1, 2, 3}};
  Value flatFilter =
      tensor::CollapseShapeOp::create(rewriter, loc, filter, flattenFilter);
  int64_t reduction = ic * kh * kw;
  Value transposedFilterInit =
      tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{reduction, oc},
                              filterType.getElementType());
  Value transposedFilter =
      linalg::TransposeOp::create(rewriter, loc, flatFilter,
                                  transposedFilterInit, ArrayRef<int64_t>{1, 0})
          .getResult()[0];
  auto packedFilterType =
      RankedTensorType::get({ic, kh, kw, oc}, filterType.getElementType());
  SmallVector<ReassociationIndices> expandFilter = {{0, 1, 2}, {3}};
  Value packedFilter = tensor::ExpandShapeOp::create(
      rewriter, loc, packedFilterType, transposedFilter, expandFilter);

  auto formedF32Type =
      RankedTensorType::get({n, oh, ow, oc}, outputType.getElementType());
  Value formedInit = tensor::EmptyOp::create(
      rewriter, loc, formedF32Type.getShape(), formedF32Type.getElementType());
  Value formedBias =
      linalg::BroadcastOp::create(rewriter, loc, bias, formedInit,
                                  ArrayRef<int64_t>{0, 1, 2})
          .getResult()[0];

  MLIRContext *ctx = rewriter.getContext();
  SmallVector<AffineExpr> d;
  for (unsigned i = 0; i < 7; ++i)
    d.push_back(getAffineDimExpr(i, ctx));
  AffineExpr inputH = d[1] * kh + d[4];
  AffineExpr inputW = d[2] * kw + d[5];
  SmallVector<AffineMap> maps = {
      AffineMap::get(7, 0, {d[0], d[3], inputH, inputW}, ctx),
      AffineMap::get(7, 0, {d[3], d[4], d[5], d[6]}, ctx),
      AffineMap::get(7, 0, {d[0], d[1], d[2], d[6]}, ctx)};
  SmallVector<utils::IteratorType> iterators = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::reduction,
      utils::IteratorType::reduction, utils::IteratorType::reduction,
      utils::IteratorType::parallel};
  auto formedConv = linalg::GenericOp::create(
      rewriter, loc, TypeRange{formedF32Type},
      ValueRange{conv.getInputs()[0], packedFilter}, ValueRange{formedBias},
      maps, iterators, /*bodyBuild=*/nullptr);
  rewriter.cloneRegionBefore(conv.getRegion(), formedConv.getRegion(),
                             formedConv.getRegion().begin());
  formedConv->setAttr("alps.p5i.patch_conv_formation", rewriter.getUnitAttr());
  formedConv->setAttr("alps.p5i.contiguous_output_channel",
                      rewriter.getI64IntegerAttr(oc));
  formedConv->setAttr("alps.p5i.patch_reduction",
                      rewriter.getI64IntegerAttr(reduction));

  auto formedF16Type = RankedTensorType::get(formedF32Type.getShape(),
                                             tokenType.getElementType());
  Value truncInit = tensor::EmptyOp::create(
      rewriter, loc, formedF16Type.getShape(), formedF16Type.getElementType());
  AffineMap identity = AffineMap::getMultiDimIdentityMap(4, ctx);
  SmallVector<utils::IteratorType> truncIterators(
      4, utils::IteratorType::parallel);
  auto formedTrunc = linalg::GenericOp::create(
      rewriter, trunc.getLoc(), TypeRange{formedF16Type},
      ValueRange{formedConv.getResult(0)}, ValueRange{truncInit},
      ArrayRef<AffineMap>{identity, identity}, truncIterators,
      /*bodyBuild=*/nullptr);
  rewriter.cloneRegionBefore(trunc.getRegion(), formedTrunc.getRegion(),
                             formedTrunc.getRegion().begin());
  formedTrunc->setAttr("alps.p5i.patch_conv_truncate", rewriter.getUnitAttr());

  auto formedTokensType =
      RankedTensorType::get({n, oh * ow, oc}, tokenType.getElementType());
  SmallVector<ReassociationIndices> collapseTokens = {{0}, {1, 2}, {3}};
  Value formedTokens = tensor::CollapseShapeOp::create(
      rewriter, trunc.getLoc(), formedTokensType, formedTrunc.getResult(0),
      collapseTokens);
  rewriter.replaceOp(transpose, formedTokens);

  if (collapse->use_empty())
    rewriter.eraseOp(collapse);
  if (trunc->use_empty())
    rewriter.eraseOp(trunc);
  if (conv->use_empty())
    rewriter.eraseOp(conv);
  eliminatedBytes = n * oc * oh * ow * 2;
  return success();
}

struct AlpsPatchConvFormationPass final
    : ::impl::AlpsPatchConvFormationBase<AlpsPatchConvFormationPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<linalg::Conv2DNchwFchwOp> candidates;
    function.walk(
        [&](linalg::Conv2DNchwFchwOp op) { candidates.push_back(op); });
    int64_t matched = 0;
    int64_t eliminatedBytes = 0;
    PatternRewriter rewriter(function.getContext());
    for (linalg::Conv2DNchwFchwOp conv : candidates) {
      if (!conv->getBlock())
        continue;
      int64_t bytes = 0;
      if (succeeded(formPatchConv(conv, rewriter, bytes))) {
        ++matched;
        eliminatedBytes += bytes;
      }
    }
    Builder builder(function.getContext());
    function->setAttr("alps.p5i.patch_conv_formed",
                      builder.getI64IntegerAttr(matched));
    function->setAttr("alps.p5i.eliminated_output_transpose_bytes",
                      builder.getI64IntegerAttr(eliminatedBytes));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5I] function=" << function.getName()
                 << " candidates=" << candidates.size() << " formed=" << matched
                 << " eliminated_output_transpose_bytes=" << eliminatedBytes
                 << '\n';
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsPatchConvFormationPass() {
  return std::make_unique<AlpsPatchConvFormationPass>();
}
