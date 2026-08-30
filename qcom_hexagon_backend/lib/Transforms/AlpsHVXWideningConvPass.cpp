//===- AlpsHVXWideningConvPass.cpp ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// ALPS C: helper-free mixed F16/F32 convolution for HVX.
//
// The stock NCHW convolution loop nest reduces over the small kernel-width
// loop.  LLVM consequently scalarizes every F16->F32 conversion and emits two
// __extendhfsf2 calls per MAC.  This pass instead vectorizes independent output
// columns.  Each lane retains the original IC/KH/KW reduction order.  A
// 64xf16 operand occupies exactly one 128-byte HVX vector and widens to a
// 64xf32 accumulator pair.  This is the native V73 shape for
// Vdd.sf=vcvt(Vu.hf) and avoids scalar half-conversion fallbacks caused by a
// partial (32xf16) operand vector.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSHVXWIDENINGCONV
#include "hexagon/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kF16Lanes = 64;

static bool isStaticMixedF16F32Conv(linalg::Conv2DNchwFchwOp conv,
                                    MemRefType &inputType,
                                    MemRefType &filterType,
                                    MemRefType &outputType) {
  if (!conv.hasPureBufferSemantics() || conv.getInputs().size() != 2 ||
      conv.getOutputs().size() != 1)
    return false;
  inputType = dyn_cast<MemRefType>(conv.getInputs()[0].getType());
  filterType = dyn_cast<MemRefType>(conv.getInputs()[1].getType());
  outputType = dyn_cast<MemRefType>(conv.getOutputs()[0].getType());
  if (!inputType || !filterType || !outputType ||
      !inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outputType.hasStaticShape() || inputType.getRank() != 4 ||
      filterType.getRank() != 4 || outputType.getRank() != 4 ||
      !inputType.getElementType().isF16() ||
      !filterType.getElementType().isF16() ||
      !outputType.getElementType().isF32())
    return false;

  ArrayRef<int64_t> in = inputType.getShape();
  ArrayRef<int64_t> filter = filterType.getShape();
  ArrayRef<int64_t> out = outputType.getShape();
  return in[0] == out[0] && in[1] == filter[1] && filter[0] == out[1] &&
         out[2] > 0 && out[3] > 0 && filter[2] > 0 && filter[3] > 0;
}

static bool isStaticMixedF16F32Conv(linalg::Conv1DNcwFcwOp conv,
                                    MemRefType &inputType,
                                    MemRefType &filterType,
                                    MemRefType &outputType) {
  if (!conv.hasPureBufferSemantics() || conv.getInputs().size() != 2 ||
      conv.getOutputs().size() != 1)
    return false;
  inputType = dyn_cast<MemRefType>(conv.getInputs()[0].getType());
  filterType = dyn_cast<MemRefType>(conv.getInputs()[1].getType());
  outputType = dyn_cast<MemRefType>(conv.getOutputs()[0].getType());
  if (!inputType || !filterType || !outputType ||
      !inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outputType.hasStaticShape() || inputType.getRank() != 3 ||
      filterType.getRank() != 3 || outputType.getRank() != 3 ||
      !inputType.getElementType().isF16() ||
      !filterType.getElementType().isF16() ||
      !outputType.getElementType().isF32())
    return false;

  ArrayRef<int64_t> in = inputType.getShape();
  ArrayRef<int64_t> filter = filterType.getShape();
  ArrayRef<int64_t> out = outputType.getShape();
  return in[0] == out[0] && in[1] == filter[1] && filter[0] == out[1] &&
         out[2] > 0 && filter[2] > 0;
}

static Value indexMulAdd(OpBuilder &builder, Location loc, Value lhs,
                         int64_t lhsScale, Value rhs, int64_t rhsScale) {
  Value result = lhs;
  if (lhsScale != 1) {
    Value scale = builder.create<arith::ConstantIndexOp>(loc, lhsScale);
    result = builder.create<arith::MulIOp>(loc, lhs, scale);
  }
  Value rhsTerm = rhs;
  if (rhsScale != 1) {
    Value scale = builder.create<arith::ConstantIndexOp>(loc, rhsScale);
    rhsTerm = builder.create<arith::MulIOp>(loc, rhs, scale);
  }
  return builder.create<arith::AddIOp>(loc, result, rhsTerm);
}

static LogicalResult rewriteConv(linalg::Conv2DNchwFchwOp conv,
                                 IRRewriter &rewriter) {
  MemRefType inputType, filterType, outputType;
  if (!isStaticMixedF16F32Conv(conv, inputType, filterType, outputType))
    return failure();

  SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
  SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
  if (strides.size() != 2 || dilations.size() != 2 || strides[0] <= 0 ||
      strides[1] <= 0 || dilations[0] <= 0 || dilations[1] <= 0)
    return failure();

  Location loc = conv.getLoc();
  Value input = conv.getInputs()[0];
  Value filter = conv.getInputs()[1];
  Value output = conv.getOutputs()[0];
  ArrayRef<int64_t> outShape = outputType.getShape();
  ArrayRef<int64_t> filterShape = filterType.getShape();

  // A small non-overlapping patch embedding (for example DINO's 14x14,
  // stride-14 convolution producing only 16 output columns) is not a
  // native-width sliding-convolution opportunity.  Rewriting it as one
  // mostly-masked 64-lane gather both wastes lanes and leaves an unsupported
  // gather/address cast in the final Hexagon lowering.  Keep it for the
  // consumer-driven patch/direct-formation path instead.  Overlapping patch
  // convolutions such as 7/4 and 3/2 remain eligible.
  if (outShape[3] < kF16Lanes &&
      strides[0] == filterShape[2] && strides[1] == filterShape[3])
    return failure();

  Type f16 = rewriter.getF16Type();
  Type f32 = rewriter.getF32Type();
  Type i32 = rewriter.getI32Type();
  auto vecF16 = VectorType::get({kF16Lanes}, f16);
  auto vecF32 = VectorType::get({kF16Lanes}, f32);
  auto vecI32 = VectorType::get({kF16Lanes}, i32);
  auto maskType = VectorType::get({kF16Lanes}, rewriter.getI1Type());

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value c64 = rewriter.create<arith::ConstantIndexOp>(loc, kF16Lanes);
  Value nEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[0]);
  Value ocEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[1]);
  Value ohEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[2]);
  Value owEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[3]);
  Value icEnd = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[1]);
  Value khEnd = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[2]);
  Value kwEnd = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[3]);

  SmallVector<int32_t> laneOffsets;
  laneOffsets.reserve(kF16Lanes);
  for (int32_t lane = 0; lane < kF16Lanes; ++lane)
    laneOffsets.push_back(lane * static_cast<int32_t>(strides[1]));
  auto offsetAttr = DenseIntElementsAttr::get(vecI32, laneOffsets);

  Value zeroVecF16 = rewriter.create<arith::ConstantOp>(
      loc, vecF16,
      DenseElementsAttr::get(vecF16, rewriter.getFloatAttr(f16, 0.0)));
  Value zeroVecF32 = rewriter.create<arith::ConstantOp>(
      loc, vecF32,
      DenseElementsAttr::get(vecF32, rewriter.getFloatAttr(f32, 0.0)));
  Value offsets = rewriter.create<arith::ConstantOp>(loc, vecI32, offsetAttr);

  rewriter.setInsertionPoint(conv);
  scf::buildLoopNest(
      rewriter, loc, ValueRange{c0, c0, c0, c0},
      ValueRange{nEnd, ocEnd, ohEnd, owEnd},
      ValueRange{c1, c1, c1, c64},
      [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs) {
        Value n = ivs[0], oc = ivs[1], oh = ivs[2], ow = ivs[3];
        Value remaining = builder.create<arith::SubIOp>(bodyLoc, owEnd, ow);
        remaining =
            builder.create<arith::MinUIOp>(bodyLoc, remaining, c64);
        Value mask = vector::CreateMaskOp::create(builder, bodyLoc, maskType,
                                                  ValueRange{remaining});
        Value acc = vector::MaskedLoadOp::create(
            builder, bodyLoc, vecF32, output, ValueRange{n, oc, oh, ow}, mask,
            zeroVecF32);

        scf::LoopNest reductions = scf::buildLoopNest(
            builder, bodyLoc, ValueRange{c0, c0, c0},
            ValueRange{icEnd, khEnd, kwEnd}, ValueRange{c1, c1, c1},
            ValueRange{acc},
            [&](OpBuilder &reduceBuilder, Location reduceLoc,
                ValueRange reductionsIvs,
                ValueRange iterArgs) -> SmallVector<Value> {
              Value ic = reductionsIvs[0];
              Value kh = reductionsIvs[1];
              Value kw = reductionsIvs[2];
              Value inputH = indexMulAdd(reduceBuilder, reduceLoc, oh,
                                         strides[0], kh, dilations[0]);
              Value inputW = indexMulAdd(reduceBuilder, reduceLoc, ow,
                                         strides[1], kw, dilations[1]);
              Value activation = vector::GatherOp::create(
                  reduceBuilder, reduceLoc, vecF16, input,
                  ValueRange{n, ic, inputH, inputW}, offsets, mask,
                  zeroVecF16);
              Value weight = reduceBuilder.create<memref::LoadOp>(
                  reduceLoc, filter, ValueRange{oc, ic, kh, kw});
              Value weightVector = vector::BroadcastOp::create(
                  reduceBuilder, reduceLoc, vecF16, weight);
              Value activationF32 = reduceBuilder.create<arith::ExtFOp>(
                  reduceLoc, vecF32, activation);
              Value weightF32 = reduceBuilder.create<arith::ExtFOp>(
                  reduceLoc, vecF32, weightVector);
              Value product = reduceBuilder.create<arith::MulFOp>(
                  reduceLoc, activationF32, weightF32);
              Value next = reduceBuilder.create<arith::AddFOp>(
                  reduceLoc, iterArgs.front(), product);
              return SmallVector<Value>{next};
            });

        vector::MaskedStoreOp::create(builder, bodyLoc, output,
                                      ValueRange{n, oc, oh, ow}, mask,
                                      reductions.results.front());
      });

  auto function = conv->getParentOfType<FunctionOpInterface>();
  if (function) {
    auto oldCount = function->getAttrOfType<IntegerAttr>(
        "alps.c.hvx_widening_convs");
    int64_t count = oldCount ? oldCount.getInt() : 0;
    function->setAttr("alps.c.hvx_widening_convs",
                      rewriter.getI64IntegerAttr(count + 1));
  }
  rewriter.eraseOp(conv);
  return success();
}

static LogicalResult rewriteConv(linalg::Conv1DNcwFcwOp conv,
                                 IRRewriter &rewriter) {
  MemRefType inputType, filterType, outputType;
  if (!isStaticMixedF16F32Conv(conv, inputType, filterType, outputType))
    return failure();

  SmallVector<int64_t> strides(conv.getStrides().getValues<int64_t>());
  SmallVector<int64_t> dilations(conv.getDilations().getValues<int64_t>());
  if (strides.size() != 1 || dilations.size() != 1 || strides[0] <= 0 ||
      dilations[0] <= 0)
    return failure();

  Location loc = conv.getLoc();
  Value input = conv.getInputs()[0];
  Value filter = conv.getInputs()[1];
  Value output = conv.getOutputs()[0];
  ArrayRef<int64_t> outShape = outputType.getShape();
  ArrayRef<int64_t> filterShape = filterType.getShape();

  Type f16 = rewriter.getF16Type();
  Type f32 = rewriter.getF32Type();
  Type i32 = rewriter.getI32Type();
  auto vecF16 = VectorType::get({kF16Lanes}, f16);
  auto vecF32 = VectorType::get({kF16Lanes}, f32);
  auto vecI32 = VectorType::get({kF16Lanes}, i32);
  auto maskType = VectorType::get({kF16Lanes}, rewriter.getI1Type());

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value c64 = rewriter.create<arith::ConstantIndexOp>(loc, kF16Lanes);
  Value nEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[0]);
  Value ocEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[1]);
  Value owEnd = rewriter.create<arith::ConstantIndexOp>(loc, outShape[2]);
  Value icEnd = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[1]);
  Value kwEnd = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[2]);

  SmallVector<int32_t> laneOffsets;
  laneOffsets.reserve(kF16Lanes);
  for (int32_t lane = 0; lane < kF16Lanes; ++lane)
    laneOffsets.push_back(lane * static_cast<int32_t>(strides[0]));
  Value offsets = rewriter.create<arith::ConstantOp>(
      loc, vecI32, DenseIntElementsAttr::get(vecI32, laneOffsets));
  Value zeroVecF16 = rewriter.create<arith::ConstantOp>(
      loc, vecF16,
      DenseElementsAttr::get(vecF16, rewriter.getFloatAttr(f16, 0.0)));
  Value zeroVecF32 = rewriter.create<arith::ConstantOp>(
      loc, vecF32,
      DenseElementsAttr::get(vecF32, rewriter.getFloatAttr(f32, 0.0)));

  rewriter.setInsertionPoint(conv);
  scf::buildLoopNest(
      rewriter, loc, ValueRange{c0, c0, c0},
      ValueRange{nEnd, ocEnd, owEnd}, ValueRange{c1, c1, c64},
      [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs) {
        Value n = ivs[0], oc = ivs[1], ow = ivs[2];
        Value remaining = builder.create<arith::SubIOp>(bodyLoc, owEnd, ow);
        remaining = builder.create<arith::MinUIOp>(bodyLoc, remaining, c64);
        Value mask = vector::CreateMaskOp::create(builder, bodyLoc, maskType,
                                                  ValueRange{remaining});
        Value acc = vector::MaskedLoadOp::create(
            builder, bodyLoc, vecF32, output, ValueRange{n, oc, ow}, mask,
            zeroVecF32);

        scf::LoopNest reductions = scf::buildLoopNest(
            builder, bodyLoc, ValueRange{c0, c0}, ValueRange{icEnd, kwEnd},
            ValueRange{c1, c1}, ValueRange{acc},
            [&](OpBuilder &reduceBuilder, Location reduceLoc,
                ValueRange reductionsIvs,
                ValueRange iterArgs) -> SmallVector<Value> {
              Value ic = reductionsIvs[0];
              Value kw = reductionsIvs[1];
              Value inputW = indexMulAdd(reduceBuilder, reduceLoc, ow,
                                         strides[0], kw, dilations[0]);
              Value activation = vector::GatherOp::create(
                  reduceBuilder, reduceLoc, vecF16, input,
                  ValueRange{n, ic, inputW}, offsets, mask, zeroVecF16);
              Value weight = reduceBuilder.create<memref::LoadOp>(
                  reduceLoc, filter, ValueRange{oc, ic, kw});
              Value weightVector = vector::BroadcastOp::create(
                  reduceBuilder, reduceLoc, vecF16, weight);
              Value activationF32 = reduceBuilder.create<arith::ExtFOp>(
                  reduceLoc, vecF32, activation);
              Value weightF32 = reduceBuilder.create<arith::ExtFOp>(
                  reduceLoc, vecF32, weightVector);
              Value product = reduceBuilder.create<arith::MulFOp>(
                  reduceLoc, activationF32, weightF32);
              Value next = reduceBuilder.create<arith::AddFOp>(
                  reduceLoc, iterArgs.front(), product);
              return SmallVector<Value>{next};
            });

        vector::MaskedStoreOp::create(builder, bodyLoc, output,
                                      ValueRange{n, oc, ow}, mask,
                                      reductions.results.front());
      });

  auto function = conv->getParentOfType<FunctionOpInterface>();
  if (function) {
    auto oldCount = function->getAttrOfType<IntegerAttr>(
        "alps.c.hvx_widening_convs");
    int64_t count = oldCount ? oldCount.getInt() : 0;
    function->setAttr("alps.c.hvx_widening_convs",
                      rewriter.getI64IntegerAttr(count + 1));
  }
  rewriter.eraseOp(conv);
  return success();
}

struct AlpsHVXWideningConvPass
    : public ::impl::AlpsHVXWideningConvBase<AlpsHVXWideningConvPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<linalg::Conv2DNchwFchwOp> candidates;
    function.walk([&](linalg::Conv2DNchwFchwOp conv) {
      candidates.push_back(conv);
    });
    IRRewriter rewriter(function.getContext());
    int64_t rewritten = 0;
    for (linalg::Conv2DNchwFchwOp conv : candidates) {
      rewriter.setInsertionPoint(conv);
      if (succeeded(rewriteConv(conv, rewriter)))
        ++rewritten;
    }
    SmallVector<linalg::Conv1DNcwFcwOp> candidates1D;
    function.walk([&](linalg::Conv1DNcwFcwOp conv) {
      candidates1D.push_back(conv);
    });
    for (linalg::Conv1DNcwFcwOp conv : candidates1D) {
      rewriter.setInsertionPoint(conv);
      if (succeeded(rewriteConv(conv, rewriter)))
        ++rewritten;
    }
    function->setAttr("alps.c.hvx_widening_rewritten",
                      IntegerAttr::get(IntegerType::get(function.getContext(),
                                                        64),
                                       rewritten));
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createAlpsHVXWideningConvPass() {
  return std::make_unique<AlpsHVXWideningConvPass>();
}
