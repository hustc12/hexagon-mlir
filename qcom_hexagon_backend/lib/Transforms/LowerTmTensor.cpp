//===- LowerTmTensorPass.cpp: convert tm_tensor dialect ops to mlir core. -===//

//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion of tm_tensor dialect ops e.g.
// `tm_tensor.attention` to combination of mlir core ops.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-lower-tm-tensor"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::tm_tensor;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONLOWERTMTENSOR
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct HexagonLowerTmTensorPass
    : public ::impl::HexagonLowerTmTensorBase<HexagonLowerTmTensorPass> {
  explicit HexagonLowerTmTensorPass(
      const HexagonLowerTmTensorOptions &options)
      : HexagonLowerTmTensorBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tm_tensor::TmTensorDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

// Pattern to lower tm_tensor.attention to linalg operations
struct LowerAttentionOp : public OpRewritePattern<AttentionOp> {
  LowerAttentionOp(MLIRContext *context, bool emitKvCacheMetadata)
      : OpRewritePattern(context),
        emitKvCacheMetadata(emitKvCacheMetadata) {}

  bool emitKvCacheMetadata;

  LogicalResult matchAndRewrite(AttentionOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    // Input tensors
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    Value mask = op.getMask();
    Value opsInit = op.getOut();

    // Shapes and element type.
    auto queryType = cast<RankedTensorType>(query.getType());
    auto keyType = cast<RankedTensorType>(key.getType());
    auto valueType = cast<RankedTensorType>(value.getType());
    auto maskType = cast<RankedTensorType>(mask.getType());

    // Op verify ensures shape constraints are satisfied.
    // Q: [batch, seq_q, head_dim]
    // K: [batch, seq_kv, head_dim]
    // QK^T: [batch, seq_q, seq_kv]
    // V : [batch, seq_kv, head_dim]
    // M : [batch, seq_q, seq_kv]
    ArrayRef<int64_t> queryShape = queryType.getShape();
    ArrayRef<int64_t> keyShape = keyType.getShape();
    int64_t batch = queryShape[0];
    int64_t seq_q = queryShape[1];
    int64_t head_dim = queryShape[2];
    int64_t seq_kv = keyShape[1];
    auto elType = queryType.getElementType();

    // Derived shapes
    SmallVector<int64_t> keyTshape = {batch, head_dim, seq_kv};
    SmallVector<int64_t> qkTshape = {batch, seq_q, seq_kv};
    SmallVector<int64_t> outShape = {batch, seq_q, head_dim};
    SmallVector<int64_t> maxShape = {batch, seq_q};

    // Affine maps and iterator types
    auto d0 = rewriter.getAffineDimExpr(0);
    auto d1 = rewriter.getAffineDimExpr(1);
    auto d2 = rewriter.getAffineDimExpr(2);
    auto ctx = rewriter.getContext();
    auto parallel = utils::IteratorType::parallel;

    // Constants
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elType));
    Value scale = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(
                 elType, 1.0 / std::sqrt(static_cast<double>(head_dim))));

    // Compute K^T
    Value keyTinit = rewriter.create<tensor::EmptyOp>(loc, keyTshape, elType);
    Value keyT = rewriter
                     .create<linalg::TransposeOp>(loc, key, keyTinit,
                                                  ArrayRef<int64_t>{0, 2, 1})
                     .getResult()[0];

    // Compute batch matmul QK^T
    Value qkTempty = rewriter.create<tensor::EmptyOp>(loc, qkTshape, elType);
    Value qkTinit =
        rewriter.create<linalg::FillOp>(loc, zero, qkTempty).getResult(0);
    auto qkTtype = RankedTensorType::get(qkTshape, elType);
    auto qkOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, qkTtype, ValueRange{query, keyT}, ValueRange{qkTinit});
    // Preserve the semantic identity of the K stream after lowering. Item 7
    // consumes this marker after bufferization to insert page-coalesced cache
    // hints without guessing from generic contraction indexing maps.
    if (emitKvCacheMetadata) {
      qkOp->setAttr("omni_fetch.kv_cache_role",
                    rewriter.getStringAttr("key"));
      qkOp->setAttr("omni_fetch.kv_cache_operand",
                    rewriter.getI64IntegerAttr(1));
    }
    Value qkT = qkOp.getResult(0);
    DBG(" batch-matmul: " << qkT);

    // Scale QK^T by 1/sqrt(head_dim)
    Value qkTScaled =
        rewriter
            .create<linalg::GenericOp>(
                loc, qkTtype, ValueRange{qkT}, ValueRange{qkTempty},
                ArrayRef<AffineMap>{AffineMap::get(3, 0, {d0, d1, d2}, ctx),
                                    AffineMap::get(3, 0, {d0, d1, d2}, ctx)},
                ArrayRef<utils::IteratorType>{parallel, parallel, parallel},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value mul = b.create<arith::MulFOp>(loc, args[0], scale);
                  b.create<linalg::YieldOp>(loc, mul);
                })
            .getResult(0);
    DBG("Scaled QK^T: " << qkTScaled);

    // Apply mask to scaled QK^T
    Value qkTMasked =
        rewriter
            .create<linalg::AddOp>(loc, ValueRange{qkTScaled, mask}, qkTempty)
            .getResult(0);
    DBG("Masked QK^T: " << qkTMasked);

    // - Softmax -
    // Step 1: Find max for numerical stability
    Value maxEmpty = rewriter.create<tensor::EmptyOp>(loc, maxShape, elType);
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elType,
                                   -std::numeric_limits<double>::infinity()));
    Value maxInit =
        rewriter.create<linalg::FillOp>(loc, negInf, maxEmpty).getResult(0);
    Value maxVals = rewriter
                        .create<linalg::ReduceOp>(
                            loc, qkTMasked, maxInit, ArrayRef<int64_t>{2},
                            [&](OpBuilder &b, Location loc, ValueRange args) {
                              Value max = b.create<arith::MaximumFOp>(
                                  loc, args[0], args[1]);
                              b.create<linalg::YieldOp>(loc, max);
                            })
                        .getResult(0);

    // Step 2: Compute `exp(xi-max)` with implicit broadcast
    Value qkTSub =
        rewriter
            .create<linalg::GenericOp>(
                loc, qkTtype, ValueRange{qkTMasked, maxVals},
                ValueRange{qkTempty},
                ArrayRef<AffineMap>{AffineMap::get(3, 0, {d0, d1, d2}, ctx),
                                    AffineMap::get(3, 0, {d0, d1}, ctx),
                                    AffineMap::get(3, 0, {d0, d1, d2}, ctx)},
                ArrayRef<utils::IteratorType>{parallel, parallel, parallel},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value sub = b.create<arith::SubFOp>(loc, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, sub);
                })
            .getResult(0);
    Value qkTStable =
        rewriter.create<linalg::ExpOp>(loc, qkTSub, qkTempty).getResult(0);

    // Step 3: Compute `sum(exp(xi-max))` along the last dimension
    Value sumInit =
        rewriter.create<linalg::FillOp>(loc, zero, maxEmpty).getResult(0);
    Value sumVals =
        rewriter
            .create<linalg::ReduceOp>(
                loc, qkTStable, sumInit, ArrayRef<int64_t>{2},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, add);
                })
            .getResult(0);

    // Step 4:  div to get softmax
    Value softmaxResult =
        rewriter
            .create<linalg::GenericOp>(
                loc, qkTtype, ValueRange{qkTStable, sumVals},
                ValueRange{qkTempty},
                ArrayRef<AffineMap>{AffineMap::get(3, 0, {d0, d1, d2}, ctx),
                                    AffineMap::get(3, 0, {d0, d1}, ctx),
                                    AffineMap::get(3, 0, {d0, d1, d2}, ctx)},
                ArrayRef<utils::IteratorType>{parallel, parallel, parallel},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value div = b.create<arith::DivFOp>(loc, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, div);
                })
            .getResult(0);
    DBG("Softmax result: " << softmaxResult);

    // Lastly, `softmax(QK^T)*V`
    auto outType = RankedTensorType::get(outShape, elType);
    auto avOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, outType, ValueRange{softmaxResult, value}, ValueRange{opsInit});
    if (emitKvCacheMetadata) {
      avOp->setAttr("omni_fetch.kv_cache_role",
                    rewriter.getStringAttr("value"));
      avOp->setAttr("omni_fetch.kv_cache_operand",
                    rewriter.getI64IntegerAttr(1));
    }
    Value result = avOp.getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Preserve item-7 applicability for models exported with eager attention.
///
/// Those graphs no longer contain tm_tensor.attention, but their two rank-3
/// contractions still have distinctive static shapes:
///   QK^T: [B, S, H] x [B, H, S] -> [B, S, S]
///   AV:   [B, S, S] x [B, S, H] -> [B, S, H]
///
/// Require S != H so a square, non-attention batch matmul is never guessed.
/// The explicit metadata emitted by LowerAttentionOp always takes precedence.
static int64_t annotateEagerAttentionKvStreams(FunctionOpInterface func) {
  int64_t inferred = 0;
  func.walk([&](linalg::BatchMatmulOp op) {
    if (op->hasAttr("omni_fetch.kv_cache_role") ||
        op->getNumOperands() < 3)
      return;

    auto lhs = dyn_cast<ShapedType>(op->getOperand(0).getType());
    auto rhs = dyn_cast<ShapedType>(op->getOperand(1).getType());
    auto out = dyn_cast<ShapedType>(op->getOperand(2).getType());
    if (!lhs || !rhs || !out || !lhs.hasStaticShape() ||
        !rhs.hasStaticShape() || !out.hasStaticShape() ||
        lhs.getRank() != 3 || rhs.getRank() != 3 || out.getRank() != 3)
      return;

    ArrayRef<int64_t> a = lhs.getShape();
    ArrayRef<int64_t> b = rhs.getShape();
    ArrayRef<int64_t> c = out.getShape();
    if (a[0] != b[0] || a[0] != c[0] || a[1] != c[1] ||
        a[2] != b[1] || b[2] != c[2])
      return;

    StringRef role;
    if (c[1] == c[2] && a[1] != a[2])
      role = "key";
    else if (a[1] == a[2] && c[1] != c[2])
      role = "value";
    else
      return;

    Builder bld(op.getContext());
    op->setAttr("omni_fetch.kv_cache_role", bld.getStringAttr(role));
    op->setAttr("omni_fetch.kv_cache_operand", bld.getI64IntegerAttr(1));
    ++inferred;
  });
  return inferred;
}

void HexagonLowerTmTensorPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LowerAttentionOp>(patterns.getContext(), emitKvCacheMetadata);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  if (emitKvCacheMetadata) {
    int64_t inferred = annotateEagerAttentionKvStreams(getOperation());
    llvm::errs() << "[KVCacheMetadata] function="
                 << getOperation()->getName() << " eager_inferred="
                 << inferred << "\n";
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonLowerTmTensorPass(
    const HexagonLowerTmTensorOptions &options) {
  return std::make_unique<HexagonLowerTmTensorPass>(options);
}
