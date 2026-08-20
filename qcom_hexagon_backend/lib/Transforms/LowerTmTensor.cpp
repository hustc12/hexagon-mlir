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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
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
  LowerAttentionOp(MLIRContext *context, bool emitKvCacheMetadata,
                   bool emitKvFusionBoundary,
                   bool emitKvElementwiseFusionBoundary,
                   bool emitKvMultiUseFusionBoundary,
                   bool emitKvSplitReductionBoundary)
      : OpRewritePattern(context),
        emitKvCacheMetadata(emitKvCacheMetadata),
        emitKvFusionBoundary(emitKvFusionBoundary),
        emitKvElementwiseFusionBoundary(emitKvElementwiseFusionBoundary),
        emitKvMultiUseFusionBoundary(emitKvMultiUseFusionBoundary),
        emitKvSplitReductionBoundary(emitKvSplitReductionBoundary) {}

  bool emitKvCacheMetadata;
  bool emitKvFusionBoundary;
  bool emitKvElementwiseFusionBoundary;
  bool emitKvMultiUseFusionBoundary;
  bool emitKvSplitReductionBoundary;

  void attachTopologyAttrs(Operation *op, PatternRewriter &rewriter) const {
    if (emitKvFusionBoundary)
      op->setAttr("alps.kv_fusion_boundary", rewriter.getUnitAttr());
    if (emitKvElementwiseFusionBoundary)
      op->setAttr("alps.kv_elementwise_fusion_boundary",
                  rewriter.getUnitAttr());
    if (emitKvMultiUseFusionBoundary)
      op->setAttr("alps.kv_multi_use_fusion_boundary",
                  rewriter.getUnitAttr());
    if (emitKvSplitReductionBoundary)
      op->setAttr("alps.kv_split_reduction_boundary",
                  rewriter.getUnitAttr());
  }

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
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(elType));
    Value scale = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elType,
                              1.0 / std::sqrt(static_cast<double>(head_dim))));

    // Compute K^T
    Value keyTinit = tensor::EmptyOp::create(rewriter, loc, keyTshape, elType);
    Value keyT = linalg::TransposeOp::create(rewriter, loc, key, keyTinit,
                                             ArrayRef<int64_t>{0, 2, 1})
                     .getResult()[0];

    // Compute batch matmul QK^T
    Value qkTempty = tensor::EmptyOp::create(rewriter, loc, qkTshape, elType);
    Value qkTinit =
        linalg::FillOp::create(rewriter, loc, zero, qkTempty).getResult(0);
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
      attachTopologyAttrs(qkOp.getOperation(), rewriter);
    }
    Value qkT = qkOp.getResult(0);
    DBG(" batch-matmul: " << qkT);

    // Scale QK^T by 1/sqrt(head_dim)
    Value qkTScaled =
        linalg::GenericOp::create(
            rewriter, loc, qkTtype, ValueRange{qkT}, ValueRange{qkTempty},
            ArrayRef<AffineMap>{AffineMap::get(3, 0, {d0, d1, d2}, ctx),
                                AffineMap::get(3, 0, {d0, d1, d2}, ctx)},
            ArrayRef<utils::IteratorType>{parallel, parallel, parallel},
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value mul = arith::MulFOp::create(b, loc, args[0], scale);
              linalg::YieldOp::create(b, loc, mul);
            })
            .getResult(0);
    DBG("Scaled QK^T: " << qkTScaled);

    // Apply mask to scaled QK^T
    Value qkTMasked = linalg::AddOp::create(
                          rewriter, loc, ValueRange{qkTScaled, mask}, qkTempty)
                          .getResult(0);
    DBG("Masked QK^T: " << qkTMasked);

    // - Softmax -
    // Step 1: Find max for numerical stability
    Value maxEmpty = tensor::EmptyOp::create(rewriter, loc, maxShape, elType);
    Value negInf = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(elType,
                              -std::numeric_limits<double>::infinity()));
    Value maxInit =
        linalg::FillOp::create(rewriter, loc, negInf, maxEmpty).getResult(0);
    Value maxVals =
        linalg::ReduceOp::create(
            rewriter, loc, qkTMasked, maxInit, ArrayRef<int64_t>{2},
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value max = arith::MaximumFOp::create(b, loc, args[0], args[1]);
              linalg::YieldOp::create(b, loc, max);
            })
            .getResult(0);

    // Step 2: Compute `exp(xi-max)` with implicit broadcast
    Value qkTSub =
        linalg::GenericOp::create(
            rewriter, loc, qkTtype, ValueRange{qkTMasked, maxVals},
            ValueRange{qkTempty},
            ArrayRef<AffineMap>{AffineMap::get(3, 0, {d0, d1, d2}, ctx),
                                AffineMap::get(3, 0, {d0, d1}, ctx),
                                AffineMap::get(3, 0, {d0, d1, d2}, ctx)},
            ArrayRef<utils::IteratorType>{parallel, parallel, parallel},
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value sub = arith::SubFOp::create(b, loc, args[0], args[1]);
              linalg::YieldOp::create(b, loc, sub);
            })
            .getResult(0);
    Value qkTStable =
        linalg::ExpOp::create(rewriter, loc, qkTSub, qkTempty).getResult(0);

    // Step 3: Compute `sum(exp(xi-max))` along the last dimension
    Value sumInit =
        linalg::FillOp::create(rewriter, loc, zero, maxEmpty).getResult(0);
    Value sumVals = linalg::ReduceOp::create(
                        rewriter, loc, qkTStable, sumInit, ArrayRef<int64_t>{2},
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          Value add =
                              arith::AddFOp::create(b, loc, args[0], args[1]);
                          linalg::YieldOp::create(b, loc, add);
                        })
                        .getResult(0);

    // Step 4:  div to get softmax
    Value softmaxResult =
        linalg::GenericOp::create(
            rewriter, loc, qkTtype, ValueRange{qkTStable, sumVals},
            ValueRange{qkTempty},
            ArrayRef<AffineMap>{AffineMap::get(3, 0, {d0, d1, d2}, ctx),
                                AffineMap::get(3, 0, {d0, d1}, ctx),
                                AffineMap::get(3, 0, {d0, d1, d2}, ctx)},
            ArrayRef<utils::IteratorType>{parallel, parallel, parallel},
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value div = arith::DivFOp::create(b, loc, args[0], args[1]);
              linalg::YieldOp::create(b, loc, div);
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
      attachTopologyAttrs(avOp.getOperation(), rewriter);
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
/// Square attention (S == H), which occurs in audio encoders such as HuBERT
/// with S=64 and head_dim=64, needs structural evidence because its shapes
/// alone are indistinguishable from a generic square batch matmul. Recognize
/// QK only when the RHS originates at a last-two-dim transpose, then recognize
/// AV only when its probability input depends on that already-marked QK.
/// The explicit metadata emitted by LowerAttentionOp always takes precedence.
static Operation *findLastDpsWriterBefore(Value buffer, Operation *consumer) {
  Operation *last = nullptr;
  for (Operation *user : buffer.getUsers()) {
    if (user->getBlock() != consumer->getBlock() ||
        !user->isBeforeInBlock(consumer))
      continue;
    auto dps = dyn_cast<DestinationStyleOpInterface>(user);
    if (!dps)
      continue;
    if (llvm::any_of(dps.getDpsInitsMutable(),
                     [&](OpOperand &init) { return init.get() == buffer; }) &&
        (!last || last->isBeforeInBlock(user)))
      last = user;
  }
  return last;
}

static bool isLastTwoDimTransposeGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic.getNumReductionLoops() != 0)
    return false;
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || maps[0].getNumResults() < 2 ||
      !maps[1].isIdentity() ||
      maps[0].getNumResults() != maps[1].getNumResults())
    return false;
  const unsigned rank = maps[0].getNumResults();
  for (unsigned i = 0; i < rank - 2; ++i)
    if (maps[0].getResult(i) != getAffineDimExpr(i, generic.getContext()))
      return false;
  return maps[0].getResult(rank - 2) ==
             getAffineDimExpr(rank - 1, generic.getContext()) &&
         maps[0].getResult(rank - 1) ==
             getAffineDimExpr(rank - 2, generic.getContext());
}

static bool hasLastTwoDimTransposeAncestor(Value value, Operation *consumer,
                                           unsigned depth = 0) {
  if (depth > 8)
    return false;
  Operation *def = value.getDefiningOp();
  if (isa<MemRefType>(value.getType())) {
    if (Operation *writer = findLastDpsWriterBefore(value, consumer)) {
      if (isa<linalg::TransposeOp>(writer))
        return true;
      if (auto generic = dyn_cast<linalg::GenericOp>(writer))
        if (isLastTwoDimTransposeGeneric(generic))
          return true;
      def = writer;
    }
  }
  if (!def)
    return false;
  if (auto transpose = dyn_cast<linalg::TransposeOp>(def)) {
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    if (permutation.size() >= 2) {
      const int64_t rank = permutation.size();
      bool prefixIdentity = true;
      for (int64_t i = 0; i < rank - 2; ++i)
        prefixIdentity &= permutation[i] == i;
      if (prefixIdentity && permutation[rank - 2] == rank - 1 &&
          permutation[rank - 1] == rank - 2)
        return true;
    }
    return false;
  }
  if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp>(def))
    return false;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return hasLastTwoDimTransposeAncestor(operand, def, depth + 1);
  });
}

static bool hasBshdToBhsdTransposeAncestor(Value value,
                                           unsigned depth = 0) {
  if (depth > 8)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (auto transpose = dyn_cast<linalg::TransposeOp>(def)) {
    ArrayRef<int64_t> p = transpose.getPermutation();
    return p.size() == 4 && p[0] == 0 && p[1] == 2 && p[2] == 1 &&
           p[3] == 3;
  }
  if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp>(def))
    return false;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return hasBshdToBhsdTransposeAncestor(operand, depth + 1);
  });
}

static bool hasFoldedBatchKtAccess(linalg::LinalgOp op) {
  if (!isa<linalg::GenericOp>(op.getOperation()))
    return false;
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (op.getNumLoops() != 4 || maps.size() < 2 ||
      maps[1].getNumResults() != 3)
    return false;
  MLIRContext *ctx = op.getContext();
  return maps[1].getResult(0) == getAffineDimExpr(0, ctx) &&
         maps[1].getResult(1) == getAffineDimExpr(2, ctx) &&
         maps[1].getResult(2) == getAffineDimExpr(3, ctx);
}

static bool dependsOnMarkedAttentionKey(Value value, Operation *consumer,
                                        unsigned depth = 0) {
  if (depth > 32)
    return false;
  Operation *def = value.getDefiningOp();
  if (isa<MemRefType>(value.getType()))
    if (Operation *writer = findLastDpsWriterBefore(value, consumer))
      def = writer;
  if (!def)
    return false;
  if (auto role =
          def->getAttrOfType<StringAttr>("omni_fetch.kv_cache_role"))
    if (role.getValue() == "key")
      return true;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return dependsOnMarkedAttentionKey(operand, def, depth + 1);
  });
}

static bool dependsOnSoftmaxLike(Value value, Operation *consumer,
                                 unsigned depth = 0) {
  if (depth > 32)
    return false;
  Operation *def = value.getDefiningOp();
  if (isa<MemRefType>(value.getType()))
    if (Operation *writer = findLastDpsWriterBefore(value, consumer))
      def = writer;
  if (!def)
    return false;
  bool softmaxMath = false;
  def->walk([&](Operation *nested) {
    if (isa<math::ExpOp, arith::DivFOp>(nested))
      softmaxMath = true;
  });
  if (softmaxMath)
    return true;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return dependsOnSoftmaxLike(operand, def, depth + 1);
  });
}

static int64_t annotateEagerAttentionKvStreams(
    FunctionOpInterface func, bool emitKvFusionBoundary,
    bool emitKvElementwiseFusionBoundary, bool emitKvMultiUseFusionBoundary,
    bool emitKvSplitReductionBoundary) {
  int64_t inferred = 0;
  int64_t keys = 0;
  int64_t values = 0;
  DenseMap<Block *, Operation *> pendingSquareKey;
  func.walk([&](linalg::LinalgOp op) {
    // Earlier scheduling/generalization may preserve semantic identity but
    // intentionally omit the independent topology policy.  Apply the ALPS
    // fusion marker without re-running shape inference in that case.
    if (op->hasAttr("omni_fetch.kv_cache_role")) {
      if (emitKvFusionBoundary)
        op->setAttr("alps.kv_fusion_boundary",
                    UnitAttr::get(op.getContext()));
      if (emitKvElementwiseFusionBoundary)
        op->setAttr("alps.kv_elementwise_fusion_boundary",
                    UnitAttr::get(op.getContext()));
      if (emitKvMultiUseFusionBoundary)
        op->setAttr("alps.kv_multi_use_fusion_boundary",
                    UnitAttr::get(op.getContext()));
      if (emitKvSplitReductionBoundary)
        op->setAttr("alps.kv_split_reduction_boundary",
                    UnitAttr::get(op.getContext()));
      return;
    }
    if (op.getNumReductionLoops() == 0 || op.getDpsInputs().size() < 2 ||
        op.getDpsInits().empty())
      return;

    auto out = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
    if (!out || !out.hasStaticShape())
      return;

    ArrayRef<int64_t> c = out.getShape();
    StringRef role;
    int64_t operandIndex = -1;
    StringRef layout = "sequence_head";

    // Fusion can inline Q/K/V projection epilogues into attention.  The
    // physical stream then remains [B,S,H,D], while the contraction output is
    // [B,H,S,S] (QK) or [B,H,S,D] (AV).  Select the K/V activation operand
    // directly instead of forcing materialization merely to recover [B,H,S,D].
    if (out.getRank() == 4 && c[0] > 0 && c[1] > 0 && c[2] > 0) {
      SmallVector<int64_t> bshdOperands;
      bool hasAttentionProb = false;
      for (auto [index, input] : llvm::enumerate(op.getDpsInputs())) {
        auto type = dyn_cast<ShapedType>(input.getType());
        if (!type || !type.hasStaticShape() || type.getRank() != 4)
          continue;
        ArrayRef<int64_t> shape = type.getShape();
        if (shape[0] == c[0] && shape[1] == c[2] &&
            shape[2] == c[1] && shape[3] > 0)
          bshdOperands.push_back(index);
        if (shape[0] == c[0] && shape[1] == c[1] &&
            shape[2] == c[2] && shape[3] == c[2])
          hasAttentionProb = true;
      }
      if (c[2] == c[3] && bshdOperands.size() >= 2) {
        role = "key";
        operandIndex = bshdOperands.back();
        layout = "bshd";
      } else if (c[2] != c[3] && hasAttentionProb &&
                 !bshdOperands.empty()) {
        role = "value";
        operandIndex = bshdOperands.back();
        layout = "bshd";
      }
    }

    // One-shot bufferization can fold the unit batch dimension, yielding
    // physical [S,H,D] streams and [H,S,S]/[H,S,D] contractions.
    if (operandIndex < 0 && out.getRank() == 3) {
      SmallVector<int64_t> shdOperands;
      bool hasAttentionProb = false;
      for (auto [index, input] : llvm::enumerate(op.getDpsInputs())) {
        auto type = dyn_cast<ShapedType>(input.getType());
        if (!type || !type.hasStaticShape() || type.getRank() != 3)
          continue;
        ArrayRef<int64_t> shape = type.getShape();
        if (shape[0] == c[1] && shape[1] == c[0] && shape[2] > 0)
          shdOperands.push_back(index);
        if (shape[0] == c[0] && shape[1] == c[1] &&
            shape[2] == c[1])
          hasAttentionProb = true;
      }
      if (c[1] == c[2] && shdOperands.size() >= 2) {
        role = "key";
        operandIndex = shdOperands.back();
        layout = "shd";
      } else if (c[1] != c[2] && hasAttentionProb &&
                 !shdOperands.empty()) {
        role = "value";
        operandIndex = shdOperands.back();
        layout = "shd";
      }
    }

    if (operandIndex < 0 && out.getRank() == 3) {
      auto lhs = dyn_cast<ShapedType>(op.getDpsInputs()[0].getType());
      auto rhs = dyn_cast<ShapedType>(op.getDpsInputs()[1].getType());
      if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
          lhs.getRank() != 3 || rhs.getRank() != 3)
        return;
      ArrayRef<int64_t> a = lhs.getShape();
      ArrayRef<int64_t> b = rhs.getShape();
      if (a[0] != b[0] || a[0] != c[0])
        return;

      // QK^T may carry explicit [B,H,S] K^T, or scheduling may have folded
      // the transpose into the indexing map and restored physical [B,S,H].
      bool rhsIsExplicitKt = b[1] == a[2] && b[2] == a[1];
      bool rhsIsFoldedK = b[1] == a[1] && b[2] == a[2];
      if (c[1] == c[2] && c[1] == a[1] && a[1] != a[2] &&
          (rhsIsExplicitKt || rhsIsFoldedK))
        role = "key";
      else if (a[1] == a[2] && a[1] == b[1] && b[1] == c[1] &&
               b[2] == c[2] && c[1] != c[2])
        role = "value";
      else if (a[1] == a[2] && b[1] == b[2] && c[1] == c[2] &&
               a[1] == b[1] && b[1] == c[1]) {
        if (hasFoldedBatchKtAccess(op) ||
            hasLastTwoDimTransposeAncestor(op.getDpsInputs()[1], op))
          role = "key";
        else if (hasBshdToBhsdTransposeAncestor(op.getDpsInputs()[1]))
          role = "value";
        else if (dependsOnMarkedAttentionKey(op.getDpsInputs()[0], op) ||
                 dependsOnSoftmaxLike(op.getDpsInputs()[0], op))
          role = "value";
        else if (Operation *key = pendingSquareKey.lookup(op->getBlock())) {
          unsigned distance = 0;
          for (Operation *cursor = key->getNextNode(); cursor && cursor != op;
               cursor = cursor->getNextNode())
            ++distance;
          if (distance <= 32)
            role = "value";
        }
      }
      if (!role.empty())
        operandIndex = 1;
    }

    if (operandIndex < 0 || role.empty())
      return;

    Builder bld(op.getContext());
    op->setAttr("omni_fetch.kv_cache_role", bld.getStringAttr(role));
    op->setAttr("omni_fetch.kv_cache_operand",
                bld.getI64IntegerAttr(operandIndex));
    op->setAttr("omni_fetch.kv_cache_layout", bld.getStringAttr(layout));
    if (emitKvFusionBoundary)
      op->setAttr("alps.kv_fusion_boundary", bld.getUnitAttr());
    if (emitKvElementwiseFusionBoundary)
      op->setAttr("alps.kv_elementwise_fusion_boundary", bld.getUnitAttr());
    if (emitKvMultiUseFusionBoundary)
      op->setAttr("alps.kv_multi_use_fusion_boundary", bld.getUnitAttr());
    if (emitKvSplitReductionBoundary)
      op->setAttr("alps.kv_split_reduction_boundary", bld.getUnitAttr());
    if (role == "key")
      pendingSquareKey[op->getBlock()] = op;
    else if (role == "value")
      pendingSquareKey.erase(op->getBlock());
    keys += role == "key";
    values += role == "value";
    ++inferred;
  });
  if (inferred)
    llvm::errs() << "[KVCacheMetadataRoles] key=" << keys
                 << " value=" << values << "\n";
  return inferred;
}

void HexagonLowerTmTensorPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LowerAttentionOp>(
      patterns.getContext(), emitKvCacheMetadata, emitKvFusionBoundary,
      emitKvElementwiseFusionBoundary, emitKvMultiUseFusionBoundary,
      emitKvSplitReductionBoundary);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  if (emitKvCacheMetadata) {
    int64_t inferred = annotateEagerAttentionKvStreams(
        getOperation(), emitKvFusionBoundary,
        emitKvElementwiseFusionBoundary, emitKvMultiUseFusionBoundary,
        emitKvSplitReductionBoundary);
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
