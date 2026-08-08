//===- ScheduleMatmulForHVXPass.cpp - Scheduling matmul ops          ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file specifies the rules to schedule matmul op and its variants.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/LinalgUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "schedule-matmul-for-hvx"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SCHEDULEMATMULFORHVX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static void copyOmniFetchAttrs(Operation *from, Operation *to) {
  for (StringRef name :
       {"omni_fetch.kv_cache_role", "omni_fetch.kv_cache_operand",
        "omni_fetch.kv_cache_layout",
        "omni_fetch.n1_weight_stationary", "omni_fetch.n1_mkn",
        "omni_fetch.n1_baseline_weight_bytes",
        "omni_fetch.n1_stationary_weight_bytes",
        "omni_fetch.n1_added_transpose_bytes",
        "omni_fetch.n1_predicted_saved_bytes"})
    if (Attribute attr = from->getAttr(name))
      to->setAttr(name, attr);
}

struct ScheduleMatmulForHVXPass
    : public ::impl::ScheduleMatmulForHVXBase<ScheduleMatmulForHVXPass> {
  explicit ScheduleMatmulForHVXPass() = default;
  explicit ScheduleMatmulForHVXPass(
      const ScheduleMatmulForHVXOptions &options)
      : ScheduleMatmulForHVXBase(options) {}
  void runOnOperation() override;
  FailureOr<linalg::GenericOp> GeneralizeOp(IRRewriter &rewriter,
                                            linalg::LinalgOp linalgOp);
};

struct WeightStationaryLedger {
  int64_t candidates = 0;
  int64_t admitted = 0;
  int64_t baselineWeightBytes = 0;
  int64_t stationaryWeightBytes = 0;
  int64_t addedTransposeBytes = 0;
  int64_t predictedSavedBytes = 0;
};

struct ActivationMulticastLedger {
  int64_t candidates = 0;
  int64_t admitted = 0;
  int64_t consumersFused = 0;
  int64_t estimatedVectorActivationBytesSaved = 0;
};

static bool isAfterInSameBlock(Operation *anchor, Operation *user) {
  return anchor->getBlock() == user->getBlock() &&
         anchor->isBeforeInBlock(user);
}

/// Fuse two equal-shape sibling projections that consume the identical
/// activation:
///
///   C0 += A * B0; C1 += A * B1
///
/// into one five-operand generic.  Its scalar/vector body receives A once and
/// uses it in both FMAs.  This is deliberately limited to two consumers to
/// bound VRF pressure; N6's final-object gate decides whether the fused pair
/// actually removed loads without introducing spills.
static bool rewriteActivationMulticast(IRRewriter &rewriter,
                                       linalg::MatmulOp first,
                                       linalg::MatmulOp second,
                                       ActivationMulticastLedger &ledger) {
  if (!first || !second || first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) || !first.hasPureTensorSemantics() ||
      !second.hasPureTensorSemantics() ||
      first->getNumResults() != 1 || second->getNumResults() != 1 ||
      first->hasAttr("omni_fetch.n2_activation_multicast") ||
      second->hasAttr("omni_fetch.n2_activation_multicast") ||
      first.getInputs()[0] != second.getInputs()[0])
    return false;

  auto aType = dyn_cast<RankedTensorType>(first.getInputs()[0].getType());
  auto b0Type = dyn_cast<RankedTensorType>(first.getInputs()[1].getType());
  auto b1Type = dyn_cast<RankedTensorType>(second.getInputs()[1].getType());
  auto c0Type = dyn_cast<RankedTensorType>(first.getOutputs()[0].getType());
  auto c1Type = dyn_cast<RankedTensorType>(second.getOutputs()[0].getType());
  if (!aType || !b0Type || !b1Type || !c0Type || !c1Type ||
      aType.getRank() != 2 || b0Type.getRank() != 2 ||
      b1Type.getRank() != 2 || c0Type.getRank() != 2 ||
      c1Type.getRank() != 2 || !aType.hasStaticShape() ||
      !b0Type.hasStaticShape() || !b1Type.hasStaticShape() ||
      !c0Type.hasStaticShape() || !c1Type.hasStaticShape() ||
      aType.getElementType() != b0Type.getElementType() ||
      aType.getElementType() != b1Type.getElementType() ||
      !aType.getElementType().isF16() ||
      !c0Type.getElementType().isF32() ||
      c0Type != c1Type || b0Type != b1Type)
    return false;

  int64_t m = aType.getShape()[0];
  int64_t k = aType.getShape()[1];
  int64_t n = b0Type.getShape()[1];
  constexpr int64_t kF16HvxElements = 64;
  constexpr int64_t kMinPredicatedF16Elements = 32;
  if (m <= 0 || n < kMinPredicatedF16Elements ||
      n % kMinPredicatedF16Elements != 0 ||
      b0Type.getShape()[0] != k ||
      c0Type.getShape()[0] != m || c0Type.getShape()[1] != n)
    return false;

  ++ledger.candidates;
  // Moving the fused operation to the second site is legal only if the first
  // result has no intervening use.
  if (!llvm::all_of(first->getUsers(), [&](Operation *user) {
        return isAfterInSameBlock(second, user);
      }))
    return false;

  MLIRContext *ctx = rewriter.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr d1 = getAffineDimExpr(1, ctx);
  AffineExpr d2 = getAffineDimExpr(2, ctx);
  SmallVector<AffineMap> maps{
      AffineMap::get(3, 0, {d0, d1}, ctx),
      AffineMap::get(3, 0, {d1, d2}, ctx),
      AffineMap::get(3, 0, {d1, d2}, ctx),
      AffineMap::get(3, 0, {d0, d2}, ctx),
      AffineMap::get(3, 0, {d0, d2}, ctx)};
  SmallVector<utils::IteratorType> iterators{
      utils::IteratorType::parallel, utils::IteratorType::reduction,
      utils::IteratorType::parallel};

  rewriter.setInsertionPoint(second);
  auto fused = rewriter.create<linalg::GenericOp>(
      second.getLoc(), TypeRange{c0Type, c1Type},
      ValueRange{first.getInputs()[0], first.getInputs()[1],
                 second.getInputs()[1]},
      ValueRange{first.getOutputs()[0], second.getOutputs()[0]}, maps,
      iterators, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value a = b.create<arith::ExtFOp>(loc, c0Type.getElementType(),
                                         args[0]);
        Value b0 = b.create<arith::ExtFOp>(loc, c0Type.getElementType(),
                                          args[1]);
        Value b1 = b.create<arith::ExtFOp>(loc, c0Type.getElementType(),
                                          args[2]);
        Value product0 = b.create<arith::MulFOp>(loc, a, b0);
        Value product1 = b.create<arith::MulFOp>(loc, a, b1);
        Value sum0 = b.create<arith::AddFOp>(loc, args[3], product0);
        Value sum1 = b.create<arith::AddFOp>(loc, args[4], product1);
        b.create<linalg::YieldOp>(loc, ValueRange{sum0, sum1});
      });
  fused->setAttr("omni_fetch.n2_activation_multicast",
                 rewriter.getUnitAttr());
  fused->setAttr("omni_fetch.n2_mkn",
                 rewriter.getDenseI64ArrayAttr({m, k, n}));
  int64_t vectorChunks = (n + kF16HvxElements - 1) / kF16HvxElements;
  int64_t savedBytes = m * k * vectorChunks * 2;
  fused->setAttr("omni_fetch.n2_estimated_vector_activation_bytes_saved",
                 rewriter.getI64IntegerAttr(savedBytes));

  rewriter.replaceOp(first, fused.getResult(0));
  rewriter.replaceOp(second, fused.getResult(1));
  ++ledger.admitted;
  ledger.consumersFused += 2;
  ledger.estimatedVectorActivationBytesSaved += savedBytes;
  return true;
}

/// Rewrite A[M,K] * B[K,N] -> transpose(
///   transpose(B)[N,K] * transpose(A)[K,M]).
///
/// The inner contiguous HVX dimension becomes M, so one weight element serves
/// a vector of sequence/image positions.  The explicit transposes are the
/// conservative N1 implementation boundary; later in-situ layout production
/// can fold them into producers/consumers.  A byte ledger prevents applying
/// this schedule where transpose traffic would erase the predicted weight-read
/// saving.
static bool rewriteWeightStationary(IRRewriter &rewriter,
                                    linalg::MatmulOp op,
                                    WeightStationaryLedger &ledger) {
  if (!op.hasPureTensorSemantics() || op->getNumResults() != 1 ||
      op->hasAttr("omni_fetch.n1_weight_stationary"))
    return false;

  auto aType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto bType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto cType = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
  if (!aType || !bType || !cType || aType.getRank() != 2 ||
      bType.getRank() != 2 || cType.getRank() != 2 ||
      !aType.hasStaticShape() || !bType.hasStaticShape() ||
      !cType.hasStaticShape() || !aType.getElementType().isF16() ||
      !bType.getElementType().isF16())
    return false;

  int64_t m = aType.getShape()[0];
  int64_t k = aType.getShape()[1];
  int64_t n = bType.getShape()[1];
  if (m <= 0 || k <= 0 || n <= 0 || bType.getShape()[0] != k ||
      cType.getShape()[0] != m || cType.getShape()[1] != n)
    return false;

  // Attention score/context contractions need an online tiled design (N3),
  // not projection transposition.  N1 targets prefill projection/MLP/vocab.
  if (k == m || n == m)
    return false;
  constexpr int64_t kF16HvxElements = 64;
  if (m < kF16HvxElements || m % kF16HvxElements != 0)
    return false;

  ++ledger.candidates;
  int64_t outBytes = cType.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t baselineWeightBytes = m * k * n * 2;
  int64_t stationaryWeightBytes = k * n * 2;
  // Read+write A, B, init-C, and final C transposes.  This deliberately
  // overestimates cost; canonicalization may eliminate constant/fill cases.
  int64_t transposeBytes =
      2 * m * k * 2 + 2 * k * n * 2 + 4 * m * n * outBytes;
  int64_t predictedSaved =
      baselineWeightBytes - stationaryWeightBytes - transposeBytes;
  if (predictedSaved <= transposeBytes)
    return false;

  Location loc = op.getLoc();
  SmallVector<Value> noDynamicDims;
  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value init = op.getOutputs()[0];

  auto makeTranspose = [&](Value source, ArrayRef<int64_t> shape,
                           Type elementType) -> Value {
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc, shape, elementType, noDynamicDims);
    return rewriter
        .create<linalg::TransposeOp>(loc, source, empty,
                                     ArrayRef<int64_t>{1, 0})
        .getResult()[0];
  };

  rewriter.setInsertionPoint(op);
  Value bt = makeTranspose(b, {n, k}, bType.getElementType());
  Value at = makeTranspose(a, {k, m}, aType.getElementType());
  Value ctInit = makeTranspose(init, {n, m}, cType.getElementType());
  auto ctType =
      RankedTensorType::get({n, m}, cType.getElementType());
  auto stationary = rewriter.create<linalg::MatmulOp>(
      loc, ctType, ValueRange{bt, at}, ValueRange{ctInit});
  stationary->setAttr("omni_fetch.n1_weight_stationary",
                      rewriter.getUnitAttr());
  stationary->setAttr("omni_fetch.n1_mkn",
                      rewriter.getDenseI64ArrayAttr({m, k, n}));
  stationary->setAttr(
      "omni_fetch.n1_baseline_weight_bytes",
      rewriter.getI64IntegerAttr(baselineWeightBytes));
  stationary->setAttr(
      "omni_fetch.n1_stationary_weight_bytes",
      rewriter.getI64IntegerAttr(stationaryWeightBytes));
  stationary->setAttr("omni_fetch.n1_added_transpose_bytes",
                      rewriter.getI64IntegerAttr(transposeBytes));
  stationary->setAttr("omni_fetch.n1_predicted_saved_bytes",
                      rewriter.getI64IntegerAttr(predictedSaved));

  Value result = makeTranspose(stationary.getResult(0), {m, n},
                               cType.getElementType());
  rewriter.replaceOp(op, result);

  ++ledger.admitted;
  ledger.baselineWeightBytes += baselineWeightBytes;
  ledger.stationaryWeightBytes += stationaryWeightBytes;
  ledger.addedTransposeBytes += transposeBytes;
  ledger.predictedSavedBytes += predictedSaved;
  return true;
}

FailureOr<linalg::GenericOp>
ScheduleMatmulForHVXPass::GeneralizeOp(IRRewriter &rewriter,
                                       linalg::LinalgOp linalgOp) {
  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::GenericOp> generalizedOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);
  if (failed(generalizedOp)) {
    linalgOp->emitOpError("failed to generalize linalg named op");
    signalPassFailure();
    return failure(); // Return a failure result
  }
  copyOmniFetchAttrs(linalgOp, *generalizedOp);
  return generalizedOp;
}

void ScheduleMatmulForHVXPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  ActivationMulticastLedger n2Ledger;
  if (enableActivationMulticast) {
    SmallVector<linalg::MatmulOp> originalMatmuls;
    funcOp.walk(
        [&](linalg::MatmulOp op) { originalMatmuls.push_back(op); });
    llvm::SmallDenseSet<Operation *> consumed;
    for (auto [index, first] : llvm::enumerate(originalMatmuls)) {
      if (consumed.contains(first.getOperation()))
        continue;
      for (linalg::MatmulOp second :
           llvm::drop_begin(originalMatmuls, index + 1)) {
        if (consumed.contains(second.getOperation()))
          continue;
        if (rewriteActivationMulticast(rewriter, first, second, n2Ledger)) {
          consumed.insert(first.getOperation());
          consumed.insert(second.getOperation());
          break;
        }
      }
    }
    Builder b(&getContext());
    funcOp->setAttr("omni_fetch.n2_candidates",
                    b.getI64IntegerAttr(n2Ledger.candidates));
    funcOp->setAttr("omni_fetch.n2_admitted",
                    b.getI64IntegerAttr(n2Ledger.admitted));
    funcOp->setAttr(
        "omni_fetch.n2_estimated_vector_activation_bytes_saved",
        b.getI64IntegerAttr(
            n2Ledger.estimatedVectorActivationBytesSaved));
    std::string ledgerLine;
    llvm::raw_string_ostream ledgerStream(ledgerLine);
    ledgerStream << "[OmniFetchN2] function=" << funcOp.getName()
                 << " candidates=" << n2Ledger.candidates
                 << " admitted=" << n2Ledger.admitted
                 << " consumers_fused=" << n2Ledger.consumersFused
                 << " estimated_vector_activation_bytes_saved="
                 << n2Ledger.estimatedVectorActivationBytesSaved << "\n";
    ledgerStream.flush();
    llvm::errs() << ledgerLine;
  }

  WeightStationaryLedger n1Ledger;
  if (enableWeightStationary) {
    SmallVector<linalg::MatmulOp> originalMatmuls;
    funcOp.walk(
        [&](linalg::MatmulOp op) { originalMatmuls.push_back(op); });
    for (linalg::MatmulOp op : originalMatmuls)
      rewriteWeightStationary(rewriter, op, n1Ledger);

    Builder b(&getContext());
    funcOp->setAttr("omni_fetch.n1_candidates",
                    b.getI64IntegerAttr(n1Ledger.candidates));
    funcOp->setAttr("omni_fetch.n1_admitted",
                    b.getI64IntegerAttr(n1Ledger.admitted));
    funcOp->setAttr(
        "omni_fetch.n1_predicted_saved_bytes",
        b.getI64IntegerAttr(n1Ledger.predictedSavedBytes));
    std::string ledgerLine;
    llvm::raw_string_ostream ledgerStream(ledgerLine);
    ledgerStream << "[OmniFetchN1] function=" << funcOp.getName()
                 << " candidates=" << n1Ledger.candidates
                 << " admitted=" << n1Ledger.admitted
                 << " baseline_weight_bytes="
                 << n1Ledger.baselineWeightBytes
                 << " stationary_weight_bytes="
                 << n1Ledger.stationaryWeightBytes
                 << " added_transpose_bytes="
                 << n1Ledger.addedTransposeBytes
                 << " predicted_saved_bytes="
                 << n1Ledger.predictedSavedBytes << "\n";
    ledgerStream.flush();
    llvm::errs() << ledgerLine;
  }

  // Optimize below Matmul variants
  //   linalg.matmul(a, transpose_b) -> linalg.matmul_transpose_b
  //   linalg.matmul(transpose_a, b) -> linalg.matmul_transpose_a
  //   linalg.batch_matmul(a, transpose_b) -> linalg.batch_matmul_transpose_b
  //   linalg.batch_matmul(transpose_a, b) -> linalg.batch_matmul_transpose_a
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::MatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->hasAttr("omni_fetch.n1_weight_stationary") &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (secondOperandDef && isa<linalg::TransposeOp>(secondOperandDef)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul_transpose_b
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp = rewriter.create<linalg::MatmulTransposeBOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{linalgOp.getDpsInputs()[0], transposeOp.getOperand(0)},
            linalgOp.getDpsInits());
        copyOmniFetchAttrs(linalgOp, matmulTransposeBOp);
        rewriter.replaceOp(linalgOp, matmulTransposeBOp);
        rewriter.eraseOp(transposeOp);
      } else if (firstOperandDef && isa<linalg::TransposeOp>(firstOperandDef)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul_transpose_a
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp = rewriter.create<linalg::MatmulTransposeAOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{transposeOp.getOperand(0), linalgOp.getDpsInputs()[1]},
            linalgOp.getDpsInits());
        copyOmniFetchAttrs(linalgOp, matmulTransposeAOp);
        rewriter.replaceOp(linalgOp, matmulTransposeAOp);
        rewriter.eraseOp(transposeOp);
      }
    } else if (isa_and_nonnull<linalg::BatchMatmulOp>(
                   linalgOp.getOperation()) &&
               !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (secondOperandDef && isa<linalg::TransposeOp>(secondOperandDef)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul_transpose_b op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp =
            rewriter.create<linalg::BatchMatmulTransposeBOp>(
                linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
                ValueRange{linalgOp.getDpsInputs()[0],
                           transposeOp.getOperand(0)},
                linalgOp.getDpsInits());
        copyOmniFetchAttrs(linalgOp, matmulTransposeBOp);
        rewriter.replaceOp(linalgOp, matmulTransposeBOp);
        rewriter.eraseOp(transposeOp);
      } else if (firstOperandDef && isa<linalg::TransposeOp>(firstOperandDef)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul_transpose_a op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp =
            rewriter.create<linalg::BatchMatmulTransposeAOp>(
                linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
                ValueRange{transposeOp.getOperand(0),
                           linalgOp.getDpsInputs()[1]},
                linalgOp.getDpsInits());
        copyOmniFetchAttrs(linalgOp, matmulTransposeAOp);
        rewriter.replaceOp(linalgOp, matmulTransposeAOp);
        rewriter.eraseOp(transposeOp);
      }
    }
  });

  SmallVector<linalg::LinalgOp> batchMatmulOps;
  // Collect all linalg::BatchMatmulOps that do not have the
  // "library_call" attribute
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::BatchMatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      batchMatmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : batchMatmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    auto permutation = getBatchMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine batch matmul permutation");
      return signalPassFailure();
    }
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }

  SmallVector<linalg::LinalgOp> matmulOps;
  // Collect all linalg::MatmulOps that do not have the "library_call" attribute
  funcOp.walk([&](linalg::MatmulOp linalgOp) {
    if (isa_and_nonnull<linalg::LinalgOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      matmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : matmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    auto permutation = getMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine matmul permutation");
      return signalPassFailure();
    }
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createScheduleMatmulForHVXPass(
    const ScheduleMatmulForHVXOptions &options) {
  return std::make_unique<ScheduleMatmulForHVXPass>(options);
}
