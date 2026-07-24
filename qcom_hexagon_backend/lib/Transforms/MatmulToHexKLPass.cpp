//===-- MatmulToHexKLPass.cpp - linalg.matmul to hexkl ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to transform linalg::MatmulOp to hexkl ops.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "-matmul-to-hexkl"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_MATMULTOHEXKL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToHexKL(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // HMX micro-kernels tile in 32x32 blocks (see DecomposeHexKLMatmulPass).
    // Converting unaligned shapes (e.g. GPT-2 lm_head N=50257) faults on
    // device (adb exit 13).  Keep those on the HVX path instead.
    auto lhsTy = dyn_cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
    auto rhsTy = dyn_cast<ShapedType>(op.getDpsInputOperand(1)->get().getType());
    if (!lhsTy || !rhsTy || !lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic matmul shape");
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 matmul");
    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);
    if (K != rhsTy.getDimSize(0))
      return rewriter.notifyMatchFailure(op, "K mismatch");
    constexpr int64_t kHmxTile = 32;
    if ((M % kHmxTile) != 0 || (K % kHmxTile) != 0 || (N % kHmxTile) != 0) {
      DBG("skip HexKL: unaligned MxKxN=" << M << "x" << K << "x" << N);
      return rewriter.notifyMatchFailure(
          op, "M/K/N not divisible by HMX tile size 32");
    }

    // Attention score / context matmuls (after ReduceContractionRank collapses
    // batch=1 batch_matmul) are tile-aligned at seq=32 but HMX on those shapes
    // faults on device (Bad VA / exit 13), e.g. QK^T 32x64x32 (N==M) and
    // AV 32x32x64 (K==M). Keep projections / FFN / lm_head on HexKL only.
    if (K == M || N == M) {
      DBG("skip HexKL: attention-like MxKxN=" << M << "x" << K << "x" << N);
      return rewriter.notifyMatchFailure(
          op, "attention-like matmul (K==M or N==M); keep HVX");
    }

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];
    rewriter.replaceOpWithNewOp<hexkl::MatmulOp>(op, C.getType(), A, B, C);
    return success();
  }
};

void populateMatmulToHexKLPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToHexKL>(patterns.getContext());
}

struct MatmulToHexKLPass : public ::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMatmulToHexKLPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createMatmulToHexKLPass() {
  return std::make_unique<MatmulToHexKLPass>();
}
