//===-- MatmulToHexKLPass.cpp - linalg.matmul to hexkl ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "matmul-to-hexkl"
#define DBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_MATMULTOHEXKL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToHexKL(MLIRContext *ctx, bool enableAttentionHmx, bool enableMPadHmx)
      : OpRewritePattern(ctx), enableAttentionHmx(enableAttentionHmx),
        enableMPadHmx(enableMPadHmx) {}

  bool enableAttentionHmx;
  bool enableMPadHmx;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
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
    // K and N must be tile-aligned.  M (rows / tokens) may be padded up to the
    // next tile multiple when enableMPadHmx: the extra A rows are independent
    // dot products whose padded output rows are discarded on unpad, so M-pad is
    // the safest pad direction (correctness does not depend on zero-fill, unlike
    // K/N pad).  Unaligned N (e.g. GPT-2 lm_head 50257) still faults on device
    // for large padded buffers, so those matmuls stay on HVX.
    if ((K % kHmxTile) != 0 || (N % kHmxTile) != 0) {
      DBG("skip HexKL: unaligned K/N in MxKxN=" << M << "x" << K << "x" << N);
      return rewriter.notifyMatchFailure(
          op, "K or N not divisible by HMX tile size 32");
    }
    const bool mUnaligned = (M % kHmxTile) != 0;
    if (mUnaligned && !enableMPadHmx) {
      DBG("skip HexKL: unaligned M=" << M << " (enableMPadHmx off)");
      return rewriter.notifyMatchFailure(
          op, "M not divisible by HMX tile size 32; enableMPadHmx off");
    }
    // M-pad allocates a fresh contiguous A/result buffer sized to the padded M.
    // At large N the resulting total function-frame pressure trips a Hexagon
    // frame-lowering stack-coloring defect (over-aligned dynamic frame clobbers
    // the sret spill slot -> Bad VA on device).  The threshold is total-frame,
    // not per-matmul, so N-tiling within one function does not help.  Until the
    // frame defect is root-caused, keep M-pad matmuls with N>1024 on HVX.
    constexpr int64_t kMaxMPadN = 1024;
    if (mUnaligned && enableMPadHmx && N > kMaxMPadN) {
      DBG("skip HexKL: M-pad with N=" << N << " > " << kMaxMPadN
                                      << " (frame defect); keep HVX");
      return rewriter.notifyMatchFailure(
          op, "M-pad with N>1024 trips Hexagon frame defect; keep HVX");
    }

    // Attention score / context matmuls (after ReduceContractionRank collapses
    // batch=1 batch_matmul) are tile-aligned at seq=32 but HMX on those shapes
    // faults on device (Bad VA / exit 13), e.g. QK^T 32x64x32 (N==M) and
    // AV 32x32x64 (K==M).  With enableAttentionHmx, pad to break K==M/N==M
    // then unpad the result (§2.5/§4.6).
    const bool attentionLike = (K == M || N == M);
    if (attentionLike && !enableAttentionHmx) {
      DBG("skip HexKL: attention-like MxKxN=" << M << "x" << K << "x" << N);
      return rewriter.notifyMatchFailure(
          op, "attention-like matmul (K==M or N==M); keep HVX");
    }

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];

    if (!attentionLike) {
      // Emit HexKL directly even when M is unaligned (gated above by
      // enableMPadHmx).  DecomposeHexKLMatmul pads M up to a tile multiple
      // internally with fresh contiguous buffers (mirroring its N-pad path),
      // which avoids the tensor.pad/extract_slice buffers that faulted on
      // device at large N.
      rewriter.replaceOpWithNewOp<hexkl::MatmulOp>(op, C.getType(), A, B, C);
      return success();
    }

    int64_t padK = K;
    int64_t padN = N;
    if (N == M)
      padN = N + kHmxTile;
    if (K == M)
      padK = K + kHmxTile;
    DBG("attention HMX pad MxKxN=" << M << "x" << K << "x" << N << " -> "
                                   << M << "x" << padK << "x" << padN);

    auto elemA = cast<MemRefType>(A.getType()).getElementType();
    auto elemB = cast<MemRefType>(B.getType()).getElementType();
    auto elemC = cast<MemRefType>(C.getType()).getElementType();
    auto aPadTy = MemRefType::get({M, padK}, elemA);
    auto bPadTy = MemRefType::get({padK, padN}, elemB);
    auto cPadTy = MemRefType::get({M, padN}, elemC);

    Value aPad = rewriter.create<memref::AllocOp>(loc, aPadTy);
    Value bPad = rewriter.create<memref::AllocOp>(loc, bPadTy);
    Value cPad = rewriter.create<memref::AllocOp>(loc, cPadTy);

    Value zeroA =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemA));
    Value zeroB =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemB));

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value vM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value vK = rewriter.create<arith::ConstantIndexOp>(loc, K);
    Value vN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value vPadK = rewriter.create<arith::ConstantIndexOp>(loc, padK);
    Value vPadN = rewriter.create<arith::ConstantIndexOp>(loc, padN);

    auto copy2d = [&](Value src, Value dst, Value rows, Value cols) {
      SmallVector<OpFoldResult> zeros = {rewriter.getIndexAttr(0),
                                         rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {rows, cols};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
      Value sv =
          rewriter.create<memref::SubViewOp>(loc, dst, zeros, sizes, strides);
      rewriter.create<memref::CopyOp>(loc, src, sv);
    };
    copy2d(A, aPad, vM, vK);
    copy2d(B, bPad, vK, vN);

    // Zero A padding columns [K, padK).
    if (padK != K) {
      rewriter.create<scf::ForOp>(
          loc, c0, vM, c1, ValueRange{},
          [&](OpBuilder &bb, Location loc, Value r, ValueRange) {
            bb.create<scf::ForOp>(
                loc, vK, vPadK, c1, ValueRange{},
                [&](OpBuilder &bbb, Location loc, Value c, ValueRange) {
                  bbb.create<memref::StoreOp>(loc, zeroA, aPad,
                                              ValueRange{r, c});
                  bbb.create<scf::YieldOp>(loc);
                });
            bb.create<scf::YieldOp>(loc);
          });
      // Zero B padding rows [K, padK) across padN.
      rewriter.create<scf::ForOp>(
          loc, vK, vPadK, c1, ValueRange{},
          [&](OpBuilder &bb, Location loc, Value r, ValueRange) {
            bb.create<scf::ForOp>(
                loc, c0, vPadN, c1, ValueRange{},
                [&](OpBuilder &bbb, Location loc, Value c, ValueRange) {
                  bbb.create<memref::StoreOp>(loc, zeroB, bPad,
                                              ValueRange{r, c});
                  bbb.create<scf::YieldOp>(loc);
                });
            bb.create<scf::YieldOp>(loc);
          });
    }
    // Zero B padding columns [N, padN) for rows [0, K).
    if (padN != N) {
      rewriter.create<scf::ForOp>(
          loc, c0, vK, c1, ValueRange{},
          [&](OpBuilder &bb, Location loc, Value r, ValueRange) {
            bb.create<scf::ForOp>(
                loc, vN, vPadN, c1, ValueRange{},
                [&](OpBuilder &bbb, Location loc, Value c, ValueRange) {
                  bbb.create<memref::StoreOp>(loc, zeroB, bPad,
                                              ValueRange{r, c});
                  bbb.create<scf::YieldOp>(loc);
                });
            bb.create<scf::YieldOp>(loc);
          });
    }

    rewriter.create<hexkl::MatmulOp>(loc, cPad.getType(), aPad, bPad, cPad);

    // Unpad: copy C[:, :N] back.
    SmallVector<OpFoldResult> zeros = {rewriter.getIndexAttr(0),
                                       rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> outSizes = {vM, vN};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
    Value cSv =
        rewriter.create<memref::SubViewOp>(loc, cPad, zeros, outSizes, strides);
    rewriter.create<memref::CopyOp>(loc, cSv, C);
    rewriter.create<memref::DeallocOp>(loc, aPad);
    rewriter.create<memref::DeallocOp>(loc, bPad);
    rewriter.create<memref::DeallocOp>(loc, cPad);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateMatmulToHexKLPatterns(RewritePatternSet &patterns,
                                   bool enableAttentionHmx, bool enableMPadHmx) {
  patterns.add<MatmulToHexKL>(patterns.getContext(), enableAttentionHmx,
                              enableMPadHmx);
}

struct MatmulToHexKLPass : public ::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  using MatmulToHexKLBase::MatmulToHexKLBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMatmulToHexKLPatterns(patterns, enableAttentionHmx, enableMPadHmx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createMatmulToHexKLPass(const MatmulToHexKLOptions &options) {
  return std::make_unique<MatmulToHexKLPass>(options);
}
