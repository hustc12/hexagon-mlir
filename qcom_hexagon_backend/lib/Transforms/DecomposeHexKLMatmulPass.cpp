//===-- DecomposeHexKLMatmulPass.cpp - Decompose hexkl.matmul to micro ops ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to decompose hexkl::MatmulOp into hexkl micro HMX operations.
// This pass implements the tiling strategy from hexkl_matmul_f16f16_f32.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Alps/IR/AlpsDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "decompose-hexkl-matmul"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DECOMPOSEHEXKLMATMUL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// V73 UserDMA 2-D ROI dimensions and strides are 16-bit descriptor fields.
// Keep this compiler-side predicate in lockstep with UserDMA::copy2D so an
// illegal async drain remains on the existing synchronous copy path.
static constexpr int64_t kUserDMA2DFieldMax = 0xFFFF;

static bool isAsyncDrainDMA2DLegal(int64_t outputColumns) {
  return outputColumns > 0 && outputColumns <= kUserDMA2DFieldMax / 2;
}

struct DecomposeHexKLMatmul final : public OpRewritePattern<hexkl::MatmulOp> {
  DecomposeHexKLMatmul(MLIRContext *ctx, bool enableWeightPrepack,
                       bool enableVtcmLifetimeColoring,
                       bool enableDmaToVtcm,
                       bool enableDirectOutputFormation,
                       bool enableF16BiasEpilogueFormation,
                       bool enableAsyncDrain, Value sharedVtcm)
      : OpRewritePattern(ctx), enableWeightPrepack(enableWeightPrepack),
        enableVtcmLifetimeColoring(enableVtcmLifetimeColoring),
        enableDmaToVtcm(enableDmaToVtcm),
        enableDirectOutputFormation(enableDirectOutputFormation),
        enableF16BiasEpilogueFormation(enableF16BiasEpilogueFormation),
        enableAsyncDrain(enableAsyncDrain),
        sharedVtcm(sharedVtcm) {}

  bool enableWeightPrepack;
  bool enableVtcmLifetimeColoring;
  bool enableDmaToVtcm;
  bool enableDirectOutputFormation;
  bool enableF16BiasEpilogueFormation;
  bool enableAsyncDrain;
  /// Non-null when the pass allocated a function-scoped VTCM arena.
  Value sharedVtcm;

  LogicalResult matchAndRewrite(hexkl::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result = op.getOuts();

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());
    auto resultType = cast<MemRefType>(result.getType());

    // P5l survives one-shot bufferization as an explicit consumer-contract
    // op. Allocation-liveness optimization may reuse the same memref for many
    // sequential matmul/epilogue pairs, so the global SSA user list does not
    // describe one physical value version. Match the first lexical operation
    // that touches this matmul's output version instead: only an immediate
    // bias epilogue is legal. A later matmul starts a new value version and its
    // users must not reject the current pair.
    hexkl::F16BiasEpilogueOp biasEpilogue;
    Value bias;
    if (enableF16BiasEpilogueFormation &&
        resultType.getElementType().isF16()) {
      for (Operation *cursor = op->getNextNode(); cursor;
           cursor = cursor->getNextNode()) {
        if (!llvm::is_contained(cursor->getOperands(), result))
          continue;
        if (isa<memref::DimOp>(cursor))
          continue;
        if (auto candidate = dyn_cast<hexkl::F16BiasEpilogueOp>(cursor);
            candidate && candidate.getSrc() == result)
          biasEpilogue = candidate;
        break;
      }
      if (biasEpilogue) {
        bias = biasEpilogue.getBias();
        Value final = biasEpilogue.getOuts();
        auto biasType = dyn_cast<MemRefType>(bias.getType());
        auto finalType = dyn_cast<MemRefType>(final.getType());
        if (!biasType || biasType.getRank() != 1 ||
            !biasType.getElementType().isF16() || !finalType ||
            finalType.getRank() != 2 || !finalType.getElementType().isF16())
          biasEpilogue = {};
        // Never move HMX computation across mutable memref operations merely
        // to make a late consumer operand dominate.  P5l formation creates
        // its final destination before the producer; malformed/external IR is
        // conservatively left unfused.
        auto isLateSameBlockDef = [&](Value value) {
          Operation *def = value.getDefiningOp();
          return def && def->getBlock() == op->getBlock() &&
                 op->isBeforeInBlock(def);
        };
        if (biasEpilogue &&
            (isLateSameBlockDef(bias) || isLateSameBlockDef(final)))
          biasEpilogue = {};
      }
    }

    // Validate rank (must be 2D)
    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    if (lhsShape.size() != 2 || rhsShape.size() != 2 ||
        resultShape.size() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matmul supported");
    }

    // Validate static shape compatibility if available
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
        resultType.hasStaticShape()) {
      int64_t M = lhsShape[0];
      int64_t K_lhs = lhsShape[1];
      int64_t K_rhs = rhsShape[0];
      int64_t N = rhsShape[1];
      int64_t M_out = resultShape[0];
      int64_t N_out = resultShape[1];

      // Validate dimensions match: lhs(M×K) × rhs(K×N) = result(M×N)
      if (K_lhs != K_rhs) {
        return rewriter.notifyMatchFailure(
            op, "inner dimensions mismatch: lhs K != rhs K");
      }
      if (M != M_out || N != N_out) {
        return rewriter.notifyMatchFailure(op, "output dimensions mismatch");
      }
    }

    // Create constants
    auto i32Ty = rewriter.getI32Type();

    Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value idx32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Value idx31 = rewriter.create<arith::ConstantIndexOp>(loc, 31);

    Value i32_0 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0);
    Value i32_1 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 1);
    Value i32_2 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 2);
    Value i32_2048 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 2048);
    Value i32_32 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 32);
    Value i32_4096 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 4096);
    Value idx4096 = rewriter.create<arith::ConstantIndexOp>(loc, 4096);

    // Get dimensions dynamically
    Value dimMOrig = rewriter.create<memref::DimOp>(loc, lhs, idx0);
    Value dimK = rewriter.create<memref::DimOp>(loc, lhs, idx1);
    Value dimNOrig = rewriter.create<memref::DimOp>(loc, rhs, idx1);

    // Pad M (rows/tokens) and/or N (columns) up to a multiple of 32 so
    // unaligned-token encoders (M, e.g. DINOv2's 257) and lm_head-class shapes
    // (N, e.g. 50257) run on HMX.  MatmulToHexKL only converts static shapes, so
    // padding is decided statically here and always materializes fresh
    // contiguous buffers: the micro-HMX lowering models each operand as a dense
    // row-major buffer whose row stride equals its logical dim, so a
    // subview/offset operand (e.g. a tensor.pad result) would fault at large N.
    bool doMPad = false, doNPad = false;
    int64_t staticMAligned = -1, staticNAligned = -1;
    if (lhsType.hasStaticShape()) {
      int64_t staticM = lhsShape[0];
      staticMAligned = (staticM + 31) / 32 * 32;
      doMPad = staticMAligned != staticM;
    }
    if (rhsType.hasStaticShape()) {
      int64_t staticN = rhsShape[1];
      staticNAligned = (staticN + 31) / 32 * 32;
      doNPad = staticNAligned != staticN;
    }

    Value dimM =
        doMPad ? rewriter.create<arith::ConstantIndexOp>(loc, staticMAligned)
               : dimMOrig;
    Value dimN =
        doNPad ? rewriter.create<arith::ConstantIndexOp>(loc, staticNAligned)
               : dimNOrig;

    Value lhsWork = lhs;
    Value rhsWork = rhs;
    Value resultWork = result;
    if (biasEpilogue)
      resultWork = biasEpilogue.getOuts();
    Value lhsPadAlloc, rhsPadAlloc, resultPadAlloc;

    SmallVector<OpFoldResult> padZeros = {rewriter.getIndexAttr(0),
                                          rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> padStrides = {rewriter.getIndexAttr(1),
                                            rewriter.getIndexAttr(1)};

    if (doMPad) {
      // Fresh [padM, K] activation buffer: copy valid rows, zero the pad rows.
      // The padded output rows are discarded on copy-back.
      auto lhsPadTy = MemRefType::get(
          ArrayRef<int64_t>{staticMAligned, lhsShape[1]},
          lhsType.getElementType(), MemRefLayoutAttrInterface{},
          lhsType.getMemorySpace());
      lhsPadAlloc = rewriter.create<memref::AllocOp>(loc, lhsPadTy);
      Value zeroA = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(lhsType.getElementType()));
      SmallVector<OpFoldResult> lhsSizes = {dimMOrig, dimK};
      Value lhsSv = rewriter.create<memref::SubViewOp>(loc, lhsPadAlloc, padZeros,
                                                       lhsSizes, padStrides);
      rewriter.create<memref::CopyOp>(loc, lhs, lhsSv);
      rewriter.create<scf::ForOp>(
          loc, dimMOrig, dimM, idx1, ValueRange{},
          [&](OpBuilder &bb, Location loc, Value r, ValueRange) {
            bb.create<scf::ForOp>(
                loc, idx0, dimK, idx1, ValueRange{},
                [&](OpBuilder &bbb, Location loc, Value c, ValueRange) {
                  bbb.create<memref::StoreOp>(loc, zeroA, lhsPadAlloc,
                                              ValueRange{r, c});
                  bbb.create<scf::YieldOp>(loc);
                });
            bb.create<scf::YieldOp>(loc);
          });
      lhsWork = lhsPadAlloc;
    }

    if (doNPad) {
      auto rhsPadTy = MemRefType::get(
          ArrayRef<int64_t>{rhsShape[0], staticNAligned},
          rhsType.getElementType(), MemRefLayoutAttrInterface{},
          rhsType.getMemorySpace());
      rhsPadAlloc = rewriter.create<memref::AllocOp>(loc, rhsPadTy);
      Value zeroW = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(rhsType.getElementType()));
      // Copy valid K×NOrig weights, then zero only the padding columns.
      SmallVector<OpFoldResult> rhsSizes = {dimK, dimNOrig};
      Value rhsSv = rewriter.create<memref::SubViewOp>(loc, rhsPadAlloc, padZeros,
                                                       rhsSizes, padStrides);
      rewriter.create<memref::CopyOp>(loc, rhs, rhsSv);
      rewriter.create<scf::ForOp>(
          loc, idx0, dimK, idx1, ValueRange{},
          [&](OpBuilder &bb, Location loc, Value r, ValueRange) {
            bb.create<scf::ForOp>(
                loc, dimNOrig, dimN, idx1, ValueRange{},
                [&](OpBuilder &bbb, Location loc, Value c, ValueRange) {
                  bbb.create<memref::StoreOp>(loc, zeroW, rhsPadAlloc,
                                              ValueRange{r, c});
                  bbb.create<scf::YieldOp>(loc);
                });
            bb.create<scf::YieldOp>(loc);
          });
      rhsWork = rhsPadAlloc;
    }

    bool formDirectOutput = biasEpilogue ||
                            (enableDirectOutputFormation && (doMPad || doNPad));
    bool useAsyncDrain =
        enableAsyncDrain && !enableWeightPrepack && !biasEpilogue &&
        resultType.getElementType().isF16() && lhsType.hasStaticShape() &&
        rhsType.hasStaticShape() && resultType.hasStaticShape() &&
        resultShape[0] >= 32 &&
        isAsyncDrainDMA2DLegal(resultShape[1]) &&
        (((resultShape[0] + 31) / 32) * ((resultShape[1] + 31) / 32) >= 2) &&
        (!(doMPad || doNPad) || formDirectOutput);
    if ((doMPad || doNPad) && !formDirectOutput) {
      // Result buffer spans the padded extents; the valid [Morig×Norig] region
      // is copied back after compute.
      int64_t rM = doMPad ? staticMAligned : resultShape[0];
      int64_t rN = doNPad ? staticNAligned : resultShape[1];
      auto resultPadTy = MemRefType::get(
          ArrayRef<int64_t>{rM, rN}, resultType.getElementType(),
          MemRefLayoutAttrInterface{}, resultType.getMemorySpace());
      resultPadAlloc = rewriter.create<memref::AllocOp>(loc, resultPadTy);
      resultWork = resultPadAlloc;
    }

    Value M = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimM);
    Value K = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimK);
    Value N = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimN);
    Value outputRows =
        formDirectOutput
            ? rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimMOrig)
            : M;
    Value outputCols =
        formDirectOutput
            ? rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimNOrig)
            : N;

    // Calculate numKTiles = (k + 31) / 32
    Value kPlus31 = rewriter.create<arith::AddIOp>(loc, dimK, idx31);
    Value kTiles = rewriter.create<arith::DivUIOp>(loc, kPlus31, idx32);
    Value kTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, kTiles);
    Value nPlus31 = rewriter.create<arith::AddIOp>(loc, dimN, idx31);
    Value nTiles = rewriter.create<arith::DivUIOp>(loc, nPlus31, idx32);
    Value nTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, nTiles);

    // VTCM layout (legacy HexKL reuses scratch as weight ping-pong after act load):
    //   default:  act[0..kTiles) | scratch/w0/w1/flat/acc starting at kTiles
    //             (= (2*kTiles+7)*4096 budget)
    //   prepack:  act[0..kTiles) | scratch[kTiles..2*kTiles) | wh[2*kTiles..3*kTiles)
    //             | flat | acc | extra
    //             WH for one column stays in VTCM; RmToWh once per (col,kt).
    Value twoKTiles = rewriter.create<arith::MulIOp>(
        loc, kTiles, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value vtcmTiles;
    if (enableVtcmLifetimeColoring && enableWeightPrepack) {
      // [0,K): AH, [K,2K): persistent WH. One scratch tile at 2K is
      // sufficient because CopySubmatrix -> RmToAh consumes it immediately.
      vtcmTiles = rewriter.create<arith::AddIOp>(
          loc, twoKTiles, rewriter.create<arith::ConstantIndexOp>(loc, 1));
    } else if (enableVtcmLifetimeColoring) {
      // [0,K): AH. Scratch, w0 and w1 share the post-AH phase colors K/K+1.
      // Output flat/acc reuse colors 0/1 after the final HMX consume.
      Value extraTiles = rewriter.create<arith::ConstantIndexOp>(
          loc, enableDmaToVtcm ? 3 : 2);
      vtcmTiles = rewriter.create<arith::AddIOp>(loc, kTiles, extraTiles);
    } else if (enableWeightPrepack) {
      Value threeK = rewriter.create<arith::AddIOp>(loc, twoKTiles, kTiles);
      vtcmTiles = rewriter.create<arith::AddIOp>(
          loc, threeK, rewriter.create<arith::ConstantIndexOp>(loc, 5));
    } else {
      Value dataTiles = rewriter.create<arith::AddIOp>(
          loc, twoKTiles, rewriter.create<arith::ConstantIndexOp>(loc, 4));
      vtcmTiles = rewriter.create<arith::AddIOp>(
          loc, dataTiles, rewriter.create<arith::ConstantIndexOp>(loc, 3));
    }
    Value asyncDrainBase;
    if (useAsyncDrain) {
      Value vtcmTilesI32ForDrain =
          rewriter.create<arith::IndexCastOp>(loc, i32Ty, vtcmTiles);
      asyncDrainBase = rewriter.create<arith::MulIOp>(
          loc, vtcmTilesI32ForDrain, i32_4096);
      // One tile holds the two 2048-byte slots. Keep a second tile after it so
      // the HexKL config block (defined relative to slab end) cannot overlap
      // the slots after the descriptor size grows.
      vtcmTiles = rewriter.create<arith::AddIOp>(
          loc, vtcmTiles, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    }
    Value vtcmBytes = rewriter.create<arith::MulIOp>(loc, vtcmTiles, idx4096);

    Value vtcm;
    bool ownsVtcm = false;
    if (sharedVtcm) {
      vtcm = sharedVtcm;
    } else {
      auto vtcmType =
          MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                          MemRefLayoutAttrInterface{},
                          IntegerAttr::get(rewriter.getI32Type(), 1));
      auto vtcmAlloc =
          rewriter.create<hexagonmem::AllocOp>(loc, vtcmType, vtcmBytes);
      vtcmAlloc->setAttr("bufferization.manual_deallocation",
                         rewriter.getUnitAttr());
      vtcm = vtcmAlloc.getResult();
      ownsVtcm = true;
    }

    rewriter.create<hexkl::MicroHMXSetupAccReadF16Op>(loc, vtcm);

    // Default HexKL: weights reuse scratch base (kTiles*4096). Prepack: WH after
    // full scratch bank (2*kTiles*4096) so act reload does not clobber WH.
    Value wOff0 = rewriter.create<arith::MulIOp>(loc, kTilesI32, i32_4096);
    Value wOff1 = rewriter.create<arith::AddIOp>(loc, wOff0, i32_4096);
    Value wRegionBase = enableWeightPrepack && !enableVtcmLifetimeColoring
                            ? rewriter.create<arith::MulIOp>(
                                  loc,
                                  rewriter.create<arith::AddIOp>(loc, kTilesI32,
                                                                 kTilesI32),
                                  i32_4096)
                            : wOff0;
    Value flatOff;
    Value accOff;
    if (enableVtcmLifetimeColoring) {
      // In the default M-outer schedule AH remains live across every N tile,
      // while each column's WH ping-pong slots die before its readback. Reuse
      // WH colors there. In the prepack N-outer schedule WH remains live
      // across M rows, while AH dies before each row's readback; reuse AH.
      flatOff = enableWeightPrepack ? i32_0 : wOff0;
      accOff = enableWeightPrepack ? i32_4096 : wOff1;
    } else if (enableWeightPrepack) {
      Value threeKI32 = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::AddIOp>(loc, kTilesI32, kTilesI32),
          kTilesI32);
      flatOff = rewriter.create<arith::MulIOp>(loc, threeKI32, i32_4096);
      accOff = rewriter.create<arith::AddIOp>(loc, flatOff, i32_4096);
    } else {
      Value kTilesPlus2 = rewriter.create<arith::AddIOp>(
          loc, kTilesI32, rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 2));
      flatOff = rewriter.create<arith::MulIOp>(loc, kTilesPlus2, i32_4096);
      Value kTilesPlus3 = rewriter.create<arith::AddIOp>(
          loc, kTilesI32, rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 3));
      accOff = rewriter.create<arith::MulIOp>(loc, kTilesPlus3, i32_4096);
    }

    scf::ForOp outerFor;
    if (enableWeightPrepack) {
      // Column-outer: pack all K WH tiles for this column into VTCM cache,
      // then stream all M rows against the cached weights.
      outerFor = rewriter.create<scf::ForOp>(
          loc, idx0, dimN, idx32, ValueRange{},
          [&](OpBuilder &b, Location loc, Value col, ValueRange) {
            Value colI32 = b.create<arith::IndexCastOp>(loc, i32Ty, col);
            Value colTile = b.create<arith::DivUIOp>(loc, colI32, i32_32);

            b.create<scf::ForOp>(
                loc, idx0, kTiles, idx1, ValueRange{},
                [&](OpBuilder &bb, Location loc, Value ktIdx, ValueRange) {
                  Value kt = bb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);
                  Value wOff = bb.create<arith::AddIOp>(
                      loc, wRegionBase,
                      bb.create<arith::MulIOp>(loc, kt, i32_4096));
                  bb.create<hexkl::MicroHMXRmToWhF16Op>(loc, vtcm, wOff,
                                                        rhsWork, kt, colTile, N);
                  bb.create<scf::YieldOp>(loc);
                });

            b.create<scf::ForOp>(
                loc, idx0, dimM, idx32, ValueRange{},
                [&](OpBuilder &bb, Location loc, Value row, ValueRange) {
                  Value rowI32 =
                      bb.create<arith::IndexCastOp>(loc, i32Ty, row);
                  Value rowTile =
                      bb.create<arith::DivUIOp>(loc, rowI32, i32_32);

                  bb.create<scf::ForOp>(
                      loc, idx0, kTiles, idx1, ValueRange{},
                      [&](OpBuilder &bbb, Location loc, Value ktIdx,
                          ValueRange) {
                        Value kt =
                            bbb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);
                        Value scrIdx = enableVtcmLifetimeColoring
                            ? bbb.create<arith::AddIOp>(
                                  loc, kTilesI32, kTilesI32)
                            : bbb.create<arith::AddIOp>(
                                  loc, kTilesI32, kt);
                        Value scrOff =
                            bbb.create<arith::MulIOp>(loc, scrIdx, i32_4096);
                        bbb.create<hexkl::MicroHMXCopySubmatrixToF16Op>(
                            loc, vtcm, scrOff, lhsWork, rowTile, kt, M, K);
                        Value actOff =
                            bbb.create<arith::MulIOp>(loc, kt, i32_4096);
                        bbb.create<hexkl::MicroHMXRmToAhF16Op>(loc, vtcm,
                                                               actOff, scrOff);
                        bbb.create<scf::YieldOp>(loc);
                      });

                  bb.create<hexkl::MicroHMXAccClearF16Op>(loc);
                  bb.create<scf::ForOp>(
                      loc, idx0, kTiles, idx1, ValueRange{},
                      [&](OpBuilder &bbb, Location loc, Value ktIdx,
                          ValueRange) {
                        Value kt =
                            bbb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);
                        Value wOff = bbb.create<arith::AddIOp>(
                            loc, wRegionBase,
                            bbb.create<arith::MulIOp>(loc, kt, i32_4096));
                        Value actOff =
                            bbb.create<arith::MulIOp>(loc, kt, i32_4096);
                        bbb.create<hexkl::MicroHMXMmF16Op>(loc, vtcm, actOff,
                                                           wOff);
                        bbb.create<scf::YieldOp>(loc);
                      });

                  bb.create<hexkl::MicroHMXAccReadF16Op>(loc, vtcm, accOff);
                  if (useAsyncDrain) {
                    Value rowEnd = bb.create<arith::AddIOp>(loc, row, idx32);
                    Value fullRow = bb.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::ule, rowEnd, dimMOrig);
                    Value shortRow = bb.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::ugt, rowEnd, dimMOrig);
                    Value linearTile = bb.create<arith::AddIOp>(
                        loc,
                        bb.create<arith::MulIOp>(loc, rowTile, nTilesI32),
                        colTile);
                    Value slot =
                        bb.create<arith::RemUIOp>(loc, linearTile, i32_2);
                    Value slotByteOffset = bb.create<arith::MulIOp>(
                        loc, slot, i32_2048);
                    Value asyncFlatOff = bb.create<arith::AddIOp>(
                        loc, asyncDrainBase, slotByteOffset);
                    bb.create<scf::IfOp>(
                        loc, fullRow,
                        [&](OpBuilder &thenBuilder, Location thenLoc) {
                          thenBuilder
                              .create<hexkl::MicroHMXAsyncDrainWaitSlotOp>(
                                  thenLoc, slot);
                          thenBuilder.create<hexkl::MicroHMXAhToRmF16Op>(
                              thenLoc, vtcm, asyncFlatOff, accOff);
                          thenBuilder
                              .create<hexkl::MicroHMXAsyncDrainStartF16Op>(
                                  thenLoc, vtcm, asyncFlatOff, resultWork,
                                  rowTile, colTile, outputRows, outputCols,
                                  slot);
                          thenBuilder.create<scf::YieldOp>(thenLoc);
                        });
                    bb.create<scf::IfOp>(
                        loc, shortRow,
                        [&](OpBuilder &thenBuilder, Location thenLoc) {
                          thenBuilder.create<hexkl::MicroHMXAhToRmF16Op>(
                              thenLoc, vtcm, flatOff, accOff);
                          thenBuilder
                              .create<hexkl::MicroHMXCopyF16ToSubmatrixOp>(
                                  thenLoc, vtcm, flatOff, resultWork, rowTile,
                                  colTile, outputRows, outputCols);
                          thenBuilder.create<scf::YieldOp>(thenLoc);
                        });
                  } else if (biasEpilogue) {
                    bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                          accOff);
                    bb.create<hexkl::MicroHMXCopyF16BiasToSubmatrixOp>(
                        loc, vtcm, flatOff, bias, resultWork, rowTile, colTile,
                        outputRows, outputCols);
                  } else if (resultType.getElementType().isF16()) {
                    bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                          accOff);
                    bb.create<hexkl::MicroHMXCopyF16ToSubmatrixOp>(
                        loc, vtcm, flatOff, resultWork, rowTile, colTile,
                        outputRows, outputCols);
                  } else {
                    bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                          accOff);
                    bb.create<hexkl::MicroHMXCopyF16ToF32SubmatrixOp>(
                        loc, vtcm, flatOff, resultWork, rowTile, colTile,
                        outputRows, outputCols);
                  }
                  bb.create<scf::YieldOp>(loc);
                });
            b.create<scf::YieldOp>(loc);
          });
    } else {
      // Default: M-outer with dual ping-pong weight slots for Alps.
      (void)wOff1;
      outerFor = rewriter.create<scf::ForOp>(
          loc, idx0, dimM, idx32, ValueRange{},
          [&](OpBuilder &b, Location loc, Value row, ValueRange) {
            Value rowI32 = b.create<arith::IndexCastOp>(loc, i32Ty, row);
            Value rowTile = b.create<arith::DivUIOp>(loc, rowI32, i32_32);

            b.create<scf::ForOp>(
                loc, idx0, kTiles, idx1, ValueRange{},
                [&](OpBuilder &bb, Location loc, Value ktIdx, ValueRange) {
                  Value kt = bb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);
                  Value scrIdx = enableVtcmLifetimeColoring
                      ? kTilesI32
                      : bb.create<arith::AddIOp>(loc, kTilesI32, kt);
                  Value scrOff =
                      bb.create<arith::MulIOp>(loc, scrIdx, i32_4096);
                  bb.create<hexkl::MicroHMXCopySubmatrixToF16Op>(
                      loc, vtcm, scrOff, lhsWork, rowTile, kt, M, K);
                  Value actOff = bb.create<arith::MulIOp>(loc, kt, i32_4096);
                  bb.create<hexkl::MicroHMXRmToAhF16Op>(loc, vtcm, actOff,
                                                        scrOff);
                  bb.create<scf::YieldOp>(loc);
                });

            b.create<scf::ForOp>(
                loc, idx0, dimN, idx32, ValueRange{},
                [&](OpBuilder &bb, Location loc, Value col, ValueRange) {
                  Value colI32 =
                      bb.create<arith::IndexCastOp>(loc, i32Ty, col);
                  Value colTile =
                      bb.create<arith::DivUIOp>(loc, colI32, i32_32);
                  bb.create<hexkl::MicroHMXAccClearF16Op>(loc);
                  bb.create<scf::ForOp>(
                      loc, idx0, kTiles, idx1, ValueRange{},
                      [&](OpBuilder &bbb, Location loc, Value ktIdx,
                          ValueRange) {
                        Value kt =
                            bbb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);
                        Value phase =
                            bbb.create<arith::RemUIOp>(loc, kt, i32_2);
                        Value isOdd = bbb.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ne, phase, i32_0);
                        Value curW = bbb.create<arith::SelectOp>(
                            loc, isOdd, wOff1, wOff0);
                        bbb.create<hexkl::MicroHMXRmToWhF16Op>(
                            loc, vtcm, curW, rhsWork, kt, colTile, N);
                        Value actOff2 =
                            bbb.create<arith::MulIOp>(loc, kt, i32_4096);
                        bbb.create<hexkl::MicroHMXMmF16Op>(loc, vtcm, actOff2,
                                                           curW);
                        bbb.create<scf::YieldOp>(loc);
                      });
                  bb.create<hexkl::MicroHMXAccReadF16Op>(loc, vtcm, accOff);
                  if (useAsyncDrain) {
                    Value rowEnd = bb.create<arith::AddIOp>(loc, row, idx32);
                    Value fullRow = bb.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::ule, rowEnd, dimMOrig);
                    Value shortRow = bb.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::ugt, rowEnd, dimMOrig);
                    Value linearTile = bb.create<arith::AddIOp>(
                        loc,
                        bb.create<arith::MulIOp>(loc, rowTile, nTilesI32),
                        colTile);
                    Value slot =
                        bb.create<arith::RemUIOp>(loc, linearTile, i32_2);
                    Value slotByteOffset = bb.create<arith::MulIOp>(
                        loc, slot, i32_2048);
                    Value asyncFlatOff = bb.create<arith::AddIOp>(
                        loc, asyncDrainBase, slotByteOffset);
                    bb.create<scf::IfOp>(
                        loc, fullRow,
                        [&](OpBuilder &thenBuilder, Location thenLoc) {
                          thenBuilder
                              .create<hexkl::MicroHMXAsyncDrainWaitSlotOp>(
                                  thenLoc, slot);
                          thenBuilder.create<hexkl::MicroHMXAhToRmF16Op>(
                              thenLoc, vtcm, asyncFlatOff, accOff);
                          thenBuilder
                              .create<hexkl::MicroHMXAsyncDrainStartF16Op>(
                                  thenLoc, vtcm, asyncFlatOff, resultWork,
                                  rowTile, colTile, outputRows, outputCols,
                                  slot);
                          thenBuilder.create<scf::YieldOp>(thenLoc);
                        });
                    bb.create<scf::IfOp>(
                        loc, shortRow,
                        [&](OpBuilder &thenBuilder, Location thenLoc) {
                          thenBuilder.create<hexkl::MicroHMXAhToRmF16Op>(
                              thenLoc, vtcm, flatOff, accOff);
                          thenBuilder
                              .create<hexkl::MicroHMXCopyF16ToSubmatrixOp>(
                                  thenLoc, vtcm, flatOff, resultWork, rowTile,
                                  colTile, outputRows, outputCols);
                          thenBuilder.create<scf::YieldOp>(thenLoc);
                        });
                  } else {
                    bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                          accOff);
                  if (biasEpilogue)
                    bb.create<hexkl::MicroHMXCopyF16BiasToSubmatrixOp>(
                        loc, vtcm, flatOff, bias, resultWork, rowTile, colTile,
                        outputRows, outputCols);
                  else if (resultType.getElementType().isF16())
                    bb.create<hexkl::MicroHMXCopyF16ToSubmatrixOp>(
                        loc, vtcm, flatOff, resultWork, rowTile, colTile,
                        outputRows, outputCols);
                  else
                    bb.create<hexkl::MicroHMXCopyF16ToF32SubmatrixOp>(
                        loc, vtcm, flatOff, resultWork, rowTile, colTile,
                        outputRows, outputCols);
                  }
                  bb.create<scf::YieldOp>(loc);
                });
            b.create<scf::YieldOp>(loc);
          });
    }

    // Explicitly deallocate VTCM buffer to avoid relying on ConvertToHexagonmem
    // rewriting of generic memref.dealloc for dynamic VTCM types.
    rewriter.setInsertionPointAfter(outerFor);
    if (useAsyncDrain)
      rewriter.create<hexkl::MicroHMXAsyncDrainFlushOp>(loc);
    if ((doMPad || doNPad) && !formDirectOutput) {
      // Copy the valid [Morig×Norig] region back into the caller's result and
      // release the padded scratch buffers.
      SmallVector<OpFoldResult> zeros = {rewriter.getIndexAttr(0),
                                         rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> outSizes = {dimMOrig, dimNOrig};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
      Value outSv = rewriter.create<memref::SubViewOp>(
          loc, resultPadAlloc, zeros, outSizes, strides);
      rewriter.create<memref::CopyOp>(loc, outSv, result);
      if (doMPad)
        rewriter.create<memref::DeallocOp>(loc, lhsPadAlloc);
      if (doNPad)
        rewriter.create<memref::DeallocOp>(loc, rhsPadAlloc);
      rewriter.create<memref::DeallocOp>(loc, resultPadAlloc);
    } else if (formDirectOutput) {
      if (doMPad)
        rewriter.create<memref::DeallocOp>(loc, lhsPadAlloc);
      if (doNPad)
        rewriter.create<memref::DeallocOp>(loc, rhsPadAlloc);
    }
    if (ownsVtcm)
      rewriter.create<hexagonmem::DeallocOp>(loc, vtcm);

    if (biasEpilogue)
      rewriter.eraseOp(biasEpilogue);
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower P5l contracts that cannot be safely fused at the producer point back
// to their ordinary elementwise semantics.  This is the conservative
// admission fallback: a late/reused destination never causes HMX computation
// to move across mutable-buffer lifetimes merely to obtain a fused drain.
struct DecomposeF16BiasEpilogue final
    : public OpRewritePattern<hexkl::F16BiasEpilogueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hexkl::F16BiasEpilogueOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSrc().getType());
    auto biasType = dyn_cast<MemRefType>(op.getBias().getType());
    auto dstType = dyn_cast<MemRefType>(op.getOuts().getType());
    if (!srcType || !biasType || !dstType || srcType.getRank() != 2 ||
        biasType.getRank() != 1 || dstType.getRank() != 2 ||
        !srcType.getElementType().isF16() ||
        !biasType.getElementType().isF16() ||
        !dstType.getElementType().isF16())
      return rewriter.notifyMatchFailure(op, "expected rank-2/rank-1 F16 memrefs");

    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value rows = rewriter.create<memref::DimOp>(loc, op.getSrc(), zero);
    Value cols = rewriter.create<memref::DimOp>(loc, op.getSrc(), one);
    rewriter.create<scf::ForOp>(
        loc, zero, rows, one, ValueRange{},
        [&](OpBuilder &rowBuilder, Location rowLoc, Value row, ValueRange) {
          rowBuilder.create<scf::ForOp>(
              rowLoc, zero, cols, one, ValueRange{},
              [&](OpBuilder &colBuilder, Location colLoc, Value col,
                  ValueRange) {
                Value value = colBuilder.create<memref::LoadOp>(
                    colLoc, op.getSrc(), ValueRange{row, col});
                Value bias = colBuilder.create<memref::LoadOp>(
                    colLoc, op.getBias(), ValueRange{col});
                Value sum = colBuilder.create<arith::AddFOp>(colLoc, value, bias);
                colBuilder.create<memref::StoreOp>(
                    colLoc, sum, op.getOuts(), ValueRange{row, col});
                colBuilder.create<scf::YieldOp>(colLoc);
              });
          rowBuilder.create<scf::YieldOp>(rowLoc);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

struct VtcmLiveRegion {
  int64_t tiles;
  int beginPhase;
  int endPhase;
  int64_t color = -1;
};

/// First-fit interval coloring with contiguous tile ranges. Two regions
/// interfere only when both their phase intervals and assigned tile ranges
/// overlap. Phases are: 0=activation/prepack, 1=HMX compute preparation,
/// 2=HMX consume, 3=accumulator readback, 4=result copy.
static int64_t computeColoredVtcmTiles(int64_t kTiles, bool weightPrepack,
                                       bool dmaToVtcm) {
  SmallVector<VtcmLiveRegion> regions;
  if (weightPrepack) {
    regions.push_back({kTiles, 0, 5}); // WH retained across all M rows
    regions.push_back({kTiles, 1, 3}); // AH bank
    regions.push_back({1, 1, 2});      // one reusable RM scratch tile
  } else {
    regions.push_back({kTiles, 0, 5}); // AH retained across all N columns
    regions.push_back({1, 0, 1});      // RM scratch, dead before WH
    regions.push_back({2, 1, 3});      // WH ping-pong slots
    if (dmaToVtcm)
      regions.push_back({1, 1, 2});    // non-aliasing async DMA stage
  }
  regions.push_back({1, 3, 4}); // accumulator read tile
  regions.push_back({1, 3, 5}); // flat result tile; overlaps accumulator

  int64_t peak = 0;
  for (VtcmLiveRegion &region : regions) {
    for (int64_t candidate = 0;; ++candidate) {
      bool conflicts = false;
      for (const VtcmLiveRegion &placed : regions) {
        if (placed.color < 0)
          continue;
        bool lifetimeOverlap =
            region.beginPhase < placed.endPhase &&
            placed.beginPhase < region.endPhase;
        bool addressOverlap =
            candidate < placed.color + placed.tiles &&
            placed.color < candidate + region.tiles;
        if (lifetimeOverlap && addressOverlap) {
          conflicts = true;
          break;
        }
      }
      if (!conflicts) {
        region.color = candidate;
        peak = std::max(peak, candidate + region.tiles);
        break;
      }
    }
  }
  return peak;
}

/// VTCM bytes for a HexKL matmul with static K. The legacy layout reserves
/// (2*kTiles+7) tiles by default or (3*kTiles+5) for prepack. Item 6 uses the
/// interference graph above to compute the compact peak.
static std::optional<int64_t>
estimateVtcmBytes(hexkl::MatmulOp op, bool enableWeightPrepack,
                  bool enableVtcmLifetimeColoring, bool enableDmaToVtcm,
                  bool enableAsyncDrain) {
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  if (!lhsType || !lhsType.hasStaticShape() || lhsType.getRank() != 2)
    return std::nullopt;
  int64_t K = lhsType.getShape()[1];
  if (K <= 0)
    return std::nullopt;
  int64_t kTiles = (K + 31) / 32;
  if (enableVtcmLifetimeColoring)
    return (computeColoredVtcmTiles(kTiles, enableWeightPrepack,
                                    enableDmaToVtcm) +
            (enableAsyncDrain ? 2 : 0)) *
           4096;
  int64_t defBytes = (kTiles * 2 + 4 + 3) * 4096;
  int64_t prepackBytes = (kTiles * 3 + 5) * 4096;
  return (enableWeightPrepack ? prepackBytes : defBytes) +
         (enableAsyncDrain ? 8192 : 0);
}

/// P5m is deliberately analysis-only.  A future implementation may replace
/// the synchronous HMX result copy with ping-pong 2D UserDMA, but only for the
/// portion whose descriptor cost can overlap a following HMX tile.  In
/// particular, a DINO-style M=257 tail is kept synchronous: issuing DMA for a
/// single 64-byte row would turn prefetch into pure launch overhead.
struct AsyncDrainLedger {
  int64_t sites = 0;
  int64_t admittedSites = 0;
  int64_t rejectedSites = 0;
  int64_t drainBytes = 0;
  int64_t descriptors = 0;
  int64_t fullTiles = 0;
  int64_t boundaryTiles = 0;
  int64_t admittedDescriptors = 0;
  int64_t admittedBytes = 0;
  int64_t boundaryBytes = 0;
  int64_t overlapDescriptors = 0;
  int64_t overlapBytes = 0;
  int64_t overlapHmxCalls = 0;
  int64_t maxDestinationStrideBytes = 0;
  bool dma2dLegal = true;
};

static void analyzeAsyncDrain(hexkl::MatmulOp op,
                              AsyncDrainLedger &ledger) {
  ++ledger.sites;
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  auto outType = dyn_cast<MemRefType>(op.getOuts().getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape() || !lhsType.getElementType().isF16() ||
      !rhsType.getElementType().isF16() ||
      !outType.getElementType().isF16()) {
    ++ledger.rejectedSites;
    return;
  }

  int64_t M = outType.getShape()[0];
  int64_t N = outType.getShape()[1];
  int64_t K = lhsType.getShape()[1];
  if (M <= 0 || N <= 0 || K <= 0 || lhsType.getShape()[0] != M ||
      rhsType.getShape()[0] != K || rhsType.getShape()[1] != N) {
    ++ledger.rejectedSites;
    ledger.dma2dLegal = false;
    return;
  }

  int64_t mTiles = (M + 31) / 32;
  int64_t nTiles = (N + 31) / 32;
  int64_t kTiles = (K + 31) / 32;
  int64_t siteDescriptors = mTiles * nTiles;
  int64_t siteFullTiles = (M / 32) * (N / 32);
  int64_t siteDrainBytes = M * N * 2;

  ledger.drainBytes += siteDrainBytes;
  ledger.descriptors += siteDescriptors;
  ledger.fullTiles += siteFullTiles;
  ledger.boundaryTiles += siteDescriptors - siteFullTiles;
  ledger.maxDestinationStrideBytes =
      std::max(ledger.maxDestinationStrideBytes, N * 2);

  // Each drain descriptor uses width <= 64 bytes, height = 32,
  // src_stride = 64, and dst_stride = N * sizeof(f16).  Only the destination
  // stride can exceed the 16-bit V73 descriptor field for these fixed tiles.
  // Record the potential traffic, but admit none of an illegal site.
  if (!isAsyncDrainDMA2DLegal(N)) {
    ++ledger.rejectedSites;
    ledger.dma2dLegal = false;
    ledger.boundaryBytes += siteDrainBytes;
    return;
  }

  // All complete 32-row bands are admitted.  A final short N tile remains a
  // legal 2D transfer because width shrinks while both strides stay fixed.
  // Short M tails stay synchronous to avoid tiny-height DMA descriptors.
  int64_t fullRowBands = M / 32;
  int64_t siteAdmittedDescriptors = fullRowBands * nTiles;
  int64_t siteAdmittedBytes = fullRowBands * 32 * N * 2;
  ledger.admittedDescriptors += siteAdmittedDescriptors;
  ledger.admittedBytes += siteAdmittedBytes;
  ledger.boundaryBytes += siteDrainBytes - siteAdmittedBytes;

  // The final admitted descriptor has no independently-proven following HMX
  // compute in this matmul, so it is never counted as hidden work.
  if (siteAdmittedDescriptors >= 2) {
    int64_t lastWidth = (N % 32 == 0) ? 32 : N % 32;
    int64_t lastDescriptorBytes = 32 * lastWidth * 2;
    ledger.overlapDescriptors += siteAdmittedDescriptors - 1;
    ledger.overlapBytes += siteAdmittedBytes - lastDescriptorBytes;
    ledger.overlapHmxCalls += (siteAdmittedDescriptors - 1) * kTiles;
    ++ledger.admittedSites;
  } else {
    ++ledger.rejectedSites;
  }
}

void populateDecomposeHexKLMatmulPatterns(RewritePatternSet &patterns,
                                          bool enableWeightPrepack,
                                          bool enableVtcmLifetimeColoring,
                                          bool enableDmaToVtcm,
                                          bool enableDirectOutputFormation,
                                          bool enableF16BiasEpilogueFormation,
                                          bool enableAsyncDrain,
                                          Value sharedVtcm) {
  patterns.add<DecomposeHexKLMatmul>(
      patterns.getContext(), enableWeightPrepack,
      enableVtcmLifetimeColoring, enableDmaToVtcm,
      enableDirectOutputFormation, enableF16BiasEpilogueFormation,
      enableAsyncDrain, sharedVtcm);
}

struct DecomposeHexKLMatmulPass
    : public ::impl::DecomposeHexKLMatmulBase<DecomposeHexKLMatmulPass> {
  using DecomposeHexKLMatmulBase::DecomposeHexKLMatmulBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, hexagonmem::HexagonMemDialect,
                alps::AlpsDialect, arith::ArithDialect,
                scf::SCFDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    Value sharedVtcm;

    if (enableAsyncDrainAnalysis) {
      AsyncDrainLedger ledger;
      func.walk([&](hexkl::MatmulOp op) { analyzeAsyncDrain(op, ledger); });
      Builder b(func.getContext());
      func->setAttr("alps.p5m.sites", b.getI64IntegerAttr(ledger.sites));
      func->setAttr("alps.p5m.admitted_sites",
                    b.getI64IntegerAttr(ledger.admittedSites));
      func->setAttr("alps.p5m.drain_bytes",
                    b.getI64IntegerAttr(ledger.drainBytes));
      func->setAttr("alps.p5m.admitted_bytes",
                    b.getI64IntegerAttr(ledger.admittedBytes));
      func->setAttr("alps.p5m.overlap_bytes",
                    b.getI64IntegerAttr(ledger.overlapBytes));
      StringRef decision = ledger.admittedSites > 0 ? "admit" : "reject";
      // Function passes may execute concurrently. Build the record locally
      // and emit it in one write so individual ledgers cannot interleave.
      std::string record;
      llvm::raw_string_ostream os(record);
      os << "[ALPS-P5M-ANALYSIS] function=" << func.getName()
         << " sites=" << ledger.sites
         << " admitted_sites=" << ledger.admittedSites
         << " rejected_sites=" << ledger.rejectedSites
         << " drain_bytes=" << ledger.drainBytes
         << " descriptors=" << ledger.descriptors
         << " full_tiles=" << ledger.fullTiles
         << " boundary_tiles=" << ledger.boundaryTiles
         << " admitted_descriptors=" << ledger.admittedDescriptors
         << " admitted_bytes=" << ledger.admittedBytes
         << " boundary_bytes=" << ledger.boundaryBytes
         << " overlap_descriptors=" << ledger.overlapDescriptors
         << " overlap_bytes=" << ledger.overlapBytes
         << " overlap_hmx_calls=" << ledger.overlapHmxCalls
         // One tile holds two 2 KiB slots. A second tile preserves the HexKL
         // configuration tail at the end of the enlarged VTCM slab.
         << " extra_vtcm_bytes=8192"
         << " max_dst_stride_bytes=" << ledger.maxDestinationStrideBytes
         << " dma2d_legal=" << (ledger.dma2dLegal ? 1 : 0)
         << " decision=" << decision << "\n";
      os.flush();
      llvm::errs() << record;
    }

    // #4: hoist one max-sized VTCM slab to the function entry when every HexKL
    // matmul has a static K.  Per-matmul alloc/dealloc churn is replaced by a
    // single arena reused sequentially across matmuls.
    if (enablePersistentVtcm) {
      SmallVector<hexkl::MatmulOp> matmuls;
      func.walk([&](hexkl::MatmulOp op) { matmuls.push_back(op); });
      std::optional<int64_t> maxBytes;
      bool allStatic = !matmuls.empty();
      for (hexkl::MatmulOp op : matmuls) {
        auto bytes = estimateVtcmBytes(
            op, enableWeightPrepack, enableVtcmLifetimeColoring,
            enableDmaToVtcm, enableAsyncDrain);
        if (!bytes) {
          allStatic = false;
          break;
        }
        maxBytes = maxBytes ? std::max(*maxBytes, *bytes) : *bytes;
      }
      if (allStatic && maxBytes && *maxBytes > 0) {
        Block &entry = func.getFunctionBody().front();
        OpBuilder b(func.getContext());
        b.setInsertionPointToStart(&entry);
        Location loc = func.getLoc();
        Value bytesVal =
            b.create<arith::ConstantIndexOp>(loc, *maxBytes);
        auto vtcmType = MemRefType::get(
            {ShapedType::kDynamic}, b.getI8Type(), MemRefLayoutAttrInterface{},
            IntegerAttr::get(b.getI32Type(), 1));
        auto alloc = b.create<hexagonmem::AllocOp>(loc, vtcmType, bytesVal);
        alloc->setAttr("bufferization.manual_deallocation", b.getUnitAttr());
        sharedVtcm = alloc.getResult();
        LLVM_DEBUG(DBGS() << "persistent VTCM arena bytes=" << *maxBytes
                          << " matmuls=" << matmuls.size() << "\n");
      }
    }

    RewritePatternSet patterns(&getContext());
    // Report the peak static arena reduction before rewriting the matmuls.
    int64_t legacyPeak = 0;
    int64_t coloredPeak = 0;
    int64_t staticSites = 0;
    func.walk([&](hexkl::MatmulOp op) {
      auto legacy = estimateVtcmBytes(op, enableWeightPrepack,
                                      /*coloring=*/false,
                                      enableDmaToVtcm, enableAsyncDrain);
      auto colored = estimateVtcmBytes(op, enableWeightPrepack,
                                       /*coloring=*/true,
                                       enableDmaToVtcm, enableAsyncDrain);
      if (!legacy || !colored)
        return;
      legacyPeak = std::max(legacyPeak, *legacy);
      coloredPeak = std::max(coloredPeak, *colored);
      ++staticSites;
    });
    if (enableVtcmLifetimeColoring && staticSites > 0) {
      Builder b(func.getContext());
      func->setAttr("alps.vtcm_coloring_enabled", b.getUnitAttr());
      func->setAttr("alps.vtcm_legacy_peak_bytes",
                    b.getI64IntegerAttr(legacyPeak));
      func->setAttr("alps.vtcm_colored_peak_bytes",
                    b.getI64IntegerAttr(coloredPeak));
      func->setAttr("alps.vtcm_saved_peak_bytes",
                    b.getI64IntegerAttr(legacyPeak - coloredPeak));
      func->setAttr("alps.vtcm_colored_sites",
                    b.getI64IntegerAttr(staticSites));
      llvm::errs() << "[VTCMLifetimeColoring] function=" << func.getName()
                   << " sites=" << staticSites
                   << " legacy_peak=" << legacyPeak
                   << " colored_peak=" << coloredPeak
                   << " saved_peak=" << (legacyPeak - coloredPeak) << "\n";
    }

    populateDecomposeHexKLMatmulPatterns(
        patterns, enableWeightPrepack, enableVtcmLifetimeColoring,
        enableDmaToVtcm, enableDirectOutputFormation,
        enableF16BiasEpilogueFormation, enableAsyncDrain, sharedVtcm);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }

    if (enableF16BiasEpilogueFormation) {
      // Give producer fusion the first opportunity.  Only contracts left
      // after every matmul rewrite has reached a fixed point take the
      // conservative elementwise fallback.
      RewritePatternSet fallbackPatterns(&getContext());
      fallbackPatterns.add<DecomposeF16BiasEpilogue>(&getContext());
      if (failed(applyPatternsGreedily(func, std::move(fallbackPatterns))))
        return signalPassFailure();

      int64_t remaining = 0;
      func.walk([&](hexkl::F16BiasEpilogueOp) { ++remaining; });
      llvm::errs() << "[ALPS-P5L-DECOMPOSE] function=" << func.getName()
                   << " remaining_bias_epilogues=" << remaining << "\n";
      if (remaining != 0) {
        func.emitError("P5l left an unmatched f16 bias epilogue after HMX "
                       "decomposition");
        return signalPassFailure();
      }
    }

    if (sharedVtcm) {
      // Dealloc once per return after all matmul bodies have used the arena.
      func.walk([&](func::ReturnOp ret) {
        OpBuilder b(ret);
        b.create<hexagonmem::DeallocOp>(ret.getLoc(), sharedVtcm);
      });
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createDecomposeHexKLMatmulPass(
    const DecomposeHexKLMatmulOptions &options) {
  return std::make_unique<DecomposeHexKLMatmulPass>(options);
}
