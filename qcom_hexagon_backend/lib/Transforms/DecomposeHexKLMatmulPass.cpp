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
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
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
#include <optional>

#define DEBUG_TYPE "decompose-hexkl-matmul"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DECOMPOSEHEXKLMATMUL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct DecomposeHexKLMatmul final : public OpRewritePattern<hexkl::MatmulOp> {
  DecomposeHexKLMatmul(MLIRContext *ctx, bool enableWeightPrepack,
                       bool enableVtcmLifetimeColoring,
                       bool enableDmaToVtcm, Value sharedVtcm)
      : OpRewritePattern(ctx), enableWeightPrepack(enableWeightPrepack),
        enableVtcmLifetimeColoring(enableVtcmLifetimeColoring),
        enableDmaToVtcm(enableDmaToVtcm),
        sharedVtcm(sharedVtcm) {}

  bool enableWeightPrepack;
  bool enableVtcmLifetimeColoring;
  bool enableDmaToVtcm;
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
    Value i32_32 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 32);
    Value i32_4096 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 4096);
    Value idx4096 = rewriter.create<arith::ConstantIndexOp>(loc, 4096);

    // Get dimensions dynamically
    Value dimM = rewriter.create<memref::DimOp>(loc, lhs, idx0);
    Value dimK = rewriter.create<memref::DimOp>(loc, lhs, idx1);
    Value dimNOrig = rewriter.create<memref::DimOp>(loc, rhs, idx1);

    // Pad N up to a multiple of 32 so lm_head-class shapes (e.g. 50257) run on
    // HMX.  MatmulToHexKL only converts static shapes, so padding is decided
    // statically here.
    bool doNPad = false;
    int64_t staticNAligned = -1;
    if (rhsType.hasStaticShape()) {
      int64_t staticN = rhsShape[1];
      staticNAligned = (staticN + 31) / 32 * 32;
      doNPad = staticNAligned != staticN;
    }

    Value dimN = dimNOrig;
    Value rhsWork = rhs;
    Value resultWork = result;
    Value rhsPadAlloc, resultPadAlloc;
    if (doNPad) {
      dimN = rewriter.create<arith::ConstantIndexOp>(loc, staticNAligned);

      auto rhsPadTy = MemRefType::get(
          ArrayRef<int64_t>{rhsShape[0], staticNAligned},
          rhsType.getElementType(), MemRefLayoutAttrInterface{},
          rhsType.getMemorySpace());
      auto resultPadTy = MemRefType::get(
          ArrayRef<int64_t>{resultShape[0], staticNAligned},
          resultType.getElementType(), MemRefLayoutAttrInterface{},
          resultType.getMemorySpace());

      rhsPadAlloc = rewriter.create<memref::AllocOp>(loc, rhsPadTy);
      resultPadAlloc = rewriter.create<memref::AllocOp>(loc, resultPadTy);
      Value zeroW = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(rhsType.getElementType()));
      // Copy valid K×NOrig weights, then zero only the padding columns.
      SmallVector<OpFoldResult> zeros = {rewriter.getIndexAttr(0),
                                         rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> rhsSizes = {dimK, dimNOrig};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
      Value rhsSv = rewriter.create<memref::SubViewOp>(loc, rhsPadAlloc, zeros,
                                                       rhsSizes, strides);
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
      resultWork = resultPadAlloc;
    }

    Value M = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimM);
    Value K = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimK);
    Value N = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimN);

    // Calculate numKTiles = (k + 31) / 32
    Value kPlus31 = rewriter.create<arith::AddIOp>(loc, dimK, idx31);
    Value kTiles = rewriter.create<arith::DivUIOp>(loc, kPlus31, idx32);
    Value kTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, kTiles);

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
                            loc, vtcm, scrOff, lhs, rowTile, kt, M, K);
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
                  bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                        accOff);
                  bb.create<hexkl::MicroHMXCopyF16ToF32SubmatrixOp>(
                      loc, vtcm, flatOff, resultWork, rowTile, colTile, M, N);
                  bb.create<scf::YieldOp>(loc);
                });
            b.create<scf::YieldOp>(loc);
          });
    } else {
      // Default: M-outer with dual ping-pong weight slots for OmniFetch.
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
                      loc, vtcm, scrOff, lhs, rowTile, kt, M, K);
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
                  bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                        accOff);
                  bb.create<hexkl::MicroHMXCopyF16ToF32SubmatrixOp>(
                      loc, vtcm, flatOff, resultWork, rowTile, colTile, M, N);
                  bb.create<scf::YieldOp>(loc);
                });
            b.create<scf::YieldOp>(loc);
          });
    }

    // Explicitly deallocate VTCM buffer to avoid relying on ConvertToHexagonmem
    // rewriting of generic memref.dealloc for dynamic VTCM types.
    rewriter.setInsertionPointAfter(outerFor);
    if (doNPad) {
      SmallVector<OpFoldResult> zeros = {rewriter.getIndexAttr(0),
                                         rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> outSizes = {dimM, dimNOrig};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
      Value outSv = rewriter.create<memref::SubViewOp>(
          loc, resultPadAlloc, zeros, outSizes, strides);
      rewriter.create<memref::CopyOp>(loc, outSv, result);
      rewriter.create<memref::DeallocOp>(loc, rhsPadAlloc);
      rewriter.create<memref::DeallocOp>(loc, resultPadAlloc);
    }
    if (ownsVtcm)
      rewriter.create<hexagonmem::DeallocOp>(loc, vtcm);

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
                  bool enableVtcmLifetimeColoring, bool enableDmaToVtcm) {
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  if (!lhsType || !lhsType.hasStaticShape() || lhsType.getRank() != 2)
    return std::nullopt;
  int64_t K = lhsType.getShape()[1];
  if (K <= 0)
    return std::nullopt;
  int64_t kTiles = (K + 31) / 32;
  if (enableVtcmLifetimeColoring)
    return computeColoredVtcmTiles(kTiles, enableWeightPrepack,
                                   enableDmaToVtcm) *
           4096;
  int64_t defBytes = (kTiles * 2 + 4 + 3) * 4096;
  int64_t prepackBytes = (kTiles * 3 + 5) * 4096;
  return enableWeightPrepack ? prepackBytes : defBytes;
}

void populateDecomposeHexKLMatmulPatterns(RewritePatternSet &patterns,
                                          bool enableWeightPrepack,
                                          bool enableVtcmLifetimeColoring,
                                          bool enableDmaToVtcm,
                                          Value sharedVtcm) {
  patterns.add<DecomposeHexKLMatmul>(
      patterns.getContext(), enableWeightPrepack,
      enableVtcmLifetimeColoring, enableDmaToVtcm, sharedVtcm);
}

struct DecomposeHexKLMatmulPass
    : public ::impl::DecomposeHexKLMatmulBase<DecomposeHexKLMatmulPass> {
  using DecomposeHexKLMatmulBase::DecomposeHexKLMatmulBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, hexagonmem::HexagonMemDialect,
                omni_fetch::OmniFetchDialect, arith::ArithDialect,
                scf::SCFDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    Value sharedVtcm;

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
            enableDmaToVtcm);
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
                                      enableDmaToVtcm);
      auto colored = estimateVtcmBytes(op, enableWeightPrepack,
                                       /*coloring=*/true,
                                       enableDmaToVtcm);
      if (!legacy || !colored)
        return;
      legacyPeak = std::max(legacyPeak, *legacy);
      coloredPeak = std::max(coloredPeak, *colored);
      ++staticSites;
    });
    if (enableVtcmLifetimeColoring && staticSites > 0) {
      Builder b(func.getContext());
      func->setAttr("omni_fetch.vtcm_coloring_enabled", b.getUnitAttr());
      func->setAttr("omni_fetch.vtcm_legacy_peak_bytes",
                    b.getI64IntegerAttr(legacyPeak));
      func->setAttr("omni_fetch.vtcm_colored_peak_bytes",
                    b.getI64IntegerAttr(coloredPeak));
      func->setAttr("omni_fetch.vtcm_saved_peak_bytes",
                    b.getI64IntegerAttr(legacyPeak - coloredPeak));
      func->setAttr("omni_fetch.vtcm_colored_sites",
                    b.getI64IntegerAttr(staticSites));
      llvm::errs() << "[VTCMLifetimeColoring] function=" << func.getName()
                   << " sites=" << staticSites
                   << " legacy_peak=" << legacyPeak
                   << " colored_peak=" << coloredPeak
                   << " saved_peak=" << (legacyPeak - coloredPeak) << "\n";
    }

    populateDecomposeHexKLMatmulPatterns(
        patterns, enableWeightPrepack, enableVtcmLifetimeColoring,
        enableDmaToVtcm, sharedVtcm);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
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
