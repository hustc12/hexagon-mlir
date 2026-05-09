//===- VDAEDecouplePass.cpp - V-DAE prefetch insertion  -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Implements the "Virtual Decoupled Access-Execute" (V-DAE) transformation
// for DNN inference on Hexagon NPU.
//
// Motivation
// ----------
// Memory latency on edge NPUs far exceeds compute latency for large weight
// tensors.  The "layout wall" compounds this: a tensor fetched in standard
// row-major format still requires a reshape before HMX can consume it.
//
// This pass addresses both problems by:
//
//   1. Moving data movement AHEAD of compute (software prefetching).
//   2. Requesting an in-situ layout reshape DURING the fetch via
//      `omni_fetch.prefetch_in_situ`, so the VTCM tile is already in
//      HMX-preferred format when compute begins.
//   3. Using Hexagon hardware semaphores (via `omni_fetch.signal/wait`)
//      for zero-overhead synchronisation between the Access Thread and the
//      Execute Thread.
//   4. Optionally emitting `omni_fetch.adaptive_control` at the loop-back
//      edge to let the runtime tune the prefetch distance based on live
//      PMU feedback (AXI stall counters).
//
// Prerequisites
// -------------
// The pass should run AFTER `decompose-hexkl-matmul` (or any pass that
// produces tiled `scf.for` loops over VTCM-allocated tile buffers).
// It can also be applied after `hexagon-double-buffer-generic-s1` — in
// that case it augments the existing double-buffer with layout-aware
// prefetch semantics.
//
// Transformation overview
// -----------------------
// For each qualifying `scf.for` loop the pass:
//
//   A. DETECTS the weight and activation memrefs feeding into HMX ops.
//
//   B. ALLOCATES a VTCM "shadow" buffer sized to one tile (if not already
//      allocated by a prior double-buffer pass).
//
//   C. EMITS PROLOGUE before the loop:
//        %sem = omni_fetch.create_sem
//        // prefetch iteration 0 … lookahead-1
//        omni_fetch.prefetch_in_situ %src_ddr[0], %vtcm_tile,
//            {layout_transform = HMXWeight, lookahead = K}
//        omni_fetch.signal %sem
//
//   D. REWRITES THE LOOP BODY:
//        omni_fetch.wait %sem           // ensure tile[i] ready
//        <original HMX compute on vtcm_tile[i]>
//        omni_fetch.prefetch_in_situ %src_ddr[i+K], %vtcm_tile,
//            {layout_transform = HMXWeight, lookahead = K}
//        omni_fetch.signal %sem
//        %dist = omni_fetch.adaptive_control(%dist)  // optional
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "omni-fetch-vdae-insert"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::omni_fetch;
using namespace hexagon;

#define GEN_PASS_DEF_OMNIFETCHVDAEINSERT
#include "hexagon/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Pattern-matching helpers
//===----------------------------------------------------------------------===//

/// Returns true if `op` or any op in its nest uses a hexkl micro-HMX op.
static bool containsHMXCompute(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) {
    if (llvm::isa<hexkl::MicroHMXMmF16Op,
                  hexkl::MicroHMXSetupAccReadF16Op,
                  hexkl::MatmulOp>(inner)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Collect all `memref.subview` or plain memref-typed block arguments that
/// are read (but not defined) inside `loop` and live in DDR (address space 0).
static SmallVector<Value> collectDDRInputs(scf::ForOp loop) {
  SmallVector<Value> inputs;
  llvm::SmallPtrSet<Value, 8> seen;

  loop.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!seen.insert(operand).second)
        continue;
      auto memTy = dyn_cast<MemRefType>(operand.getType());
      if (!memTy)
        continue;
      // DDR = address space 0 (or unspecified)
      if (memTy.getMemorySpaceAsInt() != 0)
        continue;
      // Must be defined outside the loop (it's a "load" from DDR)
      if (loop->isAncestor(operand.getParentBlock()->getParentOp()))
        continue;
      inputs.push_back(operand);
    }
  });
  return inputs;
}

/// Collect all memref inputs regardless of address space.
/// This is used as a fallback when no DDR inputs are found (e.g., when
/// bufferization allocated everything to VTCM).
static SmallVector<Value> collectAllMemrefInputs(scf::ForOp loop) {
  SmallVector<Value> inputs;
  llvm::SmallPtrSet<Value, 8> seen;

  loop.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!seen.insert(operand).second)
        continue;
      auto memTy = dyn_cast<MemRefType>(operand.getType());
      if (!memTy)
        continue;
      // Accept memrefs in any address space (DDR, VTCM, etc.)
      // Must be defined outside the loop
      if (loop->isAncestor(operand.getParentBlock()->getParentOp()))
        continue;
      inputs.push_back(operand);
    }
  });
  return inputs;
}

/// Guess whether `memref` is a weight or activation based on its use inside
/// the loop body.  Heuristic: if it feeds into the `lhs` of a hexkl.matmul
/// (or micro-setup op), it is treated as a weight; otherwise, activation.
static LayoutTransform inferLayoutTransform(Value memref, scf::ForOp loop) {
  for (auto *user : memref.getUsers()) {
    if (auto matmul = dyn_cast<hexkl::MatmulOp>(user)) {
      if (matmul.getRhs() == memref)
        return LayoutTransform::HMXWeight;
      return LayoutTransform::HMXActivation;
    }
    // Micro-HMX Mm op (matrix-multiply) reads from a weight tile.
    if (llvm::isa<hexkl::MicroHMXMmF16Op>(user))
      return LayoutTransform::HMXWeight;
  }
  return LayoutTransform::HMXActivation;
}

//===----------------------------------------------------------------------===//
// Tile-subview helper
//===----------------------------------------------------------------------===//

/// Build a subview of `base` for iteration `iterVal` with static `tileSize`
/// along the outermost dimension.  This models fetching one tile per loop
/// iteration.
static Value buildTileSubview(OpBuilder &builder, Location loc, Value base,
                              Value iterVal, int64_t tileSize) {
  auto baseType = cast<MemRefType>(base.getType());
  auto rank     = static_cast<unsigned>(baseType.getRank());

  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(rank);
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

  // Outer dim: offset = iter * tileSize, size = tileSize
  Value tileSizeVal = builder.create<arith::ConstantIndexOp>(loc, tileSize);
  Value offsetVal = builder.create<arith::MulIOp>(loc, iterVal, tileSizeVal);
  offsets[0] = offsetVal;
  sizes[0] = builder.getIndexAttr(tileSize);

  // Inner dims: full extent
  for (unsigned i = 1; i < rank; ++i)
    sizes[i] = builder.getIndexAttr(baseType.getShape()[i]);

  auto subviewType = memref::SubViewOp::inferResultType(
      baseType, offsets, sizes, strides);
  return builder.create<memref::SubViewOp>(
      loc, cast<MemRefType>(subviewType), base, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// VTCM shadow buffer allocation
//===----------------------------------------------------------------------===//

/// Allocate a one-tile VTCM buffer for prefetching.  The shape is the same
/// as the `src` memref but with the outermost dimension clamped to `tileSize`.
///
/// We emit a plain `memref.alloc` with address space 1 (VTCM).  The existing
/// `convert-to-hexagonmem` pass will later canonicalise this into a
/// `hexagonmem.alloc` op — so this pass does not need to depend on the
/// hexagonmem::AllocOp builder directly.
static Value allocateVTCMTile(OpBuilder &builder, Location loc, Value src,
                              int64_t tileSize) {
  auto srcType = cast<MemRefType>(src.getType());

  // Build tile shape: [tileSize, inner…]
  SmallVector<int64_t> tileShape;
  tileShape.push_back(tileSize);
  for (int64_t i = 1; i < srcType.getRank(); ++i)
    tileShape.push_back(srcType.getShape()[i]);

  // Address space 1 = VTCM on Hexagon
  auto vtcmMemref = MemRefType::get(
      tileShape, srcType.getElementType(),
      /*layout=*/MemRefLayoutAttrInterface{},
      /*memorySpace=*/IntegerAttr::get(
          IntegerType::get(builder.getContext(), 64), 1));

  return builder.create<memref::AllocOp>(loc, vtcmMemref);
}

//===----------------------------------------------------------------------===//
// Core transformation: transform one qualifying scf.for loop
//===----------------------------------------------------------------------===//

static void transformLoop(scf::ForOp loop, int lookahead, bool enableAdaptive,
                          bool enableLayoutAware) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  MLIRContext *ctx = loop.getContext();

  llvm::errs() << "\n[VDAEInsert] transformLoop: lookahead=" << lookahead 
               << " adaptive=" << enableAdaptive 
               << " layoutAware=" << enableLayoutAware << "\n";

  // ----- Collect DDR inputs -----------------------------------------------
  SmallVector<Value> ddrInputs = collectDDRInputs(loop);
  
  llvm::errs() << "[VDAEInsert]   Found " << ddrInputs.size() << " DDR inputs\n";
  
  // RELAXED CONDITION: If no DDR inputs, try all memref inputs.
  // This handles cases where bufferization allocated everything to VTCM.
  if (ddrInputs.empty()) {
    llvm::errs() << "[VDAEInsert]   No DDR inputs, trying all memref inputs...\n";
    ddrInputs = collectAllMemrefInputs(loop);
    llvm::errs() << "[VDAEInsert]   Found " << ddrInputs.size() << " total memref inputs\n";
    
    if (!ddrInputs.empty()) {
      // Print their address spaces for debugging
      for (Value input : ddrInputs) {
        auto memTy = cast<MemRefType>(input.getType());
        int memSpace = memTy.getMemorySpaceAsInt();
        llvm::errs() << "[VDAEInsert]     Using memref with address space " << memSpace 
                     << " (0=DDR, 1=VTCM, 2=Other)\n";
      }
    }
  }
  
  if (ddrInputs.empty()) {
    llvm::errs() << "[VDAEInsert]   → No memref inputs found, skipping loop\n";
    return;
  }

  llvm::errs() << "[VDAEInsert]   Transforming loop with " << ddrInputs.size() << " inputs\n";

  // ----- Create semaphore -------------------------------------------------
  Value sem = builder.create<CreateSemOp>(loc,
                                                    builder.getIndexType());

  // ----- Determine tile size (heuristic: 32 rows = one HMX tile) ----------
  const int64_t kTileSize = 32;

  // ----- Emit prologue prefetches (iterations 0 … lookahead-1) -----------
  // We insert `lookahead` prefetches immediately before the loop so that by
  // the time the loop body first executes, the VTCM tile is ready.
  SmallVector<Value> vtcmTiles;

  for (Value src : ddrInputs) {
    auto srcType = cast<MemRefType>(src.getType());
    if (srcType.getRank() < 1 || srcType.getShape()[0] < kTileSize) {
      llvm::errs() << "[VDAEInsert]     Skipping small/scalar input\n";
      continue;
    }

    llvm::errs() << "[VDAEInsert]     Processing input with shape rank=" << srcType.getRank() << "\n";

    // Allocate VTCM shadow tile (2x for double-buffering)
    Value vtcmTile = allocateVTCMTile(builder, loc, src, kTileSize * 2);
    vtcmTiles.push_back(vtcmTile);

    LayoutTransform lt = enableLayoutAware ? inferLayoutTransform(src, loop)
                                           : LayoutTransform::None;
    llvm::errs() << "[VDAEInsert]     Layout transform: " << (int)lt << "\n";
    
    // Compute static index map for this layout transform
    auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());
    SmallVector<int32_t> idxMapVec =
        computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

    llvm::errs() << "[VDAEInsert]     Index map size: " << idxMapVec.size() << "\n";

    // Prologue: prefetch tile 0
    Value iterZero = loop.getLowerBound();
    Value tileSubview = buildTileSubview(builder, loc, src, iterZero, kTileSize);

    auto prefetchOp = builder.create<PrefetchInSituOp>(
        loc, tileSubview, vtcmTile,
        lt, static_cast<uint32_t>(lookahead),
        idxMapVec.empty() ? DenseI32ArrayAttr{}
                          : DenseI32ArrayAttr::get(ctx, idxMapVec));
    (void)prefetchOp;
    llvm::errs() << "[VDAEInsert]     Created prologue prefetch\n";
  }

  // Signal after prologue prefetches
  builder.create<SignalOp>(loc, sem);
  llvm::errs() << "[VDAEInsert]   Prologue complete, inserted signal\n";

  // ----- Rewrite loop body ------------------------------------------------
  // We insert:
  //   1. wait at top of body
  //   2. After the last compute op: prefetch(i + lookahead), signal, adaptive
  Block *body = loop.getBody();
  Value iv = loop.getInductionVar();

  // Insert wait at the very top of the body.
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(body);
    builder.create<WaitOp>(loc, sem);
    llvm::errs() << "[VDAEInsert]   Inserted wait at loop body start\n";
  }

  // Insert prefetch + signal + adaptive at the bottom (before yield).
  {
    OpBuilder::InsertionGuard g(builder);
    Operation *yieldOp = body->getTerminator();
    builder.setInsertionPoint(yieldOp);

    // next iteration index: iv + step * lookahead
    Value step = loop.getStep();
    Value kVal = builder.create<arith::ConstantIndexOp>(loc, lookahead);
    Value stepTimesK = builder.create<arith::MulIOp>(loc, step, kVal);
    Value nextIter = builder.create<arith::AddIOp>(loc, iv, stepTimesK);
    
    llvm::errs() << "[VDAEInsert]   Computing next iteration index (iv + step * " << lookahead << ")\n";

    for (size_t i = 0; i < vtcmTiles.size(); ++i) {
      Value src = ddrInputs[i];
      Value vtcmTile = vtcmTiles[i];

      LayoutTransform lt = enableLayoutAware ? inferLayoutTransform(src, loop)
                                             : LayoutTransform::None;
      auto srcType = cast<MemRefType>(src.getType());
      auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());
      SmallVector<int32_t> idxMapVec =
          computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

      // Build guarded subview: only prefetch if nextIter < ub
      Value ub = loop.getUpperBound();
      Value inBounds = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, nextIter, ub);

      auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder();

      Value nextSubview =
          buildTileSubview(thenBuilder, loc, src, nextIter, kTileSize);

      thenBuilder.create<PrefetchInSituOp>(
          loc, nextSubview, vtcmTile,
          lt, static_cast<uint32_t>(lookahead),
          idxMapVec.empty() ? DenseI32ArrayAttr{}
                            : DenseI32ArrayAttr::get(ctx, idxMapVec));
    }

    // Signal after issuing next-iteration prefetch
    builder.create<SignalOp>(loc, sem);

    // Adaptive control (optional)
    if (enableAdaptive) {
      Value initDist =
          builder.create<arith::ConstantIntOp>(loc, (int64_t)lookahead, 32u);
      builder.create<AdaptiveControlOp>(loc, builder.getI32Type(), initDist);
      llvm::errs() << "[VDAEInsert]   Inserted adaptive control\n";
    }
  }

  llvm::errs() << "[VDAEInsert] V-DAE transformation complete for loop\n";
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct OmniFetchVDAEInsertPass
    : public ::impl::OmniFetchVDAEInsertBase<OmniFetchVDAEInsertPass> {

  explicit OmniFetchVDAEInsertPass() = default;
  explicit OmniFetchVDAEInsertPass(const OmniFetchVDAEInsertOptions &options)
      : OmniFetchVDAEInsertBase(options) {}

  void runOnOperation() override {
    auto func = cast<func::FuncOp>(getOperation());
    SmallVector<scf::ForOp> candidates;

    llvm::errs() << "\n[VDAEInsert] === Pass Running ===\n";
    llvm::errs() << "[VDAEInsert] Function: " << func.getName() << "\n";
    llvm::errs() << "[VDAEInsert] lookahead=" << lookahead 
                 << " enableAdaptive=" << enableAdaptive
                 << " enableLayoutAware=" << enableLayoutAware << "\n";
    
    int totalLoops = 0;
    int innermostLoops = 0;
    int loopsWithHMX = 0;

    func.walk([&](scf::ForOp loop) {
      totalLoops++;
      // Only transform inner-most loops (no nested for ops) with HMX compute.
      bool hasNestedFor = false;
      loop.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
      
      if (!hasNestedFor) {
        innermostLoops++;
        bool hasHMX = containsHMXCompute(loop);
        llvm::errs() << "[VDAEInsert]   Innermost loop: hasHMX=" << hasHMX << "\n";
        
        if (hasHMX) {
          loopsWithHMX++;
          candidates.push_back(loop);
        }
      }
    });

    llvm::errs() << "[VDAEInsert] Found " << totalLoops << " total loops, " 
                 << innermostLoops << " innermost, " 
                 << loopsWithHMX << " with HMX compute\n";
    llvm::errs() << "[VDAEInsert] Transforming " << candidates.size() << " candidate loops\n";

    for (auto loop : candidates)
      transformLoop(loop, lookahead, enableAdaptive, enableLayoutAware);
    
    llvm::errs() << "[VDAEInsert] === Pass Complete ===\n\n";
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public factory
//===----------------------------------------------------------------------===//
std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createOmniFetchVDAEInsertPass(
    const OmniFetchVDAEInsertOptions &options) {
  return std::make_unique<OmniFetchVDAEInsertPass>(options);
}
