//===- PrefetchInsertPass.cpp - Insert prefetch operations ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass inserts prefetch operations to preload data from DDR to VTCM.
//
// The pass:
// 1. Detects loops with HMX compute operations
// 2. Identifies DDR inputs that need to be prefetched
// 3. Allocates VTCM shadow buffers
// 4. Inserts prefetch operations in prologue and loop body
//
// The prefetch can optionally perform in-situ layout transformation during
// the DDR→VTCM transfer (controlled by enableLayoutAware option).
//
// This pass is independent of V-DAE. It only inserts prefetch operations
// without any synchronization mechanism. The V-DAE pass (if enabled) will
// later add semaphore synchronization around these prefetch operations.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "prefetch-insert"

using namespace mlir;
using namespace mlir::omni_fetch;
using namespace hexagon;

#define GEN_PASS_DEF_PREFETCHINSERT
#include "hexagon/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
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
      // Must be defined outside the loop
      if (loop->isAncestor(operand.getParentBlock()->getParentOp()))
        continue;
      inputs.push_back(operand);
    }
  });
  return inputs;
}

/// Guess whether `memref` is a weight or activation based on its use.
static LayoutTransform inferLayoutTransform(Value memref, scf::ForOp loop) {
  for (auto *user : memref.getUsers()) {
    if (auto matmul = dyn_cast<hexkl::MatmulOp>(user)) {
      if (matmul.getRhs() == memref)
        return LayoutTransform::HMXWeight;
      return LayoutTransform::HMXActivation;
    }
    if (llvm::isa<hexkl::MicroHMXMmF16Op>(user))
      return LayoutTransform::HMXWeight;
  }
  return LayoutTransform::HMXActivation;
}

/// Build a subview of `base` for iteration `iterVal` with static `tileSize`.
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

/// Allocate a one-tile VTCM buffer for prefetching.
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
// Core transformation
//===----------------------------------------------------------------------===//

static void insertPrefetchForLoop(scf::ForOp loop, int lookahead,
                                  bool enableLayoutAware) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  MLIRContext *ctx = loop.getContext();

  // Collect DDR inputs
  SmallVector<Value> ddrInputs = collectDDRInputs(loop);
  
  // Fallback: try all memref inputs if no DDR inputs found
  if (ddrInputs.empty()) {
    ddrInputs = collectAllMemrefInputs(loop);
  }
  
  if (ddrInputs.empty())
    return;

  // Tile size (heuristic: 32 rows = one HMX tile)
  const int64_t kTileSize = 32;

  // For each DDR input, insert prefetch operations
  SmallVector<Value> vtcmTiles;

  for (Value src : ddrInputs) {
    auto srcType = cast<MemRefType>(src.getType());
    if (srcType.getRank() < 1 || srcType.getShape()[0] < kTileSize)
      continue;

    // Allocate VTCM shadow tile (2x for double-buffering)
    Value vtcmTile = allocateVTCMTile(builder, loc, src, kTileSize * 2);
    vtcmTiles.push_back(vtcmTile);

    // Determine layout transform
    LayoutTransform lt = enableLayoutAware ? inferLayoutTransform(src, loop)
                                           : LayoutTransform::None;
    
    // Compute index map
    auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());
    SmallVector<int32_t> idxMapVec =
        computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

    // === PROLOGUE: prefetch tile 0 ===
    Value iterZero = loop.getLowerBound();
    Value tileSubview = buildTileSubview(builder, loc, src, iterZero, kTileSize);

    builder.create<PrefetchInSituOp>(
        loc, tileSubview, vtcmTile,
        lt, static_cast<uint32_t>(lookahead),
        idxMapVec.empty() ? DenseI32ArrayAttr{}
                          : DenseI32ArrayAttr::get(ctx, idxMapVec));
  }

  // === LOOP BODY: prefetch tile i+lookahead ===
  Block *body = loop.getBody();
  Value iv = loop.getInductionVar();

  {
    OpBuilder::InsertionGuard g(builder);
    Operation *yieldOp = body->getTerminator();
    builder.setInsertionPoint(yieldOp);

    // next iteration index: iv + step * lookahead
    Value step = loop.getStep();
    Value kVal = builder.create<arith::ConstantIndexOp>(loc, lookahead);
    Value stepTimesK = builder.create<arith::MulIOp>(loc, step, kVal);
    Value nextIter = builder.create<arith::AddIOp>(loc, iv, stepTimesK);

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
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct PrefetchInsertPass
    : public ::impl::PrefetchInsertBase<PrefetchInsertPass> {

  explicit PrefetchInsertPass() = default;
  explicit PrefetchInsertPass(const PrefetchInsertOptions &options)
      : PrefetchInsertBase(options) {}

  void runOnOperation() override {
    auto func = cast<func::FuncOp>(getOperation());
    SmallVector<scf::ForOp> candidates;

    // Find all innermost loops with HMX compute
    func.walk([&](scf::ForOp loop) {
      // Only transform inner-most loops with HMX compute
      bool hasNestedFor = false;
      loop.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
      
      if (!hasNestedFor && containsHMXCompute(loop)) {
        candidates.push_back(loop);
      }
    });

    // Insert prefetch for each candidate loop
    for (auto loop : candidates) {
      insertPrefetchForLoop(loop, lookahead, enableLayoutAware);
    }
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public factory
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createPrefetchInsertPass(const PrefetchInsertOptions &options) {
  return std::make_unique<PrefetchInsertPass>(options);
}
