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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
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

/// Returns true if `op` or any op in its nest uses HMX or HVX compute operations.
/// This includes:
/// - HMX operations: hexkl micro-HMX ops, matmul
/// - HVX operations: linalg ops (matmul, conv, generic), vector ops
static bool containsAcceleratorCompute(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) {
    // Check for HMX operations
    if (llvm::isa<hexkl::MicroHMXMmF16Op,
                  hexkl::MicroHMXSetupAccReadF16Op,
                  hexkl::MatmulOp>(inner)) {
      found = true;
      return WalkResult::interrupt();
    }
    
    // Check for HVX operations (linalg compute ops)
    if (llvm::isa<linalg::MatmulOp,
                  linalg::BatchMatmulOp,
                  linalg::Conv2DNhwcHwcfOp,
                  linalg::Conv2DNchwFchwOp,
                  linalg::GenericOp>(inner)) {
      // For linalg.generic, check if it's a compute operation (not just a copy/transpose)
      if (auto genericOp = dyn_cast<linalg::GenericOp>(inner)) {
        // Check if the body contains arithmetic operations
        bool hasCompute = false;
        genericOp.getBody()->walk([&](Operation *bodyOp) {
          if (llvm::isa<arith::MulFOp, arith::AddFOp, arith::SubFOp,
                        arith::MulIOp, arith::AddIOp, arith::SubIOp,
                        math::ExpOp, math::TanhOp, math::SqrtOp>(bodyOp)) {
            hasCompute = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (hasCompute) {
          found = true;
          return WalkResult::interrupt();
        }
      } else {
        // Other linalg ops are compute operations
        found = true;
        return WalkResult::interrupt();
      }
    }
    
    // Check for vector operations (HVX)
    if (inner->getDialect() && 
        inner->getDialect()->getNamespace() == "vector") {
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

/// Infer layout transform based on how the memref is used.
/// For HMX operations, we can infer specific layout transforms.
/// For HVX operations, we use a heuristic based on the operation type.
static LayoutTransform inferLayoutTransform(Value memref, scf::ForOp loop) {
  // Check direct users of the memref
  for (auto *user : memref.getUsers()) {
    // HMX operations
    if (auto matmul = dyn_cast<hexkl::MatmulOp>(user)) {
      if (matmul.getRhs() == memref)
        return LayoutTransform::HMXWeight;
      return LayoutTransform::HMXActivation;
    }
    if (llvm::isa<hexkl::MicroHMXMmF16Op>(user))
      return LayoutTransform::HMXWeight;
    
    // HVX linalg operations
    if (auto linalgMatmul = dyn_cast<linalg::MatmulOp>(user)) {
      // For linalg.matmul, check if this is the RHS (weight)
      if (linalgMatmul.getInputs().size() >= 2 && 
          linalgMatmul.getInputs()[1] == memref)
        return LayoutTransform::HMXWeight;
      return LayoutTransform::HMXActivation;
    }
    
    if (auto batchMatmul = dyn_cast<linalg::BatchMatmulOp>(user)) {
      // For batch_matmul, check if this is the RHS (weight)
      if (batchMatmul.getInputs().size() >= 2 && 
          batchMatmul.getInputs()[1] == memref)
        return LayoutTransform::HMXWeight;
      return LayoutTransform::HMXActivation;
    }
    
    // For generic linalg ops, check the indexing maps
    if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
      // Try to infer from indexing maps
      // If the memref is an input with a permuted access pattern, 
      // it might benefit from layout transformation
      auto indexingMaps = genericOp.getIndexingMapsArray();
      for (size_t i = 0; i < genericOp.getNumDpsInputs(); ++i) {
        if (genericOp.getDpsInputOperand(i)->get() == memref) {
          if (i < indexingMaps.size() && indexingMaps[i].isPermutation()) {
            // This input has a permuted access pattern
            return LayoutTransform::HMXActivation;
          }
        }
      }
    }
  }
  
  // Check indirect users (through subview/cast operations)
  for (auto *user : memref.getUsers()) {
    if (isa<memref::SubViewOp, memref::CastOp>(user)) {
      for (Value result : user->getResults()) {
        LayoutTransform lt = inferLayoutTransform(result, loop);
        if (lt != LayoutTransform::None)
          return lt;
      }
    }
  }
  
  // Default: no layout transform needed
  // Only apply layout transform when we can positively identify the operation
  return LayoutTransform::None;
}

/// Build a subview of `base` along `tiledDim` starting at `iterVal` with
/// static `tileSize`.  After Hexagon tiling, scf.for IVs are element offsets
/// into the tiled dimension (0, step, 2*step, …), not tile indices.
static Value buildTileSubview(OpBuilder &builder, Location loc, Value base,
                              Value iterVal, int64_t tileSize,
                              unsigned tiledDim) {
  auto baseType = cast<MemRefType>(base.getType());
  auto rank = static_cast<unsigned>(baseType.getRank());
  assert(tiledDim < rank && "tiledDim out of range");

  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(rank);
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

  offsets[tiledDim] = iterVal;
  sizes[tiledDim] = builder.getIndexAttr(tileSize);

  for (unsigned i = 0; i < rank; ++i) {
    if (i == tiledDim)
      continue;
    int64_t dim = baseType.getShape()[i];
    if (ShapedType::isDynamic(dim))
      sizes[i] = builder.create<memref::DimOp>(loc, base, i).getResult();
    else
      sizes[i] = builder.getIndexAttr(dim);
  }

  auto subviewType =
      memref::SubViewOp::inferResultType(baseType, offsets, sizes, strides);
  return builder.create<memref::SubViewOp>(
      loc, cast<MemRefType>(subviewType), base, offsets, sizes, strides);
}

/// Allocate a one-tile shadow buffer for prefetching.
///
/// Minimal safety fix (GPT2 path): use address space 0 (DDR/heap).
/// GPT2 runs with enableConvertToHexagonmem=false; AS1 allocs are not
/// converted to real VTCMPool and have been observed to fault on device
/// (adb exit 13).  When hexagonmem conversion is enabled, a later patch
/// can restore AS1 for true VTCM.
static Value allocateVTCMTile(OpBuilder &builder, Location loc, Value src,
                              int64_t tileSize, unsigned tiledDim) {
  auto srcType = cast<MemRefType>(src.getType());
  assert(tiledDim < static_cast<unsigned>(srcType.getRank()));

  // Build tile shape: full dims, except tiledDim := tileSize.
  SmallVector<int64_t> tileShape(srcType.getShape().begin(),
                                 srcType.getShape().end());
  tileShape[tiledDim] = tileSize;

  // AS 0 = DDR/heap (safe without ConvertToHexagonmem).
  auto tileMemref = MemRefType::get(
      tileShape, srcType.getElementType(),
      /*layout=*/MemRefLayoutAttrInterface{},
      /*memorySpace=*/IntegerAttr::get(
          IntegerType::get(builder.getContext(), 64), 0));

  return builder.create<memref::AllocOp>(loc, tileMemref);
}

//===----------------------------------------------------------------------===//
// Core transformation
//===----------------------------------------------------------------------===//

static void insertPrefetchForLoop(scf::ForOp loop, int lookahead,
                                  bool enableLayoutAware) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  MLIRContext *ctx = loop.getContext();

  llvm::dbgs() << "[PrefetchInsert]   Analyzing loop body...\n";

  // Collect DDR inputs
  SmallVector<Value> ddrInputs = collectDDRInputs(loop);
  
  // Fallback: try all memref inputs if no DDR inputs found
  if (ddrInputs.empty()) {
    llvm::dbgs() << "[PrefetchInsert]   No DDR inputs found, trying all memref inputs\n";
    ddrInputs = collectAllMemrefInputs(loop);
  }
  
  llvm::dbgs() << "[PrefetchInsert]   Found " << ddrInputs.size() 
               << " memref inputs to prefetch\n";
  
  // Debug: print types of all inputs
  for (size_t i = 0; i < ddrInputs.size(); ++i) {
    auto memrefType = dyn_cast<MemRefType>(ddrInputs[i].getType());
    if (memrefType) {
      llvm::dbgs() << "[PrefetchInsert]     Input " << i << ": " 
                   << memrefType << "\n";
    }
  }
  
  if (ddrInputs.empty()) {
    llvm::dbgs() << "[PrefetchInsert]   No inputs to prefetch, skipping loop\n";
    return;
  }

  const int64_t kMaxTileSize = 32;
  const int64_t kMinTileSize = 8;
  // Skip giant tiled extents (e.g. GPT2 vocab embedding 50257).
  const int64_t kMaxTiledExtent = 4096;
  const int64_t kMaxVTCMBytes = 256 * 1024;
  int64_t vtcmUsed = 0;

  // Force linear prefetch for now.  HMX layout transforms without HexKL (and
  // without compute rewire) corrupt / OOB on device.
  (void)enableLayoutAware;
  const bool doLayoutAware = false;

  // (src, shadowTile, tiledDim)
  SmallVector<std::tuple<Value, Value, unsigned>> prefetchPairs;

  // Prefetch addressing: IV is an element offset into whichever memref dim
  // matches the loop upper bound (GPT2 weight loops commonly tile dim1).
  IntegerAttr stepAttr, lbAttr, ubAttr;
  const bool haveStep = matchPattern(loop.getStep(), m_Constant(&stepAttr));
  const bool haveLb = matchPattern(loop.getLowerBound(), m_Constant(&lbAttr));
  const bool haveUb = matchPattern(loop.getUpperBound(), m_Constant(&ubAttr));
  if (!haveStep || !haveLb || !haveUb) {
    llvm::dbgs() << "[PrefetchInsert]   Non-constant loop bounds/step, skipping\n";
    return;
  }
  const int64_t loopStep = stepAttr.getInt();
  const int64_t loopLb = lbAttr.getInt();
  const int64_t loopUb = ubAttr.getInt();
  if (loopLb != 0 || loopStep <= 0 || loopUb <= loopLb) {
    llvm::dbgs() << "[PrefetchInsert]   Unsupported loop lb/step/ub ("
                 << loopLb << "," << loopStep << "," << loopUb
                 << "), skipping\n";
    return;
  }
  if (loopStep < kMinTileSize || loopStep > kMaxTileSize) {
    llvm::dbgs() << "[PrefetchInsert]   loop step=" << loopStep
                 << " outside tile size [" << kMinTileSize << ","
                 << kMaxTileSize << "], skipping\n";
    return;
  }
  if (loopUb % loopStep != 0 || loopUb / loopStep < 2) {
    llvm::dbgs() << "[PrefetchInsert]   loop extent not a multi-tile multiple "
                 << "of step, skipping\n";
    return;
  }

  for (size_t i = 0; i < ddrInputs.size(); ++i) {
    Value src = ddrInputs[i];
    auto srcType = cast<MemRefType>(src.getType());
    if (srcType.getRank() < 2) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": rank < 2, skipping\n";
      continue;
    }

    // Only floating-point payloads (skip i1/i64 index masks etc.).
    if (!isa<FloatType>(srcType.getElementType())) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": non-float element, skipping\n";
      continue;
    }

    // Contiguous identity layout only.  Strided / dynamic-offset views
    // (common after subview) have caused DSP Bad VA (adb exit 13).
    if (!srcType.getLayout().isIdentity()) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": non-identity layout, skipping\n";
      continue;
    }
    if (!srcType.hasStaticShape()) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": dynamic shape, skipping\n";
      continue;
    }

    // Find the dimension this loop tiles (must match ub).
    int tiledDim = -1;
    for (int d = 0; d < srcType.getRank(); ++d) {
      if (srcType.getShape()[d] == loopUb) {
        tiledDim = d;
        break;
      }
    }
    if (tiledDim < 0) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": no dim matches loop ub=" << loopUb << ", skipping\n";
      continue;
    }

    int64_t tiledExtent = srcType.getShape()[tiledDim];
    if (tiledExtent > kMaxTiledExtent) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": tiled extent "
                   << tiledExtent << " > " << kMaxTiledExtent
                   << ", skipping\n";
      continue;
    }

    int64_t tileSize = loopStep;
    llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": using tileSize="
                 << tileSize << " on dim" << tiledDim << " (extent="
                 << tiledExtent << ")\n";

    // Dest size must equal src tile size (OmniFetchToLLVM uses dest.num_elems).
    Value vtcmTile = allocateVTCMTile(builder, loc, src, tileSize,
                                      static_cast<unsigned>(tiledDim));

    auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());
    int64_t tileBytes = 1;
    for (int64_t dim : vtcmTileType.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        tileBytes = -1;
        break;
      }
      tileBytes *= dim;
    }
    if (tileBytes > 0)
      tileBytes *= vtcmTileType.getElementTypeBitWidth() / 8;

    if (tileBytes < 0 || vtcmUsed + tileBytes > kMaxVTCMBytes) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": exceeds budget or dynamic, skipping\n";
      vtcmTile.getDefiningOp()->erase();
      continue;
    }

    vtcmUsed += tileBytes;
    llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": allocated "
                 << tileBytes / 1024 << " KB (total: " << vtcmUsed / 1024
                 << " KB)\n";

    LayoutTransform lt =
        doLayoutAware ? inferLayoutTransform(src, loop) : LayoutTransform::None;

    llvm::dbgs() << "[PrefetchInsert]   Processing memref " << i
                 << " layout=" << static_cast<int>(lt) << "\n";

    if (srcType.getElementType() != vtcmTileType.getElementType()) {
      vtcmUsed -= tileBytes;
      vtcmTile.getDefiningOp()->erase();
      continue;
    }

    SmallVector<int32_t> idxMapVec =
        computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

    // Prologue: prefetch at lower bound offset.
    Value iterZero = loop.getLowerBound();
    Value tileSubview =
        buildTileSubview(builder, loc, src, iterZero, tileSize,
                         static_cast<unsigned>(tiledDim));

    auto subviewType = cast<MemRefType>(tileSubview.getType());
    if (subviewType.getElementType() != vtcmTileType.getElementType()) {
      vtcmUsed -= tileBytes;
      vtcmTile.getDefiningOp()->erase();
      continue;
    }

    builder.create<PrefetchInSituOp>(
        loc, tileSubview, vtcmTile, lt, static_cast<uint32_t>(lookahead),
        idxMapVec.empty() ? DenseI32ArrayAttr{}
                          : DenseI32ArrayAttr::get(ctx, idxMapVec));

    llvm::dbgs() << "[PrefetchInsert]     ✓ Prologue prefetch tileSize="
                 << tileSize << " dim=" << tiledDim << "\n";
    prefetchPairs.emplace_back(src, vtcmTile, static_cast<unsigned>(tiledDim));
  }

  llvm::dbgs() << "[PrefetchInsert]   Total prefetch pairs created: " << prefetchPairs.size() << "\n";
  llvm::dbgs() << "[PrefetchInsert]   Total VTCM used: " << vtcmUsed / 1024 << " KB / " 
               << kMaxVTCMBytes / 1024 << " KB\n";

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

    // Process each prefetch pair
    for (auto [src, vtcmTile, tiledDim] : prefetchPairs) {
      LayoutTransform lt = LayoutTransform::None;
      auto srcType = cast<MemRefType>(src.getType());
      auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());

      int64_t tileSize = loopStep;
      if (tileSize < kMinTileSize)
        continue;

      SmallVector<int32_t> idxMapVec =
          computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

      // Require nextIter + tileSize <= ub (exclusive upper bound).
      Value ub = loop.getUpperBound();
      Value tileSizeVal =
          builder.create<arith::ConstantIndexOp>(loc, tileSize);
      Value nextEnd =
          builder.create<arith::AddIOp>(loc, nextIter, tileSizeVal);
      Value inBounds = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ule, nextEnd, ub);

      auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder();

      Value nextSubview = buildTileSubview(thenBuilder, loc, src, nextIter,
                                           tileSize, tiledDim);

      thenBuilder.create<PrefetchInSituOp>(
          loc, nextSubview, vtcmTile, lt, static_cast<uint32_t>(lookahead),
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

    llvm::dbgs() << "\n[PrefetchInsert] ========== PASS STARTING ==========\n";
    llvm::dbgs() << "[PrefetchInsert] Function: " << func.getName() << "\n";
    llvm::dbgs() << "[PrefetchInsert] Options: lookahead=" << lookahead
                 << ", enableLayoutAware=" << enableLayoutAware
                 << " (forced off in insert for safety)\n";

    func.walk([&](scf::ForOp loop) {
      bool hasNestedFor = false;
      loop.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
      if (!hasNestedFor && containsAcceleratorCompute(loop))
        candidates.push_back(loop);
    });

    llvm::dbgs() << "[PrefetchInsert] Found " << candidates.size()
                 << " candidate loops for prefetch insertion\n";

    int loopIdx = 0;
    for (auto loop : candidates) {
      llvm::dbgs() << "\n[PrefetchInsert] --- Processing loop " << loopIdx++
                   << " at " << loop.getLoc() << " ---\n";
      insertPrefetchForLoop(loop, lookahead, enableLayoutAware);
    }

    if (candidates.empty()) {
      llvm::dbgs() << "[PrefetchInsert] No candidate loops found - "
                   << "no prefetch operations inserted\n";
    }
    llvm::dbgs() << "[PrefetchInsert] ========== PASS COMPLETE ==========\n\n";
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
