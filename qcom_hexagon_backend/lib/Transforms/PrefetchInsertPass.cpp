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

/// Allocate a one-tile buffer for prefetching.
/// Uses address space 0 (DDR/heap) because memref::AllocOp does not map
/// to VTCM – true VTCM allocation requires the VTCMPool runtime API which
/// is not yet wired into the compiler-generated alloc path.
/// The prefetch will perform a DDR→DDR copy (still useful for hiding latency
/// via software pipelining, and correct for layout transformation).
static Value allocateVTCMTile(OpBuilder &builder, Location loc, Value src,
                              int64_t tileSize) {
  auto srcType = cast<MemRefType>(src.getType());

  // Build tile shape: [tileSize, inner…]
  SmallVector<int64_t> tileShape;
  tileShape.push_back(tileSize);
  for (int64_t i = 1; i < srcType.getRank(); ++i)
    tileShape.push_back(srcType.getShape()[i]);

  // Address space 0 = DDR (heap). AllocOp lowers to malloc which is safe.
  // TODO: switch to VTCM once VTCMPool alloc is exposed to the compiler.
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

  // Tile size - make it adaptive based on the actual shape
  // For small batch sizes (e.g., batch=1, num_heads=16), use smaller tiles
  const int64_t kMaxTileSize = 32;  // Maximum tile size (one HMX tile)
  const int64_t kMinTileSize = 8;   // Minimum tile size (increased from 4 to avoid too small tiles)
  
  // VTCM budget management: limit total allocation to avoid OOM on DSP heap.
  // DSP heap is limited; keep total prefetch buffers small.
  // Each tile is allocated 2x (double-buffering), so budget = max single alloc.
  const int64_t kMaxVTCMBytes = 64 * 1024;  // 64 KB total across all loops
  int64_t vtcmUsed = 0;

  // For each DDR input, insert prefetch operations
  // Use a vector of pairs to maintain correct src-to-vtcmTile mapping
  SmallVector<std::pair<Value, Value>> prefetchPairs;  // (src, vtcmTile)

  for (size_t i = 0; i < ddrInputs.size(); ++i) {
    Value src = ddrInputs[i];
    auto srcType = cast<MemRefType>(src.getType());
    if (srcType.getRank() < 1)
      continue;
    
    // Determine tile size based on the actual shape
    int64_t shape0 = srcType.getShape()[0];
    if (shape0 < kMinTileSize) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": shape[0]=" << shape0 
                   << " < " << kMinTileSize << ", skipping\n";
      continue;
    }
    
    // Use adaptive tile size: min(maxTileSize, shape[0] / 2)
    // We need at least 2 tiles for double-buffering to be effective
    int64_t tileSize = std::min(kMaxTileSize, shape0 / 2);
    if (tileSize < kMinTileSize) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": computed tileSize=" << tileSize 
                   << " < " << kMinTileSize << ", skipping\n";
      continue;
    }
    
    llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": using tileSize=" << tileSize 
                 << " (shape[0]=" << shape0 << ")\n";

    // Allocate VTCM shadow tile (2x for double-buffering)
    Value vtcmTile = allocateVTCMTile(builder, loc, src, tileSize * 2);
    
    // Calculate VTCM usage for this tile
    auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());
    int64_t tileBytes = 1;
    for (int64_t dim : vtcmTileType.getShape()) {
      tileBytes *= dim;
    }
    tileBytes *= vtcmTileType.getElementTypeBitWidth() / 8;
    
    // Check if we exceed VTCM budget
    if (vtcmUsed + tileBytes > kMaxVTCMBytes) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": would exceed VTCM budget ("
                   << (vtcmUsed + tileBytes) / 1024 << " KB > " << kMaxVTCMBytes / 1024 
                   << " KB), skipping\n";
      continue;
    }
    
    vtcmUsed += tileBytes;
    llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": allocated " << tileBytes / 1024 
                 << " KB VTCM (total: " << vtcmUsed / 1024 << " KB)\n";

    // Determine layout transform
    LayoutTransform lt = enableLayoutAware ? inferLayoutTransform(src, loop)
                                           : LayoutTransform::None;
    
    llvm::dbgs() << "[PrefetchInsert]   Processing memref " << i << ":\n";
    llvm::dbgs() << "[PrefetchInsert]     enableLayoutAware=" << enableLayoutAware << "\n";
    llvm::dbgs() << "[PrefetchInsert]     Inferred layout transform: " << static_cast<int>(lt) 
                 << " (0=None, 1=HMXWeight, 2=HMXActivation)\n";
    
    // Verify element types match
    if (srcType.getElementType() != vtcmTileType.getElementType()) {
      llvm::dbgs() << "[PrefetchInsert]     WARNING: Element type mismatch - skipping this memref\n";
      llvm::dbgs() << "[PrefetchInsert]       src type: " << srcType << "\n";
      llvm::dbgs() << "[PrefetchInsert]       vtcm type: " << vtcmTileType << "\n";
      vtcmUsed -= tileBytes;  // Rollback VTCM usage
      continue;
    }
    
    // Compute index map
    SmallVector<int32_t> idxMapVec =
        computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

    // === PROLOGUE: prefetch tile 0 ===
    Value iterZero = loop.getLowerBound();
    Value tileSubview = buildTileSubview(builder, loc, src, iterZero, tileSize);
    
    // Verify subview element type matches
    auto subviewType = cast<MemRefType>(tileSubview.getType());
    if (subviewType.getElementType() != vtcmTileType.getElementType()) {
      llvm::dbgs() << "[PrefetchInsert]   WARNING: Subview element type mismatch - skipping\n";
      llvm::dbgs() << "[PrefetchInsert]     subview type: " << subviewType << "\n";
      llvm::dbgs() << "[PrefetchInsert]     vtcm type: " << vtcmTileType << "\n";
      vtcmUsed -= tileBytes;  // Rollback VTCM usage
      continue;
    }

    builder.create<PrefetchInSituOp>(
        loc, tileSubview, vtcmTile,
        lt, static_cast<uint32_t>(lookahead),
        idxMapVec.empty() ? DenseI32ArrayAttr{}
                          : DenseI32ArrayAttr::get(ctx, idxMapVec));
    
    llvm::dbgs() << "[PrefetchInsert]     ✓ Inserted prologue prefetch operation (tileSize=" << tileSize << ")\n";
    
    // Store the pair for loop body processing
    prefetchPairs.push_back({src, vtcmTile});
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
    for (auto [src, vtcmTile] : prefetchPairs) {
      LayoutTransform lt = enableLayoutAware ? inferLayoutTransform(src, loop)
                                             : LayoutTransform::None;
      auto srcType = cast<MemRefType>(src.getType());
      auto vtcmTileType = cast<MemRefType>(vtcmTile.getType());
      
      // Recompute tile size for this memref
      int64_t shape0 = srcType.getShape()[0];
      int64_t tileSize = std::min(kMaxTileSize, shape0 / 2);
      if (tileSize < kMinTileSize)
        continue;
      
      SmallVector<int32_t> idxMapVec =
          computeHMXIndexMap(ctx, srcType, vtcmTileType, lt);

      // Build guarded subview: only prefetch if nextIter < ub
      Value ub = loop.getUpperBound();
      Value inBounds = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, nextIter, ub);

      auto ifOp = builder.create<scf::IfOp>(loc, inBounds, /*withElse=*/false);
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder();

      Value nextSubview =
          buildTileSubview(thenBuilder, loc, src, nextIter, tileSize);

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

    llvm::dbgs() << "\n[PrefetchInsert] ========== PASS STARTING ==========\n";
    llvm::dbgs() << "[PrefetchInsert] Function: " << func.getName() << "\n";
    llvm::dbgs() << "[PrefetchInsert] Options: lookahead=" << lookahead 
                 << ", enableLayoutAware=" << enableLayoutAware << "\n";

    // Find all innermost loops with HMX or HVX compute
    func.walk([&](scf::ForOp loop) {
      // Only transform inner-most loops with accelerator compute
      bool hasNestedFor = false;
      loop.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
      
      if (!hasNestedFor && containsAcceleratorCompute(loop)) {
        candidates.push_back(loop);
      }
    });

    llvm::dbgs() << "[PrefetchInsert] Found " << candidates.size() 
                 << " candidate loops for prefetch insertion\n";

    // Insert prefetch for each candidate loop
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
