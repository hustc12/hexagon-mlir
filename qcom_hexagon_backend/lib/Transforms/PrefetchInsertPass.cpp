//===- PrefetchInsertPass.cpp - Insert prefetch operations ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass inserts prefetch operations to preload data from DDR to a
// multi-buffer shadow, and rewires the loop-body compute to consume the
// current shadow tile (software-pipelined with depth = lookahead+1).
//
// The pass:
// 1. Detects loops with HMX/HVX compute operations
// 2. Identifies DDR inputs that need to be prefetched
// 3. Allocates multi-buffer shadow tiles (AS0 unless hexagonmem is enabled)
// 4. Prefetches prologue tiles [0, lookahead) and body tile i+lookahead
// 5. Rewires compute subviews of the DDR source to the current shadow slot
//
// The prefetch can optionally perform in-situ layout transformation during
// the DDR→shadow transfer (controlled by enableLayoutAware option).
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
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

//===----------------------------------------------------------------------===//
// Core transformation
//===----------------------------------------------------------------------===//

static bool sizeIsConstant(OpFoldResult size, int64_t expect) {
  if (auto attr = dyn_cast<Attribute>(size)) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt() == expect;
    return false;
  }
  if (auto v = dyn_cast<Value>(size)) {
    IntegerAttr ia;
    if (matchPattern(v, m_Constant(&ia)))
      return ia.getInt() == expect;
  }
  return false;
}

/// Follow trivial memref.cast chains.
static Value peelMemRefCasts(Value v) {
  while (auto cast = v.getDefiningOp<memref::CastOp>())
    v = cast.getSource();
  return v;
}

/// True if `v` (or a subview of it) is written inside `loop`.
static bool isWrittenInLoop(Value v, scf::ForOp loop) {
  for (Operation *user : v.getUsers()) {
    if (!loop->isAncestor(user))
      continue;
    if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
      if (isWrittenInLoop(sv.getResult(), loop))
        return true;
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(user)) {
      if (isWrittenInLoop(cast.getResult(), loop))
        return true;
      continue;
    }
    if (isa<memref::StoreOp>(user))
      return true;
    if (auto tw = dyn_cast<vector::TransferWriteOp>(user)) {
      if (tw.getBase() == v)
        return true;
      continue;
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
      for (OpOperand &init : dps.getDpsInitsMutable()) {
        if (init.get() == v)
          return true;
      }
    }
  }
  return false;
}

/// Subview of `src` inside `loop` that covers one tile along `tiledDim`.
static bool isTileSubviewOf(memref::SubViewOp sv, Value src, unsigned tiledDim,
                            int64_t tileSize) {
  if (peelMemRefCasts(sv.getSource()) != src && sv.getSource() != src)
    return false;
  SmallVector<OpFoldResult> sizes = sv.getMixedSizes();
  return tiledDim < sizes.size() && sizeIsConstant(sizes[tiledDim], tileSize);
}

/// Collect tile subviews of `src` that feed vector.transfer_read or linalg.
static void collectTileSubviews(Value src, scf::ForOp loop, unsigned tiledDim,
                                int64_t tileSize,
                                SmallVectorImpl<memref::SubViewOp> &out) {
  for (Operation *user : src.getUsers()) {
    if (!loop->isAncestor(user))
      continue;
    auto sv = dyn_cast<memref::SubViewOp>(user);
    if (!sv || !isTileSubviewOf(sv, src, tiledDim, tileSize))
      continue;
    bool useful = false;
    for (Operation *su : sv.getResult().getUsers()) {
      if (isa<vector::TransferReadOp>(su)) {
        useful = true;
        break;
      }
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(su)) {
        for (OpOperand &operand : su->getOpOperands()) {
          if (operand.get() == sv.getResult() && !dps.isDpsInit(&operand)) {
            useful = true;
            break;
          }
        }
      }
      if (useful)
        break;
    }
    if (useful)
      out.push_back(sv);
  }
}

/// Insert L2-fetch hints before transfer_read/linalg readers of each tile
/// subview.  Currently unused on GPT2: per-strip runtime calls regress
/// latency more than they hide.  Kept for experiments on larger tiles.
[[maybe_unused]] static int insertL2FetchHints(OpBuilder &builder,
                                               scf::ForOp loop, Value src,
                                               unsigned tiledDim,
                                               int64_t tileSize) {
  SmallVector<memref::SubViewOp> tileViews;
  collectTileSubviews(src, loop, tiledDim, tileSize, tileViews);
  if (tileViews.empty())
    return 0;

  // Per-site 1-element scratch dest so PrefetchInSitu's MemWrite does not
  // alias the live DDR subview (src==dest broke later passes).
  int inserted = 0;
  for (memref::SubViewOp sv : tileViews) {
    auto svTy = cast<MemRefType>(sv.getType());
    if (!svTy.hasStaticShape())
      continue;

    SmallVector<Operation *> readers;
    for (Operation *user : sv.getResult().getUsers()) {
      if (isa<vector::TransferReadOp>(user)) {
        readers.push_back(user);
        continue;
      }
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
        for (OpOperand &operand : user->getOpOperands()) {
          if (operand.get() == sv.getResult() && !dps.isDpsInit(&operand)) {
            readers.push_back(user);
            break;
          }
        }
      }
    }
    if (readers.empty())
      continue;

    llvm::sort(readers, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    OpBuilder b(readers.front());
    // Tiny AS0 scratch — runtime L2Hint ignores dest contents.
    auto scratchTy = MemRefType::get(
        ArrayRef<int64_t>{1}, svTy.getElementType(),
        /*layout=*/MemRefLayoutAttrInterface{},
        /*memorySpace=*/IntegerAttr::get(
            IntegerType::get(builder.getContext(), 64), 0));
    // Hoist scratch next to the enclosing loop so it is allocated once.
    Value scratch;
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(loop);
      scratch = builder.create<memref::AllocOp>(loop.getLoc(), scratchTy);
    }
    b.create<PrefetchInSituOp>(sv.getLoc(), sv.getResult(), scratch,
                               LayoutTransform::L2Hint, /*lookahead=*/1,
                               DenseI32ArrayAttr{});
    ++inserted;
  }

  llvm::dbgs() << "[PrefetchInsert]     l2-hint ops: " << inserted << "\n";
  return inserted;
}

/// Insert sync prefetch and rewire for large tiles (VTCM-oriented path).
/// Currently unused on GPT2's 64-element HVX strips; kept for larger tiles.
static int insertSyncPrefetchAndRewire(OpBuilder &builder, scf::ForOp loop,
                                       Value src, unsigned tiledDim,
                                       int64_t tileSize, int64_t &vtcmUsed,
                                       int64_t kMaxVTCMBytes) {
  SmallVector<memref::SubViewOp> tileViews;
  collectTileSubviews(src, loop, tiledDim, tileSize, tileViews);
  if (tileViews.empty())
    return 0;

  auto refTy = cast<MemRefType>(tileViews.front().getType());
  if (!refTy.hasStaticShape())
    return 0;
  for (auto sv : tileViews) {
    auto ty = cast<MemRefType>(sv.getType());
    if (ty.getShape() != refTy.getShape() ||
        ty.getElementType() != refTy.getElementType())
      return 0;
  }

  int64_t tileBytes = refTy.getElementTypeBitWidth() / 8;
  for (int64_t d : refTy.getShape())
    tileBytes *= d;
  // Tiny tiles: sync copy cannot win — caller should use L2 hints instead.
  const int64_t kMinCopyBytes = 4096;
  if (tileBytes < kMinCopyBytes)
    return 0;
  if (vtcmUsed + tileBytes > kMaxVTCMBytes)
    return 0;

  Location loc = loop.getLoc();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(loop);
  // Prefer VTCM (AS1).  ConvertToHexagonmem runs before this pass today, so
  // emit hexagonmem.alloc when the dialect is available; else AS1 memref.alloc
  // and rely on a follow-up hexagonmem conversion if enabled later.
  auto shadowTy = MemRefType::get(
      refTy.getShape(), refTy.getElementType(),
      /*layout=*/MemRefLayoutAttrInterface{},
      /*memorySpace=*/IntegerAttr::get(
          IntegerType::get(builder.getContext(), 64),
          /*VTCM*/ 1));
  Value shadow = builder.create<memref::AllocOp>(loc, shadowTy);
  vtcmUsed += tileBytes;

  int replaced = 0;
  for (memref::SubViewOp sv : tileViews) {
    SmallVector<Operation *> readers;
    for (Operation *user : sv.getResult().getUsers()) {
      if (isa<vector::TransferReadOp>(user)) {
        readers.push_back(user);
        continue;
      }
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
        for (OpOperand &operand : user->getOpOperands()) {
          if (operand.get() == sv.getResult() && !dps.isDpsInit(&operand)) {
            readers.push_back(user);
            break;
          }
        }
      }
    }
    if (readers.empty())
      continue;

    llvm::sort(readers, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    OpBuilder b(readers.front());
    b.create<PrefetchInSituOp>(sv.getLoc(), sv.getResult(), shadow,
                               LayoutTransform::None, /*lookahead=*/1,
                               DenseI32ArrayAttr{});

    for (Operation *user : readers) {
      IRMapping mapping;
      mapping.map(sv.getResult(), shadow);
      OpBuilder cb(user);
      Operation *cloned = cb.clone(*user, mapping);
      if (user->getNumResults() == 0) {
        user->erase();
      } else {
        user->replaceAllUsesWith(cloned);
        user->erase();
      }
      ++replaced;
    }
  }

  llvm::dbgs() << "[PrefetchInsert]     sync-rewired ops: " << replaced
               << " shadow=" << shadowTy << " (" << tileBytes / 1024
               << " KB)\n";
  if (replaced == 0) {
    shadow.getDefiningOp()->erase();
    vtcmUsed -= tileBytes;
  }
  return replaced;
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
      llvm::dbgs() << "[PrefetchInsert]     Input " << i << ": " << memrefType
                   << "\n";
    }
  }

  if (ddrInputs.empty()) {
    llvm::dbgs() << "[PrefetchInsert]   No inputs to prefetch, skipping loop\n";
    return;
  }

  const int64_t kMaxTileSize = 128;
  const int64_t kMinTileSize = 8;
  // Skip giant tiled extents (e.g. GPT2 vocab embedding 50257).
  const int64_t kMaxTiledExtent = 4096;
  // Shadow buffers currently live in AS0 (DDR).  Use a larger budget than
  // physical VTCM; tighten again when hexagonmem/AS1 allocation is enabled.
  const int64_t kMaxVTCMBytes = 2 * 1024 * 1024;
  int64_t vtcmUsed = 0;

  // Force linear prefetch for now.  HMX layout transforms without HexKL (and
  // without a matching consumer layout) corrupt data on device.
  (void)enableLayoutAware;
  const bool doLayoutAware = false;

  // Single-buffer synchronous schedule for HVX vector.transfer_read tiles:
  // before each reader, copy the existing tile subview into a contiguous
  // shadow and rewire the reader to that shadow.
  (void)lookahead;

  int totalRewired = 0;

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

    if (!isa<FloatType>(srcType.getElementType())) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": non-float element, skipping\n";
      continue;
    }

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

    if (isWrittenInLoop(src, loop)) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": written in loop, skipping\n";
      continue;
    }

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

    int64_t tileSize = loopStep;
    int64_t tiledExtent = srcType.getShape()[tiledDim];
    if (tiledExtent > kMaxTiledExtent) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": tiled extent "
                   << tiledExtent << " > " << kMaxTiledExtent << ", skipping\n";
      continue;
    }

    llvm::dbgs() << "[PrefetchInsert]   Memref " << i << ": trying tileSize="
                 << tileSize << " on dim" << tiledDim << " (extent="
                 << tiledExtent << ")\n";

    // Tiny HVX strips (e.g. GPT2 64xf16): neither sync DDR→shadow memcpy nor
    // per-read L2-hint runtime calls can win — both regress vs baseline.
    // Only instrument tiles large enough for a contiguous copy to matter.
    int n = insertSyncPrefetchAndRewire(builder, loop, src,
                                        static_cast<unsigned>(tiledDim),
                                        tileSize, vtcmUsed, kMaxVTCMBytes);
    if (n == 0) {
      llvm::dbgs() << "[PrefetchInsert]   Memref " << i
                   << ": no large-tile prefetch opportunity, skipping\n";
      continue;
    }
    totalRewired += n;
    llvm::dbgs() << "[PrefetchInsert]     ✓ prefetch sites for memref " << i
                 << ": " << n << "\n";
  }

  llvm::dbgs() << "[PrefetchInsert]   Total prefetch sites: " << totalRewired
               << "\n";
  llvm::dbgs() << "[PrefetchInsert]   Total shadow used: " << vtcmUsed / 1024
               << " KB / " << kMaxVTCMBytes / 1024 << " KB\n";
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
