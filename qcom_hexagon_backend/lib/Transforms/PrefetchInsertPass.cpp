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

#include <algorithm>
#include <limits>
#include <optional>

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

enum class TransformMode {
  Native,
  SyncInSitu,
  AsyncInSitu,
  Persistent,
};

struct TransformDecision {
  TransformMode mode = TransformMode::Native;
  int64_t score = 0;
  int64_t usefulTiles = 0;
  int64_t outerReuse = -1;
  bool persistentCandidate = false;
};

struct TransformStats {
  int64_t native = 0;
  int64_t sync = 0;
  int64_t async = 0;
  int64_t persistentCandidates = 0;
  int64_t persistent = 0;
  int64_t dequant = 0;
};

struct KvCachePrefetchStats {
  int64_t sites = 0;
  int64_t hints = 0;
  int64_t pages = 0;
  int64_t bytes = 0;
  int64_t directLayoutSites = 0;
  int64_t vtcmStages = 0;
  int64_t asyncStages = 0;
  int64_t hoistedSites = 0;
  int64_t rejectedProducedSites = 0;
};

static Operation *findLastDpsWriterBefore(Value buffer,
                                          Operation *consumer) {
  Operation *last = nullptr;
  for (Operation *user : buffer.getUsers()) {
    if (user->getBlock() != consumer->getBlock() ||
        !user->isBeforeInBlock(consumer))
      continue;
    auto dps = dyn_cast<DestinationStyleOpInterface>(user);
    if (!dps)
      continue;
    bool writesBuffer = llvm::any_of(
        dps.getDpsInitsMutable(),
        [&](OpOperand &init) { return init.get() == buffer; });
    if (writesBuffer && (!last || last->isBeforeInBlock(user)))
      last = user;
  }
  return last;
}

static StringRef stringifyTransformMode(TransformMode mode) {
  switch (mode) {
  case TransformMode::Native:
    return "native";
  case TransformMode::SyncInSitu:
    return "sync";
  case TransformMode::AsyncInSitu:
    return "async";
  case TransformMode::Persistent:
    return "persistent";
  }
  llvm_unreachable("unknown transform mode");
}

static std::optional<int64_t> getConstantInt(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

static std::optional<int64_t> getStaticTripCount(scf::ForOp loop) {
  auto lower = getConstantInt(loop.getLowerBound());
  auto upper = getConstantInt(loop.getUpperBound());
  auto step = getConstantInt(loop.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  if (*upper <= *lower)
    return 0;
  return (*upper - *lower + *step - 1) / *step;
}

static int64_t estimateOuterReuse(scf::ForOp loop) {
  int64_t reuse = 1;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto outer = dyn_cast<scf::ForOp>(parent);
    if (!outer)
      continue;
    auto trips = getStaticTripCount(outer);
    if (!trips)
      return -1;
    if (*trips != 0 &&
        reuse > std::numeric_limits<int64_t>::max() / *trips)
      return -1;
    reuse *= *trips;
  }
  return reuse;
}

/// A deliberately conservative first cost model. Native HexKL wins for a
/// one-to-one synchronous weight transform. Async is selected only when enough
/// K tiles exist to amortize queue, semaphore, and staging costs. High outer
/// reuse is reported for the future persistent-layout implementation but does
/// not enable the previously regressing loop-interchange path.
static TransformDecision decideWeightTransform(scf::ForOp loop,
                                               MemRefType weightType,
                                               int lookahead,
                                               bool enablePersistentWhCache) {
  TransformDecision decision;
  auto trips = getStaticTripCount(loop);
  decision.outerReuse = estimateOuterReuse(loop);
  if (!trips || !weightType || !weightType.hasStaticShape() ||
      weightType.getRank() != 2)
    return decision;

  decision.usefulTiles =
      std::min(weightType.getShape()[0], weightType.getShape()[1]) / 32;
  decision.persistentCandidate =
      decision.outerReuse < 0 || decision.outerReuse >= 4;

  // Approximate saved/hidden work in arbitrary stable cost units:
  // 100 per useful tile, versus ~800 units of queue/sync/staging overhead.
  constexpr int64_t kAsyncFixedCost = 800;
  constexpr int64_t kBenefitPerTile = 100;
  decision.score =
      decision.usefulTiles * kBenefitPerTile - kAsyncFixedCost;
  if (enablePersistentWhCache && decision.persistentCandidate)
    decision.mode = TransformMode::Persistent;
  else if (lookahead >= 1 && *trips >= 8 && decision.usefulTiles >= 8 &&
      decision.score >= 0)
    decision.mode = TransformMode::AsyncInSitu;
  else if (lookahead == 0)
    decision.mode = TransformMode::SyncInSitu;
  return decision;
}

static void annotateTransformDecision(Operation *op,
                                      const TransformDecision &decision) {
  Builder builder(op->getContext());
  op->setAttr("omni_fetch.transform_mode",
              builder.getStringAttr(stringifyTransformMode(decision.mode)));
  op->setAttr("omni_fetch.transform_score",
              builder.getI64IntegerAttr(decision.score));
  op->setAttr("omni_fetch.transform_useful_tiles",
              builder.getI64IntegerAttr(decision.usefulTiles));
  op->setAttr("omni_fetch.transform_outer_reuse",
              builder.getI64IntegerAttr(decision.outerReuse));
  if (decision.persistentCandidate)
    op->setAttr("omni_fetch.persistent_candidate", builder.getUnitAttr());
}

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
      // HexKL high-level / micro ops that read the tile.
      if (isa<hexkl::MatmulOp, hexkl::MicroHMXMmF16Op,
              hexkl::MicroHMXCopySubmatrixToF16Op,
              hexkl::MicroHMXRmToWhF16Op,
              hexkl::MicroHMXRmToAhF16Op>(su)) {
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
                               DenseI32ArrayAttr{}, ValueRange{});
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
  if (tileViews.empty()) {
    int svCount = 0;
    for (Operation *user : src.getUsers()) {
      if (!loop->isAncestor(user))
        continue;
      if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
        ++svCount;
        llvm::errs() << "[PrefetchInsert]     subview " << sv.getType()
                     << " users:";
        for (Operation *su : sv.getResult().getUsers())
          llvm::errs() << " " << su->getName();
        llvm::errs() << "\n";
      } else {
        llvm::errs() << "[PrefetchInsert]     non-subview user: "
                     << user->getName() << "\n";
      }
    }
    llvm::errs() << "[PrefetchInsert]     no useful tile subviews (raw sv="
                 << svCount << ")\n";
    return 0;
  }

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
  // HexKL micro-tiles are 32x32xf16 = 2KB; allow those when a HexKL consumer
  // is present.  Keep a higher floor for pure HVX strips.
  const bool hexklConsumer = llvm::any_of(tileViews, [&](memref::SubViewOp sv) {
    for (Operation *user : sv.getResult().getUsers()) {
      if (isa<hexkl::MatmulOp, hexkl::MicroHMXMmF16Op,
              hexkl::MicroHMXCopySubmatrixToF16Op,
              hexkl::MicroHMXRmToWhF16Op>(user))
        return true;
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
        for (OpOperand &operand : user->getOpOperands()) {
          if (operand.get() == sv.getResult() && !dps.isDpsInit(&operand) &&
              isa<hexkl::MatmulOp>(user))
            return true;
        }
      }
    }
    return false;
  });
  const int64_t kMinCopyBytes = hexklConsumer ? 2048 : 4096;
  if (tileBytes < kMinCopyBytes) {
    llvm::errs() << "[PrefetchInsert]     skip sync: tileBytes=" << tileBytes
                 << " < min=" << kMinCopyBytes
                 << " shape=" << refTy << "\n";
    return 0;
  }
  if (vtcmUsed + tileBytes > kMaxVTCMBytes) {
    llvm::errs() << "[PrefetchInsert]     skip sync: VTCM budget\n";
    return 0;
  }

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
                               DenseI32ArrayAttr{}, ValueRange{});

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
// HexKL MicroHMX helpers
//===----------------------------------------------------------------------===//

/// True if `op` (or anything nested in it) contains HexKL micro / matmul ops.
static bool containsHexKLCompute(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) {
    if (llvm::isa<hexkl::MicroHMXMmF16Op, hexkl::MicroHMXSetupAccReadF16Op,
                  hexkl::MicroHMXCopySubmatrixToF16Op,
                  hexkl::MicroHMXRmToWhF16Op, hexkl::MicroHMXRmToAhF16Op,
                  hexkl::MatmulOp>(inner)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// View a 32×32 f16 tile into the HexKL i8 VTCM slab at a byte offset.
/// Memory space must match `vtcm` (hexagonmem Alloc uses space 1).
static Value createVtcmF16TileView(OpBuilder &b, Location loc, Value vtcm,
                                   Value byteOffI32) {
  auto vtcmTy = cast<MemRefType>(vtcm.getType());
  Value byteOff =
      b.create<arith::IndexCastOp>(loc, b.getIndexType(), byteOffI32);
  auto tileTy = MemRefType::get(ArrayRef<int64_t>{32, 32}, b.getF16Type(),
                                /*layout=*/MemRefLayoutAttrInterface{},
                                vtcmTy.getMemorySpace());
  return b.create<memref::ViewOp>(loc, tileTy, vtcm, byteOff, ValueRange{})
      .getResult();
}

/// Build a 32×32 subview of a rank-2 f16 memref at (tileRow, tileCol).
static Value createDdrTileSubview(OpBuilder &b, Location loc, Value src,
                                  Value tileRow, Value tileCol) {
  Value c32 = b.create<arith::ConstantIndexOp>(loc, 32);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value rowIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), tileRow);
  Value colIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), tileCol);
  Value rowOff = b.create<arith::MulIOp>(loc, rowIdx, c32);
  Value colOff = b.create<arith::MulIOp>(loc, colIdx, c32);
  SmallVector<OpFoldResult> offsets = {rowOff, colOff};
  SmallVector<OpFoldResult> sizes = {c32, c32};
  SmallVector<OpFoldResult> strides = {c1, c1};
  return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides)
      .getResult();
}

/// Insert one contiguous L2 hint per leading K/V stream. Attention tensors use
/// [..., sequence, head_dim]; keeping every leading stream separate avoids
/// fetching through padding while coalescing all logically adjacent cache
/// pages into one hardware request. `kvCachePageTokens` remains visible in the
/// page count even when adjacent pages are coalesced into one hint.
static KvCachePrefetchStats
insertKvCachePrefetchHints(func::FuncOp func, int64_t kvCachePageTokens,
                           bool stageInVtcm, bool enableAsyncOverlap) {
  KvCachePrefetchStats stats;
  DenseMap<Value, Value> stagedBuffers;
  if (kvCachePageTokens <= 0) {
    func.emitWarning() << "invalid KV cache page size " << kvCachePageTokens
                       << "; item 7 disabled for this function";
    return stats;
  }

  struct KvConsumer {
    Operation *op;
    Value src;
    OpOperand *replaceOperand;
  };
  SmallVector<KvConsumer> consumers;
  int64_t markedLinalg = 0;
  int64_t markedVector = 0;
  func.walk([&](linalg::LinalgOp op) {
    if (!op->hasAttr("omni_fetch.kv_cache_role"))
      return;
    auto operandAttr =
        op->getAttrOfType<IntegerAttr>("omni_fetch.kv_cache_operand");
    if (!operandAttr)
      return;
    int64_t index = operandAttr.getInt();
    if (index < 0 || index >= static_cast<int64_t>(op.getDpsInputs().size()))
      return;
    consumers.push_back(
        {op, op.getDpsInputs()[index], op.getDpsInputOperand(index)});
    ++markedLinalg;
  });
  // Hexagon vectorization replaces the annotated contraction with transfer
  // reads before one-shot bufferization. The vectorizer propagates item-7
  // identity to the K/V read so it remains recoverable on final memref IR.
  func.walk([&](vector::TransferReadOp read) {
    if (read->hasAttr("omni_fetch.kv_cache_role"))
      consumers.push_back({read, read.getBase(), nullptr});
    if (read->hasAttr("omni_fetch.kv_cache_role"))
      ++markedVector;
  });
  int64_t markedLoops = 0;
  func.walk([&](scf::ForOp loop) {
    if (!loop->hasAttr("omni_fetch.kv_cache_role"))
      return;
    Value src;
    Operation *sourceConsumer = nullptr;
    loop.walk([&](linalg::LinalgOp nested) {
      if (src || nested.getNumReductionLoops() == 0 ||
          nested.getDpsInputs().size() < 2)
        return;
      src = nested.getDpsInputs()[1];
      sourceConsumer = nested;
    });
    if (!src) {
      SmallVector<vector::TransferReadOp> reads;
      loop.walk([&](vector::TransferReadOp read) { reads.push_back(read); });
      if (reads.size() >= 2) {
        src = reads[1].getBase();
        sourceConsumer = reads[1];
      }
    }
    if (!src)
      return;
    // Insert before the first concrete reader, while retaining role/layout
    // from the durable loop carrier. No operand replacement is needed for an
    // L2-only hint.
    consumers.push_back({sourceConsumer ? sourceConsumer : loop.getOperation(),
                         src, nullptr});
    consumers.back().op->setAttr(
        "omni_fetch.kv_cache_role",
        loop->getAttr("omni_fetch.kv_cache_role"));
    if (Attribute layout = loop->getAttr("omni_fetch.kv_cache_layout"))
      consumers.back().op->setAttr("omni_fetch.kv_cache_layout", layout);
    ++markedLoops;
  });
  llvm::errs() << "[KVPropagation] final_linalg=" << markedLinalg
               << " final_vector_reads=" << markedVector
               << " final_loops=" << markedLoops << "\n";

  SmallVector<std::pair<Value, StringRef>> emitted;

  for (KvConsumer candidate : consumers) {
    Operation *consumer = candidate.op;
    auto role =
        consumer->getAttrOfType<StringAttr>("omni_fetch.kv_cache_role");
    auto layout =
        consumer->getAttrOfType<StringAttr>("omni_fetch.kv_cache_layout");
    if (!role)
      continue;

    Value src = candidate.src;
    while (true) {
      if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
        src = subview.getSource();
        continue;
      }
      if (auto cast = src.getDefiningOp<memref::CastOp>()) {
        src = cast.getSource();
        continue;
      }
      break;
    }
    auto srcType = dyn_cast<MemRefType>(src.getType());
    if (!srcType || !srcType.hasStaticShape() || srcType.getRank() < 2 ||
        srcType.getMemorySpaceAsInt() != 0 ||
        !isa<FloatType>(srcType.getElementType()))
      continue;

    // Item 7 is a future-data optimization only for K/V state that already
    // exists when this invocation starts (for example decode past_key_values).
    // Eager prefill K/V is produced inside this same function immediately
    // before attention consumes it: prefetching after production is redundant,
    // while moving the hint before production is causally invalid. Admit only
    // entry-block arguments until an explicit decode-cache ABI/metadata marks
    // other storage as persistent.
    auto blockArg = dyn_cast<BlockArgument>(src);
    if (!blockArg || blockArg.getOwner() != &func.getBody().front()) {
      ++stats.rejectedProducedSites;
      continue;
    }

    // A tiled contraction can contain several reads of the same base stream.
    // One prefetch before its first read is sufficient and avoids turning
    // item 7 into per-vector request spam.
    if (llvm::is_contained(emitted,
                           std::make_pair(src, role.getValue())))
      continue;
    emitted.emplace_back(src, role.getValue());

    ArrayRef<int64_t> shape = srcType.getShape();
    int64_t rank = srcType.getRank();
    bool isBshd = layout && layout.getValue() == "bshd" && rank == 4;
    bool isShd = layout && layout.getValue() == "shd" && rank == 3;
    int64_t seqTokens = (isBshd || isShd) ? shape[isBshd ? 1 : 0]
                                          : shape[rank - 2];
    int64_t headDim = shape[rank - 1];
    if (seqTokens <= 0 || headDim <= 0)
      continue;

    int64_t streams = 1;
    bool reasonable = true;
    if (isBshd) {
      if (shape[0] <= 0 || shape[2] <= 0 ||
          shape[0] > 128 / shape[2])
        reasonable = false;
      else
        streams = shape[0] * shape[2];
    } else if (isShd) {
      if (shape[1] <= 0 || shape[1] > 128)
        reasonable = false;
      else
        streams = shape[1];
    } else {
      for (int64_t dim = 0; dim < rank - 2; ++dim) {
        if (shape[dim] <= 0 || streams > 128 / shape[dim]) {
          reasonable = false;
          break;
        }
        streams *= shape[dim];
      }
    }
    if (!reasonable)
      continue;

    int64_t elemBytes =
        srcType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t pagesPerStream =
        (seqTokens + kvCachePageTokens - 1) / kvCachePageTokens;
    OpBuilder b(consumer);
    Location loc = consumer->getLoc();

    if (stageInVtcm) {
      if (!candidate.replaceOperand)
        continue;
      Value shadow = stagedBuffers.lookup(src);
      if (!shadow) {
        auto shadowTy = MemRefType::get(
            srcType.getShape(), srcType.getElementType(),
            /*layout=*/MemRefLayoutAttrInterface{},
            /*memorySpace=*/IntegerAttr::get(
                IntegerType::get(b.getContext(), 64), /*VTCM=*/1));

        Operation *producer = findLastDpsWriterBefore(src, consumer);
        if (producer)
          b.setInsertionPointAfter(producer);
        else
          b.setInsertionPoint(consumer);
        shadow = b.create<memref::AllocOp>(loc, shadowTy);

        if (enableAsyncOverlap) {
          auto tagTy = MemRefType::get({1}, b.getI32Type());
          Value tag = b.create<memref::AllocOp>(loc, tagTy);
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          Value numElements = b.create<arith::ConstantIndexOp>(
              loc, srcType.getNumElements());
          SmallVector<Value> zeroIndices(srcType.getRank(), zero);
          SmallVector<Value> tagIndex{zero};
          b.create<memref::DmaStartOp>(
              loc, src, zeroIndices, shadow, zeroIndices, numElements, tag,
              tagIndex);
          b.setInsertionPoint(consumer);
          b.create<memref::DmaWaitOp>(loc, tag, tagIndex, numElements);
          ++stats.asyncStages;
        } else {
          b.create<memref::CopyOp>(loc, src, shadow);
        }

        stagedBuffers[src] = shadow;
        ++stats.vtcmStages;
        stats.bytes += streams * seqTokens * headDim * elemBytes;
      }
      // Replacing a tiled vector subview with the full VTCM allocation needs a
      // matching subview reconstruction. Keep the initial causal experiment
      // L2-only; the DMA/VTCM variant is admitted for untiled linalg consumers.
      candidate.replaceOperand->set(shadow);
      consumer->setAttr("omni_fetch.kv_layout",
                        b.getStringAttr(enableAsyncOverlap
                                            ? "vtcm_dma_overlapped"
                                            : "vtcm_staged"));
      ++stats.sites;
      continue;
    }

    // Vectorization can leave the semantic attention consumer inside several
    // strip-mined loops. Inserting at that consumer turns one logical K/V
    // prefetch into a command on every dynamic vector tile. Hoist across every
    // loop for which the recovered base buffer is invariant; this gives each
    // (buffer, role) pair one preheader issue. If the source is produced inside
    // a loop, stop at that loop rather than moving the hint before its data is
    // available.
    Operation *hintAnchor = consumer;
    Operation *scope = consumer;
    while (auto loop = scope->getParentOfType<scf::ForOp>()) {
      if (!loop.isDefinedOutsideOfLoop(src))
        break;
      hintAnchor = loop;
      scope = loop;
    }
    if (hintAnchor != consumer) {
      b.setInsertionPoint(hintAnchor);
      ++stats.hoistedSites;
    }

    // V73 l2fetch cannot generate across the 4-KiB page containing its start
    // address. A whole [sequence, head_dim] stream therefore degenerates into
    // a clipped command (and large strided geometries have caused DSP faults
    // on full vector graphs). Warm only the first aligned demand line here.
    // Subsequent pages require a future loop-progress-aware scheduler rather
    // than pretending that one preheader command can cover the full stream.
    constexpr int64_t kKvHintBudgetBytes = 128;
    int64_t budgetHeadDim =
        std::min(headDim, kKvHintBudgetBytes / elemBytes);
    if (budgetHeadDim <= 0)
      continue;
    constexpr int64_t budgetSeqTokens = 1;

    for (int64_t linearStream = 0; linearStream < streams; ++linearStream) {
      SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
      sizes.reserve(rank);
      if (isBshd) {
        offsets[0] = b.getIndexAttr(linearStream / shape[2]);
        offsets[2] = b.getIndexAttr(linearStream % shape[2]);
        sizes.push_back(b.getIndexAttr(1));
        sizes.push_back(b.getIndexAttr(budgetSeqTokens));
        sizes.push_back(b.getIndexAttr(1));
        sizes.push_back(b.getIndexAttr(budgetHeadDim));
      } else if (isShd) {
        offsets[1] = b.getIndexAttr(linearStream);
        sizes.push_back(b.getIndexAttr(budgetSeqTokens));
        sizes.push_back(b.getIndexAttr(1));
        sizes.push_back(b.getIndexAttr(budgetHeadDim));
      } else {
        int64_t remaining = linearStream;
        for (int64_t dim = rank - 3; dim >= 0; --dim) {
          offsets[dim] = b.getIndexAttr(remaining % shape[dim]);
          remaining /= shape[dim];
        }
        for (int64_t dim = 0; dim < rank - 2; ++dim)
          sizes.push_back(b.getIndexAttr(1));
        sizes.push_back(b.getIndexAttr(budgetSeqTokens));
        sizes.push_back(b.getIndexAttr(budgetHeadDim));
      }

      Value stream = b.create<memref::SubViewOp>(
          loc, src, offsets, sizes, strides);
      b.create<L2HintOp>(loc, stream, /*lookahead=*/1);
      ++stats.hints;
    }

    consumer->setAttr("omni_fetch.kv_page_tokens",
                      b.getI64IntegerAttr(kvCachePageTokens));
    consumer->setAttr("omni_fetch.kv_pages",
                      b.getI64IntegerAttr(streams * pagesPerStream));
    consumer->setAttr("omni_fetch.kv_layout",
                      b.getStringAttr("budgeted_first_demand_line"));
    ++stats.sites;
    ++stats.directLayoutSites;
    stats.pages += streams * pagesPerStream;
    stats.bytes += streams * budgetSeqTokens * budgetHeadDim * elemBytes;
  }

  Builder b(func.getContext());
  func->setAttr("omni_fetch.kv_prefetch_sites",
                b.getI64IntegerAttr(stats.sites));
  func->setAttr("omni_fetch.kv_prefetch_hints",
                b.getI64IntegerAttr(stats.hints));
  func->setAttr("omni_fetch.kv_prefetch_pages",
                b.getI64IntegerAttr(stats.pages));
  func->setAttr("omni_fetch.kv_prefetch_bytes",
                b.getI64IntegerAttr(stats.bytes));
  func->setAttr("omni_fetch.kv_direct_layout_sites",
                b.getI64IntegerAttr(stats.directLayoutSites));
  func->setAttr("omni_fetch.kv_vtcm_stages",
                b.getI64IntegerAttr(stats.vtcmStages));
  func->setAttr("omni_fetch.kv_async_stages",
                b.getI64IntegerAttr(stats.asyncStages));
  func->setAttr("omni_fetch.kv_hoisted_sites",
                b.getI64IntegerAttr(stats.hoistedSites));
  func->setAttr("omni_fetch.kv_rejected_produced_sites",
                b.getI64IntegerAttr(stats.rejectedProducedSites));
  llvm::errs() << "[KVCachePrefetch] function=" << func.getName()
               << " sites=" << stats.sites << " hints=" << stats.hints
               << " pages=" << stats.pages << " bytes=" << stats.bytes
               << " vtcm_stages=" << stats.vtcmStages
               << " async_stages=" << stats.asyncStages
               << " hoisted_sites=" << stats.hoistedSites
               << " rejected_produced_sites=" << stats.rejectedProducedSites
               << " page_tokens=" << kvCachePageTokens << "\n";
  return stats;
}

/// HexKL OmniFetch insertion.
/// - layoutAware=false: L2Hint warmup (Phase 1; no HexKL op removal)
/// - layoutAware=true:  replace RmToWh / (Copy+RmToAh) with in-situ HMX layout
///   prefetch into the VTCM tile slot (Phase 2a).  With lookahead>=1, emit a
///   prologue prefetch and a body prefetch of tile i+lookahead into the idle
///   ping-pong weight slot (Phase 2b software pipeline; runtime async DMA
///   overlaps the transfer with Mm when lookahead>0).
static int insertHexKLMicroPrefetchHints(OpBuilder &builder, scf::ForOp loop,
                                         bool enableLayoutAware,
                                         int lookahead, bool enableDmaToVtcm,
                                         bool enablePersistentWhCache,
                                         bool enableTwoDimPipeline,
                                         bool enableDequantReshape,
                                         TransformStats &stats) {
  if (!enableLayoutAware) {
    // ----- Phase 1 path: L2 hints only -----
    SmallVector<Operation *> ddrLoads;
    for (Operation &op : *loop.getBody()) {
      if (isa<hexkl::MicroHMXCopySubmatrixToF16Op, hexkl::MicroHMXRmToWhF16Op>(
              &op))
        ddrLoads.push_back(&op);
    }
    if (ddrLoads.empty())
      return 0;

    int inserted = 0;
    for (Operation *op : ddrLoads) {
      Value src, tileRow, tileCol;
      if (auto copy = dyn_cast<hexkl::MicroHMXCopySubmatrixToF16Op>(op)) {
        src = copy.getSrc();
        tileRow = copy.getTileRow();
        tileCol = copy.getTileCol();
      } else {
        auto wh = cast<hexkl::MicroHMXRmToWhF16Op>(op);
        src = wh.getSrc();
        tileRow = wh.getTileRow();
        tileCol = wh.getTileCol();
      }
      auto srcTy = dyn_cast<MemRefType>(src.getType());
      if (!srcTy || srcTy.getRank() != 2 || !srcTy.getElementType().isF16())
        continue;

      Location loc = op->getLoc();
      OpBuilder b(op);
      Value tileSv = createDdrTileSubview(b, loc, src, tileRow, tileCol);
      auto scratchTy = MemRefType::get(
          ArrayRef<int64_t>{1}, srcTy.getElementType(),
          /*layout=*/MemRefLayoutAttrInterface{},
          /*memorySpace=*/IntegerAttr::get(
              IntegerType::get(b.getContext(), 64), 0));
      Value scratch;
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(loop);
        scratch = builder.create<memref::AllocOp>(loop.getLoc(), scratchTy);
      }
      b.create<PrefetchInSituOp>(loc, tileSv, scratch, LayoutTransform::L2Hint,
                                 /*lookahead=*/1, DenseI32ArrayAttr{},
                                 ValueRange{});
      ++inserted;
    }
    if (inserted)
      llvm::errs() << "[PrefetchInsert]     HexKL L2-hint sites: " << inserted
                   << "\n";
    return inserted;
  }

  // ----- Phase 2a/2b: layout-aware path -----
  // Weight: replace RmToWh with prefetch_in_situ → hexkl_micro WH layout.
  // Phase 2b software pipeline (lookahead>=1, dual weight slots):
  //   always sync-fill current (correctness)
  //   async kick of tile kt+1 into the idle slot (dma2d→stage; WH in wait)
  // Activation: replace CopySubmatrix+RmToAh with HexKL-accurate HMXActivation
  //   prefetch into the VTCM slab (sync only; no Mm overlap in this loop).
  int inserted = 0;
  int erased = 0;
  SmallVector<hexkl::MicroHMXRmToWhF16Op> whOps;
  for (Operation &op : *loop.getBody())
    if (auto wh = dyn_cast<hexkl::MicroHMXRmToWhF16Op>(&op))
      whOps.push_back(wh);

  for (auto wh : whOps) {
    Location loc = wh.getLoc();
    hexkl::MicroHMXMmF16Op mmOp;
    for (Operation *op = wh->getNextNode(); op; op = op->getNextNode()) {
      if (auto mm = dyn_cast<hexkl::MicroHMXMmF16Op>(op)) {
        if (mm.getHmxBlock() == wh.getHmxBlock() &&
            mm.getWeightOffset() == wh.getWeightOffset()) {
          mmOp = mm;
          break;
        }
      }
    }

    Value srcMem = wh.getSrc();
    Value vtcmMem = wh.getHmxBlock();
    Value curOff = wh.getWeightOffset();
    Value ktVal = wh.getTileRow();
    Value colVal = wh.getTileCol();
    Value wtCols = wh.getWtCols();

    OpBuilder b(wh);
    auto wtTy = dyn_cast<MemRefType>(srcMem.getType());
    TransformDecision decision =
        decideWeightTransform(loop, wtTy, lookahead,
                              enablePersistentWhCache);
    // Item 5 composes with item 4: profitable sites use the load/reshape/
    // compute pipeline, while the runtime cache services or populates each
    // pipelined tile using the compiler-assigned site identity.
    // The explicit item-5 gate also enables short model-debug loops (two or
    // more K tiles). This is essential for correctness/performance gating on
    // the repository's model runners, whose reduced hidden size would
    // otherwise exercise no pipeline at all. The default cost model remains
    // unchanged when the item-5 gate is off.
    if (enableTwoDimPipeline && mmOp && wtTy && wtTy.hasStaticShape() &&
        getStaticTripCount(loop).value_or(0) >= 2 &&
        decision.usefulTiles >= 2)
      decision.mode = TransformMode::AsyncInSitu;
    // Item 8 owns a persistent W8 tile cache in the runtime.  Its first
    // implementation keeps production synchronous: a warm hit reads 1 KiB
    // of W8 data, dequantizes directly into a 2 KiB tile, and immediately
    // produces WH.  It must not enter the older FP16 DMA descriptor path.
    if (enableDequantReshape)
      decision.mode = TransformMode::SyncInSitu;
    if (decision.mode == TransformMode::AsyncInSitu && !mmOp)
      decision.mode = TransformMode::SyncInSitu;
    annotateTransformDecision(wh, decision);
    stats.persistentCandidates += decision.persistentCandidate;

    if (decision.mode == TransformMode::Native) {
      ++stats.native;
      llvm::errs() << "[TransformCostModel] kind=weight mode=native"
                   << " tiles=" << decision.usefulTiles
                   << " outer_reuse=" << decision.outerReuse
                   << " score=" << decision.score
                   << (decision.persistentCandidate
                           ? " persistent_candidate=1\n"
                           : "\n");
      continue;
    }

    if (decision.mode == TransformMode::AsyncInSitu && mmOp) {
      ++stats.async;
      auto i32Ty = b.getI32Type();
      Value c0 = b.create<arith::ConstantIntOp>(loc, i32Ty, 0);
      Value c1 = b.create<arith::ConstantIntOp>(loc, i32Ty, 1);
      Value c2 = b.create<arith::ConstantIntOp>(loc, i32Ty, 2);
      Value c4096 = b.create<arith::ConstantIntOp>(loc, i32Ty, 4096);
      Value neg4096 = b.create<arith::ConstantIntOp>(loc, i32Ty, -4096);

      Value ub = loop.getUpperBound();
      Value ubI32 = b.create<arith::IndexCastOp>(loc, i32Ty, ub);

      // Optional VTCM DMA stage at flatOff=(kTiles+2)*4096.  When disabled,
      // runtime packs into DDR omni_stage instead (tile_params size 4).
      SmallVector<Value, 5> syncParams = {ktVal, colVal, wtCols, curOff};
      Value stageOff;
      if (enableDmaToVtcm) {
        Value kTilesPlus2 = b.create<arith::AddIOp>(
            loc, ubI32, b.create<arith::ConstantIntOp>(loc, i32Ty, 2));
        stageOff = b.create<arith::MulIOp>(loc, kTilesPlus2, c4096);
        syncParams.push_back(stageOff);
      }

      Value hybridSiteId;
      if (enableTwoDimPipeline && enablePersistentWhCache &&
          decision.persistentCandidate) {
        if (!enableDmaToVtcm)
          syncParams.push_back(
              b.create<arith::ConstantIntOp>(loc, i32Ty, -1));
        hybridSiteId = b.create<arith::ConstantIntOp>(
            loc, i32Ty, stats.persistent++);
        syncParams.push_back(hybridSiteId);
      }

      auto emitSyncCurrent = [&](OpBuilder &syncBuilder, Location syncLoc) {
      auto syncPrefetch = syncBuilder.create<PrefetchInSituOp>(
            syncLoc, srcMem, vtcmMem,
            enableDequantReshape ? LayoutTransform::HMXWeightDequantI8
                                 : LayoutTransform::HMXWeight,
            hybridSiteId ? /*persistent cache*/ -1 : 0,
            DenseI32ArrayAttr{}, syncParams);
        annotateTransformDecision(syncPrefetch, decision);
      };

      if (enableTwoDimPipeline) {
        // Bootstrap only tile 0. Later current tiles were loaded and reshaped
        // by the previous iteration; repeating the synchronous WH transform
        // here would serialize and nullify the pipeline.
        Value lbI32 =
            b.create<arith::IndexCastOp>(loc, i32Ty, loop.getLowerBound());
        Value isFirst = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, ktVal, lbI32);
        b.create<scf::IfOp>(
            loc, isFirst, [&](OpBuilder &thenBuilder, Location thenLoc) {
              emitSyncCurrent(thenBuilder, thenLoc);
              thenBuilder.create<scf::YieldOp>(thenLoc);
            });
      } else {
        emitSyncCurrent(b, loc);
      }
      ++inserted;

      Value nextKtRaw = b.create<arith::AddIOp>(loc, ktVal, c1);
      Value hasNext = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, nextKtRaw, ubI32);
      b.create<scf::IfOp>(
          loc, hasNext,
          [&](OpBuilder &thenBuilder, Location thenLoc) {
            Value phase =
                thenBuilder.create<arith::RemUIOp>(thenLoc, ktVal, c2);
            Value isEven = thenBuilder.create<arith::CmpIOp>(
                thenLoc, arith::CmpIPredicate::eq, phase, c0);
            Value delta = thenBuilder.create<arith::SelectOp>(
                thenLoc, isEven, c4096, neg4096);
            Value nextOff = thenBuilder.create<arith::AddIOp>(
                thenLoc, curOff, delta);

            SmallVector<Value, 5> asyncParams = {nextKtRaw, colVal, wtCols,
                                                 nextOff};
            if (enableDmaToVtcm)
              asyncParams.push_back(stageOff);
            if (hybridSiteId) {
              if (!enableDmaToVtcm)
                asyncParams.push_back(thenBuilder.create<arith::ConstantIntOp>(
                    thenLoc, i32Ty, -1));
              asyncParams.push_back(hybridSiteId);
            }
            auto asyncPrefetch = thenBuilder.create<PrefetchInSituOp>(
                thenLoc, srcMem, vtcmMem,
                enableDequantReshape ? LayoutTransform::HMXWeightDequantI8
                                     : LayoutTransform::HMXWeight,
                /*lookahead=*/1, DenseI32ArrayAttr{}, asyncParams);
            annotateTransformDecision(asyncPrefetch, decision);
            thenBuilder.create<scf::YieldOp>(thenLoc);
          });
      ++inserted;
    } else {
      int selectedLookahead = 0;
      SmallVector<Value, 6> persistentParams = {ktVal, colVal, wtCols, curOff};
      if (enableDequantReshape) {
        Value noStage =
            b.create<arith::ConstantIntOp>(loc, b.getI32Type(), -1);
        Value siteId = b.create<arith::ConstantIntOp>(
            loc, b.getI32Type(), stats.dequant++);
        persistentParams.push_back(noStage);
        persistentParams.push_back(siteId);
        ++stats.sync;
      } else if (decision.mode == TransformMode::Persistent) {
        Value noStage =
            b.create<arith::ConstantIntOp>(loc, b.getI32Type(), -1);
        Value siteId = b.create<arith::ConstantIntOp>(
            loc, b.getI32Type(), stats.persistent);
        persistentParams.push_back(noStage);
        persistentParams.push_back(siteId);
        ++stats.persistent;
        selectedLookahead = -1;
      } else {
        ++stats.sync;
      }
      auto syncPrefetch = b.create<PrefetchInSituOp>(
          loc, srcMem, vtcmMem,
          enableDequantReshape ? LayoutTransform::HMXWeightDequantI8
                               : LayoutTransform::HMXWeight,
          selectedLookahead, DenseI32ArrayAttr{},
          persistentParams);
      annotateTransformDecision(syncPrefetch, decision);
      ++inserted;
    }
    wh->erase();
    ++erased;
  }

  // Activation fusion: CopySubmatrix + RmToAh → prefetch_in_situ(HMXActivation).
  SmallVector<hexkl::MicroHMXCopySubmatrixToF16Op> copyOps;
  for (Operation &op : *loop.getBody())
    if (auto c = dyn_cast<hexkl::MicroHMXCopySubmatrixToF16Op>(&op))
      copyOps.push_back(c);

  for (auto copy : copyOps) {
    hexkl::MicroHMXRmToAhF16Op rmAh;
    for (Operation *op = copy->getNextNode(); op; op = op->getNextNode()) {
      if (auto ah = dyn_cast<hexkl::MicroHMXRmToAhF16Op>(op)) {
        if (ah.getHmxBlock() == copy.getHmxBlock()) {
          rmAh = ah;
          break;
        }
      }
      if (isa<scf::YieldOp, hexkl::MicroHMXMmF16Op, hexkl::MicroHMXRmToWhF16Op,
              hexkl::MicroHMXCopySubmatrixToF16Op>(op))
        break;
    }
    if (!rmAh)
      continue;

    Location loc = copy.getLoc();
    OpBuilder b(rmAh);
    b.setInsertionPointAfter(rmAh);
    auto activationPrefetch = b.create<PrefetchInSituOp>(
        loc, copy.getSrc(), copy.getHmxBlock(), LayoutTransform::HMXActivation,
        /*lookahead=*/0, DenseI32ArrayAttr{},
        ValueRange{copy.getTileRow(), copy.getTileCol(), copy.getInputCols(),
                   rmAh.getActivationOutOffset(), rmAh.getFlatInOffset(),
                   copy.getInputRows()});
    TransformDecision activationDecision;
    activationDecision.mode = TransformMode::SyncInSitu;
    activationDecision.score = 100;
    activationDecision.outerReuse = estimateOuterReuse(loop);
    annotateTransformDecision(activationPrefetch, activationDecision);
    ++inserted;
    ++stats.sync;
    rmAh->erase();
    copy->erase();
    erased += 2;
  }

  if (inserted || erased)
    llvm::errs() << "[PrefetchInsert]     HexKL layout-fusion sites: "
                 << inserted << " erased_hexkl_ops=" << erased
                 << (enableDmaToVtcm ? " dma_to_vtcm=1\n" : "\n");
  return inserted;
}

//===----------------------------------------------------------------------===//
// Core transformation
//===----------------------------------------------------------------------===//

static void insertPrefetchForLoop(scf::ForOp loop, int lookahead,
                                  bool enableLayoutAware,
                                  bool enableDmaToVtcm,
                                  bool enablePersistentWhCache,
                                  bool enableTwoDimPipeline,
                                  bool enableDequantReshape,
                                  TransformStats &stats) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  MLIRContext *ctx = loop.getContext();

  llvm::dbgs() << "[PrefetchInsert]   Analyzing loop body...\n";

  // HexKL MicroHMX path first: warm / fuse DDR tiles that feed Copy / RmToWh.
  if (containsHexKLCompute(loop)) {
    int n = insertHexKLMicroPrefetchHints(builder, loop, enableLayoutAware,
                                          lookahead, enableDmaToVtcm,
                                          enablePersistentWhCache,
                                          enableTwoDimPipeline,
                                          enableDequantReshape, stats);
    llvm::errs() << "[PrefetchInsert] Total prefetch sites: " << n
                 << " shadow_kb=0\n";
    return;
  }

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
    // Always visible summary (Phase-0 / device smoke relies on this).
    llvm::errs() << "[PrefetchInsert] Total prefetch sites: " << totalRewired
                 << " shadow_kb=" << (vtcmUsed / 1024) << "\n";
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
                 << ", enableDmaToVtcm=" << enableDmaToVtcm
                 << ", enablePersistentWhCache="
                 << enablePersistentWhCache
                 << ", enableTwoDimPipeline=" << enableTwoDimPipeline
                 << ", enableInterLayerPrefetch=" << enableInterLayerPrefetch
                 << ", enableKvCachePrefetch=" << enableKvCachePrefetch
                 << ", kvCacheOnly=" << kvCacheOnly
                 << ", enableDequantReshape=" << enableDequantReshape
                 << ", kvCachePageTokens=" << kvCachePageTokens
                 << " (forced off in HVX insert for safety)\n";

    if (enableKvCachePrefetch)
      insertKvCachePrefetchHints(func, kvCachePageTokens, enableDmaToVtcm,
                                 enableTwoDimPipeline);

    const bool funcHasHexKL = containsHexKLCompute(func);

    if (!kvCacheOnly)
      func.walk([&](scf::ForOp loop) {
      bool hasNestedFor = false;
      loop.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
      // Default: innermost only.  Inter-layer mode also takes outer HexKL
      // loops so next-layer weights can be async-prefetched (§2.6/§4.4).
      if (hasNestedFor && !enableInterLayerPrefetch)
        return;
      if (funcHasHexKL) {
        // Prefer HexKL MicroHMX loops; skip Softmax/HVX strip loops that only
        // create tiny 1D transfer_read tiles and cannot win with sync copy.
        if (containsHexKLCompute(loop))
          candidates.push_back(loop);
      } else if (containsAcceleratorCompute(loop)) {
        if (!hasNestedFor)
          candidates.push_back(loop);
      }
      });

    llvm::errs() << "[PrefetchInsert] Found " << candidates.size()
                 << " candidate loops (hexkl_func="
                 << (funcHasHexKL ? 1 : 0) << ")\n";

    llvm::dbgs() << "[PrefetchInsert] Found " << candidates.size()
                 << " candidate loops for prefetch insertion\n";

    int loopIdx = 0;
    TransformStats stats;
    for (auto loop : candidates) {
      llvm::dbgs() << "\n[PrefetchInsert] --- Processing loop " << loopIdx++
                   << " at " << loop.getLoc() << " ---\n";
      insertPrefetchForLoop(loop, lookahead, enableLayoutAware,
                            enableDmaToVtcm, enablePersistentWhCache,
                            enableTwoDimPipeline, enableDequantReshape, stats);
    }

    Builder builder(func.getContext());
    func->setAttr("omni_fetch.cost_native_sites",
                  builder.getI64IntegerAttr(stats.native));
    func->setAttr("omni_fetch.cost_sync_sites",
                  builder.getI64IntegerAttr(stats.sync));
    func->setAttr("omni_fetch.cost_async_sites",
                  builder.getI64IntegerAttr(stats.async));
    func->setAttr("omni_fetch.cost_persistent_candidates",
                  builder.getI64IntegerAttr(stats.persistentCandidates));
    func->setAttr("omni_fetch.cost_persistent_sites",
                  builder.getI64IntegerAttr(stats.persistent));
    func->setAttr("omni_fetch.dequant_reshape_enabled",
                  builder.getBoolAttr(enableDequantReshape));
    func->setAttr("omni_fetch.dequant_reshape_sites",
                  builder.getI64IntegerAttr(stats.dequant));
    llvm::errs() << "[TransformCostModel] function=" << func.getName()
                 << " native=" << stats.native << " sync=" << stats.sync
                 << " async=" << stats.async
                 << " persistent=" << stats.persistent
                 << " persistent_candidates=" << stats.persistentCandidates
                 << " dequant=" << stats.dequant
                 << "\n";

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
