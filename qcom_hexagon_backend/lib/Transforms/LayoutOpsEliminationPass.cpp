//===- LayoutOpsEliminationPass.cpp - Eliminate redundant layout ops ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates redundant layout transformation operations when
// in-situ reshape is enabled during prefetch.
//
// When `alps.prefetch_in_situ` is used with `layout_transform != None`,
// the hardware performs the layout transformation during the DDR→VTCM
// transfer. This makes explicit transpose/permute/slice operations in the
// graph redundant.
//
// This pass:
// 1. Finds all prefetch_in_situ operations with layout transforms
// 2. Walks backward from each prefetch's src to find matching layout ops
// 3. Marks those ops as redundant
// 4. Deletes the marked ops
//
// Key design: This pass is INDEPENDENT of V-DAE. It detects prefetch_in_situ
// operations, not loops. Even without V-DAE, if other passes insert
// prefetch_in_situ operations, this pass will eliminate redundant layout ops.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Alps/IR/AlpsDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <limits>

#define DEBUG_TYPE "layout-ops-elimination"

using namespace mlir;
using namespace mlir::alps;
using namespace hexagon;

#define GEN_PASS_DEF_LAYOUTOPSELIMINATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// A layout-value site is the compiler-visible identity of producing a
/// particular HMX layout from a particular SSA source.  This first stage is
/// deliberately analysis-only: it records safe reuse/liveness evidence without
/// changing the generated data movement.
struct LayoutValueSite {
  Value root;
  LayoutTransform transform;
  SmallVector<Value> tileParams;
  int64_t id;
  int64_t occurrences = 0;
  int64_t estimatedExecutions = 1;
};

struct LayoutCarryStats {
  int64_t fusedSites = 0;
  int64_t bypassedViews = 0;
};

static bool hasStaticContiguousLayout(MemRefType type) {
  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()))
    return false;

  int64_t expectedStride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    if (strides[dim] != expectedStride || type.isDynamicDim(dim))
      return false;
    expectedStride *= type.getDimSize(dim);
  }
  return true;
}

/// Return true when a collapse only changes the descriptor of a statically
/// contiguous activation. HexKL activation prefetch consumes a flat base
/// pointer plus explicit row/column tile metadata, so the source rank does not
/// participate in runtime addressing.
static bool isContiguousActivationCollapse(memref::CollapseShapeOp collapse) {
  auto srcType = dyn_cast<MemRefType>(collapse.getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(collapse.getResult().getType());
  if (!srcType || !dstType || srcType.getRank() < 2 ||
      dstType.getRank() != 2 || !srcType.hasStaticShape() ||
      !dstType.hasStaticShape() ||
      srcType.getElementType() != dstType.getElementType() ||
      !hasStaticContiguousLayout(srcType) ||
      !hasStaticContiguousLayout(dstType))
    return false;

  return srcType.getNumElements() == dstType.getNumElements();
}

/// Carry an HMX activation layout request through address-only producer views.
/// This is intentionally narrower than generic reshape folding: changing the
/// base pointer is safe here because the HexKL activation runtime path indexes
/// the source with explicit [tile_row, tile_col, src_cols, ..., src_rows]
/// metadata and does not consume the source memref rank/strides.
static LayoutCarryStats
carryActivationLayoutToProducer(ArrayRef<PrefetchInSituOp> prefetches) {
  LayoutCarryStats stats;
  for (PrefetchInSituOp prefetch : prefetches) {
    if (prefetch.getLayoutTransform() != LayoutTransform::HMXActivation ||
        prefetch.getTileParams().size() < 6)
      continue;

    Value source = prefetch.getSrc();
    SmallVector<Operation *> bypassed;
    while (Operation *def = source.getDefiningOp()) {
      if (auto cast = dyn_cast<memref::CastOp>(def)) {
        source = cast.getSource();
        bypassed.push_back(def);
        continue;
      }
      auto collapse = dyn_cast<memref::CollapseShapeOp>(def);
      if (!collapse || !isContiguousActivationCollapse(collapse))
        break;
      source = collapse.getSrc();
      bypassed.push_back(def);
    }

    if (bypassed.empty())
      continue;

    prefetch->setOperand(0, source);
    Builder builder(prefetch.getContext());
    prefetch->setAttr("alps.layout_carried_from_producer",
                      builder.getUnitAttr());
    prefetch->setAttr("alps.layout_carried_view_depth",
                      builder.getI64IntegerAttr(bypassed.size()));
    ++stats.fusedSites;
    stats.bypassedViews += bypassed.size();

    for (Operation *view : bypassed)
      if (view->use_empty())
        view->erase();
  }
  return stats;
}

static Value findLayoutRoot(Value value) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *def = value.getDefiningOp()) {
    if (!visited.insert(def).second)
      break;
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<tensor::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

static StringRef classifyLayoutSource(Value root) {
  if (isa<BlockArgument>(root))
    return "argument";
  Operation *def = root.getDefiningOp();
  if (!def)
    return "unknown";
  if (isa<memref::GetGlobalOp>(def))
    return "constant";
  if (isa<memref::AllocOp, memref::AllocaOp>(def))
    return "allocation";
  return "produced";
}

static bool sameLayoutSite(const LayoutValueSite &site, Value root,
                           LayoutTransform transform,
                           ValueRange tileParams) {
  return site.root == root && site.transform == transform &&
         llvm::equal(site.tileParams, tileParams);
}

/// Conservatively estimate how many times an op executes from enclosing static
/// scf.for loops.  -1 means at least one enclosing loop is dynamic.
static int64_t estimateExecutions(Operation *op) {
  auto constantInt = [](Value value) -> std::optional<int64_t> {
    Attribute attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return std::nullopt;
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return std::nullopt;
    return intAttr.getInt();
  };
  int64_t executions = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    auto lower = constantInt(loop.getLowerBound());
    auto upper = constantInt(loop.getUpperBound());
    auto step = constantInt(loop.getStep());
    if (!lower || !upper || !step || *step <= 0)
      return -1;
    int64_t trips =
        *upper <= *lower ? 0 : (*upper - *lower + *step - 1) / *step;
    if (trips != 0 &&
        executions > std::numeric_limits<int64_t>::max() / trips)
      return -1;
    executions *= trips;
  }
  return executions;
}

static void annotateLayoutValueSites(FunctionOpInterface func,
                                     ArrayRef<PrefetchInSituOp> prefetches) {
  SmallVector<LayoutValueSite> sites;
  SmallVector<int64_t> opSiteIds;
  opSiteIds.reserve(prefetches.size());

  for (PrefetchInSituOp prefetch : prefetches) {
    Value root = findLayoutRoot(prefetch.getSrc());
    auto params = prefetch.getTileParams();
    int64_t estimatedExecutions = estimateExecutions(prefetch);
    auto it = llvm::find_if(sites, [&](const LayoutValueSite &site) {
      return sameLayoutSite(site, root, prefetch.getLayoutTransform(), params);
    });
    if (it == sites.end()) {
      int64_t id = static_cast<int64_t>(sites.size());
      sites.push_back(LayoutValueSite{
          root, prefetch.getLayoutTransform(), SmallVector<Value>(params), id,
          1, estimatedExecutions});
      opSiteIds.push_back(id);
    } else {
      ++it->occurrences;
      if (it->estimatedExecutions >= 0 && estimatedExecutions >= 0)
        it->estimatedExecutions += estimatedExecutions;
      else
        it->estimatedExecutions = -1;
      opSiteIds.push_back(it->id);
    }
  }

  Builder builder(func.getContext());
  int64_t reusableSites = 0;
  for (const LayoutValueSite &site : sites)
    reusableSites +=
        site.occurrences > 1 || site.estimatedExecutions > 1 ||
        site.estimatedExecutions < 0;

  for (auto [prefetch, siteId] : llvm::zip(prefetches, opSiteIds)) {
    const LayoutValueSite &site = sites[siteId];
    prefetch->setAttr("alps.layout_value_id",
                      builder.getI64IntegerAttr(site.id));
    prefetch->setAttr("alps.layout_site_occurrences",
                      builder.getI64IntegerAttr(site.occurrences));
    prefetch->setAttr("alps.layout_estimated_executions",
                      builder.getI64IntegerAttr(site.estimatedExecutions));
    prefetch->setAttr("alps.layout_source_kind",
                      builder.getStringAttr(classifyLayoutSource(site.root)));
    Value dest = prefetch->getOperand(1);
    prefetch->setAttr(
        "alps.layout_dest_users",
        builder.getI64IntegerAttr(static_cast<int64_t>(
            std::distance(dest.user_begin(), dest.user_end()))));
  }

  func->setAttr("alps.layout_value_sites",
                builder.getI64IntegerAttr(sites.size()));
  func->setAttr("alps.layout_reusable_sites",
                builder.getI64IntegerAttr(reusableSites));
  func->setAttr("alps.layout_prefetch_instances",
                builder.getI64IntegerAttr(prefetches.size()));

  llvm::errs() << "[LayoutValueAnalysis] function=" << func.getName()
               << " instances=" << prefetches.size()
               << " sites=" << sites.size()
               << " reusable_sites=" << reusableSites << "\n";
}

/// Returns true if `genericOp` is a transpose operation that matches the layout transform.
/// After bufferization, linalg.transpose becomes linalg.generic with transpose affine maps.
static bool isTransposeGeneric(linalg::GenericOp genericOp, LayoutTransform lt) {
  // Must be a copy operation (single input, single output)
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return false;

  // Check if body is just a copy (yield input)
  Block *body = genericOp.getBody();
  if (body->getOperations().size() != 1)
    return false;
  
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  
  if (yieldOp.getOperand(0) != body->getArgument(0))
    return false;

  // Check indexing maps for transpose pattern
  auto indexingMaps = genericOp.getIndexingMapsArray();
  if (indexingMaps.size() != 2)
    return false;

  AffineMap inputMap = indexingMaps[0];
  AffineMap outputMap = indexingMaps[1];

  // Output should be identity
  if (!outputMap.isIdentity())
    return false;

  // Input should be a permutation
  if (!inputMap.isPermutation())
    return false;

  // Check if permutation matches the layout transform
  if (lt == LayoutTransform::HMXWeight || lt == LayoutTransform::HMXActivation) {
    // Expected permutation: [0, 2, 1, 3] (BSHD → BHSD)
    // In affine map: (d0, d1, d2, d3) -> (d0, d2, d1, d3)
    if (inputMap.getNumDims() == 4) {
      auto results = inputMap.getResults();
      if (results.size() == 4) {
        // Check if it's (d0, d2, d1, d3)
        auto d0 = dyn_cast<AffineDimExpr>(results[0]);
        auto d1 = dyn_cast<AffineDimExpr>(results[1]);
        auto d2 = dyn_cast<AffineDimExpr>(results[2]);
        auto d3 = dyn_cast<AffineDimExpr>(results[3]);
        
        if (d0 && d1 && d2 && d3) {
          return (d0.getPosition() == 0 && 
                  d1.getPosition() == 2 && 
                  d2.getPosition() == 1 && 
                  d3.getPosition() == 3);
        }
      }
    }
  }

  return false;
}

/// Returns true if `op` is a layout operation that matches the layout
/// transform being performed in-situ during prefetch.
static bool isRedundantLayoutOp(Operation *op, LayoutTransform lt) {
  if (lt == LayoutTransform::None)
    return false;

  StringRef opName = op->getName().getStringRef();

  // Check for linalg.transpose (tensor level - before bufferization)
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    auto permAttr = transposeOp.getPermutation();
    if (lt == LayoutTransform::HMXWeight && permAttr.size() == 4) {
      // HMXWeight expects [B,H,S,D] from [B,S,H,D]
      // Permutation: [0,2,1,3]
      return (permAttr[0] == 0 && permAttr[1] == 2 &&
              permAttr[2] == 1 && permAttr[3] == 3);
    }
    if (lt == LayoutTransform::HMXActivation && permAttr.size() == 4) {
      // HMXActivation also expects [B,H,S,D] from [B,S,H,D]
      // Permutation: [0,2,1,3]
      return (permAttr[0] == 0 && permAttr[1] == 2 &&
              permAttr[2] == 1 && permAttr[3] == 3);
    }
    return false;
  }

  // Check for linalg.generic with transpose pattern (memref level - after bufferization)
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    return isTransposeGeneric(genericOp, lt);
  }

  // Check for generic transpose/permute ops
  if (opName.contains("transpose") || opName.contains("permute")) {
    if (auto permAttr = op->getAttrOfType<DenseI64ArrayAttr>("permutation")) {
      auto perm = permAttr.asArrayRef();
      if (lt == LayoutTransform::HMXWeight && perm.size() == 4) {
        return (perm[0] == 0 && perm[1] == 2 &&
                perm[2] == 1 && perm[3] == 3);
      }
    }
    // Conservatively mark as redundant if doing layout-aware prefetch
    return true;
  }

  // Check for contiguous() calls
  if (opName.contains("contiguous"))
    return true;

  // reshape/expand/collapse can be redundant when in-situ layout absorbs them
  if (opName.contains("reshape") || opName.contains("expand_shape") ||
      opName.contains("collapse_shape"))
    return true;

  // Do NOT treat memref.subview / tensor.extract_slice as redundant layout
  // ops.  PrefetchInsert builds 32×32 tile subviews as the *source* of
  // prefetch_in_situ; deleting them leaves dangling SSA and crashes.
  // True layout redundancy is transpose/permute/reshape above.

  return false;
}

/// Returns true if `op` can be safely removed.
static bool canSafelyRemove(Operation *op) {
  if (op->use_empty())
    return true;

  // For linalg.generic, check if all uses are also marked redundant
  if (isa<linalg::GenericOp>(op)) {
    for (OpOperand &use : op->getUses()) {
      Operation *user = use.getOwner();

      // If user is also marked redundant, it's safe
      if (user->hasAttr("alps.redundant"))
        continue;

      // If user is a memref cast/subview, it's safe
      if (isa<memref::SubViewOp, memref::CastOp, memref::AllocOp>(user))
        continue;

      // If user is a yield op in an unused loop, it's safe
      if (isa<scf::YieldOp>(user)) {
        if (user->getParentOp()->use_empty())
          continue;
      }

      // Otherwise, not safe to remove
      return false;
    }
    return true;
  }

  // Original logic for other operations
  for (OpOperand &use : op->getUses()) {
    Operation *user = use.getOwner();

    // Never delete a value still consumed by Alps prefetch.
    if (isa<PrefetchInSituOp>(user))
      return false;

    // If user is also marked redundant, it's safe
    if (user->hasAttr("alps.redundant"))
      continue;

    // If user is a memref cast/subview, it's safe
    if (isa<memref::SubViewOp, memref::CastOp, memref::AllocOp>(user))
      continue;

    // If user is a yield op in an unused loop, it's safe
    if (isa<scf::YieldOp>(user)) {
      if (user->getParentOp()->use_empty())
        continue;
    }

    // Otherwise, not safe to remove
    return false;
  }

  return true;
}

/// Walk the def-use chain from `memref` and mark all redundant layout ops.
static void markRedundantLayoutOps(Value memref, LayoutTransform lt,
                                   FunctionOpInterface func) {
  // L2Hint only warms cache; it does not replace expand/collapse/transpose.
  if (lt == LayoutTransform::None || lt == LayoutTransform::L2Hint) {
    llvm::dbgs() << "[LayoutOpsElimination]   Layout transform is None/L2Hint, "
                    "nothing to mark\n";
    return;
  }

  llvm::dbgs() << "[LayoutOpsElimination]   Walking def-use chain to find redundant ops...\n";

  llvm::SmallPtrSet<Operation*, 16> visited;
  SmallVector<Value> worklist = {memref};
  int opsChecked = 0;
  int opsMarked = 0;

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    Operation *defOp = val.getDefiningOp();

    if (!defOp || !visited.insert(defOp).second)
      continue;

    // Only process ops in the same function
    if (defOp->getParentOfType<FunctionOpInterface>() != func)
      continue;

    opsChecked++;
    llvm::dbgs() << "[LayoutOpsElimination]     Checking op: " << defOp->getName() << "\n";

    // Check if this is a redundant layout op
    if (isRedundantLayoutOp(defOp, lt)) {
      llvm::dbgs() << "[LayoutOpsElimination]       ✓ Marked as redundant\n";
      defOp->setAttr("alps.redundant",
                     UnitAttr::get(defOp->getContext()));
      opsMarked++;

      // Continue walking backward through operands
      for (Value operand : defOp->getOperands()) {
        if (isa<MemRefType, RankedTensorType>(operand.getType())) {
          worklist.push_back(operand);
        }
      }
      
      // For linalg.generic, also walk through results
      if (auto genericOp = dyn_cast<linalg::GenericOp>(defOp)) {
        for (Value result : genericOp.getResults()) {
          if (isa<MemRefType, RankedTensorType>(result.getType())) {
            // Mark users of this result for further analysis
            for (Operation *user : result.getUsers()) {
              if (visited.insert(user).second) {
                for (Value userOperand : user->getOperands()) {
                  if (isa<MemRefType, RankedTensorType>(userOperand.getType())) {
                    worklist.push_back(userOperand);
                  }
                }
              }
            }
          }
        }
      }
    } else {
      llvm::dbgs() << "[LayoutOpsElimination]       ✗ Not redundant\n";
    }

    // Walk through cast/subview ops
    if (isa<memref::SubViewOp, memref::CastOp, tensor::CastOp>(defOp)) {
      llvm::dbgs() << "[LayoutOpsElimination]       → Walking through cast/subview\n";
      for (Value operand : defOp->getOperands()) {
        if (isa<MemRefType, RankedTensorType>(operand.getType())) {
          worklist.push_back(operand);
        }
      }
    }
  }
  
  llvm::dbgs() << "[LayoutOpsElimination]   Checked " << opsChecked 
               << " operations, marked " << opsMarked << " as redundant\n";
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LayoutOpsEliminationPass
    : public ::impl::LayoutOpsEliminationBase<LayoutOpsEliminationPass> {

  void runOnOperation() override {
    auto func = getOperation();

    llvm::dbgs() << "\n[LayoutOpsElimination] ========== PASS STARTING ==========\n";
    llvm::dbgs() << "[LayoutOpsElimination] Function: " << func.getName() << "\n";

    // Step 1: Collect all prefetch_in_situ operations with layout transforms
    SmallVector<PrefetchInSituOp> prefetches;
    func.walk([&](PrefetchInSituOp op) {
      llvm::dbgs() << "[LayoutOpsElimination] Found prefetch_in_situ operation:\n";
      llvm::dbgs() << "[LayoutOpsElimination]   Location: " << op.getLoc() << "\n";
      llvm::dbgs() << "[LayoutOpsElimination]   Layout transform: " 
                   << static_cast<int>(op.getLayoutTransform()) 
                   << " (0=None, 1=HMXWeight, 2=HMXActivation)\n";
      
      auto lt = op.getLayoutTransform();
      if (lt != LayoutTransform::None && lt != LayoutTransform::L2Hint) {
        prefetches.push_back(op);
        llvm::dbgs() << "[LayoutOpsElimination]   ✓ Has layout transform, will process\n";
      } else {
        llvm::dbgs() << "[LayoutOpsElimination]   ✗ No layout transform (or L2Hint), skipping\n";
      }
    });

    if (prefetches.empty()) {
      // No in-situ reshape, nothing to eliminate
      llvm::dbgs() << "[LayoutOpsElimination] No prefetch operations with layout transforms found\n";
      llvm::dbgs() << "[LayoutOpsElimination] ========== PASS COMPLETE (no work) ==========\n\n";
      return;
    }

    // DEBUG: Report how many prefetch operations found
    llvm::dbgs() << "[LayoutOpsElimination] Found " << prefetches.size() 
                 << " prefetch operations with layout transforms\n";

    // Step 2: Carry activation layout requests through provably address-only
    // producer views before assigning layout identities.
    LayoutCarryStats carryStats = carryActivationLayoutToProducer(prefetches);
    Builder builder(func.getContext());
    func->setAttr("alps.layout_carried_sites",
                  builder.getI64IntegerAttr(carryStats.fusedSites));
    func->setAttr("alps.layout_carried_views",
                  builder.getI64IntegerAttr(carryStats.bypassedViews));
    llvm::errs() << "[LayoutCarryFusion] function=" << func.getName()
                 << " fused_sites=" << carryStats.fusedSites
                 << " bypassed_views=" << carryStats.bypassedViews << "\n";

    // Step 3: Give every transformed source a compiler-visible site identity
    // and record conservative liveness/reuse evidence.  Codegen is unchanged.
    annotateLayoutValueSites(func, prefetches);

    // Step 4: For each prefetch, mark redundant layout ops
    int prefetchIdx = 0;
    for (auto prefetch : prefetches) {
      Value src = prefetch.getSrc();
      LayoutTransform lt = prefetch.getLayoutTransform();
      
      llvm::dbgs() << "\n[LayoutOpsElimination] --- Processing prefetch " << prefetchIdx++ << " ---\n";
      llvm::dbgs() << "[LayoutOpsElimination]   Layout transform: " << static_cast<int>(lt) << "\n";
      llvm::dbgs() << "[LayoutOpsElimination]   Source type: " << src.getType() << "\n";
      
      markRedundantLayoutOps(src, lt, func);
    }

    // Step 5: Collect all ops marked as redundant
    SmallVector<Operation*> toDelete;
    func.walk([&](Operation *op) {
      if (op->hasAttr("alps.redundant")) {
        llvm::dbgs() << "[LayoutOpsElimination] Marked redundant: " 
                     << op->getName() << "\n";
        if (canSafelyRemove(op)) {
          toDelete.push_back(op);
          llvm::dbgs() << "[LayoutOpsElimination]   -> Will delete\n";
        } else {
          llvm::dbgs() << "[LayoutOpsElimination]   -> Cannot delete (has uses)\n";
        }
      }
    });

    // Step 6: Delete in reverse order (child before parent)
    llvm::dbgs() << "\n[LayoutOpsElimination] Deleting " << toDelete.size() 
                 << " redundant operations\n";
    
    for (auto it = toDelete.rbegin(); it != toDelete.rend(); ++it) {
      llvm::dbgs() << "[LayoutOpsElimination]   Deleting: " << (*it)->getName() << "\n";
      (*it)->erase();
    }
    
    llvm::dbgs() << "[LayoutOpsElimination] ========== PASS COMPLETE ==========\n\n";
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public factory
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createLayoutOpsEliminationPass() {
  return std::make_unique<LayoutOpsEliminationPass>();
}
