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
// When `omni_fetch.prefetch_in_situ` is used with `layout_transform != None`,
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

#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "layout-ops-elimination"

using namespace mlir;
using namespace mlir::omni_fetch;
using namespace hexagon;

#define GEN_PASS_DEF_LAYOUTOPSELIMINATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

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
      if (user->hasAttr("omni_fetch.redundant"))
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

    // Never delete a value still consumed by OmniFetch prefetch.
    if (isa<PrefetchInSituOp>(user))
      return false;

    // If user is also marked redundant, it's safe
    if (user->hasAttr("omni_fetch.redundant"))
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
      defOp->setAttr("omni_fetch.redundant",
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

    // Step 2: For each prefetch, mark redundant layout ops
    int prefetchIdx = 0;
    for (auto prefetch : prefetches) {
      Value src = prefetch.getSrc();
      LayoutTransform lt = prefetch.getLayoutTransform();
      
      llvm::dbgs() << "\n[LayoutOpsElimination] --- Processing prefetch " << prefetchIdx++ << " ---\n";
      llvm::dbgs() << "[LayoutOpsElimination]   Layout transform: " << static_cast<int>(lt) << "\n";
      llvm::dbgs() << "[LayoutOpsElimination]   Source type: " << src.getType() << "\n";
      
      markRedundantLayoutOps(src, lt, func);
    }

    // Step 3: Collect all ops marked as redundant
    SmallVector<Operation*> toDelete;
    func.walk([&](Operation *op) {
      if (op->hasAttr("omni_fetch.redundant")) {
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

    // Step 4: Delete in reverse order (child before parent)
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
