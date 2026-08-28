//===- Vectorization.cpp - Implementation of Vectorization Pass  ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of inner loop to vector form so that vector
// to llvm pass can convert to vectorized llvm-ir.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#define DEBUG_TYPE "hexagon-vectorize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONVECTORIZATION
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

static LogicalResult vectorizeLinalgOp(linalg::LinalgOp op) {
  auto fnName = op->getAttrOfType<StringAttr>("library_call");
  if (fnName) {
    DBG("-> skipping vectorization. Op will be replaced with a library call.");
    return failure();
  }

  IRRewriter rewriter(op.getContext());

  auto numLoops = op.getNumLoops();
  if (numLoops < 1) {
    return failure();
  }
  auto innerLoopDim = op.getStaticLoopRanges()[numLoops - 1];

  const bool hasRegisterTileContract =
      op->hasAttr("alps.p2g.register_tile_contract");
  auto dataTileSize = computeDataTileSize(op);
  if (!hasRegisterTileContract &&
      (!dataTileSize ||
       !perfectlyVectorizable(dataTileSize.value(), innerLoopDim))) {
    DBG("-> skipping vectorization. data tile size and loop mismatch");
    return failure();
  }

  const bool hasConsumerLayoutContract =
      op->hasAttr("alps.p2f.consumer_layout_contract");
  if (hasConsumerLayoutContract) {
    auto contiguousLoop =
        op->getAttrOfType<IntegerAttr>("alps.p2f.contiguous_loop");
    if (!contiguousLoop || contiguousLoop.getInt() != numLoops - 1 ||
        !op.getIndexingMapsArray().back().isIdentity()) {
      op.emitWarning("invalid ALPS P2f contract before HVX vectorization");
      return failure();
    }
  }

  SmallVector<int64_t> vecSizes(numLoops, 1);
  SmallVector<bool> scalableDims(numLoops, false);
  if (hasRegisterTileContract) {
    auto requested =
        op->getAttrOfType<DenseI64ArrayAttr>("alps.p2g.register_tile_sizes");
    auto ranges = op.getStaticLoopRanges();
    if (numLoops < 2 || !requested || requested.size() != 2 ||
        ShapedType::isDynamic(ranges[numLoops - 2]) ||
        ShapedType::isDynamic(ranges[numLoops - 1]) ||
        ranges[numLoops - 2] > static_cast<uint64_t>(requested[0]) ||
        ranges[numLoops - 1] > static_cast<uint64_t>(requested[1]))
      return failure();
    vecSizes[numLoops - 2] = ranges[numLoops - 2];
    vecSizes[numLoops - 1] = ranges[numLoops - 1];
  } else {
    vecSizes[numLoops - 1] = innerLoopDim;
  }

  auto role = op->getAttrOfType<StringAttr>("omni_fetch.kv_cache_role");
  auto operand =
      op->getAttrOfType<IntegerAttr>("omni_fetch.kv_cache_operand");
  auto layout = op->getAttrOfType<StringAttr>("omni_fetch.kv_cache_layout");
  Value kvSource;
  if (role && operand && operand.getInt() >= 0 &&
      operand.getInt() < static_cast<int64_t>(op.getDpsInputs().size()))
    kvSource = op.getDpsInputs()[operand.getInt()];

  // linalg::vectorize may materialize transfers inside newly created nested
  // regions rather than as direct siblings of the source linalg op.  Snapshot
  // all existing vector operations so P2g-c can mark exactly the operations
  // created by this vectorization call, independent of their nesting depth.
  SmallPtrSet<Operation *, 16> vectorOpsBefore;
  func::FuncOp parentFunction = op->getParentOfType<func::FuncOp>();
  if (hasRegisterTileContract && parentFunction)
    parentFunction.walk([&](Operation *candidate) {
      if (isa<vector::TransferReadOp, vector::TransferWriteOp,
              vector::TransposeOp>(candidate))
        vectorOpsBefore.insert(candidate);
    });

  Operation *previous = op->getPrevNode();
  FailureOr<mlir::linalg::VectorizationResult> vectorResults =
      linalg::vectorize(rewriter, op, vecSizes, scalableDims);
  if (failed(vectorResults))
    return failure();
  if (role)
    llvm::errs() << "[KVPropagation] vectorization=succeeded role="
                 << role.getValue() << "\n";
  if (kvSource) {
    Operation *created = previous ? previous->getNextNode()
                                  : &op->getBlock()->front();
    while (created && created != op) {
      if (auto read = dyn_cast<vector::TransferReadOp>(created)) {
        if (read.getBase() == kvSource) {
          read->setAttr("omni_fetch.kv_cache_role", role);
          read->setAttr("omni_fetch.kv_cache_layout", layout);
        }
      }
      created = created->getNextNode();
    }
  }
  if (hasConsumerLayoutContract) {
    Operation *created = previous ? previous->getNextNode()
                                  : &op->getBlock()->front();
    while (created && created != op) {
      if (isa<vector::TransferReadOp, vector::TransferWriteOp>(created))
        created->setAttr("alps.p2f.consumer_layout_contract",
                         rewriter.getUnitAttr());
      created = created->getNextNode();
    }
  }
  if (hasRegisterTileContract) {
    parentFunction.walk([&](Operation *created) {
      if (!vectorOpsBefore.contains(created) &&
          isa<vector::TransferReadOp, vector::TransferWriteOp,
              vector::TransposeOp>(created))
        created->setAttr("alps.p2g.register_tile", rewriter.getUnitAttr());
    });
  }
  // Replace the original op with the vectorized op.
  rewriter.replaceOp(op, vectorResults->replacements);
  return success();
}

struct HexagonVectorizationPass
    : public ::impl::HexagonVectorizationBase<HexagonVectorizationPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    int64_t p2fCandidates = 0;
    int64_t p2fVectorized = 0;
    int64_t p2gRegisterCandidates = 0;
    int64_t p2gRegisterVectorized = 0;
    moduleOp.walk([&](linalg::LinalgOp op) {
      DBG("vectorization candidate: " << op << "\n");
      bool isP2f = op->hasAttr("alps.p2f.consumer_layout_contract");
      bool isP2gRegister =
          op->hasAttr("alps.p2g.register_tile_contract");
      p2fCandidates += isP2f;
      p2gRegisterCandidates += isP2gRegister;
      if (succeeded(vectorizeLinalgOp(op))) {
        DBG(" -> vectorization succeeded.\n");
        p2fVectorized += isP2f;
        p2gRegisterVectorized += isP2gRegister;
      } else {
        DBG(" -> vectorization failed.\n");
      }
      return WalkResult::advance();
    });
    if (p2fCandidates)
      llvm::errs() << "[ALPS-P2F] vectorization candidates=" << p2fCandidates
                   << " succeeded=" << p2fVectorized
                   << " failed=" << (p2fCandidates - p2fVectorized) << '\n';
    if (p2gRegisterCandidates)
      llvm::errs() << "[ALPS-P2G-C] vectorization candidates="
                   << p2gRegisterCandidates
                   << " succeeded=" << p2gRegisterVectorized
                   << " failed="
                   << (p2gRegisterCandidates - p2gRegisterVectorized) << '\n';
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonVectorizationPass() {
  return std::make_unique<HexagonVectorizationPass>();
}
