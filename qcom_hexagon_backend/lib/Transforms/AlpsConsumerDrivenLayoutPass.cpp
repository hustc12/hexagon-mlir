//===- AlpsConsumerDrivenLayoutPass.cpp - consumer layout contracts ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <mutex>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSCONSUMERDRIVENLAYOUT
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static std::mutex reportMutex;

struct ContractStats {
  int64_t demands = 0;
  int64_t hvxConsumers = 0;
  int64_t hmxConsumers = 0;
  int64_t mixedConsumers = 0;
  int64_t producerDirect = 0;
  int64_t native = 0;
  int64_t eliminatedBytes = 0;
};

static std::optional<int64_t> staticBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dim : shaped.getShape())
    if (dim < 0 || llvm::MulOverflow(elements, dim, elements))
      return std::nullopt;
  int64_t bits = shaped.getElementTypeBitWidth();
  int64_t totalBits = 0;
  if (bits <= 0 || llvm::MulOverflow(elements, bits, totalBits) ||
      totalBits > std::numeric_limits<int64_t>::max() - 7)
    return std::nullopt;
  return llvm::divideCeilSigned(totalBits, int64_t{8});
}

static bool isDescriptorOnly(Operation *op) {
  return isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ExtractSliceOp>(op);
}

enum class Engine { Unknown, HVX, HMX, Mixed };

static Engine mergeEngine(Engine lhs, Engine rhs) {
  if (lhs == Engine::Unknown)
    return rhs;
  if (rhs == Engine::Unknown || lhs == rhs)
    return lhs;
  return Engine::Mixed;
}

static Engine classifyTerminal(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.starts_with("hexkl."))
    return Engine::HMX;
  if (name.starts_with("vector.") || isa<linalg::LinalgOp>(op))
    return Engine::HVX;
  return Engine::Unknown;
}

static void collectTerminalConsumers(Value value,
                                     SmallPtrSetImpl<Operation *> &visited,
                                     SmallVectorImpl<Operation *> &terminals) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isDescriptorOnly(user)) {
      for (Value result : user->getResults())
        collectTerminalConsumers(result, visited, terminals);
      continue;
    }
    terminals.push_back(user);
  }
}

/// Retarget an immediate, single-use, parallel tensor producer to the layout
/// already demanded by a transpose's consumers.  Keeping the innermost
/// dimension fixed is a deliberately conservative HVX rule: it removes one
/// full materialization without turning unit-stride element access into a
/// strided vector stream.
static LogicalResult makeProducerDirect(linalg::TransposeOp transpose,
                                        PatternRewriter &rewriter,
                                        int64_t &eliminatedBytes) {
  Value producerResult = transpose.getInput();
  tensor::ExpandShapeOp expand =
      producerResult.getDefiningOp<tensor::ExpandShapeOp>();
  if (expand) {
    if (!expand.getResult().hasOneUse())
      return failure();
    producerResult = expand.getSrc();
  }
  auto producer = producerResult.getDefiningOp<linalg::GenericOp>();
  if (!producer || !producer.hasPureTensorSemantics() ||
      producer->getNumResults() != 1 || !producer.getResult(0).hasOneUse() ||
      producer.getNumDpsInits() != 1)
    return failure();

  auto sourceType = dyn_cast<RankedTensorType>(producer.getResult(0).getType());
  auto expandedType =
      dyn_cast<RankedTensorType>(transpose.getInput().getType());
  auto targetType = dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
  if (!sourceType || !expandedType || !targetType ||
      !sourceType.hasStaticShape() || !expandedType.hasStaticShape() ||
      !targetType.hasStaticShape() ||
      expandedType.getRank() != targetType.getRank() ||
      producer.getNumLoops() != sourceType.getRank())
    return failure();
  int64_t sourceRank = sourceType.getRank();
  int64_t targetRank = targetType.getRank();
  if (sourceRank < 2 || targetRank > 4)
    return failure();

  ArrayRef<int64_t> permutation = transpose.getPermutation();
  if (static_cast<int64_t>(permutation.size()) != targetRank ||
      permutation.back() != targetRank - 1)
    return failure();
  for (utils::IteratorType iterator : producer.getIteratorTypesArray())
    if (iterator != utils::IteratorType::parallel)
      return failure();

  SmallVector<AffineMap> maps = producer.getIndexingMapsArray();
  if (maps.size() != producer.getNumDpsInputs() + 1 ||
      !maps.back().isIdentity())
    return failure();

  SmallVector<int64_t> inverse(targetRank, -1);
  for (auto [newDim, oldDim] : llvm::enumerate(permutation)) {
    if (oldDim < 0 || oldDim >= targetRank || inverse[oldDim] != -1)
      return failure();
    inverse[oldDim] = newDim;
  }
  SmallVector<AffineExpr> expandedLoops;
  expandedLoops.reserve(targetRank);
  for (int64_t oldDim = 0; oldDim < targetRank; ++oldDim)
    expandedLoops.push_back(
        getAffineDimExpr(inverse[oldDim], rewriter.getContext()));

  SmallVector<AffineExpr> sourceLoops;
  if (!expand) {
    if (sourceRank != targetRank)
      return failure();
    sourceLoops = expandedLoops;
  } else {
    auto reassociation = expand.getReassociationIndices();
    if (static_cast<int64_t>(reassociation.size()) != sourceRank)
      return failure();
    for (ArrayRef<int64_t> group : reassociation) {
      if (group.empty())
        return failure();
      AffineExpr flattened = expandedLoops[group.front()];
      for (int64_t position = 1; position < static_cast<int64_t>(group.size());
           ++position) {
        int64_t expandedDim = group[position];
        int64_t extent = expandedType.getDimSize(expandedDim);
        if (extent <= 0)
          return failure();
        flattened = flattened * extent + expandedLoops[expandedDim];
      }
      sourceLoops.push_back(flattened);
    }
  }
  AffineMap newToOld = AffineMap::get(targetRank, 0, sourceLoops,
                                      rewriter.getContext());
  for (unsigned index = 0; index < producer.getNumDpsInputs(); ++index)
    maps[index] = maps[index].compose(newToOld);
  maps.back() =
      AffineMap::getMultiDimIdentityMap(targetRank, rewriter.getContext());

  SmallVector<utils::IteratorType> targetIterators(
      targetRank, utils::IteratorType::parallel);

  auto direct = linalg::GenericOp::create(
      rewriter, transpose.getLoc(), TypeRange{targetType},
      producer.getDpsInputs(), transpose.getDpsInits(), maps,
      targetIterators, /*bodyBuild=*/nullptr,
      linalg::getPrunedAttributeList(producer));
  rewriter.cloneRegionBefore(producer.getRegion(), direct.getRegion(),
                             direct.getRegion().begin());
  eliminatedBytes = staticBytes(sourceType).value_or(0);

  rewriter.replaceOp(transpose, direct.getResults());
  if (expand && expand->use_empty())
    rewriter.eraseOp(expand);
  if (producer->use_empty())
    rewriter.eraseOp(producer);
  return success();
}

struct AlpsConsumerDrivenLayoutPass final
    : ::impl::AlpsConsumerDrivenLayoutBase<AlpsConsumerDrivenLayoutPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<linalg::TransposeOp> transposes;
    function.walk([&](linalg::TransposeOp op) { transposes.push_back(op); });

    ContractStats stats;
    PatternRewriter rewriter(function.getContext());
    for (linalg::TransposeOp transpose : transposes) {
      if (!transpose->getBlock())
        continue;
      SmallPtrSet<Operation *, 8> visited;
      SmallVector<Operation *> terminals;
      collectTerminalConsumers(transpose.getResult()[0], visited, terminals);
      Engine engine = Engine::Unknown;
      bool unsupported = terminals.empty();
      for (Operation *terminal : terminals) {
        Engine current = classifyTerminal(terminal);
        unsupported |= current == Engine::Unknown;
        engine = mergeEngine(engine, current);
      }

      ++stats.demands;
      if (engine == Engine::HVX)
        ++stats.hvxConsumers;
      else if (engine == Engine::HMX)
        ++stats.hmxConsumers;
      else if (engine == Engine::Mixed)
        ++stats.mixedConsumers;

      int64_t bytes = 0;
      rewriter.setInsertionPoint(transpose);
      if (!unsupported && engine != Engine::Mixed &&
          succeeded(makeProducerDirect(transpose, rewriter, bytes))) {
        ++stats.producerDirect;
        stats.eliminatedBytes += bytes;
        continue;
      }
      ++stats.native;
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p2e.demands",
                      builder.getI64IntegerAttr(stats.demands));
    function->setAttr("alps.p2e.hvx_consumers",
                      builder.getI64IntegerAttr(stats.hvxConsumers));
    function->setAttr("alps.p2e.hmx_consumers",
                      builder.getI64IntegerAttr(stats.hmxConsumers));
    function->setAttr("alps.p2e.mixed_consumers",
                      builder.getI64IntegerAttr(stats.mixedConsumers));
    function->setAttr("alps.p2e.producer_direct",
                      builder.getI64IntegerAttr(stats.producerDirect));
    function->setAttr("alps.p2e.native",
                      builder.getI64IntegerAttr(stats.native));
    function->setAttr("alps.p2e.eliminated_materialization_bytes",
                      builder.getI64IntegerAttr(stats.eliminatedBytes));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P2E] function=" << function.getName()
                 << " demands=" << stats.demands
                 << " hvx=" << stats.hvxConsumers
                 << " hmx=" << stats.hmxConsumers
                 << " mixed=" << stats.mixedConsumers
                 << " producer_direct=" << stats.producerDirect
                 << " native=" << stats.native
                 << " eliminated_materialization_bytes="
                 << stats.eliminatedBytes << '\n';
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsConsumerDrivenLayoutPass() {
  return std::make_unique<AlpsConsumerDrivenLayoutPass>();
}
