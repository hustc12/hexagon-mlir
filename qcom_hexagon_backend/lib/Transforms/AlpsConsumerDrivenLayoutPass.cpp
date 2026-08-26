//===- AlpsConsumerDrivenLayoutPass.cpp - consumer layout contracts ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <mutex>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSCONSUMERDRIVENLAYOUT
#define GEN_PASS_DEF_ALPSCONTRACTDISCHARGELEDGER
#define GEN_PASS_DEF_ALPSLAYOUTSUPPLYPREFETCH
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static std::mutex reportMutex;

static std::string locationKey(Location location) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  location.print(stream);
  std::string result = stream.str();
  if (StringRef(result).starts_with("loc(") &&
      StringRef(result).ends_with(")"))
    return result.substr(4, result.size() - 5);
  return result;
}

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
                                        int64_t &eliminatedBytes,
                                        bool propagateCodegenContract,
                                        bool emitDischargeContract,
                                        StringRef contractId) {
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
  if (propagateCodegenContract) {
    direct->setAttr("alps.p2f.consumer_layout_contract",
                    rewriter.getStringAttr("hvx_innermost_unit_stride"));
    direct->setAttr("alps.p2f.permutation",
                    rewriter.getDenseI64ArrayAttr(permutation));
    direct->setAttr("alps.p2f.contiguous_loop",
                    rewriter.getI64IntegerAttr(targetRank - 1));
  }
  if (emitDischargeContract)
    direct->setAttr("alps.p5a.contract_id",
                    rewriter.getStringAttr(contractId));
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
  explicit AlpsConsumerDrivenLayoutPass(
      const AlpsConsumerDrivenLayoutOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<linalg::TransposeOp> transposes;
    function.walk([&](linalg::TransposeOp op) { transposes.push_back(op); });

    ContractStats stats;
    SmallVector<Attribute> dischargeContracts;
    SmallVector<Attribute> nativeDemands;
    Builder builder(function.getContext());
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
      std::string contractId =
          (Twine(function.getName()) + ":" + Twine(stats.demands - 1)).str();
      std::string origin = locationKey(transpose.getLoc());
      SmallVector<int64_t> permutation(transpose.getPermutation());
      auto targetType =
          dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
      rewriter.setInsertionPoint(transpose);
      if (!unsupported && engine != Engine::Mixed &&
          succeeded(makeProducerDirect(transpose, rewriter, bytes,
                                       propagateCodegenContract,
                                       emitDischargeContracts, contractId))) {
        ++stats.producerDirect;
        stats.eliminatedBytes += bytes;
        if (emitDischargeContracts) {
          NamedAttrList record;
          record.set("id", builder.getStringAttr(contractId));
          record.set("origin", builder.getStringAttr(origin));
          record.set("bytes", builder.getI64IntegerAttr(bytes));
          record.set("permutation",
                     builder.getDenseI64ArrayAttr(permutation));
          if (targetType)
            record.set("target_shape",
                       builder.getDenseI64ArrayAttr(targetType.getShape()));
          dischargeContracts.push_back(
              DictionaryAttr::get(function.getContext(), record));
        }
        continue;
      }
      ++stats.native;
      if (emitDischargeContracts) {
        NamedAttrList record;
        record.set("id", builder.getStringAttr(
                             (Twine(function.getName()) + ":native:" +
                              Twine(stats.demands - 1))
                                 .str()));
        record.set("origin", builder.getStringAttr(origin));
        record.set("bytes", builder.getI64IntegerAttr(
                                staticBytes(transpose.getInput().getType())
                                    .value_or(0)));
        record.set("permutation", builder.getDenseI64ArrayAttr(permutation));
        if (targetType)
          record.set("target_shape",
                     builder.getDenseI64ArrayAttr(targetType.getShape()));
        nativeDemands.push_back(
            DictionaryAttr::get(function.getContext(), record));
      }
    }

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
    if (emitDischargeContracts)
      function->setAttr("alps.p5a.contracts",
                        builder.getArrayAttr(dischargeContracts));
    if (emitDischargeContracts)
      function->setAttr("alps.p5a.native_demands",
                        builder.getArrayAttr(nativeDemands));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P2E] function=" << function.getName()
                 << " demands=" << stats.demands
                 << " hvx=" << stats.hvxConsumers
                 << " hmx=" << stats.hmxConsumers
                 << " mixed=" << stats.mixedConsumers
                 << " producer_direct=" << stats.producerDirect
                 << " codegen_contract="
                 << (propagateCodegenContract ? stats.producerDirect : 0)
                 << " discharge_contract="
                 << (emitDischargeContracts ? stats.producerDirect : 0)
                 << " native=" << stats.native
                 << " eliminated_materialization_bytes="
                 << stats.eliminatedBytes << '\n';
  }
};

struct AlpsContractDischargeLedgerPass final
    : ::impl::AlpsContractDischargeLedgerBase<
          AlpsContractDischargeLedgerPass> {
  explicit AlpsContractDischargeLedgerPass(
      const AlpsContractDischargeLedgerOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    auto contracts = function->getAttrOfType<ArrayAttr>("alps.p5a.contracts");
    if (!contracts)
      return;

    llvm::StringSet<> explicitIds;
    struct LocatedOperation {
      std::string location;
      bool physicalTransform;
    };
    SmallVector<LocatedOperation> locations;
    function.walk([&](Operation *op) {
      if (auto id = op->getAttrOfType<StringAttr>("alps.p5a.contract_id"))
        explicitIds.insert(id.getValue());
      bool physicalTransform = isa<linalg::TransposeOp>(op);
      if (auto read = dyn_cast<vector::TransferReadOp>(op))
        physicalTransform |= !read.getPermutationMap().isMinorIdentity();
      if (auto write = dyn_cast<vector::TransferWriteOp>(op))
        physicalTransform |= !write.getPermutationMap().isMinorIdentity();
      locations.push_back({locationKey(op->getLoc()), physicalTransform});
    });

    int64_t explicitSurvivors = 0;
    int64_t locationCarriers = 0;
    int64_t physicalTransforms = 0;
    int64_t untraceable = 0;
    std::lock_guard<std::mutex> lock(reportMutex);
    for (Attribute attr : contracts) {
      auto record = dyn_cast<DictionaryAttr>(attr);
      if (!record)
        continue;
      StringRef id = record.getAs<StringAttr>("id").getValue();
      StringRef origin = record.getAs<StringAttr>("origin").getValue();
      StringRef status = "untraceable";
      if (explicitIds.contains(id)) {
        ++explicitSurvivors;
        status = "explicit_survivor";
      } else if (origin != "unknown") {
        bool carrier = false;
        bool transform = false;
        for (const auto &[candidate, isPhysicalTransform] : locations) {
          if (candidate.find(origin.str()) == std::string::npos)
            continue;
          carrier = true;
          transform |= isPhysicalTransform;
        }
        if (transform) {
          ++physicalTransforms;
          status = "physical_transform_remains";
        } else if (carrier) {
          ++locationCarriers;
          status = "location_carrier";
        } else {
          ++untraceable;
        }
      } else {
        ++untraceable;
      }
      llvm::errs() << "[ALPS-P5A-CONTRACT] phase=" << phase
                   << " function=" << function.getName() << " id=" << id
                   << " status=" << status << '\n';
    }
    llvm::errs() << "[ALPS-P5A-SUMMARY] phase=" << phase
                 << " function=" << function.getName()
                 << " total=" << contracts.size()
                 << " explicit=" << explicitSurvivors
                 << " location_carrier=" << locationCarriers
                 << " physical_transform=" << physicalTransforms
                 << " untraceable=" << untraceable << '\n';

    auto nativeDemands =
        function->getAttrOfType<ArrayAttr>("alps.p5a.native_demands");
    int64_t nativePhysical = 0;
    int64_t nativeCarrier = 0;
    int64_t nativeUntraceable = 0;
    if (nativeDemands) {
      for (Attribute attr : nativeDemands) {
        auto record = dyn_cast<DictionaryAttr>(attr);
        if (!record)
          continue;
        StringRef origin = record.getAs<StringAttr>("origin").getValue();
        bool carrier = false;
        bool transform = false;
        if (origin != "unknown") {
          for (const auto &[candidate, isPhysicalTransform] : locations) {
            if (candidate.find(origin.str()) == std::string::npos)
              continue;
            carrier = true;
            transform |= isPhysicalTransform;
          }
        }
        if (transform)
          ++nativePhysical;
        else if (carrier)
          ++nativeCarrier;
        else
          ++nativeUntraceable;
      }
      llvm::errs() << "[ALPS-P5D-NATIVE-SUMMARY] phase=" << phase
                   << " function=" << function.getName()
                   << " total=" << nativeDemands.size()
                   << " physical_transform=" << nativePhysical
                   << " location_carrier=" << nativeCarrier
                   << " untraceable=" << nativeUntraceable << '\n';
    }

    if (!analyzeInputs || phase != "post-bufferization")
      return;

    DenseMap<Operation *, int64_t> ordinals;
    int64_t nextOrdinal = 0;
    function.walk([&](Operation *op) { ordinals[op] = nextOrdinal++; });

    int64_t carriers = 0;
    int64_t inputs = 0;
    int64_t admitted = 0;
    int64_t admittedBytes = 0;
    function.walk([&](linalg::LinalgOp carrier) {
      std::string carrierLocation = locationKey(carrier.getLoc());
      SmallVector<StringRef> matchedContracts;
      for (Attribute attr : contracts) {
        auto record = dyn_cast<DictionaryAttr>(attr);
        if (!record)
          continue;
        StringRef origin = record.getAs<StringAttr>("origin").getValue();
        if (origin != "unknown" &&
            carrierLocation.find(origin.str()) != std::string::npos)
          matchedContracts.push_back(
              record.getAs<StringAttr>("id").getValue());
      }
      if (matchedContracts.empty())
        return;
      ++carriers;

      SmallVector<Value> outputs(carrier.getDpsInits());
      for (auto [operandIndex, source] :
           llvm::enumerate(carrier.getDpsInputs())) {
        ++inputs;
        bool aliasesOutput = llvm::is_contained(outputs, source);
        auto type = dyn_cast<MemRefType>(source.getType());
        SmallVector<int64_t> strides;
        int64_t offset = 0;
        bool contiguous =
            type && succeeded(type.getStridesAndOffset(strides, offset)) &&
            !strides.empty() && strides.back() == 1;
        int64_t bytes = staticBytes(source.getType()).value_or(0);

        Operation *consumer = carrier.getOperation();
        int64_t consumerOrdinal = ordinals.lookup(consumer);
        int64_t lastWriteOrdinal = -1;
        if (Operation *def = source.getDefiningOp())
          if (def->getBlock() == consumer->getBlock())
            lastWriteOrdinal = ordinals.lookup(def);
        for (Operation *user : source.getUsers()) {
          if (user == consumer || user->getBlock() != consumer->getBlock())
            continue;
          int64_t userOrdinal = ordinals.lookup(user);
          if (userOrdinal >= consumerOrdinal)
            continue;
          bool writesSource = false;
          if (auto writer = dyn_cast<linalg::LinalgOp>(user))
            writesSource = llvm::is_contained(writer.getDpsInits(), source);
          else if (auto copy = dyn_cast<memref::CopyOp>(user))
            writesSource = copy.getTarget() == source;
          if (writesSource)
            lastWriteOrdinal = std::max(lastWriteOrdinal, userOrdinal);
        }
        int64_t leadOps = consumerOrdinal - lastWriteOrdinal - 1;
        bool sameBlockAvailable = !source.getDefiningOp() ||
                                  source.getDefiningOp()->getBlock() ==
                                      consumer->getBlock();
        bool candidate = !aliasesOutput && type && type.hasStaticShape() &&
                         contiguous && sameBlockAvailable &&
                         leadOps >= minLeadOps && bytes >= minBytes;
        if (candidate) {
          ++admitted;
          admittedBytes += bytes;
        }
        llvm::errs() << "[ALPS-P5B-INPUT] function=" << function.getName()
                     << " contracts=";
        llvm::interleaveComma(matchedContracts, llvm::errs());
        llvm::errs() << " operand=" << operandIndex
                     << " bytes=" << bytes << " lead_ops=" << leadOps
                     << " contiguous=" << contiguous
                     << " aliases_output=" << aliasesOutput
                     << " uses=" << std::distance(source.use_begin(),
                                                   source.use_end())
                     << " decision=" << (candidate ? "admit" : "reject")
                     << '\n';
      }
    });

    // At the post-bufferization boundary most admitted P2e producers have
    // already been tiled/vectorized. Their physical input streams are now
    // vector.transfer_read operations rather than Linalg operands, so inspect
    // those final HVX-facing reads instead of declaring the contract gone.
    function.walk([&](vector::TransferReadOp read) {
      std::string carrierLocation = locationKey(read.getLoc());
      SmallVector<StringRef> matchedContracts;
      for (Attribute attr : contracts) {
        auto record = dyn_cast<DictionaryAttr>(attr);
        if (!record)
          continue;
        StringRef origin = record.getAs<StringAttr>("origin").getValue();
        if (origin != "unknown" &&
            carrierLocation.find(origin.str()) != std::string::npos)
          matchedContracts.push_back(
              record.getAs<StringAttr>("id").getValue());
      }
      if (matchedContracts.empty())
        return;
      ++carriers;
      ++inputs;

      Value source = read.getBase();
      auto type = dyn_cast<MemRefType>(source.getType());
      SmallVector<int64_t> strides;
      int64_t offset = 0;
      bool contiguous =
          type && succeeded(type.getStridesAndOffset(strides, offset)) &&
          !strides.empty() && strides.back() == 1;
      int64_t sourceBytes = staticBytes(source.getType()).value_or(0);
      int64_t tileBytes = staticBytes(read.getVectorType()).value_or(0);

      Operation *consumer = read.getOperation();
      int64_t consumerOrdinal = ordinals.lookup(consumer);
      int64_t lastWriteOrdinal = -1;
      if (Operation *def = source.getDefiningOp())
        if (def->getBlock() == consumer->getBlock())
          lastWriteOrdinal = ordinals.lookup(def);
      for (Operation *user : source.getUsers()) {
        if (user == consumer || user->getBlock() != consumer->getBlock())
          continue;
        int64_t userOrdinal = ordinals.lookup(user);
        if (userOrdinal >= consumerOrdinal)
          continue;
        bool writesSource = false;
        if (auto writer = dyn_cast<linalg::LinalgOp>(user))
          writesSource = llvm::is_contained(writer.getDpsInits(), source);
        else if (auto copy = dyn_cast<memref::CopyOp>(user))
          writesSource = copy.getTarget() == source;
        if (writesSource)
          lastWriteOrdinal = std::max(lastWriteOrdinal, userOrdinal);
      }
      int64_t leadOps = consumerOrdinal - lastWriteOrdinal - 1;
      bool sameBlockAvailable = !source.getDefiningOp() ||
                                source.getDefiningOp()->getBlock() ==
                                    consumer->getBlock();
      bool candidate = type && type.hasStaticShape() && contiguous &&
                       sameBlockAvailable && leadOps >= minLeadOps &&
                       tileBytes >= minBytes;
      if (candidate) {
        ++admitted;
        admittedBytes += tileBytes;
      }
      llvm::errs() << "[ALPS-P5B-INPUT] function=" << function.getName()
                   << " contracts=";
      llvm::interleaveComma(matchedContracts, llvm::errs());
      llvm::errs() << " op=vector.transfer_read"
                   << " source_bytes=" << sourceBytes
                   << " tile_bytes=" << tileBytes
                   << " lead_ops=" << leadOps
                   << " contiguous=" << contiguous
                   << " uses="
                   << std::distance(source.use_begin(), source.use_end())
                   << " decision=" << (candidate ? "admit" : "reject")
                   << '\n';
    });
    llvm::errs() << "[ALPS-P5B-SUMMARY] function=" << function.getName()
                 << " carriers=" << carriers << " inputs=" << inputs
                 << " admitted=" << admitted
                 << " admitted_bytes=" << admittedBytes
                 << " min_lead_ops=" << minLeadOps
                 << " min_bytes=" << minBytes << '\n';
  }
};

static std::optional<int64_t> p5ConstantIndex(Value value) {
  IntegerAttr attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  return attr.getInt();
}

static std::optional<int64_t> p5ConstantFoldResult(OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return integer.getInt();
    return std::nullopt;
  }
  return p5ConstantIndex(cast<Value>(value));
}

struct AlpsLayoutSupplyPrefetchPass final
    : ::impl::AlpsLayoutSupplyPrefetchBase<
          AlpsLayoutSupplyPrefetchPass> {
  explicit AlpsLayoutSupplyPrefetchPass(
      const AlpsLayoutSupplyPrefetchOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    auto contracts = function->getAttrOfType<ArrayAttr>("alps.p5a.contracts");
    if (!contracts)
      return;
    if (distance <= 0 || maxBytes <= 0) {
      function.emitError("ALPS P5c requires positive distance and byte budget");
      return signalPassFailure();
    }

    int64_t matched = 0;
    int64_t admitted = 0;
    int64_t requestedBytes = 0;
    int64_t rejectedNoView = 0;
    int64_t rejectedCausal = 0;
    int64_t rejectedAddress = 0;
    int64_t rejectedBounds = 0;
    int64_t rejectedSize = 0;
    SmallVector<vector::TransferReadOp> reads;
    function.walk([&](vector::TransferReadOp read) {
      std::string readLocation = locationKey(read.getLoc());
      bool hasContract = llvm::any_of(contracts, [&](Attribute attr) {
        auto record = dyn_cast<DictionaryAttr>(attr);
        if (!record)
          return false;
        StringRef origin = record.getAs<StringAttr>("origin").getValue();
        return origin != "unknown" &&
               readLocation.find(origin.str()) != std::string::npos;
      });
      if (hasContract)
        reads.push_back(read);
    });

    for (vector::TransferReadOp read : reads) {
      ++matched;
      auto view = read.getBase().getDefiningOp<memref::SubViewOp>();
      if (!view) {
        ++rejectedNoView;
        continue;
      }
      scf::ForOp loop = view->getParentOfType<scf::ForOp>();
      if (!loop || !loop->isAncestor(view)) {
        ++rejectedAddress;
        continue;
      }
      if (Operation *sourceDef = view.getSource().getDefiningOp()) {
        if (loop->isAncestor(sourceDef)) {
          ++rejectedCausal;
          continue;
        }
      }
      auto sourceType = dyn_cast<MemRefType>(view.getSource().getType());
      auto viewType = dyn_cast<MemRefType>(view.getType());
      if (!sourceType || !viewType || !sourceType.hasStaticShape() ||
          !viewType.hasStaticShape() || sourceType.getMemorySpaceAsInt() != 0 ||
          !read.getPermutationMap().isMinorIdentity()) {
        ++rejectedAddress;
        continue;
      }
      auto lower = p5ConstantIndex(loop.getLowerBound());
      auto upper = p5ConstantIndex(loop.getUpperBound());
      auto step = p5ConstantIndex(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower) {
        ++rejectedBounds;
        continue;
      }

      SmallVector<OpFoldResult> offsets = view.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = view.getMixedSizes();
      SmallVector<OpFoldResult> strides = view.getMixedStrides();
      int64_t ivDimension = -1;
      int64_t tileElements = 1;
      bool valid = true;
      for (auto [index, offset] : llvm::enumerate(offsets)) {
        auto value = dyn_cast<Value>(offset);
        if (value && value == loop.getInductionVar()) {
          if (ivDimension >= 0)
            valid = false;
          ivDimension = index;
        }
        auto size = p5ConstantFoldResult(sizes[index]);
        auto stride = p5ConstantFoldResult(strides[index]);
        if (!size || !stride || *size <= 0 || *stride != 1 ||
            llvm::MulOverflow(tileElements, *size, tileElements))
          valid = false;
      }
      if (!valid || ivDimension < 0) {
        ++rejectedAddress;
        continue;
      }
      auto ivTileSize = p5ConstantFoldResult(sizes[ivDimension]);
      if (!ivTileSize || *ivTileSize > *step) {
        ++rejectedBounds;
        continue;
      }
      int64_t elemBits = viewType.getElementTypeBitWidth();
      int64_t tileBytes = llvm::divideCeilSigned(tileElements * elemBits,
                                                 int64_t{8});
      if (elemBits <= 0 || tileBytes <= 0 || tileBytes > maxBytes) {
        ++rejectedSize;
        continue;
      }

      OpBuilder builder(view);
      Location loc = read.getLoc();
      Value delta = builder.create<arith::ConstantIndexOp>(
          loc, distance * *step);
      Value futureIv = builder.create<arith::AddIOp>(
          loc, loop.getInductionVar(), delta);
      Value tileSize = builder.create<arith::ConstantIndexOp>(loc,
                                                               *ivTileSize);
      Value futureEnd = builder.create<arith::AddIOp>(loc, futureIv, tileSize);
      Value inBounds = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, futureEnd, loop.getUpperBound());
      builder.create<scf::IfOp>(
          loc, inBounds, [&](OpBuilder &thenBuilder, Location thenLoc) {
            SmallVector<OpFoldResult> futureOffsets = offsets;
            futureOffsets[ivDimension] = futureIv;
            Value futureView = thenBuilder.create<memref::SubViewOp>(
                thenLoc, view.getSource(), futureOffsets, sizes, strides);
            auto hint = thenBuilder.create<omni_fetch::L2HintOp>(
                thenLoc, futureView, static_cast<int32_t>(distance));
            hint->setAttr("alps.p5c.layout_supply",
                          thenBuilder.getUnitAttr());
            hint->setAttr("alps.p5c.requested_bytes",
                          thenBuilder.getI64IntegerAttr(tileBytes));
            thenBuilder.create<scf::YieldOp>(thenLoc);
          });
      ++admitted;
      requestedBytes += tileBytes;
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5c.matched",
                      builder.getI64IntegerAttr(matched));
    function->setAttr("alps.p5c.admitted",
                      builder.getI64IntegerAttr(admitted));
    function->setAttr("alps.p5c.requested_bytes",
                      builder.getI64IntegerAttr(requestedBytes));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5C-SUMMARY] function=" << function.getName()
                 << " matched=" << matched << " admitted=" << admitted
                 << " hints=" << admitted
                 << " requested_bytes=" << requestedBytes
                 << " reject_no_view=" << rejectedNoView
                 << " reject_causal=" << rejectedCausal
                 << " reject_address=" << rejectedAddress
                 << " reject_bounds=" << rejectedBounds
                 << " reject_size=" << rejectedSize << '\n';
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsConsumerDrivenLayoutPass(
    const AlpsConsumerDrivenLayoutOptions &options) {
  return std::make_unique<AlpsConsumerDrivenLayoutPass>(options);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsContractDischargeLedgerPass(
    const AlpsContractDischargeLedgerOptions &options) {
  return std::make_unique<AlpsContractDischargeLedgerPass>(options);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsLayoutSupplyPrefetchPass(
    const AlpsLayoutSupplyPrefetchOptions &options) {
  return std::make_unique<AlpsLayoutSupplyPrefetchPass>(options);
}
