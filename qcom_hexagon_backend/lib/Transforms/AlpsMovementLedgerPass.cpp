//===- AlpsMovementLedgerPass.cpp - ALPS P1 movement analysis ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSMOVEMENTLEDGER
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct LedgerTotals {
  int64_t candidates = 0;
  int64_t descriptorSites = 0;
  int64_t physicalTransformSites = 0;
  int64_t copySites = 0;
  int64_t allocSites = 0;
  int64_t staticReadBytes = 0;
  int64_t staticWriteBytes = 0;
  int64_t staticMaterializationBytes = 0;
  int64_t dynamicSites = 0;
  int64_t accessSites = 0;
  int64_t logicalReadUpperBytes = 0;
  int64_t logicalWriteUpperBytes = 0;
};

static std::optional<int64_t> getStaticBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank() || !shaped.hasStaticShape())
    return std::nullopt;
  int64_t bits = shaped.getElementTypeBitWidth();
  if (bits <= 0)
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dim : shaped.getShape()) {
    if (dim < 0 || llvm::MulOverflow(elements, dim, elements))
      return std::nullopt;
  }
  int64_t totalBits = 0;
  if (llvm::MulOverflow(elements, bits, totalBits) ||
      totalBits > std::numeric_limits<int64_t>::max() - 7)
    return std::nullopt;
  return llvm::divideCeilSigned(totalBits, int64_t{8});
}

static std::string shapeString(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return "unranked";
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto [index, dim] : llvm::enumerate(shaped.getShape())) {
    if (index)
      os << 'x';
    if (ShapedType::isDynamic(dim))
      os << '?';
    else
      os << dim;
  }
  os << 'x' << shaped.getElementType();
  return result;
}

static std::string sourceLines(Location location) {
  SmallVector<unsigned> lines;
  std::function<void(Location)> collect = [&](Location loc) {
    if (auto file = dyn_cast<FileLineColLoc>(loc)) {
      lines.push_back(file.getLine());
      return;
    }
    if (auto fused = dyn_cast<FusedLoc>(loc)) {
      for (Location child : fused.getLocations())
        collect(child);
      return;
    }
    if (auto call = dyn_cast<CallSiteLoc>(loc)) {
      collect(call.getCallee());
      collect(call.getCaller());
      return;
    }
    if (auto named = dyn_cast<NameLoc>(loc))
      collect(named.getChildLoc());
  };
  collect(location);
  std::sort(lines.begin(), lines.end());
  lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
  if (lines.empty())
    return "none";
  std::string text;
  llvm::raw_string_ostream os(text);
  for (auto [index, line] : llvm::enumerate(lines)) {
    if (index)
      os << ',';
    os << line;
  }
  return text;
}

static int64_t memorySpace(Type type) {
  if (auto memref = dyn_cast<BaseMemRefType>(type))
    return memref.getMemorySpaceAsInt();
  return -1;
}

static bool isDescriptorOnly(Operation *op) {
  return isa<memref::CastOp, memref::SubViewOp, memref::CollapseShapeOp,
             memref::ExpandShapeOp, memref::ReinterpretCastOp,
             memref::TransposeOp, tensor::CastOp, tensor::CollapseShapeOp,
             tensor::ExpandShapeOp, tensor::ExtractSliceOp>(op);
}

static bool isPhysicalLayoutTransform(Operation *op) {
  return isa<linalg::TransposeOp, linalg::PackOp, linalg::UnPackOp>(op);
}

static bool isCopy(Operation *op) {
  return isa<memref::CopyOp, linalg::CopyOp,
             bufferization::CloneOp>(op);
}

static bool isAllocation(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp>(op);
}

static StringRef engineFor(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.starts_with("hexkl."))
    return "HMX";
  if (name.starts_with("vector."))
    return "HVX";
  if (isa<linalg::ContractionOpInterface>(op))
    return "HVX_or_HMX";
  if (name.starts_with("memref.") || name.starts_with("bufferization."))
    return "memory";
  return "scalar_or_HVX";
}

static StringRef layoutFor(Type type) {
  if (auto memref = dyn_cast<MemRefType>(type)) {
    if (memref.getMemorySpaceAsInt() == 1)
      return "vtcm";
    if (memref.getLayout().isIdentity())
      return "identity";
    return "strided";
  }
  if (isa<RankedTensorType>(type))
    return "tensor_logical";
  return "unknown";
}

static StringRef kvRole(Operation *op) {
  if (auto role = op->getAttrOfType<StringAttr>("alps.kv_cache_role"))
    return role.getValue();
  return "none";
}

static bool hasTopologyCandidate(Operation *op) {
  return op->hasAttr("alps.kv_fusion_boundary") ||
         op->hasAttr("alps.kv_elementwise_fusion_boundary") ||
         op->hasAttr("alps.kv_multi_use_fusion_boundary") ||
         op->hasAttr("alps.kv_split_reduction_boundary");
}

static Value getPrimaryShapedValue(Operation *op, bool candidate) {
  if (candidate) {
    if (auto operand =
            op->getAttrOfType<IntegerAttr>("alps.kv_cache_operand")) {
      int64_t index = operand.getInt();
      if (index >= 0 && index < static_cast<int64_t>(op->getNumOperands()) &&
          isa<ShapedType>(op->getOperand(index).getType()))
        return op->getOperand(index);
    }
  }
  for (Value result : op->getResults())
    if (isa<ShapedType>(result.getType()))
      return result;
  for (Value operand : op->getOperands())
    if (isa<ShapedType>(operand.getType()))
      return operand;
  return {};
}

static std::string valueVersion(Value value,
                                const DenseMap<Operation *, int64_t> &ordinal) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    std::string result;
    llvm::raw_string_ostream os(result);
    os << "arg" << argument.getArgNumber();
    return result;
  }
  if (OpResult result = dyn_cast<OpResult>(value)) {
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "op" << ordinal.lookup(result.getOwner()) << 'r'
       << result.getResultNumber();
    return text;
  }
  return "unknown";
}

static int64_t aliasDepth(Value value) {
  int64_t depth = 0;
  while (Operation *def = value.getDefiningOp()) {
    if (!isDescriptorOnly(def) || def->getNumOperands() == 0)
      break;
    Value next;
    for (Value operand : def->getOperands())
      if (isa<ShapedType>(operand.getType())) {
        next = operand;
        break;
      }
    if (!next)
      break;
    value = next;
    ++depth;
  }
  return depth;
}

/// Follow only descriptor-preserving memref/tensor views.  This deliberately
/// stops at copies and layout-forming operations: P5h needs to distinguish a
/// second name for the same storage from a genuinely materialized
/// representation.
static Value aliasRoot(Value value) {
  SmallPtrSet<Operation *, 8> visited;
  while (Operation *def = value.getDefiningOp()) {
    if (!isDescriptorOnly(def) || def->getNumOperands() == 0 ||
        !visited.insert(def).second)
      break;
    Value next;
    for (Value operand : def->getOperands()) {
      if (isa<ShapedType>(operand.getType())) {
        next = operand;
        break;
      }
    }
    if (!next)
      break;
    value = next;
  }
  return value;
}

static std::string definingOpName(Value value) {
  if (isa<BlockArgument>(value))
    return "block_argument";
  Operation *def = value.getDefiningOp();
  return def ? def->getName().getStringRef().str() : "none";
}

static std::string directUserNames(Value value, Operation *exclude = nullptr) {
  llvm::StringSet<> names;
  for (Operation *user : value.getUsers())
    if (user != exclude)
      names.insert(user->getName().getStringRef());
  if (names.empty())
    return "none";
  SmallVector<StringRef> sorted;
  sorted.reserve(names.size());
  for (const auto &entry : names)
    sorted.push_back(entry.getKey());
  llvm::sort(sorted);
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::interleaveComma(sorted, os);
  return text;
}

static Value copySource(Operation *op) {
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return copy.getSource();
  if (auto copy = dyn_cast<linalg::CopyOp>(op))
    return copy.getInputs().front();
  if (auto clone = dyn_cast<bufferization::CloneOp>(op))
    return clone.getInput();
  return {};
}

static Value copyDestination(Operation *op) {
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return copy.getTarget();
  if (auto copy = dyn_cast<linalg::CopyOp>(op))
    return copy.getOutputs().front();
  if (auto clone = dyn_cast<bufferization::CloneOp>(op))
    return clone.getOutput();
  return {};
}

static int64_t useDistance(Value value,
                           const DenseMap<Operation *, int64_t> &ordinal,
                           int64_t producerOrdinal, int64_t &lastUse,
                           int64_t &crossBlockUses) {
  int64_t firstUse = -1;
  lastUse = -1;
  crossBlockUses = 0;
  Operation *producer = value.getDefiningOp();
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    auto it = ordinal.find(owner);
    if (it == ordinal.end())
      continue;
    int64_t position = it->second;
    if (firstUse < 0 || position < firstUse)
      firstUse = position;
    lastUse = std::max(lastUse, position);
    if (producer && producer->getBlock() != owner->getBlock())
      ++crossBlockUses;
  }
  return firstUse < 0 ? -1 : std::max<int64_t>(0, firstUse - producerOrdinal);
}

static void addBytes(int64_t &total, std::optional<int64_t> bytes) {
  if (!bytes)
    return;
  if (*bytes <= std::numeric_limits<int64_t>::max() - total)
    total += *bytes;
}

static std::optional<int64_t> multiplyBytes(int64_t lhs, int64_t rhs) {
  int64_t result = 0;
  if (lhs < 0 || rhs < 0 || llvm::MulOverflow(lhs, rhs, result))
    return std::nullopt;
  return result;
}

static std::optional<int64_t> elementBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return std::nullopt;
  int64_t bits = shaped.getElementTypeBitWidth();
  if (bits <= 0)
    return std::nullopt;
  return llvm::divideCeilSigned(bits, int64_t{8});
}

static std::string loopRangesString(ArrayRef<int64_t> ranges) {
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::interleave(ranges, os, "x");
  return text;
}

struct AlpsMovementLedgerPass
    : public ::impl::AlpsMovementLedgerBase<AlpsMovementLedgerPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    std::string records;
    llvm::raw_string_ostream recordStream(records);
    DenseMap<Operation *, int64_t> ordinal;
    int64_t nextOrdinal = 0;
    function.walk([&](Operation *op) { ordinal[op] = nextOrdinal++; });

    LedgerTotals totals;
    int64_t site = 0;
    function.walk([&](Operation *op) {
      bool descriptor = isDescriptorOnly(op);
      bool transform = isPhysicalLayoutTransform(op);
      bool copy = isCopy(op);
      bool allocation = isAllocation(op);
      bool candidate = hasTopologyCandidate(op) || kvRole(op) != "none";
      if (!descriptor && !transform && !copy && !allocation && !candidate)
        return;

      StringRef kind = candidate   ? "representation_candidate"
                       : descriptor ? "descriptor_view"
                       : transform  ? "physical_layout_transform"
                       : copy       ? "physical_copy"
                                    : "allocation";
      Value primary = getPrimaryShapedValue(op, candidate);
      Type primaryType = primary ? primary.getType() : Type();
      std::optional<int64_t> bytes =
          primaryType ? getStaticBytes(primaryType) : std::nullopt;
      if (!bytes)
        ++totals.dynamicSites;
      int64_t readBytes = 0, writeBytes = 0, materializationBytes = 0;
      if (transform || copy) {
        addBytes(readBytes, bytes);
        addBytes(writeBytes, bytes);
        addBytes(materializationBytes, bytes);
      }
      totals.staticReadBytes += readBytes;
      totals.staticWriteBytes += writeBytes;
      totals.staticMaterializationBytes += materializationBytes;
      totals.candidates += candidate;
      totals.descriptorSites += descriptor;
      totals.physicalTransformSites += transform;
      totals.copySites += copy;
      totals.allocSites += allocation;

      int64_t producerOrdinal = ordinal.lookup(op);
      int64_t lastUse = -1, crossBlockUses = 0;
      int64_t firstUseDistance =
          primary ? useDistance(primary, ordinal, producerOrdinal, lastUse,
                                crossBlockUses)
                  : -1;
      int64_t uses = primary ? std::distance(primary.use_begin(),
                                             primary.use_end())
                             : 0;
      int64_t pages =
          bytes ? llvm::divideCeilSigned(*bytes, pageBytes.getValue()) : -1;
      int64_t alignment = 1;
      if (auto shaped = dyn_cast_or_null<ShapedType>(primaryType))
        alignment = std::max<int64_t>(1, shaped.getElementTypeBitWidth() / 8);
      if (bytes && *bytes % 128 == 0)
        alignment = 128;

      std::string actions = "native";
      std::string decision = "observe_only";
      if (candidate) {
        actions += "+l2_hint+in_situ_sync";
        if (bytes && *bytes <= vtcmBudgetBytes)
          actions += "+dma_vtcm_async";
        decision = bytes ? (*bytes <= vtcmBudgetBytes
                                ? "candidate_static_vtcm_fit"
                                : "candidate_exceeds_vtcm_budget")
                         : "candidate_dynamic_shape";
      } else if (descriptor) {
        decision = "descriptor_zero_physical_bytes";
      } else if (transform || copy) {
        decision = "observed_physical_movement";
      } else if (allocation) {
        decision = "capacity_only_not_movement";
      }

      Value source = copy ? copySource(op) : Value();
      Value destination = copy ? copyDestination(op) : Value();
      Value sourceRoot = source ? aliasRoot(source) : Value();
      Value destinationRoot = destination ? aliasRoot(destination) : Value();
      bool sameType = source && destination &&
                      source.getType() == destination.getType();
      bool distinctStorage = sourceRoot && destinationRoot &&
                             sourceRoot != destinationRoot;

      recordStream << "[ALPS-P1-SITE]"
                   << " phase=" << phase << " function=" << function.getName()
                   << " id=" << function.getName() << ':' << site++
                   << " ordinal=" << producerOrdinal << " kind=" << kind
                   << " value_version="
                   << (primary ? valueVersion(primary, ordinal) : "none")
                   << " consumer_ordinal=" << producerOrdinal
                   << " op=" << op->getName() << " kv_role=" << kvRole(op)
                   << " source_lines=" << sourceLines(op->getLoc())
                   << " engine=" << engineFor(op)
                   << " layout=" << (primaryType ? layoutFor(primaryType)
                                                      : StringRef("none"))
                   << " shape=" << (primaryType ? shapeString(primaryType)
                                                     : "none")
                   << " memory_space="
                   << (primaryType ? memorySpace(primaryType) : -1)
                   << " static_bytes=" << (bytes ? *bytes : -1)
                   << " read_bytes=" << readBytes
                   << " write_bytes=" << writeBytes
                   << " materialization_bytes=" << materializationBytes
                   << " descriptor_only=" << descriptor << " uses=" << uses
                   << " first_use_distance=" << firstUseDistance
                   << " last_use_ordinal=" << lastUse
                   << " cross_block_uses=" << crossBlockUses
                   << " alias_depth=" << (primary ? aliasDepth(primary) : 0)
                   << " source_version="
                   << (source ? valueVersion(source, ordinal) : "none")
                   << " source_root_version="
                   << (sourceRoot ? valueVersion(sourceRoot, ordinal) : "none")
                   << " source_def="
                   << (source ? definingOpName(source) : "none")
                   << " source_root_def="
                   << (sourceRoot ? definingOpName(sourceRoot) : "none")
                   << " source_layout="
                   << (source ? layoutFor(source.getType()) : StringRef("none"))
                   << " source_space="
                   << (source ? memorySpace(source.getType()) : -1)
                   << " source_users="
                   << (source ? directUserNames(source, op) : "none")
                   << " destination_version="
                   << (destination ? valueVersion(destination, ordinal)
                                   : "none")
                   << " destination_root_version="
                   << (destinationRoot
                           ? valueVersion(destinationRoot, ordinal)
                           : "none")
                   << " destination_def="
                   << (destination ? definingOpName(destination) : "none")
                   << " destination_root_def="
                   << (destinationRoot ? definingOpName(destinationRoot)
                                       : "none")
                   << " destination_layout="
                   << (destination ? layoutFor(destination.getType())
                                   : StringRef("none"))
                   << " destination_space="
                   << (destination ? memorySpace(destination.getType()) : -1)
                   << " destination_users="
                   << (destination ? directUserNames(destination, op) : "none")
                   << " copy_same_type=" << sameType
                   << " copy_distinct_storage=" << distinctStorage
                   << " alignment=" << alignment << " pages=" << pages
                   << " legal_actions=" << actions
                   << " decision=" << decision << '\n';
    });

    // A materialization ledger alone misses the potentially much larger
    // volume produced by repeated operand accesses inside compute loops.  For
    // every statically shaped linalg op, report a scalar-semantic upper bound:
    // one element access per used block argument per logical iteration and one
    // output write per iteration.  Tiling, vector registers, caches and HMX may
    // reduce physical traffic, so these values are deliberately named
    // `logical_*_upper_bytes`; they are not DDR/PMU measurements.
    function.walk([&](linalg::LinalgOp linalgOp) {
      SmallVector<int64_t> ranges = linalgOp.getStaticLoopRanges();
      int64_t iterations = 1;
      bool dynamic = ranges.empty();
      for (int64_t range : ranges) {
        int64_t product = 0;
        if (range < 0 || llvm::MulOverflow(iterations, range, product)) {
          dynamic = true;
          break;
        }
        iterations = product;
      }

      int64_t readUpper = 0, writeUpper = 0, uniqueBytes = 0;
      Block &payload = linalgOp->getRegion(0).front();
      auto blockArguments = payload.getArguments();
      int64_t argumentIndex = 0;
      for (OpOperand *input : linalgOp.getDpsInputOperands()) {
        addBytes(uniqueBytes, getStaticBytes(input->get().getType()));
        bool used = argumentIndex < static_cast<int64_t>(blockArguments.size())
                        ? !blockArguments[argumentIndex].use_empty()
                        : true;
        if (!dynamic && used) {
          auto bytes = elementBytes(input->get().getType());
          addBytes(readUpper,
                   bytes ? multiplyBytes(iterations, *bytes) : std::nullopt);
        }
        ++argumentIndex;
      }
      for (Value init : linalgOp.getDpsInits()) {
        addBytes(uniqueBytes, getStaticBytes(init.getType()));
        bool read = argumentIndex < static_cast<int64_t>(blockArguments.size())
                        ? !blockArguments[argumentIndex].use_empty()
                        : true;
        if (!dynamic) {
          auto bytes = elementBytes(init.getType());
          if (read)
            addBytes(readUpper,
                     bytes ? multiplyBytes(iterations, *bytes)
                           : std::nullopt);
          addBytes(writeUpper,
                   bytes ? multiplyBytes(iterations, *bytes) : std::nullopt);
        }
        ++argumentIndex;
      }
      int64_t logicalTotal = 0;
      if (!llvm::AddOverflow(readUpper, writeUpper, logicalTotal)) {
        addBytes(totals.logicalReadUpperBytes, readUpper);
        addBytes(totals.logicalWriteUpperBytes, writeUpper);
      }
      ++totals.accessSites;
      int64_t reusePermille =
          uniqueBytes > 0 && logicalTotal <=
                                 std::numeric_limits<int64_t>::max() / 1000
              ? logicalTotal * 1000 / uniqueBytes
              : -1;
      recordStream << "[ALPS-P1-ACCESS]"
                   << " phase=" << phase << " function=" << function.getName()
                   << " ordinal=" << ordinal.lookup(linalgOp)
                   << " op=" << linalgOp->getName()
                   << " source_lines=" << sourceLines(linalgOp.getLoc())
                   << " engine=" << engineFor(linalgOp)
                   << " loop_ranges="
                   << (dynamic ? StringRef("dynamic")
                               : StringRef(loopRangesString(ranges)))
                   << " logical_iterations=" << (dynamic ? -1 : iterations)
                   << " logical_read_upper_bytes="
                   << (dynamic ? -1 : readUpper)
                   << " logical_write_upper_bytes="
                   << (dynamic ? -1 : writeUpper)
                   << " unique_operand_bytes=" << uniqueBytes
                   << " logical_to_unique_permille="
                   << (dynamic ? -1 : reusePermille)
                   << " input_operands=" << linalgOp.getNumDpsInputs()
                   << " output_operands=" << linalgOp.getNumDpsInits()
                   << " interpretation=logical_upper_bound_not_physical_ddr"
                   << '\n';
    });

    recordStream << "[ALPS-P1-SUMMARY]"
                 << " phase=" << phase << " function=" << function.getName()
                 << " candidates=" << totals.candidates
                 << " descriptor_sites=" << totals.descriptorSites
                 << " physical_transform_sites="
                 << totals.physicalTransformSites
                 << " copy_sites=" << totals.copySites
                 << " alloc_sites=" << totals.allocSites
                 << " static_read_bytes=" << totals.staticReadBytes
                 << " static_write_bytes=" << totals.staticWriteBytes
                 << " static_materialization_bytes="
                 << totals.staticMaterializationBytes
                 << " dynamic_sites=" << totals.dynamicSites
                 << " access_sites=" << totals.accessSites
                 << " logical_read_upper_bytes="
                 << totals.logicalReadUpperBytes
                 << " logical_write_upper_bytes="
                 << totals.logicalWriteUpperBytes << '\n';
    recordStream.flush();
    // Nested function passes may execute concurrently. Publish one complete
    // function ledger in a serialized write so record fields never interleave.
    static std::mutex outputMutex;
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << records;
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createAlpsMovementLedgerPass(
    const AlpsMovementLedgerOptions &options) {
  return std::make_unique<AlpsMovementLedgerPass>(options);
}
