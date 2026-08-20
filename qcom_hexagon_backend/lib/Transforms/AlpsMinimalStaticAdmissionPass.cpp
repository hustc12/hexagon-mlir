//===- AlpsMinimalStaticAdmissionPass.cpp - ALPS P2d admission -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ALPSMINIMALSTATICADMISSION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct AdmissionStats {
  int64_t candidates = 0;
  int64_t noOp = 0;
  int64_t native = 0;
  int64_t l2Hint = 0;
  int64_t syncInSitu = 0;
  int64_t dmaVtcmAsync = 0;
  int64_t rejected = 0;
  int64_t materialized = 0;
  int64_t plannedBytes = 0;
};

struct Decision {
  StringRef kind = "unknown";
  StringRef action = "native";
  StringRef reason = "no_profitable_action";
  StringRef legalActions = "native";
  int64_t bytes = -1;
  int64_t reuse = 0;
  int64_t pages = -1;
  int64_t alignment = 1;
  int64_t overlapWindow = -1;
  bool vtcmFit = false;
  bool rejected = true;
  bool materialize = false;
};

static std::optional<int64_t> staticBytes(Type type) {
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

static std::optional<int64_t> constantInt(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(attr);
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

static std::optional<int64_t> staticTripCount(scf::ForOp loop) {
  auto lower = constantInt(loop.getLowerBound());
  auto upper = constantInt(loop.getUpperBound());
  auto step = constantInt(loop.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  if (*upper <= *lower)
    return 0;
  return (*upper - *lower + *step - 1) / *step;
}

static Value peelViews(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    break;
  }
  return value;
}

static Value kvSource(Operation *op) {
  if (auto read = dyn_cast<vector::TransferReadOp>(op))
    return read.getBase();
  if (auto operand =
          op->getAttrOfType<IntegerAttr>("omni_fetch.kv_cache_operand")) {
    int64_t index = operand.getInt();
    if (index >= 0 && index < static_cast<int64_t>(op->getNumOperands()))
      return op->getOperand(index);
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (linalgOp.getDpsInputs().size() >= 2)
      return linalgOp.getDpsInputs()[1];
  }
  return {};
}

static int64_t estimatedAlignment(Type type, std::optional<int64_t> bytes) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return 1;
  int64_t elementBytes = std::max<int64_t>(1, shaped.getElementTypeBitWidth() / 8);
  if (bytes && *bytes % 128 == 0)
    return 128;
  return elementBytes;
}

static void attachDecision(Operation *op, const Decision &decision) {
  Builder builder(op->getContext());
  op->setAttr("alps.p2d.candidate_kind",
              builder.getStringAttr(decision.kind));
  op->setAttr("alps.p2d.action", builder.getStringAttr(decision.action));
  op->setAttr("alps.p2d.reason", builder.getStringAttr(decision.reason));
  op->setAttr("alps.p2d.legal_actions",
              builder.getStringAttr(decision.legalActions));
  op->setAttr("alps.p2d.tile_bytes",
              builder.getI64IntegerAttr(decision.bytes));
  op->setAttr("alps.p2d.reuse", builder.getI64IntegerAttr(decision.reuse));
  op->setAttr("alps.p2d.pages", builder.getI64IntegerAttr(decision.pages));
  op->setAttr("alps.p2d.alignment",
              builder.getI64IntegerAttr(decision.alignment));
  op->setAttr("alps.p2d.overlap_window",
              builder.getI64IntegerAttr(decision.overlapWindow));
  op->setAttr("alps.p2d.vtcm_fit", builder.getBoolAttr(decision.vtcmFit));
  op->setAttr("alps.p2d.materialize",
              builder.getBoolAttr(decision.materialize));
}

static void countDecision(const Decision &decision, AdmissionStats &stats) {
  ++stats.candidates;
  if (decision.action == "no_op")
    ++stats.noOp;
  else if (decision.action == "l2_hint")
    ++stats.l2Hint;
  else if (decision.action == "in_situ_sync")
    ++stats.syncInSitu;
  else if (decision.action == "dma_vtcm_async")
    ++stats.dmaVtcmAsync;
  else
    ++stats.native;
  stats.rejected += decision.rejected;
  stats.materialized += decision.materialize;
  if (decision.materialize && decision.bytes > 0)
    stats.plannedBytes += decision.bytes;
}

struct AlpsMinimalStaticAdmissionPass final
    : ::impl::AlpsMinimalStaticAdmissionBase<
          AlpsMinimalStaticAdmissionPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    AdmissionStats stats;
    DenseMap<Operation *, int64_t> ordinal;
    int64_t nextOrdinal = 0;
    function.walk([&](Operation *op) { ordinal[op] = nextOrdinal++; });

    std::string records;
    llvm::raw_string_ostream record(records);
    bool hasStableP2aSummary = false;
    if (auto sites =
            function->getAttrOfType<IntegerAttr>("alps.p2a.zero_copy_sites")) {
      int64_t count = sites.getInt();
      int64_t bytes = -1;
      if (auto eliminated = function->getAttrOfType<IntegerAttr>(
              "alps.p2a.eliminated_transpose_materialization_bytes"))
        bytes = eliminated.getInt();
      if (count > 0) {
        hasStableP2aSummary = true;
        stats.candidates += count;
        stats.noOp += count;
        record << "[ALPS-P2D-SITE] function=" << function.getName()
               << " ordinal=-1 kind=zero_copy_representation action=no_op"
               << " reason=p2a_eliminated_transfer legal_actions=no_op"
               << " tile_bytes=" << bytes << " reuse=0 pages=0 alignment=1"
               << " vtcm_fit=0 overlap_window=0 materialize=0"
               << " count=" << count << '\n';
      }
    }
    function.walk([&](Operation *op) {
      Decision decision;
      Value source;
      bool candidate = false;

      if (!hasStableP2aSummary &&
          op->hasAttr("alps.p2a.zero_copy_attention")) {
        candidate = true;
        decision.kind = "zero_copy_representation";
        decision.action = "no_op";
        decision.reason = "p2a_eliminated_transfer";
        decision.legalActions = "no_op";
        auto eliminated = op->getAttrOfType<IntegerAttr>(
            "alps.p2a.eliminated_transpose_materialization_bytes");
        decision.bytes = eliminated ? eliminated.getInt() : -1;
        decision.rejected = false;
      } else if (op->hasAttr("omni_fetch.kv_cache_role")) {
        source = peelViews(kvSource(op));
        if (!source)
          return;
        candidate = true;
        decision.kind = "attention_kv_stream";
        decision.legalActions = "native+l2_hint";
        auto bytes = staticBytes(source.getType());
        decision.bytes = bytes.value_or(-1);
        decision.reuse = std::distance(source.use_begin(), source.use_end());
        decision.pages = bytes ? llvm::divideCeilSigned(*bytes, pageBytes.getValue()) : -1;
        decision.alignment = estimatedAlignment(source.getType(), bytes);
        int64_t consumerOrdinal = ordinal.lookup(op);
        if (Operation *producer = source.getDefiningOp())
          decision.overlapWindow = consumerOrdinal - ordinal.lookup(producer);
        else
          decision.overlapWindow = consumerOrdinal;

        auto blockArg = dyn_cast<BlockArgument>(source);
        bool entryArgument = blockArg &&
                             blockArg.getOwner() ==
                                 &function.getFunctionBody().front();
        auto memref = dyn_cast<MemRefType>(source.getType());
        if (!entryArgument) {
          decision.reason = "source_not_entry_persistent";
        } else if (!memref || memref.getMemorySpaceAsInt() != 0) {
          decision.reason = "source_not_ddr_l2";
        } else if (!bytes) {
          decision.reason = "dynamic_extent";
        } else if (*bytes < minL2Bytes) {
          decision.reason = "below_l2_byte_threshold";
        } else if (decision.overlapWindow < minOverlapOps) {
          decision.reason = "insufficient_overlap_window";
        } else {
          decision.action = "l2_hint";
          decision.reason = "persistent_page_safe_stream";
          decision.rejected = false;
          decision.materialize = true;
        }
      } else if (auto weight = dyn_cast<hexkl::MicroHMXRmToWhF16Op>(op)) {
        candidate = true;
        source = weight.getSrc();
        decision.kind = "hmx_weight_transform";
        decision.legalActions = "native+in_situ_sync+dma_vtcm_async";
        decision.bytes = 32 * 32 * 2;
        decision.pages = llvm::divideCeilSigned(decision.bytes, pageBytes.getValue());
        decision.alignment = estimatedAlignment(source.getType(), decision.bytes);
        auto loop = op->getParentOfType<scf::ForOp>();
        auto trips = loop ? staticTripCount(loop) : std::nullopt;
        decision.overlapWindow = trips.value_or(-1);
        decision.reuse = trips.value_or(0);
        decision.vtcmFit = 2 * decision.bytes <= vtcmBudgetBytes;
        // P2c proved that the current sync wrapper eliminates zero physical
        // bytes.  DMA becomes admissible only under the independent P3 exact
        // readiness gate; default P2d behavior remains unchanged.
        if (decision.bytes < minDmaBytes)
          decision.reason = "below_dma_byte_threshold";
        else if (!decision.vtcmFit)
          decision.reason = "vtcm_capacity";
        else if (!trips || *trips < minOverlapOps)
          decision.reason = "insufficient_overlap_window";
        else if (enableP3ExactReadiness) {
          decision.action = "dma_vtcm_async";
          decision.reason = "p3_exact_weight_pipeline";
          decision.rejected = false;
          decision.materialize = true;
        } else
          decision.reason = "requires_p3_readiness";
      } else if (auto activation =
                     dyn_cast<hexkl::MicroHMXCopySubmatrixToF16Op>(op)) {
        candidate = true;
        source = activation.getSrc();
        decision.kind = "hmx_activation_transform";
        decision.legalActions = "native+in_situ_sync+dma_vtcm_async";
        decision.bytes = 32 * 32 * 2;
        decision.pages = llvm::divideCeilSigned(decision.bytes, pageBytes.getValue());
        decision.alignment = estimatedAlignment(source.getType(), decision.bytes);
        auto loop = op->getParentOfType<scf::ForOp>();
        auto trips = loop ? staticTripCount(loop) : std::nullopt;
        decision.overlapWindow = trips.value_or(-1);
        decision.reuse = trips.value_or(0);
        decision.vtcmFit = 2 * decision.bytes <= vtcmBudgetBytes;
        decision.reason = "sync_has_zero_proven_byte_reduction";
      }

      if (!candidate)
        return;
      attachDecision(op, decision);
      countDecision(decision, stats);
      record << "[ALPS-P2D-SITE] function=" << function.getName()
             << " ordinal=" << ordinal.lookup(op)
             << " kind=" << decision.kind << " action=" << decision.action
             << " reason=" << decision.reason
             << " legal_actions=" << decision.legalActions
             << " tile_bytes=" << decision.bytes
             << " reuse=" << decision.reuse << " pages=" << decision.pages
             << " alignment=" << decision.alignment
             << " vtcm_fit=" << decision.vtcmFit
             << " overlap_window=" << decision.overlapWindow
             << " materialize=" << decision.materialize << '\n';
    });

    Builder builder(function.getContext());
    function->setAttr("alps.p2d.candidates",
                      builder.getI64IntegerAttr(stats.candidates));
    function->setAttr("alps.p2d.no_op",
                      builder.getI64IntegerAttr(stats.noOp));
    function->setAttr("alps.p2d.native",
                      builder.getI64IntegerAttr(stats.native));
    function->setAttr("alps.p2d.l2_hint",
                      builder.getI64IntegerAttr(stats.l2Hint));
    function->setAttr("alps.p2d.in_situ_sync",
                      builder.getI64IntegerAttr(stats.syncInSitu));
    function->setAttr("alps.p2d.dma_vtcm_async",
                      builder.getI64IntegerAttr(stats.dmaVtcmAsync));
    function->setAttr("alps.p2d.rejected",
                      builder.getI64IntegerAttr(stats.rejected));
    function->setAttr("alps.p2d.materialized",
                      builder.getI64IntegerAttr(stats.materialized));
    function->setAttr("alps.p2d.planned_bytes",
                      builder.getI64IntegerAttr(stats.plannedBytes));
    record << "[ALPS-P2D-SUMMARY] function=" << function.getName()
           << " candidates=" << stats.candidates << " no_op=" << stats.noOp
           << " native=" << stats.native << " l2_hint=" << stats.l2Hint
           << " in_situ_sync=" << stats.syncInSitu
           << " dma_vtcm_async=" << stats.dmaVtcmAsync
           << " rejected=" << stats.rejected
           << " materialized=" << stats.materialized
           << " planned_bytes=" << stats.plannedBytes << '\n';
    record.flush();
    static std::mutex outputMutex;
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << records;
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createAlpsMinimalStaticAdmissionPass(
    const AlpsMinimalStaticAdmissionOptions &options) {
  return std::make_unique<AlpsMinimalStaticAdmissionPass>(options);
}
