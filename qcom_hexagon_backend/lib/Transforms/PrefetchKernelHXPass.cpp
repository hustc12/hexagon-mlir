//===- PrefetchKernelHXPass.cpp - static prefetch-kernel baseline ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::omni_fetch;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_PREFETCHKERNELHX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct BaselineStats {
  int64_t loops = 0;
  int64_t candidates = 0;
  int64_t admitted1D = 0;
  int64_t admitted2D = 0;
  int64_t hints = 0;
  int64_t requestedBytes = 0;
  int64_t rejectedDynamic = 0;
  int64_t rejectedAddress = 0;
  int64_t rejectedBounds = 0;
  int64_t rejectedWrite = 0;
  int64_t rejectedOversize = 0;
  int64_t rejectedUnmarked = 0;
};

static std::optional<int64_t> constantIndex(Value value) {
  IntegerAttr attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  return attr.getInt();
}

static std::optional<int64_t> constantFoldResult(OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return integer.getInt();
    return std::nullopt;
  }
  return constantIndex(cast<Value>(value));
}

/// The baseline must not change program ownership. Accept a view only when it
/// has a real reader and no writer or destination-style init use.
static bool isReadOnlyCandidate(memref::SubViewOp view) {
  bool hasReader = false;
  for (Operation *user : view.getResult().getUsers()) {
    if (isa<memref::LoadOp, vector::TransferReadOp>(user)) {
      hasReader = true;
      continue;
    }
    if (isa<memref::StoreOp, vector::TransferWriteOp>(user))
      return false;
    auto dps = dyn_cast<DestinationStyleOpInterface>(user);
    if (!dps)
      return false;
    for (OpOperand &operand : user->getOpOperands()) {
      if (operand.get() != view.getResult())
        continue;
      if (dps.isDpsInit(&operand))
        return false;
      hasReader = true;
    }
  }
  return hasReader;
}

struct ClassifiedView {
  scf::ForOp loop;
  memref::SubViewOp view;
  int64_t ivDimension = -1;
  int64_t tileElements = 0;
  int64_t requestedBytes = 0;
  bool isTwoDimensional = false;
};

static std::optional<ClassifiedView>
classifyView(scf::ForOp loop, memref::SubViewOp view, int64_t maxBytes,
             bool enable2D, bool requireManualSafe, BaselineStats &stats) {
  ++stats.candidates;
  if (requireManualSafe &&
      !view->hasAttrOfType<UnitAttr>("prefetch_baseline.manual_safe") &&
      !loop->hasAttrOfType<UnitAttr>("prefetch_baseline.manual_safe")) {
    ++stats.rejectedUnmarked;
    return std::nullopt;
  }
  auto sourceType = dyn_cast<MemRefType>(view.getSource().getType());
  auto viewType = dyn_cast<MemRefType>(view.getType());
  if (!sourceType || !viewType || !sourceType.hasStaticShape() ||
      !viewType.hasStaticShape() || sourceType.getMemorySpaceAsInt() != 0) {
    ++stats.rejectedDynamic;
    return std::nullopt;
  }
  if (Operation *def = view.getSource().getDefiningOp())
    if (loop->isAncestor(def)) {
      ++stats.rejectedAddress;
      return std::nullopt;
    }
  if (!isReadOnlyCandidate(view)) {
    ++stats.rejectedWrite;
    return std::nullopt;
  }

  auto lower = constantIndex(loop.getLowerBound());
  auto upper = constantIndex(loop.getUpperBound());
  auto step = constantIndex(loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *step <= 0 ||
      *upper <= *lower) {
    ++stats.rejectedDynamic;
    return std::nullopt;
  }

  SmallVector<OpFoldResult> offsets = view.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = view.getMixedSizes();
  SmallVector<OpFoldResult> strides = view.getMixedStrides();
  int64_t ivDimension = -1;
  for (auto [index, offset] : llvm::enumerate(offsets)) {
    auto value = dyn_cast<Value>(offset);
    if (!value || value != loop.getInductionVar())
      continue;
    if (ivDimension >= 0) {
      ++stats.rejectedAddress;
      return std::nullopt;
    }
    ivDimension = index;
  }
  if (ivDimension < 0 || sourceType.getDimSize(ivDimension) != *upper) {
    ++stats.rejectedAddress;
    return std::nullopt;
  }

  int64_t tileElements = 1;
  int64_t nonUnitDimensions = 0;
  for (auto [index, size] : llvm::enumerate(sizes)) {
    auto staticSize = constantFoldResult(size);
    auto staticStride = constantFoldResult(strides[index]);
    if (!staticSize || !staticStride || *staticSize <= 0 ||
        *staticStride != 1) {
      ++stats.rejectedDynamic;
      return std::nullopt;
    }
    if (*staticSize > 1)
      ++nonUnitDimensions;
    if (tileElements > INT64_MAX / *staticSize) {
      ++stats.rejectedOversize;
      return std::nullopt;
    }
    tileElements *= *staticSize;
  }
  auto ivTileSize = constantFoldResult(sizes[ivDimension]);
  if (!ivTileSize || *ivTileSize > *step || *upper % *step != 0) {
    ++stats.rejectedBounds;
    return std::nullopt;
  }

  if (nonUnitDimensions == 0 || nonUnitDimensions > 2 ||
      (nonUnitDimensions == 2 && !enable2D)) {
    ++stats.rejectedAddress;
    return std::nullopt;
  }
  SmallVector<int64_t> physicalStrides;
  int64_t physicalOffset = 0;
  if (failed(viewType.getStridesAndOffset(physicalStrides, physicalOffset)) ||
      physicalStrides.empty() || physicalStrides.back() != 1) {
    ++stats.rejectedAddress;
    return std::nullopt;
  }

  int64_t elemBytes =
      viewType.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t requestedBytes = tileElements * elemBytes;
  int64_t widthBytes = viewType.getShape().back() * elemBytes;
  if (elemBytes <= 0 || requestedBytes <= 0 || requestedBytes > maxBytes ||
      widthBytes <= 0 || widthBytes > UINT16_MAX) {
    ++stats.rejectedOversize;
    return std::nullopt;
  }

  return ClassifiedView{loop, view, ivDimension, tileElements,
                        requestedBytes, nonUnitDimensions == 2};
}

static void emitHint(const ClassifiedView &candidate, int64_t distance,
                     StringRef baselineKind, BaselineStats &stats) {
  scf::ForOp loop = candidate.loop;
  memref::SubViewOp view = candidate.view;
  OpBuilder builder(view);
  Location loc = view.getLoc();
  int64_t step = *constantIndex(loop.getStep());
  int64_t futureDelta = distance * step;
  Value delta = builder.create<arith::ConstantIndexOp>(loc, futureDelta);
  Value futureIv = builder.create<arith::AddIOp>(
      loc, loop.getInductionVar(), delta);
  int64_t ivTileSize =
      *constantFoldResult(view.getMixedSizes()[candidate.ivDimension]);
  Value tileSize = builder.create<arith::ConstantIndexOp>(loc, ivTileSize);
  Value futureEnd = builder.create<arith::AddIOp>(loc, futureIv, tileSize);
  Value inBounds = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, futureEnd, loop.getUpperBound());

  builder.create<scf::IfOp>(
      loc, inBounds, [&](OpBuilder &thenBuilder, Location thenLoc) {
        SmallVector<OpFoldResult> futureOffsets = view.getMixedOffsets();
        futureOffsets[candidate.ivDimension] = futureIv;
        Value futureView = thenBuilder.create<memref::SubViewOp>(
            thenLoc, view.getSource(), futureOffsets, view.getMixedSizes(),
            view.getMixedStrides());
        auto hint = thenBuilder.create<L2HintOp>(thenLoc, futureView,
                                                 static_cast<int32_t>(distance));
        hint->setAttr("prefetch_baseline.kind",
                      thenBuilder.getStringAttr(baselineKind));
        hint->setAttr("prefetch_baseline.address_class",
                      thenBuilder.getStringAttr(candidate.isTwoDimensional
                                                    ? "affine_2d"
                                                    : "affine_1d"));
        hint->setAttr("prefetch_baseline.distance",
                      thenBuilder.getI64IntegerAttr(distance));
        hint->setAttr("prefetch_baseline.requested_bytes",
                      thenBuilder.getI64IntegerAttr(candidate.requestedBytes));
        hint->setAttr("prefetch_baseline.page_policy",
                      thenBuilder.getStringAttr("runtime_clip_v1"));
        thenBuilder.create<scf::YieldOp>(thenLoc);
      });

  ++stats.hints;
  stats.requestedBytes += candidate.requestedBytes;
  if (candidate.isTwoDimensional)
    ++stats.admitted2D;
  else
    ++stats.admitted1D;
}

struct PrefetchKernelHXPass
    : public ::impl::PrefetchKernelHXBase<PrefetchKernelHXPass> {
  explicit PrefetchKernelHXPass() = default;
  explicit PrefetchKernelHXPass(const PrefetchKernelHXOptions &options)
      : PrefetchKernelHXBase(options) {}

  void runOnOperation() override {
    auto func = cast<func::FuncOp>(getOperation());
    BaselineStats stats;
    if (distance <= 0 || maxCommandBytes <= 0 ||
        (baselineKind != "prefetch-kernel-hx" &&
         baselineKind != "apt-get-hx")) {
      func.emitError("prefetch baseline requires a positive distance/command "
                     "budget and a supported baseline identity");
      return signalPassFailure();
    }

    SmallVector<std::pair<scf::ForOp, memref::SubViewOp>> views;
    func.walk([&](scf::ForOp loop) {
      ++stats.loops;
      for (Operation &op : loop.getBody()->without_terminator())
        if (auto view = dyn_cast<memref::SubViewOp>(op))
          views.emplace_back(loop, view);
    });
    for (auto [loop, view] : views) {
      auto candidate = classifyView(loop, view, maxCommandBytes,
                                    enableTwoDimensional, requireManualSafe,
                                    stats);
      if (candidate)
        emitHint(*candidate, distance, baselineKind, stats);
    }

    Builder b(func.getContext());
    func->setAttr("prefetch_kernel_hx.loops", b.getI64IntegerAttr(stats.loops));
    func->setAttr("prefetch_kernel_hx.candidates",
                  b.getI64IntegerAttr(stats.candidates));
    func->setAttr("prefetch_kernel_hx.admitted_1d",
                  b.getI64IntegerAttr(stats.admitted1D));
    func->setAttr("prefetch_kernel_hx.admitted_2d",
                  b.getI64IntegerAttr(stats.admitted2D));
    func->setAttr("prefetch_kernel_hx.hints",
                  b.getI64IntegerAttr(stats.hints));
    func->setAttr("prefetch_kernel_hx.requested_bytes",
                  b.getI64IntegerAttr(stats.requestedBytes));
    func->setAttr("prefetch_kernel_hx.rejected_dynamic",
                  b.getI64IntegerAttr(stats.rejectedDynamic));
    func->setAttr("prefetch_kernel_hx.rejected_address",
                  b.getI64IntegerAttr(stats.rejectedAddress));
    func->setAttr("prefetch_kernel_hx.rejected_bounds",
                  b.getI64IntegerAttr(stats.rejectedBounds));
    func->setAttr("prefetch_kernel_hx.rejected_write",
                  b.getI64IntegerAttr(stats.rejectedWrite));
    func->setAttr("prefetch_kernel_hx.rejected_oversize",
                  b.getI64IntegerAttr(stats.rejectedOversize));
    func->setAttr("prefetch_kernel_hx.rejected_unmarked",
                  b.getI64IntegerAttr(stats.rejectedUnmarked));
    func->setAttr("prefetch_kernel_hx.policy",
                  b.getStringAttr(baselineKind));

  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createPrefetchKernelHXPass(
    const PrefetchKernelHXOptions &options) {
  return std::make_unique<PrefetchKernelHXPass>(options);
}
