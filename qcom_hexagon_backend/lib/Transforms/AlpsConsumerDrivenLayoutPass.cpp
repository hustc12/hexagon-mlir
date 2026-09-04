//===- AlpsConsumerDrivenLayoutPass.cpp - consumer layout contracts ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Alps/IR/AlpsDialect.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
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
#define GEN_PASS_DEF_ALPSCONTINUITYAUDIT
#define GEN_PASS_DEF_ALPSLAYOUTSUPPLYPREFETCH
#define GEN_PASS_DEF_ALPSCRPSUPPLYANALYSIS
#define GEN_PASS_DEF_ALPSCRPSUPPLYPREFETCH
#define GEN_PASS_DEF_ALPSCRPVTCMFORMATION
#define GEN_PASS_DEF_ALPSCRPVTCMWINDOW
#define GEN_PASS_DEF_ALPSCRPPRODUCERDIRECTANALYSIS
#define GEN_PASS_DEF_ALPSATTENTIONDESTINATIONFORMATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static std::mutex reportMutex;

static std::string locationKey(Location location) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  location.print(stream);
  std::string result = stream.str();
  if (StringRef(result).starts_with("loc(") && StringRef(result).ends_with(")"))
    return result.substr(4, result.size() - 5);
  return result;
}

struct ContractStats {
  int64_t demands = 0;
  int64_t hvxConsumers = 0;
  int64_t hmxConsumers = 0;
  int64_t mixedConsumers = 0;
  int64_t producerDirect = 0;
  int64_t loopInterchangedDirect = 0;
  int64_t registerTileDirect = 0;
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

/// Return the constant linear coefficient of `dim` in `expr`.  Expressions
/// independent of the dimension have coefficient zero.  A missing result
/// means that the dependence is non-linear (mod/floordiv, products of two
/// non-constants, and similar gather-like addressing).
static std::optional<int64_t> linearDimCoefficient(AffineExpr expr,
                                                   unsigned dim) {
  if (!expr.isFunctionOfDim(dim))
    return 0;
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
    return dimExpr.getPosition() == dim ? std::optional<int64_t>(1)
                                        : std::optional<int64_t>(0);
  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return std::nullopt;
  if (expr.getKind() == AffineExprKind::Add) {
    auto lhs = linearDimCoefficient(binary.getLHS(), dim);
    auto rhs = linearDimCoefficient(binary.getRHS(), dim);
    int64_t sum = 0;
    if (!lhs || !rhs || llvm::AddOverflow(*lhs, *rhs, sum))
      return std::nullopt;
    return sum;
  }
  if (expr.getKind() != AffineExprKind::Mul)
    return std::nullopt;
  auto lhsConstant = dyn_cast<AffineConstantExpr>(binary.getLHS());
  auto rhsConstant = dyn_cast<AffineConstantExpr>(binary.getRHS());
  AffineExpr variable;
  int64_t factor = 0;
  if (lhsConstant) {
    factor = lhsConstant.getValue();
    variable = binary.getRHS();
  } else if (rhsConstant) {
    factor = rhsConstant.getValue();
    variable = binary.getLHS();
  } else {
    return std::nullopt;
  }
  auto coefficient = linearDimCoefficient(variable, dim);
  int64_t product = 0;
  if (!coefficient || llvm::MulOverflow(*coefficient, factor, product))
    return std::nullopt;
  return product;
}

/// A register tile may transpose the final two loop dimensions in VRF, but
/// each of those dimensions must remain a unit-step coordinate in at most one
/// result of an input map.  Outer-loop offsets (for example d1 * 32 + d2) are
/// harmless: they select the tile base and do not turn a tile lane into a
/// gather.
static bool isRegisterTileInputMapLegal(AffineMap map, unsigned outerTileDim,
                                        unsigned innerTileDim) {
  for (unsigned dim : {outerTileDim, innerTileDim}) {
    unsigned dependentResults = 0;
    for (AffineExpr expr : map.getResults()) {
      auto coefficient = linearDimCoefficient(expr, dim);
      if (!coefficient || (*coefficient != 0 && *coefficient != 1))
        return false;
      dependentResults += *coefficient == 1;
    }
    if (dependentResults > 1)
      return false;
  }
  return true;
}

/// Decompose a row-major linearized coordinate such as `d1 * 32 + d2` into
/// `{d1, d2}`.  The multiplier must exactly equal the static extent of the
/// appended dimension; arbitrary strided/gather expressions are rejected.
static FailureOr<SmallVector<unsigned>>
decomposeRowMajorCoordinate(AffineExpr expr, ArrayRef<int64_t> loopShape) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return SmallVector<unsigned>{dim.getPosition()};
  if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
    if (constant.getValue() == 0)
      return SmallVector<unsigned>{};
    return failure();
  }
  auto add = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!add || expr.getKind() != AffineExprKind::Add)
    return failure();

  auto match = [&](AffineExpr scaled,
                   AffineExpr suffix) -> FailureOr<SmallVector<unsigned>> {
    auto suffixDim = dyn_cast<AffineDimExpr>(suffix);
    auto mul = dyn_cast<AffineBinaryOpExpr>(scaled);
    if (!suffixDim || !mul || scaled.getKind() != AffineExprKind::Mul)
      return failure();
    auto lhsConstant = dyn_cast<AffineConstantExpr>(mul.getLHS());
    auto rhsConstant = dyn_cast<AffineConstantExpr>(mul.getRHS());
    int64_t factor = 0;
    AffineExpr prefix;
    if (lhsConstant) {
      factor = lhsConstant.getValue();
      prefix = mul.getRHS();
    } else if (rhsConstant) {
      factor = rhsConstant.getValue();
      prefix = mul.getLHS();
    } else {
      return failure();
    }
    unsigned suffixPosition = suffixDim.getPosition();
    if (suffixPosition >= loopShape.size() ||
        factor != loopShape[suffixPosition])
      return failure();
    FailureOr<SmallVector<unsigned>> prefixDims =
        decomposeRowMajorCoordinate(prefix, loopShape);
    if (failed(prefixDims) || llvm::is_contained(*prefixDims, suffixPosition))
      return failure();
    prefixDims->push_back(suffixPosition);
    return *prefixDims;
  };

  FailureOr<SmallVector<unsigned>> result = match(add.getLHS(), add.getRHS());
  if (succeeded(result))
    return result;
  return match(add.getRHS(), add.getLHS());
}

struct RegisterTileInput {
  Value value;
  AffineMap map;
};

/// Replace a purely descriptor-level flattened tensor dimension with an
/// expanded view whose indexing map is a projected permutation.  This lets
/// upstream linalg vectorization form a 2-D VRF tile instead of rejecting the
/// row-major affine expression.
static FailureOr<RegisterTileInput>
expandRegisterTileInput(PatternRewriter &rewriter, Location loc, Value input,
                        AffineMap map, ArrayRef<int64_t> loopShape) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape() ||
      inputType.getRank() != static_cast<int64_t>(map.getNumResults()))
    return failure();

  SmallVector<int64_t> expandedShape;
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<AffineExpr> expandedResults;
  SmallVector<bool> usedLoopDims(loopShape.size(), false);
  for (auto [inputDim, expr] : llvm::enumerate(map.getResults())) {
    FailureOr<SmallVector<unsigned>> factors =
        decomposeRowMajorCoordinate(expr, loopShape);
    if (failed(factors))
      return failure();
    ReassociationIndices group;
    int64_t product = 1;
    if (factors->empty()) {
      if (inputType.getDimSize(inputDim) != 1)
        return failure();
      group.push_back(expandedShape.size());
      expandedShape.push_back(1);
      expandedResults.push_back(
          getAffineConstantExpr(0, rewriter.getContext()));
    } else {
      for (unsigned loopDim : *factors) {
        if (loopDim >= loopShape.size() || loopShape[loopDim] <= 0 ||
            usedLoopDims[loopDim] ||
            llvm::MulOverflow(product, loopShape[loopDim], product))
          return failure();
        usedLoopDims[loopDim] = true;
        group.push_back(expandedShape.size());
        expandedShape.push_back(loopShape[loopDim]);
        expandedResults.push_back(
            getAffineDimExpr(loopDim, rewriter.getContext()));
      }
      if (product != inputType.getDimSize(inputDim))
        return failure();
    }
    reassociation.push_back(std::move(group));
  }

  AffineMap expandedMap =
      AffineMap::get(map.getNumDims(), /*symbolCount=*/0, expandedResults,
                     rewriter.getContext());
  if (!expandedMap.isProjectedPermutation(/*allowZeroInResults=*/true))
    return failure();
  if (expandedShape == inputType.getShape())
    return RegisterTileInput{input, expandedMap};

  auto expandedType = RankedTensorType::get(
      expandedShape, inputType.getElementType(), inputType.getEncoding());
  Value expanded = tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                                 input, reassociation);
  return RegisterTileInput{expanded, expandedMap};
}

/// Retarget an immediate, single-use, parallel tensor producer to the layout
/// already demanded by a transpose's consumers.  Keeping the innermost
/// dimension fixed is a deliberately conservative HVX rule: it removes one
/// full materialization without turning unit-stride element access into a
/// strided vector stream.
static LogicalResult
makeProducerDirect(linalg::TransposeOp transpose, PatternRewriter &rewriter,
                   int64_t &eliminatedBytes, bool propagateCodegenContract,
                   bool emitDischargeContract, StringRef contractId,
                   bool allowInnermostLoopInterchange,
                   bool allowRegisterTileFormation, bool &usedLoopInterchange,
                   bool &usedRegisterTile) {
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
  auto targetType =
      dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
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
  if (static_cast<int64_t>(permutation.size()) != targetRank)
    return failure();
  bool movesInnermost = permutation.back() != targetRank - 1;
  if (movesInnermost && !allowInnermostLoopInterchange &&
      !allowRegisterTileFormation)
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
  AffineMap newToOld =
      AffineMap::get(targetRank, 0, sourceLoops, rewriter.getContext());
  auto unitStrideOrInvariant = [&](AffineMap map) {
    unsigned inner = targetRank - 1;
    bool dependsOnInner = llvm::any_of(map.getResults(), [&](AffineExpr expr) {
      return expr.isFunctionOfDim(inner);
    });
    if (!dependsOnInner)
      return true;
    if (map.getNumResults() == 0)
      return false;
    for (AffineExpr expr : map.getResults().drop_back())
      if (expr.isFunctionOfDim(inner))
        return false;
    auto lastDim =
        dyn_cast<AffineDimExpr>(map.getResult(map.getNumResults() - 1));
    return lastDim && lastDim.getPosition() == inner;
  };
  bool allInputsUnitStrideOrInvariant = true;
  bool registerTileMapsLegal = true;
  SmallVector<AffineExpr> foldUnitLoops;
  foldUnitLoops.reserve(targetRank);
  for (int64_t dim = 0; dim < targetRank; ++dim)
    foldUnitLoops.push_back(
        targetType.getDimSize(dim) == 1
            ? getAffineConstantExpr(0, rewriter.getContext())
            : getAffineDimExpr(dim, rewriter.getContext()));
  for (unsigned index = 0; index < producer.getNumDpsInputs(); ++index) {
    AffineMap composed = maps[index].compose(newToOld);
    if (movesInnermost && allowRegisterTileFormation)
      composed = simplifyAffineMap(composed.replaceDimsAndSymbols(
          foldUnitLoops, /*symbolReplacements=*/{}, targetRank,
          /*numResultSyms=*/0));
    allInputsUnitStrideOrInvariant &= unitStrideOrInvariant(composed);
    // A folded static unit dimension becomes a constant-zero result.  This is
    // an invariant/broadcast lane, not an affine gather, and is therefore
    // legal for register-tile formation.
    registerTileMapsLegal &=
        isRegisterTileInputMapLegal(composed, targetRank - 2, targetRank - 1);
    maps[index] = composed;
  }
  bool cyclicInnermostPermutation =
      targetRank >= 3 && permutation.front() == 0 && permutation.back() == 1;
  for (int64_t dim = 1; dim + 1 < targetRank; ++dim)
    cyclicInnermostPermutation &= permutation[dim] == dim + 1;
  constexpr int64_t registerTileInner = 16;
  constexpr int64_t hvxVectorBits = 1024;
  int64_t elementBits = targetType.getElementTypeBitWidth();
  int64_t registerTileOuter =
      elementBits > 0
          ? std::max<int64_t>(1,
                              hvxVectorBits / (registerTileInner * elementBits))
          : 0;
  int64_t nativeVectorElements =
      elementBits > 0 ? hvxVectorBits / elementBits : 0;
  int64_t sourceInnerExtent = expandedType.getDimSize(targetRank - 1);
  // P2g-c lowers the interchanged producer input to full-width native HVX
  // loads followed by an in-register deinterleave (for example vmemu+vdeal
  // for f16).  A load is row-safe when either one physical row contains an
  // integral number of native vectors, or one native vector contains an
  // integral number of complete rows.  The second case is important for
  // Swin's 32xf16 head rows: a 128-byte load contains exactly two rows and
  // the register deinterleave never observes a partial row.  Do not admit a
  // merely contiguous allocation when neither extent divides the other;
  // Whisper's 1500xf16 row passed the older allocation-level proof, then a
  // final 64-lane load contained a partial next row and the DSP exited with
  // status 13.
  bool sourceFullVectorTailLegal =
      nativeVectorElements > 0 && sourceInnerExtent > 0 &&
      (sourceInnerExtent % nativeVectorElements == 0 ||
       nativeVectorElements % sourceInnerExtent == 0);
  // Keep the two-dimensional tile within one native v73 HVX vector.  The
  // earlier fixed 8x16 tile was 512 B for f32 and caused a full-model DSP
  // failure even though the smaller Debug extent happened to clamp to 128 B.
  registerTileOuter = std::min<int64_t>(registerTileOuter, 8);
  bool registerTileLegal =
      allowRegisterTileFormation && movesInnermost &&
      cyclicInnermostPermutation && registerTileMapsLegal &&
      registerTileOuter > 0 && sourceFullVectorTailLegal &&
      // The outer register dimension may be smaller than the preferred 8
      // lanes (for example DINOv2's six attention heads).  Tiling and
      // vectorization clamp it to the static remainder, so only reject an
      // actually empty/dynamic extent here.
      targetType.getDimSize(targetRank - 2) > 0 &&
      targetType.getDimSize(targetRank - 1) >= registerTileInner;
  SmallVector<Value> directInputs(producer.getDpsInputs());
  if (registerTileLegal && !allInputsUnitStrideOrInvariant) {
    for (unsigned index = 0; index < producer.getNumDpsInputs(); ++index) {
      FailureOr<RegisterTileInput> expanded = expandRegisterTileInput(
          rewriter, transpose.getLoc(), directInputs[index], maps[index],
          targetType.getShape());
      if (failed(expanded)) {
        registerTileLegal = false;
        break;
      }
      directInputs[index] = expanded->value;
      maps[index] = expanded->map;
    }
  }
  if (movesInnermost && !allInputsUnitStrideOrInvariant && !registerTileLegal) {
    if (allowRegisterTileFormation) {
      std::lock_guard<std::mutex> lock(reportMutex);
      llvm::errs() << "[ALPS-P2G-C-REJECT] permutation=";
      llvm::interleaveComma(permutation, llvm::errs());
      llvm::errs() << " cyclic=" << cyclicInnermostPermutation
                   << " maps_legal=" << registerTileMapsLegal
                   << " source_inner_extent=" << sourceInnerExtent
                   << " native_vector_elements=" << nativeVectorElements
                   << " source_tail_safe=" << sourceFullVectorTailLegal
                   << " outer_extent=" << targetType.getDimSize(targetRank - 2)
                   << " inner_extent=" << targetType.getDimSize(targetRank - 1)
                   << " maps=";
      llvm::interleaveComma(ArrayRef<AffineMap>(maps).drop_back(),
                            llvm::errs());
      llvm::errs() << '\n';
    }
    return failure();
  }
  maps.back() =
      AffineMap::getMultiDimIdentityMap(targetRank, rewriter.getContext());

  SmallVector<utils::IteratorType> targetIterators(
      targetRank, utils::IteratorType::parallel);

  auto direct = linalg::GenericOp::create(
      rewriter, transpose.getLoc(), TypeRange{targetType}, directInputs,
      transpose.getDpsInits(), maps, targetIterators, /*bodyBuild=*/nullptr,
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
    direct->setAttr("alps.p5a.contract_id", rewriter.getStringAttr(contractId));
  if (registerTileLegal && !allInputsUnitStrideOrInvariant) {
    direct->setAttr("alps.p2g.register_tile_contract", rewriter.getUnitAttr());
    direct->setAttr(
        "alps.p2g.register_tile_sizes",
        rewriter.getDenseI64ArrayAttr({registerTileOuter, registerTileInner}));
    direct->setAttr("alps.p2g.register_tile_permutation",
                    rewriter.getDenseI64ArrayAttr(permutation));
  }
  eliminatedBytes = staticBytes(sourceType).value_or(0);
  usedRegisterTile = registerTileLegal && !allInputsUnitStrideOrInvariant;
  usedLoopInterchange = movesInnermost && !usedRegisterTile;

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
      SmallVector<Attribute> consumerOrigins;
      llvm::StringSet<> seenConsumerOrigins;
      for (Operation *terminal : terminals) {
        std::string terminalOrigin = locationKey(terminal->getLoc());
        if (seenConsumerOrigins.insert(terminalOrigin).second)
          consumerOrigins.push_back(builder.getStringAttr(terminalOrigin));
      }
      bool movesInnermost =
          permutation.empty() ||
          permutation.back() != static_cast<int64_t>(permutation.size()) - 1;
      rewriter.setInsertionPoint(transpose);
      bool usedLoopInterchange = false;
      bool usedRegisterTile = false;
      int64_t demandId = stats.demands - 1;
      bool demandInRegisterTileWindow =
          demandId >= registerTileDemandBegin &&
          (registerTileDemandEnd < 0 || demandId < registerTileDemandEnd);
      if (!unsupported && engine != Engine::Mixed &&
          succeeded(makeProducerDirect(
              transpose, rewriter, bytes, propagateCodegenContract,
              emitDischargeContracts, contractId, allowInnermostLoopInterchange,
              allowRegisterTileFormation && demandInRegisterTileWindow,
              usedLoopInterchange,
              usedRegisterTile))) {
        ++stats.producerDirect;
        stats.loopInterchangedDirect += usedLoopInterchange;
        stats.registerTileDirect += usedRegisterTile;
        stats.eliminatedBytes += bytes;
        if (usedRegisterTile) {
          std::string record;
          llvm::raw_string_ostream os(record);
          os << "[ALPS-P2G-C-SITE] function=" << function.getName()
             << " demand=" << demandId << " origin=" << origin
             << " target_type=" << (targetType ? Type(targetType) : Type{})
             << " permutation=";
          llvm::interleaveComma(permutation, os);
          os << " eliminated_bytes=" << bytes << '\n';
          os.flush();
          std::lock_guard<std::mutex> lock(reportMutex);
          llvm::errs() << record;
        }
        if (emitDischargeContracts) {
          NamedAttrList record;
          record.set("id", builder.getStringAttr(contractId));
          record.set("origin", builder.getStringAttr(origin));
          record.set("bytes", builder.getI64IntegerAttr(bytes));
          record.set("permutation", builder.getDenseI64ArrayAttr(permutation));
          record.set("consumer_origins", builder.getArrayAttr(consumerOrigins));
          record.set("moves_innermost", builder.getBoolAttr(movesInnermost));
          record.set("formation",
                     builder.getStringAttr(usedRegisterTile ? "register_tile"
                                                            : "direct"));
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
        record.set("id",
                   builder.getStringAttr((Twine(function.getName()) +
                                          ":native:" + Twine(stats.demands - 1))
                                             .str()));
        record.set("origin", builder.getStringAttr(origin));
        record.set(
            "bytes",
            builder.getI64IntegerAttr(
                staticBytes(transpose.getInput().getType()).value_or(0)));
        record.set("permutation", builder.getDenseI64ArrayAttr(permutation));
        record.set("consumer_origins", builder.getArrayAttr(consumerOrigins));
        record.set("moves_innermost", builder.getBoolAttr(movesInnermost));
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
    function->setAttr("alps.p2g.loop_interchanged_direct",
                      builder.getI64IntegerAttr(stats.loopInterchangedDirect));
    function->setAttr("alps.p2g.register_tile_direct",
                      builder.getI64IntegerAttr(stats.registerTileDirect));
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
                 << " loop_interchanged_direct=" << stats.loopInterchangedDirect
                 << " register_tile_direct=" << stats.registerTileDirect
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
    : ::impl::AlpsContractDischargeLedgerBase<AlpsContractDischargeLedgerPass> {
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
          matchedContracts.push_back(record.getAs<StringAttr>("id").getValue());
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
        bool sameBlockAvailable =
            !source.getDefiningOp() ||
            source.getDefiningOp()->getBlock() == consumer->getBlock();
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
        llvm::errs() << " operand=" << operandIndex << " bytes=" << bytes
                     << " lead_ops=" << leadOps << " contiguous=" << contiguous
                     << " aliases_output=" << aliasesOutput << " uses="
                     << std::distance(source.use_begin(), source.use_end())
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
          matchedContracts.push_back(record.getAs<StringAttr>("id").getValue());
      }
      if (matchedContracts.empty())
        return;
      ++carriers;
      ++inputs;

      Value source = read.getBase();
      auto type = dyn_cast<MemRefType>(source.getType());
      SmallVector<int64_t> strides;
      int64_t offset = 0;
      bool contiguous = type &&
                        succeeded(type.getStridesAndOffset(strides, offset)) &&
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
      bool sameBlockAvailable =
          !source.getDefiningOp() ||
          source.getDefiningOp()->getBlock() == consumer->getBlock();
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
                   << " tile_bytes=" << tileBytes << " lead_ops=" << leadOps
                   << " contiguous=" << contiguous << " uses="
                   << std::distance(source.use_begin(), source.use_end())
                   << " decision=" << (candidate ? "admit" : "reject") << '\n';
    });
    llvm::errs() << "[ALPS-P5B-SUMMARY] function=" << function.getName()
                 << " carriers=" << carriers << " inputs=" << inputs
                 << " admitted=" << admitted
                 << " admitted_bytes=" << admittedBytes
                 << " min_lead_ops=" << minLeadOps << " min_bytes=" << minBytes
                 << '\n';
  }
};

struct ContinuityStats {
  int64_t reads = 0;
  int64_t writes = 0;
  int64_t unitStrideReads = 0;
  int64_t unitStrideWrites = 0;
  int64_t vmemuRisk = 0;
  int64_t staticTileBytes = 0;
};

static bool p2gLocationMatches(StringRef candidate, StringRef origin) {
  return origin != "unknown" && !origin.empty() && candidate.contains(origin);
}

static bool p2gUnitStride(Value base, AffineMap permutationMap) {
  auto type = dyn_cast<MemRefType>(base.getType());
  if (!type || !permutationMap.isMinorIdentity())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         !strides.empty() && strides.back() == 1;
}

static void p2gRecordRead(vector::TransferReadOp read, ContinuityStats &stats) {
  ++stats.reads;
  bool unitStride = p2gUnitStride(read.getBase(), read.getPermutationMap());
  stats.unitStrideReads += unitStride;
  stats.vmemuRisk += !unitStride;
  stats.staticTileBytes += staticBytes(read.getVectorType()).value_or(0);
}

static void p2gRecordWrite(vector::TransferWriteOp write,
                           ContinuityStats &stats) {
  ++stats.writes;
  bool unitStride = p2gUnitStride(write.getBase(), write.getPermutationMap());
  stats.unitStrideWrites += unitStride;
  stats.vmemuRisk += !unitStride;
  stats.staticTileBytes +=
      staticBytes(cast<VectorType>(write.getVector().getType())).value_or(0);
}

static StringRef p2gContinuityState(int64_t unitStride, int64_t total) {
  if (total == 0)
    return "unobserved";
  if (unitStride == total)
    return "unit_stride";
  if (unitStride == 0)
    return "strided";
  return "mixed";
}

/// P2g-a deliberately runs after vectorization and one-shot bufferization.
/// Tensor-level indexing maps are useful intent, but final vector transfers
/// are the first stable boundary at which a deleted transpose can be
/// distinguished from a strided physical stream.
struct AlpsContinuityAuditPass final
    : ::impl::AlpsContinuityAuditBase<AlpsContinuityAuditPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    auto directContracts =
        function->getAttrOfType<ArrayAttr>("alps.p5a.contracts");
    auto nativeContracts =
        function->getAttrOfType<ArrayAttr>("alps.p5a.native_demands");
    auto p2aContracts =
        function->getAttrOfType<ArrayAttr>("alps.p2g.p2a_contracts");
    if (!directContracts && !nativeContracts && !p2aContracts)
      return;

    SmallVector<vector::TransferReadOp> reads;
    SmallVector<vector::TransferWriteOp> writes;
    function.walk([&](vector::TransferReadOp op) { reads.push_back(op); });
    function.walk([&](vector::TransferWriteOp op) { writes.push_back(op); });

    int64_t totalContracts = 0;
    int64_t observedContracts = 0;
    int64_t movesInnermostContracts = 0;
    int64_t producerReadTransfers = 0;
    int64_t producerUnitReads = 0;
    int64_t producerWriteTransfers = 0;
    int64_t producerUnitWrites = 0;
    int64_t consumerReadTransfers = 0;
    int64_t consumerUnitReads = 0;
    int64_t vmemuRiskTransfers = 0;
    int64_t staticTileBytes = 0;

    auto auditRecords = [&](ArrayAttr records, StringRef kind) {
      if (!records)
        return;
      for (Attribute attr : records) {
        auto record = dyn_cast<DictionaryAttr>(attr);
        if (!record)
          continue;
        auto idAttr = record.getAs<StringAttr>("id");
        auto originAttr = record.getAs<StringAttr>("origin");
        if (!idAttr || !originAttr)
          continue;
        ++totalContracts;
        bool movesInnermost = false;
        if (auto value = record.getAs<BoolAttr>("moves_innermost"))
          movesInnermost = value.getValue();
        movesInnermostContracts += movesInnermost;

        ContinuityStats producer;
        ContinuityStats consumer;
        StringRef origin = originAttr.getValue();
        for (vector::TransferReadOp read : reads) {
          std::string location = locationKey(read.getLoc());
          if (kind != "p2a" && p2gLocationMatches(location, origin))
            p2gRecordRead(read, producer);
        }
        for (vector::TransferWriteOp write : writes) {
          std::string location = locationKey(write.getLoc());
          if (kind != "p2a" && p2gLocationMatches(location, origin))
            p2gRecordWrite(write, producer);
        }

        if (kind == "p2a") {
          for (vector::TransferReadOp read : reads) {
            std::string location = locationKey(read.getLoc());
            if (p2gLocationMatches(location, origin))
              p2gRecordRead(read, consumer);
          }
        } else if (auto consumerOrigins =
                       record.getAs<ArrayAttr>("consumer_origins")) {
          for (Attribute consumerOriginAttr : consumerOrigins) {
            auto stringAttr = dyn_cast<StringAttr>(consumerOriginAttr);
            if (!stringAttr)
              continue;
            for (vector::TransferReadOp read : reads) {
              std::string location = locationKey(read.getLoc());
              if (p2gLocationMatches(location, stringAttr.getValue()))
                p2gRecordRead(read, consumer);
            }
          }
        }

        bool observed = producer.reads || producer.writes || consumer.reads;
        observedContracts += observed;
        producerReadTransfers += producer.reads;
        producerUnitReads += producer.unitStrideReads;
        producerWriteTransfers += producer.writes;
        producerUnitWrites += producer.unitStrideWrites;
        consumerReadTransfers += consumer.reads;
        consumerUnitReads += consumer.unitStrideReads;
        vmemuRiskTransfers += producer.vmemuRisk + consumer.vmemuRisk;
        staticTileBytes += producer.staticTileBytes + consumer.staticTileBytes;

        std::lock_guard<std::mutex> lock(reportMutex);
        llvm::errs()
            << "[ALPS-P2G-CONTRACT] function=" << function.getName()
            << " kind=" << kind << " id=" << idAttr.getValue()
            << " moves_innermost=" << movesInnermost << " producer_read="
            << p2gContinuityState(producer.unitStrideReads, producer.reads)
            << " producer_write="
            << p2gContinuityState(producer.unitStrideWrites, producer.writes)
            << " consumer_read="
            << p2gContinuityState(consumer.unitStrideReads, consumer.reads)
            << " producer_reads=" << producer.reads
            << " producer_writes=" << producer.writes
            << " consumer_reads=" << consumer.reads
            << " vmemu_risk=" << producer.vmemuRisk + consumer.vmemuRisk
            << " static_tile_bytes="
            << producer.staticTileBytes + consumer.staticTileBytes << '\n';
      }
    };

    auditRecords(p2aContracts, "p2a");
    auditRecords(directContracts, "p2e_direct");
    auditRecords(nativeContracts, "p2e_native");

    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P2G-SUMMARY] function=" << function.getName()
                 << " contracts=" << totalContracts
                 << " observed=" << observedContracts
                 << " moves_innermost=" << movesInnermostContracts
                 << " producer_reads=" << producerReadTransfers
                 << " producer_unit_reads=" << producerUnitReads
                 << " producer_writes=" << producerWriteTransfers
                 << " producer_unit_writes=" << producerUnitWrites
                 << " consumer_reads=" << consumerReadTransfers
                 << " consumer_unit_reads=" << consumerUnitReads
                 << " vmemu_risk=" << vmemuRiskTransfers
                 << " static_tile_bytes=" << staticTileBytes << '\n';
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
    : ::impl::AlpsLayoutSupplyPrefetchBase<AlpsLayoutSupplyPrefetchPass> {
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
      int64_t tileBytes =
          llvm::divideCeilSigned(tileElements * elemBits, int64_t{8});
      if (elemBits <= 0 || tileBytes <= 0 || tileBytes > maxBytes) {
        ++rejectedSize;
        continue;
      }

      OpBuilder builder(view);
      Location loc = read.getLoc();
      Value delta =
          builder.create<arith::ConstantIndexOp>(loc, distance * *step);
      Value futureIv =
          builder.create<arith::AddIOp>(loc, loop.getInductionVar(), delta);
      Value tileSize = builder.create<arith::ConstantIndexOp>(loc, *ivTileSize);
      Value futureEnd = builder.create<arith::AddIOp>(loc, futureIv, tileSize);
      Value inBounds = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, futureEnd, loop.getUpperBound());
      builder.create<scf::IfOp>(
          loc, inBounds, [&](OpBuilder &thenBuilder, Location thenLoc) {
            SmallVector<OpFoldResult> futureOffsets = offsets;
            futureOffsets[ivDimension] = futureIv;
            Value futureView = thenBuilder.create<memref::SubViewOp>(
                thenLoc, view.getSource(), futureOffsets, sizes, strides);
            auto hint = thenBuilder.create<alps::L2HintOp>(
                thenLoc, futureView, static_cast<int32_t>(distance));
            hint->setAttr("alps.p5c.layout_supply", thenBuilder.getUnitAttr());
            hint->setAttr("alps.p5c.requested_bytes",
                          thenBuilder.getI64IntegerAttr(tileBytes));
            thenBuilder.create<scf::YieldOp>(thenLoc);
          });
      ++admitted;
      requestedBytes += tileBytes;
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5c.matched", builder.getI64IntegerAttr(matched));
    function->setAttr("alps.p5c.admitted", builder.getI64IntegerAttr(admitted));
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

static Value p5fRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto view = dyn_cast<memref::SubViewOp>(def))
      value = view.getSource();
    else if (auto expand = dyn_cast<memref::ExpandShapeOp>(def))
      value = expand.getSrc();
    else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def))
      value = collapse.getSrc();
    else if (auto cast = dyn_cast<memref::CastOp>(def))
      value = cast.getSource();
    else
      break;
  }
  return value;
}

static bool p5fDependsOn(Value value, Value needle,
                         SmallPtrSetImpl<Operation *> &visited) {
  if (value == needle)
    return true;
  Operation *def = value.getDefiningOp();
  if (!def || !visited.insert(def).second)
    return false;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return p5fDependsOn(operand, needle, visited);
  });
}

static bool p5fDependsOn(Value value, Value needle) {
  SmallPtrSet<Operation *, 8> visited;
  return p5fDependsOn(value, needle, visited);
}

/// P5f-a intentionally consumes only P2g-c's explicit vector marker.  It does
/// not infer candidates from location strings and cannot broaden admission to
/// unrelated graph traffic.
struct AlpsCrpSupplyAnalysisPass final
    : ::impl::AlpsCrpSupplyAnalysisBase<AlpsCrpSupplyAnalysisPass> {
  explicit AlpsCrpSupplyAnalysisPass(
      const AlpsCrpSupplyAnalysisOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    const int64_t pageSize = pageBytes;
    const int64_t minLead = minLeadIterations;
    if (pageSize <= 0 || minLead <= 0) {
      function.emitError("ALPS P5f-a requires positive page/lead settings");
      return signalPassFailure();
    }

    SmallVector<vector::TransferReadOp> reads;
    function.walk([&](vector::TransferReadOp read) {
      if (read->hasAttr("alps.p2g.register_tile"))
        reads.push_back(read);
    });

    DenseMap<Value, int64_t> rootReuse;
    for (vector::TransferReadOp read : reads)
      ++rootReuse[p5fRoot(read.getBase())];

    int64_t admitted = 0;
    int64_t admittedBytes = 0;
    int64_t rejectedStride = 0;
    int64_t rejectedLoop = 0;
    int64_t rejectedCausal = 0;
    int64_t rejectedAddress = 0;
    int64_t rejectedReadOnly = 0;
    int64_t maxWorstPages = 0;

    for (auto [ordinal, read] : llvm::enumerate(reads)) {
      Value root = p5fRoot(read.getBase());
      auto vectorBytes = staticBytes(read.getVectorType());
      bool unitStride = p2gUnitStride(read.getBase(), read.getPermutationMap());
      scf::ForOp loop = read->getParentOfType<scf::ForOp>();
      bool loopCarriedAddress = false;
      bool exactSubview = false;
      int64_t dependentSubviews = 0;
      bool sourceInvariant = false;
      bool readOnly = true;
      int64_t tripCount = 0;

      if (loop) {
        auto lower = p5ConstantIndex(loop.getLowerBound());
        auto upper = p5ConstantIndex(loop.getUpperBound());
        auto step = p5ConstantIndex(loop.getStep());
        if (lower && upper && step && *step > 0 && *upper > *lower)
          tripCount = llvm::divideCeilSigned(*upper - *lower, *step);
        sourceInvariant =
            !root.getDefiningOp() || !loop->isAncestor(root.getDefiningOp());

        Value cursor = read.getBase();
        while (Operation *def = cursor.getDefiningOp()) {
          if (auto view = dyn_cast<memref::SubViewOp>(def)) {
            int64_t dependentOffsets = 0;
            bool staticUnitSlices = true;
            for (auto [index, offset] :
                 llvm::enumerate(view.getMixedOffsets())) {
              if (auto offsetValue = dyn_cast<Value>(offset))
                dependentOffsets +=
                    p5fDependsOn(offsetValue, loop.getInductionVar());
              auto size = p5ConstantFoldResult(view.getMixedSizes()[index]);
              auto stride = p5ConstantFoldResult(view.getMixedStrides()[index]);
              staticUnitSlices &= size && *size > 0 && stride && *stride == 1;
            }
            bool exactDependent = dependentOffsets == 1 && staticUnitSlices;
            dependentSubviews += exactDependent;
            loopCarriedAddress |= exactDependent;
            exactSubview |= exactDependent;
            cursor = view.getSource();
            continue;
          }
          if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
            cursor = expand.getSrc();
            continue;
          }
          if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
            cursor = collapse.getSrc();
            continue;
          }
          if (auto cast = dyn_cast<memref::CastOp>(def)) {
            cursor = cast.getSource();
            continue;
          }
          break;
        }

        loop.walk([&](Operation *op) {
          if (auto write = dyn_cast<vector::TransferWriteOp>(op))
            readOnly &= p5fRoot(write.getBase()) != root;
          else if (auto copy = dyn_cast<memref::CopyOp>(op))
            readOnly &= p5fRoot(copy.getTarget()) != root;
          else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
            for (Value output : linalgOp.getDpsInits())
              readOnly &= p5fRoot(output) != root;
        });
      }

      int64_t bytes = vectorBytes.value_or(0);
      int64_t worstPages =
          bytes > 0 ? llvm::divideCeilSigned(bytes + pageSize - 1, pageSize)
                    : 0;
      maxWorstPages = std::max(maxWorstPages, worstPages);
      bool enoughLead = tripCount > minLead;
      bool candidate = unitStride && loop && sourceInvariant && readOnly &&
                       exactSubview && loopCarriedAddress && enoughLead &&
                       bytes > 0;
      admitted += candidate;
      admittedBytes += candidate ? bytes : 0;
      rejectedStride += !unitStride;
      rejectedLoop += !loop || !enoughLead;
      rejectedCausal += loop && !sourceInvariant;
      rejectedAddress += loop && (!exactSubview || !loopCarriedAddress);
      rejectedReadOnly += loop && !readOnly;

      std::lock_guard<std::mutex> lock(reportMutex);
      auto viewType = dyn_cast<MemRefType>(read.getBase().getType());
      Operation *rootDef = root.getDefiningOp();
      llvm::errs() << "[ALPS-P5F-A-SITE] function=" << function.getName()
                   << " id=" << ordinal << " candidate=" << candidate
                   << " unit_stride=" << unitStride
                   << " external_source=" << sourceInvariant
                   << " read_only=" << readOnly
                   << " loop_carried_address=" << loopCarriedAddress
                   << " dependent_subviews=" << dependentSubviews
                   << " trip_count=" << tripCount
                   << " lead_iterations=" << (enoughLead ? minLead : 0)
                   << " tile_bytes=" << bytes << " worst_pages=" << worstPages
                   << " reuse_reads=" << rootReuse.lookup(root)
                   << " view_type=" << (viewType ? Type(viewType) : Type{})
                   << " root_op="
                   << (rootDef ? rootDef->getName().getStringRef()
                               : StringRef("function_argument"));
      if (auto global = dyn_cast_or_null<memref::GetGlobalOp>(rootDef))
        llvm::errs() << " root_symbol=" << global.getName();
      llvm::errs() << '\n';
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5f_a.matched",
                      builder.getI64IntegerAttr(reads.size()));
    function->setAttr("alps.p5f_a.admitted",
                      builder.getI64IntegerAttr(admitted));
    function->setAttr("alps.p5f_a.admitted_bytes",
                      builder.getI64IntegerAttr(admittedBytes));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5F-A-SUMMARY] function=" << function.getName()
                 << " matched=" << reads.size() << " admitted=" << admitted
                 << " admitted_bytes=" << admittedBytes
                 << " reject_stride=" << rejectedStride
                 << " reject_loop_or_lead=" << rejectedLoop
                 << " reject_causal=" << rejectedCausal
                 << " reject_address=" << rejectedAddress
                 << " reject_read_only=" << rejectedReadOnly
                 << " max_worst_pages=" << maxWorstPages << '\n';
  }
};

/// Find the unique loop-carried subview below descriptor-only reshapes/casts.
/// These operations preserve the underlying allocation and do not move data,
/// so issuing a hint on the reconstructed subview uses the same base address as
/// the marked transfer.  Any other producer in the chain remains unsupported.
static memref::SubViewOp p5fFindExactSubview(Value value, scf::ForOp loop,
                                             bool &unsupportedChain,
                                             int64_t &dependentSubviews,
                                             std::string &chain) {
  memref::SubViewOp result;
  llvm::raw_string_ostream chainStream(chain);
  while (Operation *def = value.getDefiningOp()) {
    if (!chain.empty())
      chainStream << "->";
    chainStream << def->getName().getStringRef();
    if (auto view = dyn_cast<memref::SubViewOp>(def)) {
      bool dependsOnIv =
          llvm::any_of(view.getMixedOffsets(), [&](OpFoldResult offset) {
            if (auto offsetValue = dyn_cast<Value>(offset))
              return p5fDependsOn(offsetValue, loop.getInductionVar());
            return false;
          });
      if (dependsOnIv) {
        ++dependentSubviews;
        chainStream << "[iv]";
        if (result) {
          unsupportedChain = true;
          return {};
        }
        result = view;
      }
      value = view.getSource();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      value = expand.getSrc();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    // Once the unique IV-dependent subview has been found, its source is the
    // invariant allocation/root and need not itself be a descriptor op.  An
    // unknown op is unsafe only when it appears between the transfer and the
    // subview we must replay.
    if (result) {
      chainStream << "[root]";
      return result;
    }
    unsupportedChain = true;
    return {};
  }
  return result;
}

/// P5f-b is deliberately stricter than the analysis pass: insertion currently
/// requires the loop IV itself to be the unique loop-carried subview offset.
/// More general affine address replay can be added later, but must never be
/// approximated.
struct AlpsCrpSupplyPrefetchPass final
    : ::impl::AlpsCrpSupplyPrefetchBase<AlpsCrpSupplyPrefetchPass> {
  explicit AlpsCrpSupplyPrefetchPass(
      const AlpsCrpSupplyPrefetchOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    const int64_t lead = distance;
    const int64_t byteLimit = maxBytes;
    const int64_t pageSize = pageBytes;
    const int64_t pageLimit = maxWorstPages;
    const int64_t lineBytes = l2LineBytes;
    const int64_t minSegmentUtilization = minSegmentUtilizationPercent;
    if (lead <= 0 || byteLimit <= 0 || pageSize <= 0 || pageLimit <= 0) {
      function.emitError("ALPS P5f-b requires positive lead/byte/page limits");
      return signalPassFailure();
    }
    if (pageSafeSegmented &&
        (lineBytes <= 0 || minSegmentUtilization <= 0 ||
         minSegmentUtilization > 100)) {
      function.emitError(
          "ALPS P5f-c requires a positive L2 line size and utilization in "
          "[1,100]");
      return signalPassFailure();
    }

    SmallVector<vector::TransferReadOp> reads;
    function.walk([&](vector::TransferReadOp read) {
      if (read->hasAttr("alps.p2g.register_tile"))
        reads.push_back(read);
    });

    int64_t admitted = 0;
    int64_t requestedBytes = 0;
    int64_t rejectedView = 0;
    int64_t rejectedStride = 0;
    int64_t rejectedCausal = 0;
    int64_t rejectedReadOnly = 0;
    int64_t rejectedAddress = 0;
    int64_t rejectedBounds = 0;
    int64_t rejectedSizePage = 0;
    int64_t contiguousHints = 0;
    int64_t segmentedHints = 0;
    int64_t physicalRows = 0;
    int64_t rejectedSegmentUtilization = 0;

    for (auto [ordinal, read] : llvm::enumerate(reads)) {
      if (!p2gUnitStride(read.getBase(), read.getPermutationMap())) {
        ++rejectedStride;
        continue;
      }
      scf::ForOp loop = read->getParentOfType<scf::ForOp>();
      bool unsupportedChain = false;
      int64_t dependentSubviews = 0;
      std::string viewChain;
      auto view =
          loop ? p5fFindExactSubview(read.getBase(), loop, unsupportedChain,
                                     dependentSubviews, viewChain)
               : memref::SubViewOp{};
      if (!view || unsupportedChain) {
        ++rejectedView;
        std::lock_guard<std::mutex> lock(reportMutex);
        llvm::errs() << "[ALPS-P5F-B-VIEW] function=" << function.getName()
                     << " id=" << ordinal
                     << " dependent_subviews=" << dependentSubviews
                     << " unsupported_chain=" << unsupportedChain
                     << " chain=" << viewChain << '\n';
        continue;
      }
      if (!loop || !loop->isAncestor(view)) {
        ++rejectedAddress;
        continue;
      }

      Value root = p5fRoot(view.getSource());
      bool sourceInvariant =
          !root.getDefiningOp() || !loop->isAncestor(root.getDefiningOp());
      if (!sourceInvariant) {
        ++rejectedCausal;
        continue;
      }
      bool readOnly = true;
      loop.walk([&](Operation *op) {
        if (auto write = dyn_cast<vector::TransferWriteOp>(op))
          readOnly &= p5fRoot(write.getBase()) != root;
        else if (auto copy = dyn_cast<memref::CopyOp>(op))
          readOnly &= p5fRoot(copy.getTarget()) != root;
        else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
          for (Value output : linalgOp.getDpsInits())
            readOnly &= p5fRoot(output) != root;
      });
      if (!readOnly) {
        ++rejectedReadOnly;
        continue;
      }

      SmallVector<OpFoldResult> offsets = view.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = view.getMixedSizes();
      SmallVector<OpFoldResult> strides = view.getMixedStrides();
      int64_t ivDimension = -1;
      bool exactAddress = true;
      for (auto [index, offset] : llvm::enumerate(offsets)) {
        if (auto value = dyn_cast<Value>(offset)) {
          if (value == loop.getInductionVar()) {
            if (ivDimension >= 0)
              exactAddress = false;
            ivDimension = index;
          } else if (p5fDependsOn(value, loop.getInductionVar())) {
            exactAddress = false;
          }
        }
        auto size = p5ConstantFoldResult(sizes[index]);
        auto stride = p5ConstantFoldResult(strides[index]);
        exactAddress &= size && *size > 0 && stride && *stride == 1;
      }
      if (!exactAddress || ivDimension < 0) {
        ++rejectedAddress;
        continue;
      }

      auto lower = p5ConstantIndex(loop.getLowerBound());
      auto upper = p5ConstantIndex(loop.getUpperBound());
      auto step = p5ConstantIndex(loop.getStep());
      auto ivSize = p5ConstantFoldResult(sizes[ivDimension]);
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
          !ivSize || *ivSize <= 0 || *ivSize > *step ||
          llvm::divideCeilSigned(*upper - *lower, *step) <= lead) {
        ++rejectedBounds;
        continue;
      }

      auto viewType = dyn_cast<MemRefType>(view.getType());
      auto sourceType = dyn_cast<MemRefType>(view.getSource().getType());
      auto vectorByteCount = staticBytes(read.getVectorType());
      auto viewByteCount = viewType ? staticBytes(viewType) : std::nullopt;
      SmallVector<int64_t> physicalStrides;
      int64_t physicalOffset = 0;
      bool staticInnerContiguous =
          viewType && sourceType && viewType.hasStaticShape() &&
          sourceType.hasStaticShape() &&
          sourceType.getMemorySpaceAsInt() == 0 &&
          succeeded(
              viewType.getStridesAndOffset(physicalStrides, physicalOffset)) &&
          !physicalStrides.empty() && physicalStrides.back() == 1;
      bool physicallyContiguous = staticInnerContiguous;
      int64_t expectedStride = 1;
      int64_t rowCount = 1;
      if (staticInnerContiguous) {
        for (int64_t dim = viewType.getRank() - 1; dim >= 0; --dim) {
          int64_t size = viewType.getDimSize(dim);
          if (size > 1 && physicalStrides[dim] != expectedStride)
            physicallyContiguous = false;
          expectedStride *= size;
          if (dim < viewType.getRank() - 1)
            rowCount *= size;
        }
      }
      int64_t bytes = vectorByteCount.value_or(0);
      int64_t worstPages =
          bytes > 0 ? llvm::divideCeilSigned(bytes + pageSize - 1, pageSize)
                    : 0;
      if (!staticInnerContiguous || !viewByteCount || *viewByteCount != bytes ||
          bytes <= 0 || bytes > byteLimit || worstPages > pageLimit) {
        ++rejectedSizePage;
        continue;
      }

      // A segmented l2fetch fetches complete cache lines even when only a few
      // bytes in each physical row are useful.  DINOv2's [16x4xf16] CRP tile,
      // for example, exposes 8 useful bytes every 768 bytes: rotating across
      // its 16 rows turns 128 useful bytes into sixteen 128-byte line fills
      // (6.25% utilization) and exhausted the command budget.  Admit physical
      // row segmentation only when its useful-byte ratio is high enough.
      int64_t rowBytes =
          viewType.getDimSize(viewType.getRank() - 1) *
          viewType.getElementTypeBitWidth() / 8;
      int64_t segmentUtilization = 100;
      if (pageSafeSegmented && !physicallyContiguous) {
        int64_t physicalBytes =
            rowCount * llvm::alignTo(rowBytes, lineBytes);
        segmentUtilization =
            physicalBytes > 0 ? (bytes * 100) / physicalBytes : 0;
        if (segmentUtilization < minSegmentUtilization) {
          ++rejectedSegmentUtilization;
          std::lock_guard<std::mutex> lock(reportMutex);
          llvm::errs() << "[ALPS-P5F-C-REJECT] function="
                       << function.getName() << " id=" << ordinal
                       << " reason=segment_utilization row_bytes=" << rowBytes
                       << " physical_rows=" << rowCount
                       << " tile_bytes=" << bytes
                       << " utilization_percent=" << segmentUtilization
                       << " minimum_percent=" << minSegmentUtilization << '\n';
          continue;
        }
      }

      OpBuilder builder(view);
      Location loc = read.getLoc();
      Value delta = builder.create<arith::ConstantIndexOp>(loc, lead * *step);
      Value futureIv =
          builder.create<arith::AddIOp>(loc, loop.getInductionVar(), delta);
      Value tileSize = builder.create<arith::ConstantIndexOp>(loc, *ivSize);
      Value futureEnd = builder.create<arith::AddIOp>(loc, futureIv, tileSize);
      Value inBounds = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, futureEnd, loop.getUpperBound());
      int64_t siteOrdinal = ordinal;
      builder.create<scf::IfOp>(
          loc, inBounds, [&](OpBuilder &thenBuilder, Location thenLoc) {
            SmallVector<OpFoldResult> futureOffsets = offsets;
            futureOffsets[ivDimension] = futureIv;
            Value futureView = thenBuilder.create<memref::SubViewOp>(
                thenLoc, view.getSource(), futureOffsets, sizes, strides);
            auto hint = thenBuilder.create<alps::L2HintOp>(
                thenLoc, futureView, static_cast<int32_t>(lead));
            hint->setAttr("alps.p5f_b.crp_supply", thenBuilder.getUnitAttr());
            hint->setAttr("alps.p5f_b.requested_bytes",
                          thenBuilder.getI64IntegerAttr(bytes));
            hint->setAttr("alps.p5f_b.page_policy",
                          thenBuilder.getStringAttr("runtime_clip_v1"));
            if (pageSafeSegmented) {
              hint->setAttr("alps.p5f_c.page_safe_segmented",
                            thenBuilder.getUnitAttr());
              hint->setAttr("alps.p5f_c.physically_contiguous",
                            thenBuilder.getBoolAttr(physicallyContiguous));
              hint->setAttr("alps.p5f_c.physical_rows",
                            thenBuilder.getI64IntegerAttr(rowCount));
              hint->setAttr("alps.p5f_c.segment_utilization_percent",
                            thenBuilder.getI64IntegerAttr(segmentUtilization));
              hint->setAttr("alps.p5f_c.site_id",
                            thenBuilder.getI64IntegerAttr(siteOrdinal));
            }
            thenBuilder.create<scf::YieldOp>(thenLoc);
          });
      ++admitted;
      requestedBytes += bytes;
      if (pageSafeSegmented) {
        contiguousHints += physicallyContiguous;
        segmentedHints += !physicallyContiguous;
        physicalRows += rowCount;
        std::lock_guard<std::mutex> lock(reportMutex);
        llvm::errs() << "[ALPS-P5F-C-SITE] function=" << function.getName()
                     << " id=" << ordinal
                     << " physically_contiguous=" << physicallyContiguous
                     << " physical_rows=" << rowCount << " row_bytes="
                     << rowBytes
                     << " tile_bytes=" << bytes << " shape=" << viewType
                     << " strides=";
        llvm::interleaveComma(physicalStrides, llvm::errs());
        llvm::errs() << '\n';
      }
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5f_b.matched",
                      builder.getI64IntegerAttr(reads.size()));
    function->setAttr("alps.p5f_b.admitted",
                      builder.getI64IntegerAttr(admitted));
    function->setAttr("alps.p5f_b.requested_bytes",
                      builder.getI64IntegerAttr(requestedBytes));
    if (pageSafeSegmented) {
      function->setAttr("alps.p5f_c.contiguous_hints",
                        builder.getI64IntegerAttr(contiguousHints));
      function->setAttr("alps.p5f_c.segmented_hints",
                        builder.getI64IntegerAttr(segmentedHints));
      function->setAttr("alps.p5f_c.physical_rows",
                        builder.getI64IntegerAttr(physicalRows));
      function->setAttr(
          "alps.p5f_c.rejected_segment_utilization",
          builder.getI64IntegerAttr(rejectedSegmentUtilization));
    }
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5F-B-SUMMARY] function=" << function.getName()
                 << " matched=" << reads.size() << " admitted=" << admitted
                 << " hints=" << admitted
                 << " requested_bytes=" << requestedBytes
                 << " reject_view=" << rejectedView
                 << " reject_stride=" << rejectedStride
                 << " reject_causal=" << rejectedCausal
                 << " reject_read_only=" << rejectedReadOnly
                 << " reject_address=" << rejectedAddress
                 << " reject_bounds=" << rejectedBounds
                 << " reject_size_page=" << rejectedSizePage
                 << " p5fc=" << pageSafeSegmented
                 << " contiguous_hints=" << contiguousHints
                 << " segmented_hints=" << segmentedHints
                 << " physical_rows=" << physicalRows
                 << " reject_segment_utilization="
                 << rejectedSegmentUtilization << '\n';
  }
};

/// P5g-a: materialize an exact sparse 2-D CRP source tile as one physically
/// contiguous VTCM tile.  This is intentionally synchronous: it isolates the
/// value of physical formation/VTCM residency from lookahead.  P5g-b may add
/// ping-pong overlap only after this matched gate is correct and profitable.
struct AlpsCrpVtcmFormationPass final
    : ::impl::AlpsCrpVtcmFormationBase<AlpsCrpVtcmFormationPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<vector::TransferReadOp> reads;
    function.walk([&](vector::TransferReadOp read) {
      if (read->hasAttr("alps.p2g.register_tile"))
        reads.push_back(read);
    });

    int64_t matched = 0;
    int64_t formed = 0;
    int64_t formedBytes = 0;
    int64_t rejectedShape = 0;
    int64_t rejectedView = 0;
    int64_t rejectedNarrowDma = 0;
    for (vector::TransferReadOp read : reads) {
      ++matched;
      auto sourceType = dyn_cast<MemRefType>(read.getBase().getType());
      if (!sourceType || !sourceType.hasStaticShape() ||
          sourceType.getRank() != 4 || sourceType.getDimSize(0) != 1 ||
          sourceType.getDimSize(2) != 1 ||
          !p2gUnitStride(read.getBase(), read.getPermutationMap())) {
        ++rejectedShape;
        continue;
      }
      SmallVector<int64_t> sourceStrides;
      int64_t sourceOffset = 0;
      if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)) ||
          sourceStrides.size() != 4 ||
          sourceStrides[0] !=
              sourceStrides[1] * sourceType.getDimSize(1) ||
          sourceStrides[2] !=
              sourceStrides[3] * sourceType.getDimSize(3) ||
          sourceStrides[3] != 1) {
        ++rejectedShape;
        continue;
      }
      auto loop = read->getParentOfType<scf::ForOp>();
      if (!loop) {
        ++rejectedView;
        continue;
      }

      int64_t rows = sourceType.getDimSize(1);
      int64_t columns = sourceType.getDimSize(3);
      auto bytes = staticBytes(read.getVectorType());
      if (rows <= 1 || columns <= 0 || !bytes || *bytes <= 0) {
        ++rejectedShape;
        continue;
      }

      // A 2-D UserDMA descriptor for this CRP view transfers one physical
      // row at a time.  Do not issue the 8-byte-wide descriptors exposed by
      // DINOv2's [1,16,1,4] f16 tile: the device reports a DMA exception and
      // run_main_on_hexagon exits with status 13.  64 bytes is the smallest
      // row width already exercised by the existing P3b path; it is a
      // conservative legality gate, not a claim that 64 bytes is optimal.
      // P5g-b widens/reuses the source window before crossing this gate.
      int64_t rowBytes =
          columns * sourceType.getElementTypeBitWidth() / 8;
      constexpr int64_t kConservativeDmaRowBytes = 64;
      if (rowBytes < kConservativeDmaRowBytes) {
        ++rejectedNarrowDma;
        bool unsupportedChain = false;
        int64_t dependentSubviews = 0;
        std::string viewChain;
        auto exactView = p5fFindExactSubview(
            read.getBase(), loop, unsupportedChain, dependentSubviews,
            viewChain);
        std::lock_guard<std::mutex> lock(reportMutex);
        llvm::errs() << "[ALPS-P5G-A-NARROW] function="
                     << function.getName() << " row_bytes=" << rowBytes
                     << " required_row_bytes=" << kConservativeDmaRowBytes
                     << " loop_lower=" << loop.getLowerBound()
                     << " loop_upper=" << loop.getUpperBound()
                     << " loop_step=" << loop.getStep()
                     << " dependent_subviews=" << dependentSubviews
                     << " unsupported_chain=" << unsupportedChain
                     << " chain=" << viewChain;
        if (exactView)
          llvm::errs() << " exact_view=" << *exactView;
        llvm::errs() << '\n';
        continue;
      }

      OpBuilder builder(loop);
      Location loc = read.getLoc();
      auto tileType = MemRefType::get({rows, columns},
                                      sourceType.getElementType(), {}, 1);
      Value tile = memref::AllocOp::create(builder, loc, tileType);

      // Both reassociation groups are statically contiguous even though the
      // resulting first dimension is strided in DDR.  HexmemCpyToDMA lowers
      // this rank-2 copy to one 2-D UserDMA command (width=columns,
      // height=rows), packing the rows into contiguous VTCM.
      SmallVector<ReassociationIndices> reassociation{{0, 1}, {2, 3}};
      builder.setInsertionPoint(read);
      Value source2d =
          memref::CollapseShapeOp::create(builder, loc, read.getBase(),
                                          reassociation);
      memref::CopyOp::create(builder, loc, source2d, tile);
      auto tile4dType = MemRefType::get(sourceType.getShape(),
                                        sourceType.getElementType(), {}, 1);
      Value tile4d = memref::ExpandShapeOp::create(
          builder, loc, tile4dType, tile, reassociation);
      read.getBaseMutable().assign(tile4d);
      read->setAttr("alps.p5g_a.vtcm_contiguous", builder.getUnitAttr());

      // This pass runs before the standard ownership-based buffer
      // deallocation pipeline.  Do not introduce a competing manual free;
      // the pipeline places the VTCM release after the final loop use.
      ++formed;
      formedBytes += *bytes;
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5g_a.vtcm_formed",
                      builder.getI64IntegerAttr(formed));
    function->setAttr("alps.p5g_a.vtcm_formed_bytes",
                      builder.getI64IntegerAttr(formedBytes));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5G-A-SUMMARY] function=" << function.getName()
                 << " matched=" << matched << " formed=" << formed
                 << " formed_bytes=" << formedBytes
                 << " reject_shape=" << rejectedShape
                 << " reject_view=" << rejectedView
                 << " reject_narrow_dma=" << rejectedNarrowDma << '\n';
  }
};

/// P5g-b: coalesce a family of narrow CRP tiles into one VTCM-resident supply
/// window.  DINOv2 consumes [16,4] f16 tiles from [256,6,64].  Issuing a 2-D
/// DMA for every tile produces an unsafe 8-byte row.  Instead, copy a
/// [256,32] channel window (64-byte DMA rows) once and reuse it across eight
/// adjacent four-channel consumer iterations.
struct AlpsCrpVtcmWindowPass final
    : ::impl::AlpsCrpVtcmWindowBase<AlpsCrpVtcmWindowPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    ModuleOp module = function->getParentOfType<ModuleOp>();
    bool asyncWindow = false;
    if (auto attr = module ? module->getAttrOfType<BoolAttr>(
                                 "alps.p5g_c.vtcm_async_window")
                           : BoolAttr{})
      asyncWindow = attr.getValue();
    SmallVector<vector::TransferReadOp> reads;
    function.walk([&](vector::TransferReadOp read) {
      if (read->hasAttr("alps.p2g.register_tile"))
        reads.push_back(read);
    });

    int64_t matched = 0;
    int64_t formed = 0;
    int64_t windowBytes = 0;
    int64_t rejectedView = 0;
    int64_t rejectedShape = 0;
    int64_t rejectedLoop = 0;
    constexpr int64_t kWindowColumns = 32;

    for (vector::TransferReadOp read : reads) {
      ++matched;
      auto tokenLoop = read->getParentOfType<scf::ForOp>();
      bool unsupportedChain = false;
      int64_t dependentSubviews = 0;
      std::string viewChain;
      auto view = tokenLoop ? p5fFindExactSubview(
                                  read.getBase(), tokenLoop, unsupportedChain,
                                  dependentSubviews, viewChain)
                            : memref::SubViewOp{};
      if (!view || unsupportedChain || dependentSubviews != 1) {
        ++rejectedView;
        continue;
      }

      auto sourceType = dyn_cast<MemRefType>(view.getSource().getType());
      auto viewType = dyn_cast<MemRefType>(view.getType());
      SmallVector<int64_t> sourcePhysicalStrides;
      int64_t sourcePhysicalOffset = 0;
      if (!sourceType || !viewType || !sourceType.hasStaticShape() ||
          sourceType.getRank() != 4 || viewType.getRank() != 2 ||
          sourceType.getDimSize(0) != 1 ||
          sourceType.getDimSize(1) <= 0 ||
          sourceType.getDimSize(3) < kWindowColumns ||
          sourceType.getDimSize(3) % kWindowColumns != 0 ||
          viewType.getDimSize(0) <= 0 || viewType.getDimSize(1) <= 0) {
        ++rejectedShape;
        continue;
      }
      if (failed(sourceType.getStridesAndOffset(sourcePhysicalStrides,
                                                sourcePhysicalOffset)) ||
          sourcePhysicalStrides.size() != 4 ||
          sourcePhysicalStrides[1] <= 0 || sourcePhysicalStrides[3] != 1) {
        ++rejectedShape;
        continue;
      }
      int64_t dmaRowStride = sourcePhysicalStrides[1];

      SmallVector<OpFoldResult> offsets = view.getMixedOffsets();
      SmallVector<OpFoldResult> sizes = view.getMixedSizes();
      SmallVector<OpFoldResult> strides = view.getMixedStrides();
      auto tokenOffset = dyn_cast<Value>(offsets[1]);
      auto channelOffset = dyn_cast<Value>(offsets[3]);
      auto tokenSize = p5ConstantFoldResult(sizes[1]);
      auto channelSize = p5ConstantFoldResult(sizes[3]);
      VectorType vectorType = read.getVectorType();
      bool vectorShapeSupported = tokenSize && channelSize;
      bool sawTokenDimension = false;
      bool sawChannelDimension = false;
      if (vectorShapeSupported) {
        for (int64_t dimension : vectorType.getShape()) {
          if (!sawTokenDimension && dimension == *tokenSize) {
            sawTokenDimension = true;
          } else if (sawTokenDimension && !sawChannelDimension &&
                     dimension == *channelSize) {
            sawChannelDimension = true;
          } else if (dimension != 1) {
            vectorShapeSupported = false;
            break;
          }
        }
        vectorShapeSupported &= sawTokenDimension && sawChannelDimension;
      }
      if (!tokenOffset || tokenOffset != tokenLoop.getInductionVar() ||
          !channelOffset || !tokenSize || !channelSize || *tokenSize <= 0 ||
          *channelSize <= 0 || kWindowColumns % *channelSize != 0 ||
          !vectorShapeSupported) {
        ++rejectedLoop;
        continue;
      }

      scf::ForOp channelLoop;
      for (auto loop = tokenLoop->getParentOfType<scf::ForOp>(); loop;
           loop = loop->getParentOfType<scf::ForOp>()) {
        if (loop.getInductionVar() == channelOffset) {
          channelLoop = loop;
          break;
        }
      }
      auto tokenLower = p5ConstantIndex(tokenLoop.getLowerBound());
      auto tokenUpper = p5ConstantIndex(tokenLoop.getUpperBound());
      auto tokenStep = p5ConstantIndex(tokenLoop.getStep());
      auto channelLower = channelLoop
                              ? p5ConstantIndex(channelLoop.getLowerBound())
                              : std::optional<int64_t>{};
      auto channelUpper = channelLoop
                              ? p5ConstantIndex(channelLoop.getUpperBound())
                              : std::optional<int64_t>{};
      auto channelStep = channelLoop
                             ? p5ConstantIndex(channelLoop.getStep())
                             : std::optional<int64_t>{};
      if (!channelLoop || !tokenLower || !tokenUpper || !tokenStep ||
          !channelLower || !channelUpper || !channelStep || *tokenLower != 0 ||
          *tokenUpper != sourceType.getDimSize(1) ||
          *tokenStep != *tokenSize || *channelLower != 0 ||
          *channelUpper != sourceType.getDimSize(3) ||
          *channelStep != *channelSize) {
        ++rejectedLoop;
        continue;
      }

      Operation *sourceDef = view.getSource().getDefiningOp();
      if (sourceDef && channelLoop->isAncestor(sourceDef)) {
        ++rejectedView;
        continue;
      }

      Location loc = read.getLoc();
      OpBuilder outerBuilder(channelLoop);
      int64_t sourceRows = sourceType.getDimSize(1);
      auto windowType = MemRefType::get(
          {sourceRows, kWindowColumns}, sourceType.getElementType(), {},
          /*memorySpace=*/1);
      Value window;
      Value pingWindow;
      Value pongWindow;
      if (asyncWindow) {
        // Keep ping and pong as genuinely separate VTCM objects.  A single
        // allocation with two dynamic subviews is not sufficient: alias
        // analysis conservatively treats the next DMA destination as
        // overlapping the current HVX source and consequently sinks the DMA
        // wait in front of the vector work, serializing the intended prefetch.
        Value pingAlloc =
            memref::AllocOp::create(outerBuilder, loc, windowType).getResult();
        Value pongAlloc =
            memref::AllocOp::create(outerBuilder, loc, windowType).getResult();
        auto distinct = memref::DistinctObjectsOp::create(
            outerBuilder, loc, ValueRange{pingAlloc, pongAlloc});
        pingWindow = distinct.getResult(0);
        pongWindow = distinct.getResult(1);
      } else {
        window =
            memref::AllocOp::create(outerBuilder, loc, windowType).getResult();
      }

      auto createSourceWindow = [&](OpBuilder &builder, Location sourceLoc,
                                    OpFoldResult channel) -> Value {
        SmallVector<OpFoldResult> windowOffsets = offsets;
        SmallVector<OpFoldResult> windowSizes = sizes;
        windowOffsets[1] = builder.getIndexAttr(0);
        windowOffsets[3] = channel;
        windowSizes[1] = builder.getIndexAttr(sourceRows);
        windowSizes[3] = builder.getIndexAttr(kWindowColumns);
        Value sourceWindow = memref::SubViewOp::create(
            builder, sourceLoc, view.getSource(), windowOffsets, windowSizes,
            strides);
        SmallVector<ReassociationIndices> reassociation{{0, 1}, {2, 3}};
        return memref::CollapseShapeOp::create(
            builder, sourceLoc, sourceWindow, reassociation);
      };

      Value dmaTag;
      Value outerZero;
      if (asyncWindow) {
        auto tagType = MemRefType::get({2}, outerBuilder.getI32Type());
        dmaTag = memref::AllocaOp::create(outerBuilder, loc, tagType);
        outerZero = arith::ConstantIndexOp::create(outerBuilder, loc, 0);
        Value initialSource = createSourceWindow(
            outerBuilder, loc, outerBuilder.getIndexAttr(0));
        SmallVector<Value> tag0{outerZero};
        SmallVector<Value> sourceIndices(2, outerZero);
        SmallVector<Value> targetIndices(2, outerZero);
        Value numElements = arith::ConstantIndexOp::create(
            outerBuilder, loc, sourceRows * kWindowColumns);
        Value rowStride = arith::ConstantIndexOp::create(
            outerBuilder, loc, dmaRowStride);
        Value rowWidth = arith::ConstantIndexOp::create(
            outerBuilder, loc, kWindowColumns);
        memref::DmaStartOp::create(
            outerBuilder, loc, initialSource, sourceIndices, pingWindow,
            targetIndices, numElements, dmaTag, tag0, rowStride, rowWidth);
      }

      OpBuilder channelBuilder = OpBuilder::atBlockBegin(channelLoop.getBody());
      Value windowColumns = arith::ConstantIndexOp::create(
          channelBuilder, loc, kWindowColumns);
      Value zero = arith::ConstantIndexOp::create(channelBuilder, loc, 0);
      Value localChannel = arith::RemUIOp::create(
          channelBuilder, loc, channelLoop.getInductionVar(), windowColumns);
      Value isWindowStart = arith::CmpIOp::create(
          channelBuilder, loc, arith::CmpIPredicate::eq, localChannel, zero);
      Value currentWindow = window;
      if (asyncWindow) {
        Value two = arith::ConstantIndexOp::create(channelBuilder, loc, 2);
        Value windowIndex = arith::DivUIOp::create(
            channelBuilder, loc, channelLoop.getInductionVar(), windowColumns);
        Value currentSlot =
            arith::RemUIOp::create(channelBuilder, loc, windowIndex, two);
        Value isPing = arith::CmpIOp::create(
            channelBuilder, loc, arith::CmpIPredicate::eq, currentSlot, zero);
        Value selectedCurrent = arith::SelectOp::create(
            channelBuilder, loc, isPing, pingWindow, pongWindow);
        Value selectedNext = arith::SelectOp::create(
            channelBuilder, loc, isPing, pongWindow, pingWindow);
        // Preserve the correlated opposite-slot relation after the dynamic
        // selection itself.  Distinctness on the original allocations alone
        // is lost when the two values pass through opposing selects, which
        // made the low-level scheduler conservatively wait for the next DMA
        // before reading the current slot.  This contract is valid by
        // construction for both outcomes of isPing.
        auto selectedDistinct = memref::DistinctObjectsOp::create(
            channelBuilder, loc, ValueRange{selectedCurrent, selectedNext});
        currentWindow = selectedDistinct.getResult(0);
        Value nextWindow = selectedDistinct.getResult(1);
        scf::IfOp::create(
            channelBuilder, loc, isWindowStart,
            [&](OpBuilder &thenBuilder, Location thenLoc) {
              SmallVector<Value> currentTagIndex{currentSlot};
              Value waitElements = arith::ConstantIndexOp::create(
                  thenBuilder, thenLoc, sourceRows * kWindowColumns);
              memref::DmaWaitOp::create(thenBuilder, thenLoc, dmaTag,
                                        currentTagIndex, waitElements);
              Value nextChannel = arith::AddIOp::create(
                  thenBuilder, thenLoc, channelLoop.getInductionVar(),
                  windowColumns);
              Value hasNext = arith::CmpIOp::create(
                  thenBuilder, thenLoc, arith::CmpIPredicate::ult, nextChannel,
                  channelLoop.getUpperBound());
              scf::IfOp::create(
                  thenBuilder, thenLoc, hasNext,
                  [&](OpBuilder &nextBuilder, Location nextLoc) {
                    Value one =
                        arith::ConstantIndexOp::create(nextBuilder, nextLoc, 1);
                    Value nextSlotUnwrapped = arith::AddIOp::create(
                        nextBuilder, nextLoc, currentSlot, one);
                    Value nextSlot = arith::RemUIOp::create(
                        nextBuilder, nextLoc, nextSlotUnwrapped, two);
                    Value nextSource = createSourceWindow(
                        nextBuilder, nextLoc, OpFoldResult(nextChannel));
                    SmallVector<Value> nextTagIndex{nextSlot};
                    Value nextZero = arith::ConstantIndexOp::create(
                        nextBuilder, nextLoc, 0);
                    SmallVector<Value> sourceIndices(2, nextZero);
                    SmallVector<Value> targetIndices(2, nextZero);
                    Value numElements = arith::ConstantIndexOp::create(
                        nextBuilder, nextLoc,
                        sourceRows * kWindowColumns);
                    Value rowStride = arith::ConstantIndexOp::create(
                        nextBuilder, nextLoc, dmaRowStride);
                    Value rowWidth = arith::ConstantIndexOp::create(
                        nextBuilder, nextLoc, kWindowColumns);
                    memref::DmaStartOp::create(
                        nextBuilder, nextLoc, nextSource, sourceIndices,
                        nextWindow, targetIndices, numElements, dmaTag,
                        nextTagIndex, rowStride, rowWidth);
                    scf::YieldOp::create(nextBuilder, nextLoc);
                  });
              scf::YieldOp::create(thenBuilder, thenLoc);
            });
      } else {
        scf::IfOp::create(
            channelBuilder, loc, isWindowStart,
            [&](OpBuilder &thenBuilder, Location thenLoc) {
              Value source2d = createSourceWindow(
                  thenBuilder, thenLoc,
                  OpFoldResult(channelLoop.getInductionVar()));
              memref::CopyOp::create(thenBuilder, thenLoc, source2d, window);
              scf::YieldOp::create(thenBuilder, thenLoc);
            });
      }

      OpBuilder readBuilder(read);
      SmallVector<OpFoldResult> tileOffsets{tokenLoop.getInductionVar(),
                                            localChannel};
      SmallVector<OpFoldResult> tileSizes{readBuilder.getIndexAttr(*tokenSize),
                                          readBuilder.getIndexAttr(*channelSize)};
      SmallVector<OpFoldResult> tileStrides{readBuilder.getIndexAttr(1),
                                            readBuilder.getIndexAttr(1)};
      Value windowTile = memref::SubViewOp::create(
          readBuilder, loc, currentWindow, tileOffsets, tileSizes, tileStrides);
      // Keep the consumer on the physically contiguous 2-D VTCM tile.  The
      // original transfer already has two indices and an identity map; an
      // expand_shape would reintroduce the sparse producer representation and
      // would also require retaining its sparse indexing contract.  Normalize
      // both the already-2-D and expanded-4-D consumer forms to this local
      // contiguous tile contract.
      read.getBaseMutable().assign(windowTile);
      Value localZero =
          arith::ConstantIndexOp::create(readBuilder, loc, 0).getResult();
      SmallVector<Value> localIndices{localZero, localZero};
      read.getIndicesMutable().assign(localIndices);
      AffineExpr tokenExpr = getAffineDimExpr(0, function.getContext());
      AffineExpr channelExpr = getAffineDimExpr(1, function.getContext());
      AffineExpr zeroExpr = getAffineConstantExpr(0, function.getContext());
      bool mappedToken = false;
      bool mappedChannel = false;
      SmallVector<AffineExpr> projectedVectorMap;
      for (int64_t dimension : vectorType.getShape()) {
        if (!mappedToken && dimension == *tokenSize) {
          projectedVectorMap.push_back(tokenExpr);
          mappedToken = true;
        } else if (mappedToken && !mappedChannel &&
                   dimension == *channelSize) {
          projectedVectorMap.push_back(channelExpr);
          mappedChannel = true;
        } else {
          projectedVectorMap.push_back(zeroExpr);
        }
      }
      read.setPermutationMap(AffineMap::get(
          /*dimCount=*/2, /*symbolCount=*/0, projectedVectorMap,
          function.getContext()));
      read->setAttr(asyncWindow ? "alps.p5g_c.vtcm_async_window"
                                : "alps.p5g_b.vtcm_window",
                    readBuilder.getUnitAttr());
      ++formed;
      windowBytes += (asyncWindow ? 2 : 1) * sourceRows * kWindowColumns *
                     sourceType.getElementTypeBitWidth() / 8;
      if (asyncWindow) {
        OpBuilder releaseBuilder(channelLoop);
        releaseBuilder.setInsertionPointAfter(channelLoop);
        memref::DeallocOp::create(releaseBuilder, loc, pingWindow);
        memref::DeallocOp::create(releaseBuilder, loc, pongWindow);
      }
    }

    Builder builder(function.getContext());
    function->setAttr(asyncWindow ? "alps.p5g_c.vtcm_async_windows"
                                  : "alps.p5g_b.vtcm_windows",
                      builder.getI64IntegerAttr(formed));
    function->setAttr(asyncWindow ? "alps.p5g_c.window_bytes"
                                  : "alps.p5g_b.window_bytes",
                      builder.getI64IntegerAttr(windowBytes));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << (asyncWindow ? "[ALPS-P5G-C-SUMMARY] function="
                                 : "[ALPS-P5G-B-SUMMARY] function=")
                 << function.getName()
                 << " matched=" << matched << " formed=" << formed
                 << " window_bytes=" << windowBytes
                 << " reject_view=" << rejectedView
                 << " reject_shape=" << rejectedShape
                 << " reject_loop=" << rejectedLoop << '\n';
  }
};

static Value p5gFindRootBuffer(Value value) {
  llvm::SmallDenseSet<Value, 8> visited;
  while (visited.insert(value).second) {
    auto view = dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp());
    if (!view)
      break;
    value = view.getViewSource();
  }
  return value;
}

static Operation *p5gTopLevelOperation(Operation *op,
                                       FunctionOpInterface function) {
  while (op && op->getParentOp() != function.getOperation())
    op = op->getParentOp();
  return op;
}

static std::string p5gWriterLoopContract(vector::TransferWriteOp writer,
                                         FunctionOpInterface function) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  bool first = true;
  for (Operation *parent = writer->getParentOp(); parent &&
                                                   parent != function.getOperation();
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    if (!first)
      stream << ';';
    first = false;
    auto lower = getConstantIntValue(loop.getLowerBound());
    auto upper = getConstantIntValue(loop.getUpperBound());
    auto step = getConstantIntValue(loop.getStep());
    stream << "for(";
    if (lower)
      stream << *lower;
    else
      stream << '?';
    stream << ',';
    if (upper)
      stream << *upper;
    else
      stream << '?';
    stream << ',';
    if (step)
      stream << *step;
    else
      stream << '?';
    stream << ')';
  }
  return stream.str();
}

static std::string p5gWriterViewContract(vector::TransferWriteOp writer) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  Value value = writer.getBase();
  llvm::SmallDenseSet<Value, 8> visited;
  bool first = true;
  while (visited.insert(value).second) {
    Operation *def = value.getDefiningOp();
    auto view = dyn_cast_or_null<ViewLikeOpInterface>(def);
    if (!view)
      break;
    if (!first)
      stream << "<-";
    first = false;
    stream << def->getName().getStringRef() << ':' << value.getType();
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      auto printFoldResults = [&](ArrayRef<OpFoldResult> values) {
        stream << '[';
        llvm::interleaveComma(values, stream, [&](OpFoldResult fold) {
          if (auto attr = dyn_cast<Attribute>(fold))
            stream << attr;
          else
            stream << cast<Value>(fold);
        });
        stream << ']';
      };
      stream << " offsets=";
      printFoldResults(subview.getMixedOffsets());
      stream << " sizes=";
      printFoldResults(subview.getMixedSizes());
      stream << " strides=";
      printFoldResults(subview.getMixedStrides());
    }
    value = view.getViewSource();
  }
  return stream.str();
}

static std::optional<int64_t> p5gConstantInt(OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    if (auto integer = dyn_cast<IntegerAttr>(attr))
      return integer.getInt();
    return std::nullopt;
  }
  return getConstantIntValue(cast<Value>(value));
}

/// Prove the DINO-style complete tiled overwrite without relying on a dynamic
/// execution assumption: an unmasked identity vector covers the entire final
/// dimension, while one unit-step static loop IV covers each leading-dimension
/// subview offset exactly once.
static bool p5gProvesFullTiledOverwrite(vector::TransferWriteOp writer,
                                        Value root) {
  if (!writer || writer.getMask() ||
      !writer.getPermutationMap().isMinorIdentity() ||
      p5gFindRootBuffer(writer.getBase()) != root)
    return false;
  auto rootType = dyn_cast<MemRefType>(root.getType());
  auto vectorType = dyn_cast<VectorType>(writer.getVector().getType());
  auto subview = writer.getBase().getDefiningOp<memref::SubViewOp>();
  if (!rootType || !rootType.hasStaticShape() || rootType.getRank() < 2 ||
      !vectorType || vectorType.getRank() != 1 || !subview ||
      subview.getSource() != root)
    return false;
  ArrayRef<int64_t> shape = rootType.getShape();
  int64_t rank = rootType.getRank();
  if (vectorType.getShape()[0] != shape.back())
    return false;
  auto offsets = subview.getMixedOffsets();
  auto sizes = subview.getMixedSizes();
  auto strides = subview.getMixedStrides();
  if (offsets.size() != static_cast<size_t>(rank) ||
      sizes.size() != static_cast<size_t>(rank) ||
      strides.size() != static_cast<size_t>(rank))
    return false;
  for (int64_t dim = 0; dim < rank; ++dim) {
    auto stride = p5gConstantInt(strides[dim]);
    if (!stride || *stride != 1)
      return false;
    if (dim == rank - 1) {
      auto offset = p5gConstantInt(offsets[dim]);
      auto size = p5gConstantInt(sizes[dim]);
      if (!offset || *offset != 0 || !size || *size != shape[dim])
        return false;
      continue;
    }
    auto size = p5gConstantInt(sizes[dim]);
    if (!size || *size != 1 || !isa<Value>(offsets[dim]))
      return false;
    Value offset = cast<Value>(offsets[dim]);
    auto argument = dyn_cast<BlockArgument>(offset);
    auto loop = argument
                    ? dyn_cast_or_null<scf::ForOp>(
                          argument.getOwner()->getParentOp())
                    : scf::ForOp{};
    if (!loop || loop.getInductionVar() != offset)
      return false;
    auto lower = getConstantIntValue(loop.getLowerBound());
    auto upper = getConstantIntValue(loop.getUpperBound());
    auto step = getConstantIntValue(loop.getStep());
    if (!lower || *lower != 0 || !upper || *upper != shape[dim] || !step ||
        *step != 1)
      return false;
  }
  return true;
}

struct P5gEpochRewrite {
  Value root;
  vector::TransferWriteOp writer;
  Operation *writerTop = nullptr;
  Operation *lastReaderTop = nullptr;
  SmallVector<Operation *> readers;
};

static MemRefType p5gWithVtcmMemorySpace(MemRefType type) {
  MLIRContext *context = type.getContext();
  Attribute vtcm = IntegerAttr::get(IntegerType::get(context, 32), 1);
  return MemRefType::get(type.getShape(), type.getElementType(),
                         type.getLayout(), vtcm);
}

static FailureOr<MemRefType> p5gHeadMajorVtcmType(MemRefType type) {
  if (type.getRank() != 3 || !type.hasStaticShape())
    return failure();
  ArrayRef<int64_t> shape = type.getShape();
  if (llvm::any_of(shape, [](int64_t dim) { return dim <= 0; }))
    return failure();
  // Preserve logical [token, head, channel] indices while storing bytes as
  // [head, token, channel].  This makes one head a contiguous producer and
  // consumer stream without materializing a tensor transpose.
  SmallVector<int64_t> strides = {shape[2], shape[0] * shape[2], 1};
  auto layout = StridedLayoutAttr::get(type.getContext(), 0, strides);
  auto memorySpace =
      IntegerAttr::get(IntegerType::get(type.getContext(), 32), 1);
  return MemRefType::get(shape, type.getElementType(), layout, memorySpace);
}

static FailureOr<MemRefType>
p5gInferClonedViewType(Operation *cloned, MemRefType originalResultType) {
  if (auto subview = dyn_cast<memref::SubViewOp>(cloned))
    return memref::SubViewOp::inferRankReducedResultType(
        originalResultType.getShape(), subview.getSourceType(),
        subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides());
  if (auto expand = dyn_cast<memref::ExpandShapeOp>(cloned))
    return memref::ExpandShapeOp::computeExpandedType(
        expand.getSrcType(), originalResultType.getShape(),
        expand.getReassociationIndices());
  if (auto collapse = dyn_cast<memref::CollapseShapeOp>(cloned))
    return memref::CollapseShapeOp::computeCollapsedType(
        collapse.getSrcType(), collapse.getReassociationIndices());
  return failure();
}

static FailureOr<Value>
p5gCloneAliasInVtcm(Value original, Value root, Value vtcmRoot,
                    OpBuilder &builder, DenseMap<Value, Value> &cache) {
  if (original == root)
    return vtcmRoot;
  if (auto found = cache.find(original); found != cache.end())
    return found->second;
  Operation *def = original.getDefiningOp();
  auto view = dyn_cast_or_null<ViewLikeOpInterface>(def);
  if (!view)
    return failure();
  FailureOr<Value> source = p5gCloneAliasInVtcm(
      view.getViewSource(), root, vtcmRoot, builder, cache);
  if (failed(source))
    return failure();
  IRMapping mapping;
  mapping.map(view.getViewSource(), *source);
  Operation *cloned = builder.clone(*def, mapping);
  for (auto [oldResult, newResult] :
       llvm::zip_equal(def->getResults(), cloned->getResults())) {
    auto originalType = dyn_cast<MemRefType>(oldResult.getType());
    if (!originalType)
      return failure();
    FailureOr<MemRefType> inferred =
        p5gInferClonedViewType(cloned, originalType);
    if (failed(inferred))
      return failure();
    newResult.setType(*inferred);
  }
  for (auto [oldResult, newResult] :
       llvm::zip_equal(def->getResults(), cloned->getResults()))
    cache[oldResult] = newResult;
  auto found = cache.find(original);
  return found == cache.end() ? FailureOr<Value>(failure())
                              : FailureOr<Value>(found->second);
}

/// Interchange exactly the producer nest proven by P5g-d from
/// token-outer/head-inner to head-outer/token-inner.  The gate deliberately
/// accepts only a result-free perfect two-loop nest.  P5g-d has already proved
/// that the sole write covers disjoint [token, head, channel] rows, so changing
/// their visitation order introduces no loop-carried output dependence.
static LogicalResult p5gInterchangeHeadMajorProducer(P5gEpochRewrite &epoch) {
  auto tokenLoop = dyn_cast_or_null<scf::ForOp>(epoch.writerTop);
  auto headLoop = epoch.writer ? epoch.writer->getParentOfType<scf::ForOp>()
                               : scf::ForOp{};
  if (!tokenLoop || !headLoop || tokenLoop == headLoop ||
      headLoop->getParentOp() != tokenLoop.getOperation() ||
      !tokenLoop.getInitArgs().empty() || !headLoop.getInitArgs().empty() ||
      tokenLoop.getNumResults() != 0 || headLoop.getNumResults() != 0)
    return failure();

  Operation *onlyNested = nullptr;
  for (Operation &op : tokenLoop.getBody()->without_terminator()) {
    if (onlyNested)
      return failure();
    onlyNested = &op;
  }
  if (onlyNested != headLoop.getOperation())
    return failure();

  // The new outer loop cannot use a bound computed inside the old token loop.
  auto definedInsideTokenLoop = [&](Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return argument.getOwner() == tokenLoop.getBody();
    Operation *def = value.getDefiningOp();
    return def && tokenLoop->isAncestor(def);
  };
  if (definedInsideTokenLoop(headLoop.getLowerBound()) ||
      definedInsideTokenLoop(headLoop.getUpperBound()) ||
      definedInsideTokenLoop(headLoop.getStep()))
    return failure();

  OpBuilder outerBuilder(tokenLoop);
  Location loc = tokenLoop.getLoc();
  auto newHeadLoop = scf::ForOp::create(
      outerBuilder, loc, headLoop.getLowerBound(), headLoop.getUpperBound(),
      headLoop.getStep());
  newHeadLoop->setAttr("alps.p5g_g.head_outer",
                       outerBuilder.getUnitAttr());
  OpBuilder innerBuilder = OpBuilder::atBlockBegin(newHeadLoop.getBody());
  auto newTokenLoop = scf::ForOp::create(
      innerBuilder, loc, tokenLoop.getLowerBound(), tokenLoop.getUpperBound(),
      tokenLoop.getStep());
  newTokenLoop->setAttr("alps.p5g_g.token_inner",
                        innerBuilder.getUnitAttr());

  IRMapping mapping;
  mapping.map(tokenLoop.getInductionVar(), newTokenLoop.getInductionVar());
  mapping.map(headLoop.getInductionVar(), newHeadLoop.getInductionVar());
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newTokenLoop.getBody());
  for (Operation &op : headLoop.getBody()->without_terminator())
    bodyBuilder.clone(op, mapping);

  tokenLoop.erase();
  return success();
}

static LogicalResult p5gRewriteEpochInVtcm(P5gEpochRewrite &epoch,
                                           bool headMajor,
                                           bool interchangeProducer) {
  auto rootType = dyn_cast<MemRefType>(epoch.root.getType());
  if (!rootType || !epoch.writer || !epoch.writerTop ||
      !epoch.lastReaderTop)
    return failure();
  OpBuilder allocBuilder(epoch.writerTop);
  FailureOr<MemRefType> vtcmType = headMajor
      ? p5gHeadMajorVtcmType(rootType)
      : FailureOr<MemRefType>(p5gWithVtcmMemorySpace(rootType));
  if (failed(vtcmType))
    return failure();
  auto vtcmAlloc = memref::AllocOp::create(
      allocBuilder, epoch.writerTop->getLoc(), *vtcmType);
  // This pass deliberately bounds the VTCM lifetime to one proven writer
  // epoch.  Keep ownership-based buffer deallocation from sinking all epoch
  // frees to the function exit (which would make the 12 DINO epochs overlap).
  vtcmAlloc->setAttr("bufferization.manual_deallocation",
                     allocBuilder.getUnitAttr());

  {
    OpBuilder writerBuilder(epoch.writer);
    DenseMap<Value, Value> cache;
    FailureOr<Value> target = p5gCloneAliasInVtcm(
        epoch.writer.getBase(), epoch.root, vtcmAlloc, writerBuilder, cache);
    if (failed(target)) {
      vtcmAlloc.erase();
      return failure();
    }
    epoch.writer.getBaseMutable().assign(*target);
    epoch.writer->setAttr(headMajor
                              ? "alps.p5g_f.producer_direct_head_major_vtcm"
                              : "alps.p5g_e.producer_direct_vtcm",
                          writerBuilder.getUnitAttr());
  }

  for (Operation *reader : epoch.readers) {
    OpBuilder readerBuilder(reader);
    DenseMap<Value, Value> cache;
    for (OpOperand &operand : reader->getOpOperands()) {
      Value old = operand.get();
      if (!isa<MemRefType>(old.getType()) || p5gFindRootBuffer(old) != epoch.root)
        continue;
      FailureOr<Value> replacement = p5gCloneAliasInVtcm(
          old, epoch.root, vtcmAlloc, readerBuilder, cache);
      if (failed(replacement))
        return failure();
      operand.set(*replacement);
    }
    reader->setAttr(headMajor
                        ? "alps.p5g_f.consumer_head_major_vtcm"
                        : "alps.p5g_e.consumer_direct_vtcm",
                    readerBuilder.getUnitAttr());
  }

  OpBuilder deallocBuilder(epoch.lastReaderTop);
  deallocBuilder.setInsertionPointAfter(epoch.lastReaderTop);
  auto dealloc = memref::DeallocOp::create(
      deallocBuilder, epoch.lastReaderTop->getLoc(), vtcmAlloc);
  dealloc->setAttr("bufferization.manual_deallocation",
                   deallocBuilder.getUnitAttr());
  if (interchangeProducer &&
      failed(p5gInterchangeHeadMajorProducer(epoch)))
    return failure();
  return success();
}

static bool p5gIsFullRootOverwrite(Operation *writer, Value root) {
  auto rootBytes = staticBytes(root.getType());
  if (!rootBytes)
    return false;
  if (auto copy = dyn_cast<memref::CopyOp>(writer)) {
    return p5gFindRootBuffer(copy.getTarget()) == root &&
           staticBytes(copy.getTarget().getType()) == rootBytes;
  }
  if (auto write = dyn_cast<vector::TransferWriteOp>(writer)) {
    return p5gProvesFullTiledOverwrite(write, root) ||
           (p5gFindRootBuffer(write.getBase()) == root &&
            staticBytes(write.getVector().getType()) == rootBytes);
  }
  // A tiled/vector writer may dynamically cover the complete root, but that
  // requires a loop-domain proof. Do not promote it to a full overwrite merely
  // because it is the only static writer operation.
  return false;
}

/// P5g-d analysis-only gate.  P5g-c proved that a later DMA can overlap HVX,
/// but it retained the DDR materialization.  This pass asks the stronger
/// question: can the operation that creates the source write the
/// consumer-required representation directly into VTCM?  It intentionally
/// refuses to mutate IR until allocation ownership, writer coverage, every
/// reader contract, and footprint are all proven.
struct AlpsCrpProducerDirectAnalysisPass final
    : ::impl::AlpsCrpProducerDirectAnalysisBase<
          AlpsCrpProducerDirectAnalysisPass> {
  using Base::Base;
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<vector::TransferReadOp> reads;
    function.walk([&](vector::TransferReadOp read) {
      if (read->hasAttr("alps.p2g.register_tile"))
        reads.push_back(read);
    });

    int64_t matched = reads.size();
    int64_t analyzed = 0;
    int64_t uniqueRoots = 0;
    int64_t allocatedRoots = 0;
    int64_t uniqueWriters = 0;
    int64_t exclusiveCrpReaders = 0;
    int64_t fullOverwrites = 0;
    int64_t rewriteReady = 0;
    int64_t hmxInputRoots = 0;
    int64_t hmxOutputRoots = 0;
    int64_t mixedHmxHvxRoots = 0;
    int64_t hvxOnlyEpochs = 0;
    int64_t mixedHmxHvxEpochs = 0;
    int64_t ambiguousEpochs = 0;
    int64_t singleWriterEpochs = 0;
    int64_t vectorWriterEpochs = 0;
    int64_t epochRedirectCandidates = 0;
    int64_t coverageProvenEpochs = 0;
    int64_t headMajorIncompatibleEpochs = 0;
    int64_t rejectedView = 0;
    int64_t rejectedEscape = 0;
    int64_t rejectedFootprint = 0;
    int64_t rejectedWriter = 0;
    int64_t rejectedReaders = 0;
    constexpr int64_t kVtcmBudgetBytes = 2 * 1024 * 1024;
    llvm::SmallDenseSet<Value, 32> auditedRoots;
    SmallVector<P5gEpochRewrite> pendingRewrites;

    for (vector::TransferReadOp read : reads) {
      auto tokenLoop = read->getParentOfType<scf::ForOp>();
      bool unsupportedChain = false;
      int64_t dependentSubviews = 0;
      std::string viewChain;
      auto exactView = tokenLoop ? p5fFindExactSubview(
                                       read.getBase(), tokenLoop,
                                       unsupportedChain, dependentSubviews,
                                       viewChain)
                                 : memref::SubViewOp{};
      if (!exactView || unsupportedChain || dependentSubviews != 1) {
        ++rejectedView;
        continue;
      }
      ++analyzed;
      Value root = p5gFindRootBuffer(exactView.getSource());
      if (!auditedRoots.insert(root).second)
        continue;
      ++uniqueRoots;

      auto rootType = dyn_cast<MemRefType>(root.getType());
      auto rootBytes = staticBytes(root.getType());
      Operation *rootDef = root.getDefiningOp();
      bool allocated = rootDef && rootType && rootType.hasStaticShape() &&
                       hasSingleEffect<MemoryEffects::Allocate>(rootDef, root);
      if (allocated)
        ++allocatedRoots;
      bool footprintLegal = rootBytes && *rootBytes > 0 &&
                            *rootBytes <= kVtcmBudgetBytes;
      if (!footprintLegal)
        ++rejectedFootprint;

      SmallVector<Value> worklist{root};
      llvm::SmallDenseSet<Value, 32> aliases;
      SmallPtrSet<Operation *, 16> writers;
      SmallPtrSet<Operation *, 32> readersOfRoot;
      llvm::StringSet<> nonCrpReaderOps;
      llvm::StringSet<> readerRoles;
      llvm::StringSet<> hmxRoles;
      llvm::StringSet<> aliasOps;
      llvm::StringSet<> escapeOps;
      bool hmxInput = false;
      bool hmxOutput = false;
      bool escapes = false;
      while (!worklist.empty()) {
        Value alias = worklist.pop_back_val();
        if (!aliases.insert(alias).second)
          continue;
        for (Operation *user : alias.getUsers()) {
          if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
            if (view.getViewSource() == alias) {
              aliasOps.insert(user->getName().getStringRef());
              worklist.push_back(view.getViewDest());
              continue;
            }
          }
          bool writes = hasEffect<MemoryEffects::Write>(user, alias);
          bool reads = hasEffect<MemoryEffects::Read>(user, alias);
          // HexKL's coarse MemoryEffects trait does not associate the effects
          // with individual operands, so getEffectsOnValue cannot distinguish
          // an HMX input from an unknown escape.  Recover the destination-style
          // contract explicitly: lhs/rhs are reads and outs is a write.  This
          // is also the important cross-engine fact for P5g-d: an HMX consumer
          // is legal for VTCM placement, not evidence that the value escaped
          // the Hexagon memory hierarchy.
          if (auto matmul = dyn_cast<hexkl::MatmulOp>(user)) {
            if (matmul.getLhs() == alias) {
              reads = true;
              hmxInput = true;
              hmxRoles.insert("hexkl.matmul:lhs");
            }
            if (matmul.getRhs() == alias) {
              reads = true;
              hmxInput = true;
              hmxRoles.insert("hexkl.matmul:rhs");
            }
            if (matmul.getOuts() == alias) {
              writes = true;
              hmxOutput = true;
              hmxRoles.insert("hexkl.matmul:outs");
            }
          }
          if (writes)
            writers.insert(user);
          if (reads)
            readersOfRoot.insert(user);
          if (reads) {
            auto transfer = dyn_cast<vector::TransferReadOp>(user);
            if (!transfer || !transfer->hasAttr("alps.p2g.register_tile"))
              nonCrpReaderOps.insert(user->getName().getStringRef());
            for (OpOperand &operand : user->getOpOperands()) {
              if (operand.get() != alias)
                continue;
              std::string role;
              llvm::raw_string_ostream roleStream(role);
              roleStream << user->getName().getStringRef() << ":operand"
                         << operand.getOperandNumber();
              if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
                auto maps = linalgOp.getIndexingMapsArray();
                if (operand.getOperandNumber() < maps.size())
                  roleStream << ":map=" << maps[operand.getOperandNumber()];
              }
              readerRoles.insert(roleStream.str());
            }
          }
          if (!writes && !reads &&
              !isa<memref::DeallocOp, memref::AssumeAlignmentOp>(user)) {
            escapes = true;
            escapeOps.insert(user->getName().getStringRef());
          }
        }
      }
      if (escapes)
        ++rejectedEscape;

      if (hmxInput)
        ++hmxInputRoots;
      if (hmxOutput)
        ++hmxOutputRoots;
      bool hasCrpReader = llvm::any_of(readersOfRoot, [](Operation *reader) {
        auto transfer = dyn_cast<vector::TransferReadOp>(reader);
        return transfer && transfer->hasAttr("alps.p2g.register_tile");
      });
      bool mixedHmxHvx = (hmxInput || hmxOutput) && hasCrpReader;
      if (mixedHmxHvx)
        ++mixedHmxHvxRoots;

      // Bufferization may reuse one allocation for unrelated logical tensors.
      // Partition its uses by top-level writer epochs before interpreting a
      // root-level HMX/HVX mixture as one multi-consumer value.  This is a
      // conservative lexical proof: uses in the same top-level operation or a
      // different block remain ambiguous rather than being paired by guess.
      SmallPtrSet<Operation *, 16> writerEpochSet;
      SmallPtrSet<Operation *, 16> hmxReaderEpochSet;
      SmallPtrSet<Operation *, 16> crpReaderEpochSet;
      for (Operation *candidate : writers)
        if (Operation *top = p5gTopLevelOperation(candidate, function))
          writerEpochSet.insert(top);
      for (Operation *candidate : readersOfRoot) {
        Operation *top = p5gTopLevelOperation(candidate, function);
        if (!top)
          continue;
        if (isa<hexkl::MatmulOp>(candidate))
          hmxReaderEpochSet.insert(top);
        if (auto transfer = dyn_cast<vector::TransferReadOp>(candidate);
            transfer && transfer->hasAttr("alps.p2g.register_tile"))
          crpReaderEpochSet.insert(top);
      }
      SmallVector<Operation *> writerEpochs(writerEpochSet.begin(),
                                             writerEpochSet.end());
      llvm::sort(writerEpochs, [](Operation *lhs, Operation *rhs) {
        return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
      });
      int64_t rootHvxOnlyEpochs = 0;
      int64_t rootMixedEpochs = 0;
      int64_t rootAmbiguousEpochs = 0;
      int64_t rootSingleWriterEpochs = 0;
      int64_t rootVectorWriterEpochs = 0;
      int64_t rootRedirectCandidates = 0;
      int64_t rootCoverageProvenEpochs = 0;
      for (Operation *crpTop : crpReaderEpochSet) {
        Operation *lastWriter = nullptr;
        Operation *nextWriter = nullptr;
        for (Operation *writerTop : writerEpochs) {
          if (writerTop->getBlock() != crpTop->getBlock())
            continue;
          if (writerTop == crpTop) {
            lastWriter = nullptr;
            break;
          }
          if (writerTop->isBeforeInBlock(crpTop))
            lastWriter = writerTop;
          else {
            nextWriter = writerTop;
            break;
          }
        }
        if (!lastWriter) {
          ++rootAmbiguousEpochs;
          continue;
        }
        SmallVector<Operation *> epochWriters;
        for (Operation *candidate : writers)
          if (p5gTopLevelOperation(candidate, function) == lastWriter)
            epochWriters.push_back(candidate);
        bool singleEpochWriter = epochWriters.size() == 1;
        if (singleEpochWriter)
          ++rootSingleWriterEpochs;
        auto vectorWriter = singleEpochWriter
                                ? dyn_cast<vector::TransferWriteOp>(
                                      epochWriters.front())
                                : vector::TransferWriteOp{};
        bool vectorWriterCandidate =
            vectorWriter && !vectorWriter.getMask() &&
            vectorWriter.getPermutationMap().isMinorIdentity() &&
            p5gFindRootBuffer(vectorWriter.getBase()) == root;
        if (vectorWriterCandidate)
          ++rootVectorWriterEpochs;
        bool completeCoverage =
            vectorWriterCandidate &&
            p5gProvesFullTiledOverwrite(vectorWriter, root);
        if (completeCoverage)
          ++rootCoverageProvenEpochs;

        int64_t epochReaders = 0;
        int64_t epochCrpReaders = 0;
        int64_t epochHmxReaders = 0;
        int64_t epochOtherReaders = 0;
        SmallVector<Operation *> epochReaderOps;
        for (Operation *candidate : readersOfRoot) {
          Operation *readerTop = p5gTopLevelOperation(candidate, function);
          if (!readerTop || readerTop->getBlock() != crpTop->getBlock() ||
              readerTop == lastWriter)
            continue;
          bool afterWriter = lastWriter->isBeforeInBlock(readerTop);
          bool beforeNext = !nextWriter || readerTop->isBeforeInBlock(nextWriter);
          if (!afterWriter || !beforeNext)
            continue;
          ++epochReaders;
          epochReaderOps.push_back(candidate);
          if (isa<hexkl::MatmulOp>(candidate))
            ++epochHmxReaders;
          else if (auto transfer = dyn_cast<vector::TransferReadOp>(candidate);
                   transfer && transfer->hasAttr("alps.p2g.register_tile"))
            ++epochCrpReaders;
          else
            ++epochOtherReaders;
        }
        bool sameEpochHmx = false;
        bool ambiguousHmx = false;
        for (Operation *hmxTop : hmxReaderEpochSet) {
          if (hmxTop->getBlock() != crpTop->getBlock())
            continue;
          if (hmxTop == lastWriter || hmxTop == crpTop) {
            ambiguousHmx = true;
            continue;
          }
          bool afterWriter = lastWriter->isBeforeInBlock(hmxTop);
          bool beforeNext = !nextWriter || hmxTop->isBeforeInBlock(nextWriter);
          if (afterWriter && beforeNext)
            sameEpochHmx = true;
        }
        if (ambiguousHmx)
          ++rootAmbiguousEpochs;
        else if (sameEpochHmx)
          ++rootMixedEpochs;
        else {
          ++rootHvxOnlyEpochs;
          // This is deliberately only a redirect *candidate*.  The vector
          // writer's surrounding loop domain must still prove complete
          // coverage before a VTCM allocation or alias is rewritten.
          if (completeCoverage && epochCrpReaders > 0 &&
              epochHmxReaders == 0) {
            ++rootRedirectCandidates;
            if (rewriteEpochVtcm || rewriteEpochHeadMajorVtcm) {
              Operation *lastReaderTop = nullptr;
              for (Operation *candidate : epochReaderOps) {
                Operation *top = p5gTopLevelOperation(candidate, function);
                if (!top)
                  continue;
                if (!lastReaderTop ||
                    (lastReaderTop->getBlock() == top->getBlock() &&
                     lastReaderTop->isBeforeInBlock(top)))
                  lastReaderTop = top;
              }
              auto rootType = dyn_cast<MemRefType>(root.getType());
              // P5g-f's physical [head, token, channel] representation is
              // defined only for rank-3 roots.  Keep rank-2 speech buffers as
              // analysis-only candidates (or let P5g-e handle them when that
              // mode is selected) instead of admitting them to a rewriter
              // that must fail after analysis has completed.
              if (rewriteEpochHeadMajorVtcm &&
                  (!rootType || rootType.getRank() != 3)) {
                ++headMajorIncompatibleEpochs;
              } else {
                pendingRewrites.push_back({root, vectorWriter, lastWriter,
                                           lastReaderTop, epochReaderOps});
              }
            }
          }
        }

        std::lock_guard<std::mutex> epochLock(reportMutex);
        llvm::errs() << "[ALPS-P5G-D-EPOCH] function=" << function.getName()
                     << " writer_top=" << lastWriter->getName()
                     << " writer_count=" << epochWriters.size()
                     << " writer_op="
                     << (singleEpochWriter
                             ? epochWriters.front()->getName().getStringRef()
                             : StringRef("multiple"))
                     << " writer_vector_type="
                     << (vectorWriter ? vectorWriter.getVector().getType()
                                      : Type{})
                     << " writer_base_type="
                     << (vectorWriter ? vectorWriter.getBase().getType()
                                      : Type{})
                     << " writer_map="
                     << (vectorWriter ? vectorWriter.getPermutationMap()
                                      : AffineMap{})
                     << " writer_loops="
                     << (vectorWriter
                             ? p5gWriterLoopContract(vectorWriter, function)
                             : std::string())
                     << " writer_views="
                     << (vectorWriter ? p5gWriterViewContract(vectorWriter)
                                      : std::string())
                     << " writer_masked="
                     << static_cast<bool>(vectorWriter && vectorWriter.getMask())
                     << " readers=" << epochReaders
                     << " crp_readers=" << epochCrpReaders
                     << " hmx_readers=" << epochHmxReaders
                     << " other_readers=" << epochOtherReaders
                     << " hvx_only=" << (!sameEpochHmx && !ambiguousHmx)
                     << " redirect_candidate="
                     << (completeCoverage && epochCrpReaders > 0 &&
                         epochHmxReaders == 0)
                     << " complete_coverage=" << completeCoverage
                     << '\n';
      }
      hvxOnlyEpochs += rootHvxOnlyEpochs;
      mixedHmxHvxEpochs += rootMixedEpochs;
      ambiguousEpochs += rootAmbiguousEpochs;
      singleWriterEpochs += rootSingleWriterEpochs;
      vectorWriterEpochs += rootVectorWriterEpochs;
      epochRedirectCandidates += rootRedirectCandidates;
      coverageProvenEpochs += rootCoverageProvenEpochs;

      bool oneWriter = writers.size() == 1;
      if (oneWriter)
        ++uniqueWriters;
      else
        ++rejectedWriter;

      bool onlyCrpReaders = !readersOfRoot.empty();
      for (Operation *reader : readersOfRoot) {
        auto transfer = dyn_cast<vector::TransferReadOp>(reader);
        if (!transfer || !transfer->hasAttr("alps.p2g.register_tile")) {
          onlyCrpReaders = false;
          break;
        }
      }
      if (onlyCrpReaders)
        ++exclusiveCrpReaders;
      else
        ++rejectedReaders;

      Operation *writer = oneWriter ? *writers.begin() : nullptr;
      bool fullOverwrite = writer && p5gIsFullRootOverwrite(writer, root);
      if (fullOverwrite)
        ++fullOverwrites;
      bool ready = allocated && footprintLegal && !escapes && oneWriter &&
                   onlyCrpReaders && fullOverwrite;
      if (ready)
        ++rewriteReady;

      std::string writerNames;
      llvm::raw_string_ostream writerStream(writerNames);
      bool first = true;
      for (Operation *candidate : writers) {
        if (!first)
          writerStream << ',';
        first = false;
        writerStream << candidate->getName().getStringRef();
      }
      auto printNames = [](llvm::StringSet<> &names) {
        std::string storage;
        llvm::raw_string_ostream stream(storage);
        bool firstName = true;
        for (const auto &entry : names) {
          if (!firstName)
            stream << ',';
          firstName = false;
          stream << entry.getKey();
        }
        return stream.str();
      };
      std::lock_guard<std::mutex> lock(reportMutex);
      llvm::errs() << "[ALPS-P5G-D-ROOT] function=" << function.getName()
                   << " root_def="
                   << (rootDef ? rootDef->getName().getStringRef()
                               : StringRef("block_argument"))
                   << " root_type=" << root.getType()
                   << " root_bytes=" << (rootBytes ? *rootBytes : -1)
                   << " aliases=" << aliases.size()
                   << " writers=" << writers.size()
                   << " writer_ops=" << writerStream.str()
                   << " readers=" << readersOfRoot.size()
                   << " non_crp_reader_ops=" << printNames(nonCrpReaderOps)
                   << " reader_roles=" << printNames(readerRoles)
                   << " hmx_roles=" << printNames(hmxRoles)
                   << " alias_ops=" << printNames(aliasOps)
                   << " contract_kind="
                   << (mixedHmxHvx
                           ? (hmxOutput ? "shared_allocation_hmx_output_hvx"
                                        : "shared_allocation_hmx_input_hvx")
                           : (hmxOutput ? "hmx_output_only"
                                        : (hmxInput ? "hmx_input_only"
                                                    : "hvx_only")))
                   << " allocated=" << allocated
                   << " footprint_legal=" << footprintLegal
                   << " exclusive_crp_readers=" << onlyCrpReaders
                   << " full_overwrite=" << fullOverwrite
                   << " lifetime_partition_required="
                   << (writers.size() > 1 || mixedHmxHvx)
                   << " hvx_only_epochs=" << rootHvxOnlyEpochs
                   << " mixed_hmx_hvx_epochs=" << rootMixedEpochs
                   << " ambiguous_epochs=" << rootAmbiguousEpochs
                   << " single_writer_epochs=" << rootSingleWriterEpochs
                   << " vector_writer_epochs=" << rootVectorWriterEpochs
                   << " redirect_candidates=" << rootRedirectCandidates
                   << " coverage_proven_epochs="
                   << rootCoverageProvenEpochs
                   << " escapes=" << escapes
                   << " escape_ops=" << printNames(escapeOps)
                   << " rewrite_ready=" << ready
                   << '\n';
    }

    int64_t rewrittenEpochs = 0;
    if (rewriteEpochVtcm || rewriteEpochHeadMajorVtcm) {
      for (P5gEpochRewrite &epoch : pendingRewrites) {
        if (failed(p5gRewriteEpochInVtcm(
                epoch, static_cast<bool>(rewriteEpochHeadMajorVtcm),
                static_cast<bool>(rewriteProducerLoopOrder)))) {
          function.emitError(
              "P5g-e/f failed to clone an admitted epoch alias chain");
          return signalPassFailure();
        }
        ++rewrittenEpochs;
      }
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5g_d.unique_roots",
                      builder.getI64IntegerAttr(uniqueRoots));
    function->setAttr("alps.p5g_d.rewrite_ready",
                      builder.getI64IntegerAttr(rewriteReady));
    function->setAttr("alps.p5g_d.hmx_input_roots",
                      builder.getI64IntegerAttr(hmxInputRoots));
    function->setAttr("alps.p5g_d.hmx_output_roots",
                      builder.getI64IntegerAttr(hmxOutputRoots));
    function->setAttr("alps.p5g_d.mixed_hmx_hvx_roots",
                      builder.getI64IntegerAttr(mixedHmxHvxRoots));
    function->setAttr("alps.p5g_d.hvx_only_epochs",
                      builder.getI64IntegerAttr(hvxOnlyEpochs));
    function->setAttr("alps.p5g_d.mixed_hmx_hvx_epochs",
                      builder.getI64IntegerAttr(mixedHmxHvxEpochs));
    function->setAttr("alps.p5g_d.ambiguous_epochs",
                      builder.getI64IntegerAttr(ambiguousEpochs));
    function->setAttr("alps.p5g_d.single_writer_epochs",
                      builder.getI64IntegerAttr(singleWriterEpochs));
    function->setAttr("alps.p5g_d.vector_writer_epochs",
                      builder.getI64IntegerAttr(vectorWriterEpochs));
    function->setAttr("alps.p5g_d.epoch_redirect_candidates",
                      builder.getI64IntegerAttr(epochRedirectCandidates));
    function->setAttr("alps.p5g_d.coverage_proven_epochs",
                      builder.getI64IntegerAttr(coverageProvenEpochs));
    function->setAttr("alps.p5g_f.incompatible_rank_epochs",
                      builder.getI64IntegerAttr(headMajorIncompatibleEpochs));
    function->setAttr("alps.p5g_e.rewritten_epochs",
                      builder.getI64IntegerAttr(
                          rewriteEpochHeadMajorVtcm ? 0 : rewrittenEpochs));
    function->setAttr("alps.p5g_f.head_major_rewritten_epochs",
                      builder.getI64IntegerAttr(
                          rewriteEpochHeadMajorVtcm ? rewrittenEpochs : 0));
    function->setAttr("alps.p5g_g.interchanged_producer_epochs",
                      builder.getI64IntegerAttr(
                          rewriteProducerLoopOrder ? rewrittenEpochs : 0));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5G-D-SUMMARY] function=" << function.getName()
                 << " matched=" << matched << " analyzed=" << analyzed
                 << " unique_roots=" << uniqueRoots
                 << " allocated_roots=" << allocatedRoots
                 << " unique_writers=" << uniqueWriters
                 << " exclusive_crp_readers=" << exclusiveCrpReaders
                 << " full_overwrites=" << fullOverwrites
                 << " rewrite_ready=" << rewriteReady
                 << " hmx_input_roots=" << hmxInputRoots
                 << " hmx_output_roots=" << hmxOutputRoots
                 << " mixed_hmx_hvx_roots=" << mixedHmxHvxRoots
                 << " hvx_only_epochs=" << hvxOnlyEpochs
                 << " mixed_hmx_hvx_epochs=" << mixedHmxHvxEpochs
                 << " ambiguous_epochs=" << ambiguousEpochs
                 << " single_writer_epochs=" << singleWriterEpochs
                 << " vector_writer_epochs=" << vectorWriterEpochs
                 << " redirect_candidates=" << epochRedirectCandidates
                 << " coverage_proven_epochs=" << coverageProvenEpochs
                 << " head_major_incompatible_rank_epochs="
                 << headMajorIncompatibleEpochs
                 << " rewrite_enabled="
                 << static_cast<bool>(rewriteEpochVtcm ||
                                      rewriteEpochHeadMajorVtcm)
                 << " head_major="
                 << static_cast<bool>(rewriteEpochHeadMajorVtcm)
                 << " producer_interchange="
                 << static_cast<bool>(rewriteProducerLoopOrder)
                 << " rewritten_epochs=" << rewrittenEpochs
                 << " reject_view=" << rejectedView
                 << " reject_escape=" << rejectedEscape
                 << " reject_footprint=" << rejectedFootprint
                 << " reject_writer=" << rejectedWriter
                 << " reject_readers=" << rejectedReaders << '\n';
  }
};

static bool sameStaticSubview(memref::SubViewOp lhs, memref::SubViewOp rhs) {
  return lhs.getStaticOffsets() == rhs.getStaticOffsets() &&
         lhs.getStaticSizes() == rhs.getStaticSizes() &&
         lhs.getStaticStrides() == rhs.getStaticStrides();
}

static bool p5hIsStrictlyBetween(Operation *op, Operation *begin,
                                 Operation *end,
                                 FunctionOpInterface function) {
  Operation *top = p5gTopLevelOperation(op, function);
  return top && top != begin && top != end &&
         top->getBlock() == begin->getBlock() &&
         begin->isBeforeInBlock(top) && top->isBeforeInBlock(end);
}

/// Prove that every operation reachable through a view of `value` remains in
/// the lexical interval in which the temporary contains the active value.
/// This deliberately rejects ambiguous region/block relationships rather than
/// relying on bufferization's current placement conventions.
static bool p5hAllUsersStrictlyBetween(Value value, Operation *begin,
                                       Operation *end,
                                       FunctionOpInterface function) {
  SmallVector<Value> worklist{value};
  llvm::SmallDenseSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      if (!p5hIsStrictlyBetween(user, begin, end, function))
        return false;
      if (auto view = dyn_cast<ViewLikeOpInterface>(user))
        worklist.push_back(view->getResult(0));
    }
  }
  return true;
}

static bool p5hIsStrictlyAfter(Operation *op, Operation *point,
                               FunctionOpInterface function) {
  Operation *top = p5gTopLevelOperation(op, function);
  return top && top != point && top->getBlock() == point->getBlock() &&
         point->isBeforeInBlock(top);
}

/// Prove that an existing destination view is not observed before the old
/// writeback.  P5h moves the descriptor and initializes it at the seed point,
/// so an earlier reader would otherwise observe the value too soon.
static bool p5hAllUsersStrictlyAfter(Value value, Operation *point,
                                     Operation *allowedUser,
                                     FunctionOpInterface function) {
  SmallVector<Value> worklist{value};
  llvm::SmallDenseSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      if (user == allowedUser)
        continue;
      if (!p5hIsStrictlyAfter(user, point, function))
        return false;
      if (auto view = dyn_cast<ViewLikeOpInterface>(user))
        worklist.push_back(view->getResult(0));
    }
  }
  return true;
}

struct P5hProducerEpochAudit {
  int64_t writerCount = 0;
  int64_t readerCount = 0;
  int64_t lateOtherReaders = 0;
  bool fullOverwrite = false;
  bool redirectReady = false;
  std::string writerName = "none";
};

/// Audit the writer epoch that produces source.root for one P5h chain.  A
/// future producer-direct rewrite is only legal when the last writer before
/// the seed is unique, completely overwrites the root, and source.root has no
/// reader after the seed other than the seed/whole copies that P5h removes.
static P5hProducerEpochAudit
p5hAuditProducerEpoch(Value root, memref::CopyOp seedCopy,
                      memref::CopyOp wholeCopy,
                      FunctionOpInterface function) {
  P5hProducerEpochAudit audit;
  SmallVector<Value> worklist{root};
  llvm::SmallDenseSet<Value, 32> aliases;
  SmallPtrSet<Operation *, 16> writers;
  SmallPtrSet<Operation *, 32> readers;
  while (!worklist.empty()) {
    Value alias = worklist.pop_back_val();
    if (!aliases.insert(alias).second)
      continue;
    for (Operation *user : alias.getUsers()) {
      if (auto view = dyn_cast<ViewLikeOpInterface>(user);
          view && view.getViewSource() == alias) {
        worklist.push_back(view.getViewDest());
        continue;
      }
      bool writes = hasEffect<MemoryEffects::Write>(user, alias);
      bool reads = hasEffect<MemoryEffects::Read>(user, alias);
      if (auto matmul = dyn_cast<hexkl::MatmulOp>(user)) {
        reads |= matmul.getLhs() == alias || matmul.getRhs() == alias;
        writes |= matmul.getOuts() == alias;
      }
      if (writes)
        writers.insert(user);
      if (reads)
        readers.insert(user);
    }
  }

  Operation *seedTop = p5gTopLevelOperation(seedCopy, function);
  if (!seedTop)
    return audit;
  SmallPtrSet<Operation *, 16> writerTopSet;
  for (Operation *writer : writers)
    if (Operation *top = p5gTopLevelOperation(writer, function);
        top && top->getBlock() == seedTop->getBlock())
      writerTopSet.insert(top);
  SmallVector<Operation *> writerTops(writerTopSet.begin(),
                                      writerTopSet.end());
  llvm::sort(writerTops, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  Operation *lastWriterTop = nullptr;
  Operation *nextWriterTop = nullptr;
  for (Operation *top : writerTops) {
    if (top == seedTop)
      return audit;
    if (top->isBeforeInBlock(seedTop))
      lastWriterTop = top;
    else {
      nextWriterTop = top;
      break;
    }
  }
  if (!lastWriterTop)
    return audit;

  SmallVector<Operation *> epochWriters;
  for (Operation *writer : writers)
    if (p5gTopLevelOperation(writer, function) == lastWriterTop)
      epochWriters.push_back(writer);
  audit.writerCount = epochWriters.size();
  if (epochWriters.size() == 1) {
    audit.writerName =
        epochWriters.front()->getName().getStringRef().str();
    audit.fullOverwrite =
        p5gIsFullRootOverwrite(epochWriters.front(), root);
  }

  for (Operation *reader : readers) {
    Operation *top = p5gTopLevelOperation(reader, function);
    if (!top || top->getBlock() != seedTop->getBlock() ||
        top == lastWriterTop)
      continue;
    if (!lastWriterTop->isBeforeInBlock(top) ||
        (nextWriterTop && !top->isBeforeInBlock(nextWriterTop)))
      continue;
    ++audit.readerCount;
    if (reader == seedCopy || reader == wholeCopy)
      continue;
    if (top == seedTop || seedTop->isBeforeInBlock(top))
      ++audit.lateOtherReaders;
  }
  audit.redirectReady = audit.writerCount == 1 && audit.fullOverwrite &&
                        audit.lateOtherReaders == 0;
  return audit;
}

/// P5h removes this strictly matched bufferization artifact:
///
///   copy source.active -> temporary
///   consumer mutates temporary through subviews
///   copy source.root   -> destination.root
///   copy temporary     -> destination.active
///
/// The active computation is redirected to destination.active.  Its input is
/// still seeded from source.active, while the whole-root copy is narrowed to
/// the untouched static tail.  This first physical gate therefore removes two
/// active-sized materializations without changing the consumer algorithm.
struct AlpsAttentionDestinationFormationPass final
    : ::impl::AlpsAttentionDestinationFormationBase<
          AlpsAttentionDestinationFormationPass> {
  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    SmallVector<memref::AllocOp> allocations;
    function.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
    int64_t matched = 0, rewritten = 0, eliminatedBytes = 0,
            residualTailBytes = 0, rejectedTemporaryLifetime = 0,
            rejectedDeallocLifetime = 0, rejectedDestinationActive = 0,
            rejectedDestinationRoot = 0, producerEpochs = 0,
            producerUniqueWriters = 0, producerFullOverwrites = 0,
            producerRedirectReady = 0;
    IRRewriter rewriter(function.getContext());

    for (memref::AllocOp temporary : allocations) {
      if (!temporary->getBlock())
        continue;
      auto temporaryType = temporary.getType();
      if (!temporaryType.hasStaticShape() || temporaryType.getRank() != 3 ||
          temporaryType.getMemorySpaceAsInt() != 0)
        continue;

      memref::CopyOp seedCopy, writebackCopy;
      SmallVector<memref::SubViewOp> temporaryViews;
      SmallVector<memref::DeallocOp> temporaryDeallocs;
      bool unsupportedUser = false;
      for (Operation *user : temporary.getResult().getUsers()) {
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          if (copy.getTarget() == temporary.getResult() && !seedCopy)
            seedCopy = copy;
          else if (copy.getSource() == temporary.getResult() && !writebackCopy)
            writebackCopy = copy;
          else
            unsupportedUser = true;
        } else if (auto view = dyn_cast<memref::SubViewOp>(user)) {
          temporaryViews.push_back(view);
        } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
          temporaryDeallocs.push_back(dealloc);
        } else {
          unsupportedUser = true;
        }
      }
      if (unsupportedUser || !seedCopy || !writebackCopy ||
          temporaryViews.empty())
        continue;

      auto sourceActive =
          seedCopy.getSource().getDefiningOp<memref::SubViewOp>();
      auto destinationActive =
          writebackCopy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!sourceActive || !destinationActive ||
          !sameStaticSubview(sourceActive, destinationActive) ||
          sourceActive.getStaticOffsets().size() != 3 ||
          sourceActive.getStaticSizes().size() != 3)
        continue;
      Value sourceRoot = sourceActive.getSource();
      Value destinationRoot = destinationActive.getSource();
      auto destinationAlloc = destinationRoot.getDefiningOp<memref::AllocOp>();
      auto sourceRootType = dyn_cast<MemRefType>(sourceRoot.getType());
      auto destinationRootType = dyn_cast<MemRefType>(destinationRoot.getType());
      if (!destinationAlloc || sourceRoot == destinationRoot ||
          p5gFindRootBuffer(sourceRoot) == p5gFindRootBuffer(destinationRoot) ||
          !sourceRootType || !destinationRootType ||
          sourceRootType != destinationRootType ||
          !sourceRootType.hasStaticShape() || sourceRootType.getRank() != 3 ||
          temporaryType.getShape() != sourceActive.getStaticSizes())
        continue;

      memref::CopyOp wholeCopy;
      int64_t wholeCopyCount = 0;
      for (Operation *user : destinationRoot.getUsers())
        if (auto copy = dyn_cast<memref::CopyOp>(user);
            copy && copy.getSource() == sourceRoot &&
            copy.getTarget() == destinationRoot) {
          ++wholeCopyCount;
          wholeCopy = copy;
        }
      if (wholeCopyCount != 1 || !wholeCopy ||
          seedCopy->getBlock() != wholeCopy->getBlock() ||
          wholeCopy->getBlock() != writebackCopy->getBlock() ||
          destinationAlloc->getBlock() != wholeCopy->getBlock() ||
          !seedCopy->isBeforeInBlock(wholeCopy) ||
          !wholeCopy->isBeforeInBlock(writebackCopy))
        continue;

      // The temporary may only be consumed between its initialization and the
      // whole-root copy.  Deallocation must remain after the old writeback.
      // Existing destination users must also remain after that writeback;
      // otherwise moving/initializing destination.active early is observable.
      bool temporaryLifetimeLegal = true;
      for (memref::SubViewOp view : temporaryViews)
        temporaryLifetimeLegal &= p5hAllUsersStrictlyBetween(
            view.getResult(), seedCopy, wholeCopy, function);
      bool deallocLifetimeLegal = temporaryDeallocs.size() <= 1;
      for (memref::DeallocOp dealloc : temporaryDeallocs)
        deallocLifetimeLegal &=
            p5hIsStrictlyAfter(dealloc, writebackCopy, function);
      bool destinationActiveLegal = p5hAllUsersStrictlyAfter(
          destinationActive.getResult(), writebackCopy, writebackCopy,
          function);
      bool destinationRootLegal = true;
      for (Operation *user : destinationRoot.getUsers()) {
        if (user == wholeCopy || user == destinationActive.getOperation())
          continue;
        if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
          destinationRootLegal &= p5hAllUsersStrictlyAfter(
              view->getResult(0), writebackCopy, nullptr, function);
          continue;
        }
        // Descriptor metadata does not read or write the allocation.
        if (isa<memref::AssumeAlignmentOp>(user))
          continue;
        destinationRootLegal &=
            p5hIsStrictlyAfter(user, writebackCopy, function);
      }
      rejectedTemporaryLifetime += !temporaryLifetimeLegal;
      rejectedDeallocLifetime += !deallocLifetimeLegal;
      rejectedDestinationActive += !destinationActiveLegal;
      rejectedDestinationRoot += !destinationRootLegal;
      if (!temporaryLifetimeLegal || !deallocLifetimeLegal ||
          !destinationActiveLegal || !destinationRootLegal)
        continue;

      ArrayRef<int64_t> offsets = sourceActive.getStaticOffsets();
      ArrayRef<int64_t> sizes = sourceActive.getStaticSizes();
      ArrayRef<int64_t> strides = sourceActive.getStaticStrides();
      ArrayRef<int64_t> rootShape = sourceRootType.getShape();
      if (offsets != ArrayRef<int64_t>({0, 0, 0}) ||
          strides != ArrayRef<int64_t>({1, 1, 1}) ||
          sizes[0] != rootShape[0] || sizes[1] != rootShape[1] ||
          sizes[2] <= 0 || sizes[2] >= rootShape[2])
        continue;

      ++matched;
      P5hProducerEpochAudit producerAudit = p5hAuditProducerEpoch(
          sourceRoot, seedCopy, wholeCopy, function);
      ++producerEpochs;
      producerUniqueWriters += producerAudit.writerCount == 1;
      producerFullOverwrites += producerAudit.fullOverwrite;
      producerRedirectReady += producerAudit.redirectReady;
      {
        std::lock_guard<std::mutex> auditLock(reportMutex);
        llvm::errs() << "[ALPS-P5H-PRODUCER] function=" << function.getName()
                     << " writer=" << producerAudit.writerName
                     << " writer_count=" << producerAudit.writerCount
                     << " readers=" << producerAudit.readerCount
                     << " late_other_readers="
                     << producerAudit.lateOtherReaders
                     << " full_overwrite=" << producerAudit.fullOverwrite
                     << " redirect_ready=" << producerAudit.redirectReady
                     << '\n';
      }
      // Destination allocation and its active descriptor originally appear
      // after the in-place consumer. Move only these side-effect-free ops so
      // the original seed copy can initialize the final active subview.
      destinationAlloc->moveBefore(seedCopy);
      destinationActive->moveBefore(seedCopy);

      rewriter.setInsertionPoint(seedCopy);
      memref::CopyOp::create(rewriter, seedCopy.getLoc(), sourceActive,
                             destinationActive);

      // Rebuild every temporary-derived descriptor on destination.active at
      // the same program point. Its physical row stride may change, but the
      // innermost active dimension remains unit-stride by construction.
      for (memref::SubViewOp view : temporaryViews) {
        rewriter.setInsertionPoint(view);
        MemRefType replacementType = memref::SubViewOp::inferRankReducedResultType(
            view.getType().getShape(), destinationActive.getType(),
            view.getMixedOffsets(), view.getMixedSizes(),
            view.getMixedStrides());
        auto replacement = memref::SubViewOp::create(
            rewriter, view.getLoc(), replacementType, destinationActive,
            view.getMixedOffsets(), view.getMixedSizes(),
            view.getMixedStrides());
        view.getResult().replaceAllUsesWith(replacement.getResult());
        rewriter.eraseOp(view);
      }

      // Only the untouched tail still needs the original root value. Keep it
      // at the old whole-copy point, after the active computation, because the
      // two subviews are disjoint.
      SmallVector<OpFoldResult> tailOffsets{
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(sizes[2])};
      SmallVector<OpFoldResult> tailSizes{
          rewriter.getIndexAttr(rootShape[0]),
          rewriter.getIndexAttr(rootShape[1]),
          rewriter.getIndexAttr(rootShape[2] - sizes[2])};
      SmallVector<OpFoldResult> tailStrides(3, rewriter.getIndexAttr(1));
      rewriter.setInsertionPoint(wholeCopy);
      auto sourceTail = memref::SubViewOp::create(
          rewriter, wholeCopy.getLoc(), sourceRoot, tailOffsets, tailSizes,
          tailStrides);
      auto destinationTail = memref::SubViewOp::create(
          rewriter, wholeCopy.getLoc(), destinationRoot, tailOffsets,
          tailSizes, tailStrides);
      memref::CopyOp::create(rewriter, wholeCopy.getLoc(), sourceTail,
                             destinationTail);

      int64_t elementBytes = sourceRootType.getElementTypeBitWidth() / 8;
      int64_t activeBytes = temporaryType.getNumElements() * elementBytes;
      int64_t tailBytes = sourceTail.getType().getNumElements() * elementBytes;
      eliminatedBytes += 2 * activeBytes;
      residualTailBytes += tailBytes;
      rewriter.eraseOp(seedCopy);
      rewriter.eraseOp(wholeCopy);
      rewriter.eraseOp(writebackCopy);
      for (memref::DeallocOp dealloc : temporaryDeallocs)
        rewriter.eraseOp(dealloc);
      if (temporary->use_empty())
        rewriter.eraseOp(temporary);
      ++rewritten;
    }

    Builder builder(function.getContext());
    function->setAttr("alps.p5h.matched_chains",
                      builder.getI64IntegerAttr(matched));
    function->setAttr("alps.p5h.rewritten_chains",
                      builder.getI64IntegerAttr(rewritten));
    function->setAttr("alps.p5h.eliminated_copy_bytes",
                      builder.getI64IntegerAttr(eliminatedBytes));
    function->setAttr("alps.p5h.residual_tail_copy_bytes",
                      builder.getI64IntegerAttr(residualTailBytes));
    function->setAttr("alps.p5h.producer_redirect_ready",
                      builder.getI64IntegerAttr(producerRedirectReady));
    std::lock_guard<std::mutex> lock(reportMutex);
    llvm::errs() << "[ALPS-P5H] function=" << function.getName()
                 << " matched=" << matched << " rewritten=" << rewritten
                 << " eliminated_copy_bytes=" << eliminatedBytes
                 << " residual_tail_copy_bytes=" << residualTailBytes
                 << " reject_temporary_lifetime=" << rejectedTemporaryLifetime
                 << " reject_dealloc_lifetime=" << rejectedDeallocLifetime
                 << " reject_destination_active=" << rejectedDestinationActive
                 << " reject_destination_root=" << rejectedDestinationRoot
                 << " producer_epochs=" << producerEpochs
                 << " producer_unique_writers=" << producerUniqueWriters
                 << " producer_full_overwrites=" << producerFullOverwrites
                 << " producer_redirect_ready=" << producerRedirectReady
                 << '\n';
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
hexagon::createAlpsContinuityAuditPass() {
  return std::make_unique<AlpsContinuityAuditPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsLayoutSupplyPrefetchPass(
    const AlpsLayoutSupplyPrefetchOptions &options) {
  return std::make_unique<AlpsLayoutSupplyPrefetchPass>(options);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsCrpSupplyAnalysisPass(
    const AlpsCrpSupplyAnalysisOptions &options) {
  return std::make_unique<AlpsCrpSupplyAnalysisPass>(options);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsCrpSupplyPrefetchPass(
    const AlpsCrpSupplyPrefetchOptions &options) {
  return std::make_unique<AlpsCrpSupplyPrefetchPass>(options);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsCrpVtcmFormationPass() {
  return std::make_unique<AlpsCrpVtcmFormationPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsCrpVtcmWindowPass() {
  return std::make_unique<AlpsCrpVtcmWindowPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsCrpProducerDirectAnalysisPass(
    const AlpsCrpProducerDirectAnalysisOptions &options) {
  return std::make_unique<AlpsCrpProducerDirectAnalysisPass>(options);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createAlpsAttentionDestinationFormationPass() {
  return std::make_unique<AlpsAttentionDestinationFormationPass>();
}
