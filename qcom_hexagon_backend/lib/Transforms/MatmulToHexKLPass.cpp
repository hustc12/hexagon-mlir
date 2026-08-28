//===-- MatmulToHexKLPass.cpp - linalg.matmul to hexkl ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Converts linalg.matmul with FP16 inputs to hexkl.matmul.
//
// Macro mode: Uses hexkl_macro_* API (f16 output only, upcast to f32 if needed)
//             Requires constant weights (preprocessed at compile-time)
// Micro mode: Uses hexkl_micro_* API (more flexible I/O dtype, memory
// allocation)
//
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <map>
#include <optional>

#define DEBUG_TYPE "matmul-to-hexkl"

using namespace mlir;
using namespace hexagon;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_MATMULTOHEXKL
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  bool isMacroMode;
  bool consumerF16Epilogue;
  bool consumerF16BiasEpilogue;

  MatmulToHexKL(MLIRContext *ctx, bool macroMode, bool formConsumerF16Epilogue,
                bool formConsumerF16BiasEpilogue)
      : OpRewritePattern(ctx), isMacroMode(macroMode),
        consumerF16Epilogue(formConsumerF16Epilogue),
        consumerF16BiasEpilogue(formConsumerF16BiasEpilogue) {}

  bool isConstantWeight(Value weight) const {
    return weight.getDefiningOp<arith::ConstantOp>() != nullptr;
  }

  /// Match only the exact representation round trip that the HexKL micro
  /// implementation already performs internally: HMX accumulates/readbacks
  /// F16, copies it to an F32 tensor, and the sole identity-layout consumer
  /// immediately truncates that tensor back to F16.  Keeping this matcher
  /// deliberately narrow makes P5j a consumer-contract formation, not a
  /// general mixed-precision rewrite.
  linalg::GenericOp matchIdentityF16TruncConsumer(linalg::MatmulOp op) const {
    if (!consumerF16Epilogue || isMacroMode || op->getNumResults() != 1 ||
        !op->getResult(0).hasOneUse())
      return {};

    auto generic =
        dyn_cast<linalg::GenericOp>(*op->getResult(0).getUsers().begin());
    if (!generic || generic.getNumDpsInputs() != 1 ||
        generic.getNumDpsInits() != 1 ||
        generic.getDpsInputOperand(0)->get() != op->getResult(0) ||
        generic->getNumResults() != 1)
      return {};

    auto resultType = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    auto inputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !inputType || resultType.getShape() != inputType.getShape() ||
        !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF16())
      return {};

    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType iterator) {
                        return iterator == utils::IteratorType::parallel;
                      }))
      return {};
    for (AffineMap map : generic.getIndexingMapsArray())
      if (!map.isIdentity())
        return {};

    Block &body = generic.getRegion().front();
    if (!llvm::hasSingleElement(body.without_terminator()))
      return {};
    auto trunc = dyn_cast<arith::TruncFOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!trunc || !yield || trunc.getIn() != body.getArgument(0) ||
        !trunc.getIn().getType().isF32() ||
        !trunc.getOut().getType().isF16() || yield.getValues().size() != 1 ||
        yield.getValues()[0] != trunc.getOut())
      return {};
    return generic;
  }

  struct BiasConsumer {
    linalg::GenericOp op;
    Value bias;
  };

  /// Strict first-stage P5l contract: C[m,n] + bias[n] -> final[m,n].
  std::optional<BiasConsumer>
  matchRank2BroadcastBiasConsumer(linalg::GenericOp trunc) const {
    if (!consumerF16BiasEpilogue || !trunc || trunc->getNumResults() != 1 ||
        !trunc->getResult(0).hasOneUse())
      return std::nullopt;
    auto generic =
        dyn_cast<linalg::GenericOp>(*trunc->getResult(0).getUsers().begin());
    if (!generic || generic.getNumDpsInputs() != 2 ||
        generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
      return std::nullopt;
    auto resultType = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2 ||
        !resultType.getElementType().isF16())
      return std::nullopt;
    if (!llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      }))
      return std::nullopt;

    unsigned srcIndex = generic.getDpsInputOperand(0)->get() == trunc->getResult(0)
                            ? 0
                            : generic.getDpsInputOperand(1)->get() ==
                                      trunc->getResult(0)
                                  ? 1
                                  : 2;
    if (srcIndex == 2)
      return std::nullopt;
    unsigned biasIndex = 1 - srcIndex;
    auto biasType = dyn_cast<RankedTensorType>(
        generic.getDpsInputOperand(biasIndex)->get().getType());
    if (!biasType || biasType.getRank() != 1 ||
        !biasType.getElementType().isF16() ||
        biasType.getShape()[0] != resultType.getShape()[1])
      return std::nullopt;

    auto maps = generic.getIndexingMapsArray();
    AffineMap identity = AffineMap::getMultiDimIdentityMap(2, getContext());
    AffineMap biasMap = AffineMap::get(
        2, 0, getAffineDimExpr(1, getContext()), getContext());
    if (maps[srcIndex] != identity || maps[biasIndex] != biasMap ||
        maps[2] != identity)
      return std::nullopt;

    Block &body = generic.getRegion().front();
    if (!llvm::hasSingleElement(body.without_terminator()))
      return std::nullopt;
    auto add = dyn_cast<arith::AddFOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!add || !yield || yield.getValues().size() != 1 ||
        yield.getValues()[0] != add.getResult() ||
        !((add.getLhs() == body.getArgument(srcIndex) &&
           add.getRhs() == body.getArgument(biasIndex)) ||
          (add.getRhs() == body.getArgument(srcIndex) &&
           add.getLhs() == body.getArgument(biasIndex))))
      return std::nullopt;
    return BiasConsumer{generic,
                        generic.getDpsInputOperand(biasIndex)->get()};
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getOutputs()[0];

    auto aType = dyn_cast<RankedTensorType>(A.getType());
    auto bType = dyn_cast<RankedTensorType>(B.getType());
    auto cType = dyn_cast<RankedTensorType>(C.getType());

    if (!aType || !bType || !cType) {
      return rewriter.notifyMatchFailure(op, "not ranked tensor types");
    }

    if (aType.getRank() != 2 || bType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matmul supported");
    }

    if (llvm::any_of(aType.getShape(), ShapedType::isDynamic) ||
        llvm::any_of(bType.getShape(), ShapedType::isDynamic)) {
      return rewriter.notifyMatchFailure(op,
                                         "dynamic dimensions not supported");
    }

    Type aElemType = aType.getElementType();
    Type bElemType = bType.getElementType();
    Type cElemType = cType.getElementType();

    if (!aElemType.isF16() || !bElemType.isF16()) {
      return rewriter.notifyMatchFailure(op, "inputs must be f16");
    }

    if (!cElemType.isF16() && !cElemType.isF32()) {
      return rewriter.notifyMatchFailure(op, "output must be f16 or f32");
    }

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    if (isMacroMode) {
      // Macro mode requires constant weights
      if (!isConstantWeight(B)) {
        return rewriter.notifyMatchFailure(
            op, "macro mode requires constant weights (weight must be "
                "arith.constant)");
      }

      const int64_t MAX_N_ROW = 1600;
      const int64_t MAX_N_COL = 5120;
      const int64_t MAX_N_INNER = 76030;

      if (M > MAX_N_ROW) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: M=" + std::to_string(M) + " exceeds max " +
                    std::to_string(MAX_N_ROW));
      }

      if (N > MAX_N_COL) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: N=" + std::to_string(N) + " exceeds max " +
                    std::to_string(MAX_N_COL));
      }

      if (K > MAX_N_INNER) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: K=" + std::to_string(K) + " exceeds max " +
                    std::to_string(MAX_N_INNER));
      }

      const int64_t BLOCK_SIZE = 32;

      if ((M % BLOCK_SIZE) != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: M=" + std::to_string(M) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }

      if (N % BLOCK_SIZE != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: N=" + std::to_string(N) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }

      if (K % BLOCK_SIZE != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: K=" + std::to_string(K) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[" << DEBUG_TYPE
                   << "] Converting matmul to hexkl.matmul ("
                   << (isMacroMode ? "macro" : "micro") << " mode):\n";
      llvm::dbgs() << "  LHS: " << aType << " (M=" << M << ", K=" << K << ")\n";
      llvm::dbgs() << "  RHS: " << bType << " (K=" << K << ", N=" << N << ")\n";
      llvm::dbgs() << "  Output: " << cType << " (M=" << M << ", N=" << N
                   << ")\n";
      if (isMacroMode) {
        llvm::dbgs() << "  Weight is constant: "
                     << (isConstantWeight(B) ? "yes" : "no") << "\n";
      }
    });

    if (isMacroMode) {
      auto f16Shape = cType.getShape();
      auto f16Type = RankedTensorType::get(f16Shape, rewriter.getF16Type());
      Value f16Output = tensor::EmptyOp::create(rewriter, loc, f16Shape,
                                                rewriter.getF16Type());

      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, f16Type, A, B, f16Output);

      Value result = hexklMatmul->getResult(0);

      if (cElemType.isF32()) {
        Value f32Output = tensor::EmptyOp::create(rewriter, loc, f16Shape,
                                                  rewriter.getF32Type());

        AffineMap identityMap = rewriter.getMultiDimIdentityMap(2);
        SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
        SmallVector<utils::IteratorType> iteratorTypes = {
            utils::IteratorType::parallel, utils::IteratorType::parallel};

        auto castOp = linalg::GenericOp::create(
            rewriter, loc, cType, result, f32Output, indexingMaps,
            iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
              Value f16Val = args[0];
              Value f32Val =
                  arith::ExtFOp::create(b, loc, rewriter.getF32Type(), f16Val);
              linalg::YieldOp::create(b, loc, f32Val);
            });

        result = castOp.getResult(0);
      }

      rewriter.replaceOp(op, result);

    } else if (linalg::GenericOp truncConsumer =
                   matchIdentityF16TruncConsumer(op)) {
      std::optional<BiasConsumer> biasConsumer =
          matchRank2BroadcastBiasConsumer(truncConsumer);
      auto f16Type = RankedTensorType::get(cType.getShape(), rewriter.getF16Type());
      RankedTensorType finalType;
      Value finalOutput;
      if (biasConsumer) {
        finalType =
            cast<RankedTensorType>(biasConsumer->op->getResult(0).getType());
        // Establish the consumer-selected destination before its producer.
        // One-shot bufferization preserves this ordering, so the HMX drain can
        // form the final representation at the original matmul point without
        // moving mutable-buffer reads across the epilogue boundary.
        finalOutput = tensor::EmptyOp::create(
            rewriter, loc, finalType.getShape(), finalType.getElementType());
      }
      Value f16Output = tensor::EmptyOp::create(rewriter, loc, cType.getShape(),
                                                rewriter.getF16Type());
      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, f16Type, A, B, f16Output);
      hexklMatmul->setAttr("alps.p5j.consumer_f16_epilogue",
                           rewriter.getUnitAttr());
      if (biasConsumer) {
        auto epilogue = hexkl::F16BiasEpilogueOp::create(
            rewriter, loc, finalType, hexklMatmul->getResult(0),
            biasConsumer->bias, finalOutput);
        epilogue->setAttr("alps.p5l.consumer_bias_formation",
                          rewriter.getUnitAttr());
        rewriter.replaceOp(biasConsumer->op, epilogue->getResult(0));
        rewriter.eraseOp(truncConsumer);
      } else {
        rewriter.replaceOp(truncConsumer, hexklMatmul->getResult(0));
      }
      rewriter.eraseOp(op);
    } else {
      Value outputOperand =
          tensor::EmptyOp::create(rewriter, loc, cType.getShape(), cElemType);

      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, cType, A, B, outputOperand);

      rewriter.replaceOp(op, hexklMatmul->getResult(0));
    }

    return success();
  }
};

struct MatmulToHexKLPass
    : public hexagon::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  using hexagon::impl::MatmulToHexKLBase<MatmulToHexKLPass>::MatmulToHexKLBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    bool isMacroMode = (mode == "macro");

    if (mode != "macro" && mode != "micro") {
      emitError(getOperation()->getLoc(),
                "Invalid mode '" + mode + "'. Must be 'macro' or 'micro'");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] Running MatmulToHexKLPass (" << mode
               << " mode)\n");

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToHexKL>(&getContext(), isMacroMode,
                                consumerF16Epilogue,
                                consumerF16BiasEpilogue);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" << DEBUG_TYPE << "] Pattern application failed\n");
      return signalPassFailure();
    }

    if (consumerF16Epilogue) {
      int64_t formed = 0;
      struct ConsumerStats {
        int64_t sites = 0;
        int64_t resultBytes = 0;
      };
      std::map<std::string, ConsumerStats> consumers;
      getOperation()->walk([&](hexkl::MatmulOp matmul) {
        if (!matmul->hasAttr("alps.p5j.consumer_f16_epilogue"))
          return;
        ++formed;
        auto resultType =
            dyn_cast<RankedTensorType>(matmul->getResult(0).getType());
        int64_t resultBytes =
            resultType && resultType.hasStaticShape()
                ? resultType.getNumElements() *
                      resultType.getElementType().getIntOrFloatBitWidth() / 8
                : 0;
        if (matmul->getResult(0).use_empty()) {
          auto &stats = consumers["none"];
          ++stats.sites;
          stats.resultBytes += resultBytes;
          return;
        }
        for (Operation *user : matmul->getResult(0).getUsers()) {
          std::string signature = user->getName().getStringRef().str();
          if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
            signature += ":inputs=" +
                         std::to_string(generic.getNumDpsInputs()) + ":body=";
            bool first = true;
            for (Operation &bodyOp :
                 generic.getRegion().front().without_terminator()) {
              if (!first)
                signature += "+";
              first = false;
              signature += bodyOp.getName().getStringRef().str();
            }
            signature += ":maps=";
            first = true;
            for (AffineMap map : generic.getIndexingMapsArray()) {
              if (!first)
                signature += "/";
              first = false;
              std::string mapText;
              llvm::raw_string_ostream os(mapText);
              map.print(os);
              signature += os.str();
            }
          }
          auto &stats = consumers[signature];
          ++stats.sites;
          stats.resultBytes += resultBytes;
        }
      });
      llvm::errs() << "[ALPS-P5J] function=" << getOperation().getName()
                   << " formed_f16_epilogues=" << formed << "\n";
      for (const auto &[signature, stats] : consumers)
        llvm::errs() << "[ALPS-P5J-CONSUMER] function="
                     << getOperation().getName() << " signature=" << signature
                     << " sites=" << stats.sites
                     << " result_bytes=" << stats.resultBytes << "\n";
      if (consumerF16BiasEpilogue) {
        int64_t biasFormed = 0;
        getOperation()->walk([&](hexkl::F16BiasEpilogueOp epilogue) {
          if (epilogue->hasAttr("alps.p5l.consumer_bias_formation"))
            ++biasFormed;
        });
        llvm::errs() << "[ALPS-P5L] function=" << getOperation().getName()
                     << " formed_rank2_bias_epilogues=" << biasFormed << "\n";
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] MatmulToHexKLPass complete\n");
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createMatmulToHexKLPass(const MatmulToHexKLOptions &options) {
  return std::make_unique<MatmulToHexKLPass>(options);
}

} // namespace hexagon
} // namespace mlir
