//===- LinalgToLLVMPass.cpp - Linalg to LLVM  conversion       ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization and lowering of MLIR IR to LLVM.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexKLToLLVM/Passes.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Conversion/OmniFetchToLLVM/OmniFetchToLLVM.h"
#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "linalg-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LINALGTOLLVM
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// Keep P2g-c's rank-4 register formation domain intact while applying the
/// upstream unit-dimension folding policy to every unrelated operation.  The
/// stock pass rebuilds linalg.generic without discardable attributes, which
/// silently erased the register-tile contract before tiling.
struct AlpsAwareFoldUnitExtentDimsPass
    : PassWrapper<AlpsAwareFoldUnitExtentDimsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlpsAwareFoldUnitExtentDimsPass)

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *context = root->getContext();
    linalg::ControlDropUnitDims options;
    options.controlFn = [](Operation *op) {
      if (op->hasAttr("alps.p2g.register_tile_contract"))
        return SmallVector<unsigned>{};
      if (auto generic = dyn_cast<linalg::GenericOp>(op))
        return llvm::to_vector(llvm::seq<unsigned>(0, generic.getNumLoops()));
      if (auto pad = dyn_cast<tensor::PadOp>(op))
        return llvm::to_vector(
            llvm::seq<unsigned>(0, pad.getSourceType().getRank()));
      return SmallVector<unsigned>{};
    };

    RewritePatternSet foldingPatterns(context);
    linalg::populateFoldUnitExtentDimsPatterns(foldingPatterns, options);
    walkAndApplyPatterns(root, std::move(foldingPatterns));

    RewritePatternSet cleanupPatterns(context);
    linalg::populateMoveInitOperandsToInputPattern(cleanupPatterns);
    linalg::populateFoldUnitExtentDimsCanonicalizationPatterns(cleanupPatterns,
                                                               options);
    if (failed(applyPatternsGreedily(root, std::move(cleanupPatterns))))
      signalPassFailure();
  }
};

/// One-shot bufferization rebuilds vector.transfer_read and intentionally
/// copies only the operation's semantic attributes.  Preserve P2g-c's
/// compiler-generated provenance across that boundary with an exact
/// LocationAttr ledger; P5f-a therefore never has to guess candidates from a
/// source-location string or admit unrelated traffic.
struct AlpsRegisterTileMarkerBridgePass
    : PassWrapper<AlpsRegisterTileMarkerBridgePass,
                  OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlpsRegisterTileMarkerBridgePass)

  explicit AlpsRegisterTileMarkerBridgePass(bool restore) : restore(restore) {}

  void runOnOperation() override {
    constexpr StringLiteral kLedger = "alps.p2g.register_tile_location_ledger";
    func::FuncOp function = getOperation();
    Builder builder(function.getContext());
    if (!restore) {
      SmallVector<Attribute> locations;
      function.walk([&](vector::TransferReadOp read) {
        if (read->hasAttr("alps.p2g.register_tile"))
          locations.push_back(read->getLoc());
      });
      if (!locations.empty())
        function->setAttr(kLedger, builder.getArrayAttr(locations));
      return;
    }

    auto ledger = function->getAttrOfType<ArrayAttr>(kLedger);
    if (!ledger)
      return;
    int64_t restored = 0;
    function.walk([&](vector::TransferReadOp read) {
      if (llvm::is_contained(ledger, Attribute(read->getLoc()))) {
        read->setAttr("alps.p2g.register_tile", builder.getUnitAttr());
        ++restored;
      }
    });
    function->removeAttr(kLedger);
    llvm::errs() << "[ALPS-P2G-C] marker_bridge_restored=" << restored << '\n';
  }

private:
  bool restore;
};

struct LinalgToLLVMPass : public ::impl::LinalgToLLVMBase<LinalgToLLVMPass> {
public:
  explicit LinalgToLLVMPass(const LinalgToLLVMOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, async::AsyncDialect, tensor::TensorDialect,
                    cf::ControlFlowDialect, bufferization::BufferizationDialect,
                    vector::VectorDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect, crouton::CroutonDialect, ttx::TTXDialect,
                    tptr::HexagonTPtrDialect, hexagonmem::HexagonMemDialect,
                    hexkl::HexKLDialect, omni_fetch::OmniFetchDialect,
                    quant::QuantDialect>();
  }

  void runOnOperation() override {
    // DEBUG (disabled): pass entry banner and option dump
    // llvm::errs() << "\n[LinalgToLLVM] ========== Pass Starting ==========\n";
    // llvm::errs() << "[LinalgToLLVM] enableOmniFetchVDAE = " <<
    // enableOmniFetchVDAE << "\n"; llvm::errs() << "[LinalgToLLVM] enableHexKL
    // = " << enableHexKL << "\n"; llvm::errs() << "[LinalgToLLVM]
    // omniFetchLookahead = " << omniFetchLookahead << "\n"; llvm::errs() <<
    // "[LinalgToLLVM] enableOmniFetchAdaptive = " << enableOmniFetchAdaptive <<
    // "\n"; llvm::errs() << "[LinalgToLLVM] enableOmniFetchLayoutAware = " <<
    // enableOmniFetchLayoutAware << "\n"; llvm::errs() << "[LinalgToLLVM]
    // ==========================================\n\n";

    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    // ALPS P0 causal split.  The legacy OmniFetch item-7 option remains an
    // umbrella alias so archived scripts reproduce their original topology.
    const bool alpsKvFusionPolicy =
        enableOmniFetchKvCachePrefetch || enableAlpsKvFusionPolicy;
    const bool alpsKvElementwiseFusionPolicy =
        alpsKvFusionPolicy || enableAlpsKvElementwiseFusionPolicy;
    const bool alpsKvMultiUseFusionPolicy =
        alpsKvFusionPolicy || enableAlpsKvMultiUseFusionPolicy;
    const bool alpsKvSplitReductionPolicy =
        alpsKvFusionPolicy || enableAlpsKvSplitReductionPolicy;
    const bool alpsKvSlicingPolicy =
        enableOmniFetchKvCachePrefetch || enableAlpsKvSlicingPolicy;
    const bool alpsKvRuntimePrefetch =
        enableOmniFetchKvCachePrefetch || enableAlpsKvRuntimePrefetch;
    const bool alpsMinimalStaticAdmission = enableAlpsMinimalStaticAdmission;
    const bool alpsExactReadiness = enableAlpsExactReadiness;
    const bool alpsExactOverlap = enableAlpsExactOverlap;
    const bool alpsContractLedger = enableAlpsContractDischargeLedger ||
                                    enableAlpsRepresentationSupplyAnalysis ||
                                    enableAlpsLayoutSupplyPrefetch;
    const bool alpsP2eContracts =
        alpsContractLedger || enableAlpsContinuityAudit;
    const bool alpsKvSemanticTracking =
        enableOmniFetchKvCachePrefetch || enableAlpsKvSemanticTracking ||
        alpsKvElementwiseFusionPolicy || alpsKvMultiUseFusionPolicy ||
        alpsKvSplitReductionPolicy || alpsKvSlicingPolicy ||
        alpsKvRuntimePrefetch || alpsMinimalStaticAdmission;
    const bool alpsPrefetchPipeline = enablePrefetch || alpsKvRuntimePrefetch ||
                                      enableAlpsFusedTransformTransfer ||
                                      alpsMinimalStaticAdmission;
    const bool alpsLayoutAware = enableOmniFetchLayoutAware ||
                                 enableAlpsFusedTransformTransfer ||
                                 alpsExactOverlap;

    Builder alpsBuilder(context);
    moduleOp->setAttr("alps.p0.kv_semantic_tracking",
                      alpsBuilder.getBoolAttr(alpsKvSemanticTracking));
    moduleOp->setAttr("alps.p0.kv_fusion_policy",
                      alpsBuilder.getBoolAttr(alpsKvFusionPolicy));
    moduleOp->setAttr("alps.p0b.kv_elementwise_fusion_policy",
                      alpsBuilder.getBoolAttr(alpsKvElementwiseFusionPolicy));
    moduleOp->setAttr("alps.p0b.kv_multi_use_fusion_policy",
                      alpsBuilder.getBoolAttr(alpsKvMultiUseFusionPolicy));
    moduleOp->setAttr("alps.p0b.kv_split_reduction_policy",
                      alpsBuilder.getBoolAttr(alpsKvSplitReductionPolicy));
    moduleOp->setAttr("alps.p0.kv_slicing_policy",
                      alpsBuilder.getBoolAttr(alpsKvSlicingPolicy));
    moduleOp->setAttr("alps.p0.kv_runtime_prefetch",
                      alpsBuilder.getBoolAttr(alpsKvRuntimePrefetch));
    moduleOp->setAttr("alps.p2d.minimal_static_admission",
                      alpsBuilder.getBoolAttr(alpsMinimalStaticAdmission));
    moduleOp->setAttr("alps.p2e.consumer_driven_layout",
                      alpsBuilder.getBoolAttr(enableAlpsConsumerDrivenLayout));
    moduleOp->setAttr(
        "alps.p2f.consumer_layout_propagation",
        alpsBuilder.getBoolAttr(enableAlpsConsumerLayoutPropagation));
    moduleOp->setAttr("alps.p2g.continuity_audit",
                      alpsBuilder.getBoolAttr(enableAlpsContinuityAudit));
    moduleOp->setAttr(
        "alps.p2g.loop_interchanged_direct_formation",
        alpsBuilder.getBoolAttr(enableAlpsLoopInterchangedDirectFormation));
    moduleOp->setAttr("alps.p2g.register_tile_formation",
                      alpsBuilder.getBoolAttr(enableAlpsRegisterTileFormation));
    moduleOp->setAttr("alps.p5a.contract_discharge_ledger",
                      alpsBuilder.getBoolAttr(alpsContractLedger));
    moduleOp->setAttr(
        "alps.p5b.representation_supply_analysis",
        alpsBuilder.getBoolAttr(enableAlpsRepresentationSupplyAnalysis));
    moduleOp->setAttr("alps.p5c.layout_supply_prefetch",
                      alpsBuilder.getBoolAttr(enableAlpsLayoutSupplyPrefetch));
    moduleOp->setAttr("alps.p5f_a.crp_supply_analysis",
                      alpsBuilder.getBoolAttr(enableAlpsCrpSupplyAnalysis));
    moduleOp->setAttr("alps.p5f_b.crp_supply_prefetch",
                      alpsBuilder.getBoolAttr(enableAlpsCrpSupplyPrefetch));
    moduleOp->setAttr("alps.p5f_c.segmented_supply",
                      alpsBuilder.getBoolAttr(enableAlpsCrpSegmentedSupply));
    moduleOp->setAttr("alps.p5g_a.vtcm_formation",
                      alpsBuilder.getBoolAttr(enableAlpsCrpVtcmFormation));
    moduleOp->setAttr("alps.p5g_b.vtcm_window",
                      alpsBuilder.getBoolAttr(enableAlpsCrpVtcmWindow));
    moduleOp->setAttr("alps.p5g_c.vtcm_async_window",
                      alpsBuilder.getBoolAttr(enableAlpsCrpVtcmAsyncWindow));
    moduleOp->setAttr(
        "alps.p5g_d.producer_direct_analysis",
        alpsBuilder.getBoolAttr(enableAlpsCrpProducerDirectAnalysis));
    moduleOp->setAttr("alps.p5g_e.producer_direct_vtcm",
                      alpsBuilder.getBoolAttr(enableAlpsCrpProducerDirectVtcm));
    moduleOp->setAttr(
        "alps.p5g_f.producer_direct_head_major",
        alpsBuilder.getBoolAttr(enableAlpsCrpProducerDirectHeadMajor));
    moduleOp->setAttr(
        "alps.p5g_g.producer_loop_formation",
        alpsBuilder.getBoolAttr(enableAlpsCrpProducerLoopFormation));
    moduleOp->setAttr(
        "alps.p5h.attention_destination_formation",
        alpsBuilder.getBoolAttr(enableAlpsAttentionDestinationFormation));
    moduleOp->setAttr("alps.p5i.patch_conv_formation",
                      alpsBuilder.getBoolAttr(enableAlpsPatchConvFormation));
    moduleOp->setAttr(
        "alps.p5j.hmx_f16_epilogue_formation",
        alpsBuilder.getBoolAttr(enableAlpsHmxF16EpilogueFormation));
    moduleOp->setAttr(
        "alps.p5k.hmx_direct_output_formation",
        alpsBuilder.getBoolAttr(enableAlpsHmxDirectOutputFormation));
    moduleOp->setAttr(
        "alps.p5l.hmx_f16_bias_epilogue_formation",
        alpsBuilder.getBoolAttr(enableAlpsHmxF16BiasEpilogueFormation));
    moduleOp->setAttr(
        "alps.p5m.hmx_async_drain_analysis",
        alpsBuilder.getBoolAttr(enableAlpsHmxAsyncDrainAnalysis));
    moduleOp->setAttr("alps.p5n.hmx_async_drain",
                      alpsBuilder.getBoolAttr(enableAlpsHmxAsyncDrain));
    moduleOp->setAttr("alps.p3a.exact_readiness",
                      alpsBuilder.getBoolAttr(alpsExactReadiness));
    moduleOp->setAttr("alps.p3b.exact_overlap",
                      alpsBuilder.getBoolAttr(alpsExactOverlap));

    if (alpsExactReadiness && !alpsMinimalStaticAdmission) {
      moduleOp.emitError("ALPS P3a exact readiness requires P2d admission");
      return;
    }
    if (enableAlpsConsumerLayoutPropagation &&
        !enableAlpsConsumerDrivenLayout) {
      moduleOp.emitError("ALPS P2f layout propagation requires P2e");
      return;
    }
    if (enableAlpsLoopInterchangedDirectFormation &&
        !enableAlpsConsumerDrivenLayout) {
      moduleOp.emitError("ALPS P2g-b direct formation requires P2e");
      return;
    }
    if (enableAlpsRegisterTileFormation && !enableAlpsConsumerDrivenLayout) {
      moduleOp.emitError("ALPS P2g-c register-tile formation requires P2e");
      return;
    }
    if (alpsContractLedger && !enableAlpsConsumerDrivenLayout) {
      moduleOp.emitError("ALPS P5a/P5b contract analysis requires P2e");
      return;
    }
    if (alpsExactOverlap && !alpsExactReadiness) {
      moduleOp.emitError("ALPS P3b exact overlap requires P3a readiness");
      return;
    }

    if (enablePrefetchKernelHX && enableAPTGetHX) {
      moduleOp.emitError(
          "Prefetch-Kernel-HX and APT-GET-HX are independent baselines and "
          "cannot be enabled in the same compilation");
      return signalPassFailure();
    }
    if ((enableAlpsCrpSupplyAnalysis || enableAlpsCrpSupplyPrefetch ||
         enableAlpsCrpSegmentedSupply || enableAlpsCrpVtcmFormation ||
         enableAlpsCrpVtcmWindow || enableAlpsCrpVtcmAsyncWindow ||
         enableAlpsCrpProducerDirectAnalysis ||
         enableAlpsCrpProducerDirectVtcm ||
         enableAlpsCrpProducerDirectHeadMajor ||
         enableAlpsCrpProducerLoopFormation) &&
        !enableAlpsRegisterTileFormation) {
      moduleOp->emitError("ALPS P5f requires P2g-c register-tile formation");
      return signalPassFailure();
    }
    if (enableAlpsCrpSegmentedSupply && !enableAlpsCrpSupplyPrefetch) {
      moduleOp->emitError(
          "ALPS P5f-c segmented supply requires P5f-b CRP prefetch");
      return signalPassFailure();
    }
    if (enableAlpsCrpVtcmWindow && enableAlpsCrpVtcmAsyncWindow) {
      moduleOp->emitError(
          "ALPS P5g-b synchronous and P5g-c asynchronous VTCM windows are "
          "independent matched schemes and cannot both be enabled");
      return signalPassFailure();
    }

    setTargetTriple(moduleOp);
    setDataLayout(moduleOp);

    auto setIndexBitwidth = [&](auto passOption) {
      passOption.indexBitwidth = 32;
      return passOption;
    };

    auto setFusion = [&](auto passOption) {
      passOption.fusionAllowRecompute = fusionAllowRecompute;
      passOption.fusionDoMultiUse = fusionDoMultiUse;
      return passOption;
    };

    auto setExtendPack = [&](auto passOption) {
      passOption.upperFrontier = extendPackUpperFrontier;
      passOption.lowerFrontier = extendPackLowerFrontier;
      passOption.parallelsOnly = extendPackParallelsOnly;
      return passOption;
    };

    auto setVTCMTiling = [&](auto passOption) {
      passOption.tileSizes = tileSizes;
      passOption.vtcmBudget = scratch > 0 ? scratch : 0;
      return passOption;
    };

    auto setuseInterchangeVector = [&](auto passOption) {
      passOption.useInterchangeVector = useInterchangeVector;
      return passOption;
    };

    auto setOpSlicingFactor = [&](auto passOption) {
      passOption.slicingFactor = slicingFactor;
      return passOption;
    };

    auto setsplitTilingRange = [&](auto passOption) {
      passOption.splitTilingRange = splitTilingRange;
      return passOption;
    };

    auto setenableSplitReduction = [&](auto passOption) {
      passOption.enableSplitReduction = enableSplitReduction;
      return passOption;
    };

    auto setAllowReturnAllocs = [&](auto passOption) {
      passOption.allowReturnAllocsFromLoops = true;
      return passOption;
    };

    auto setBufferizeFunctionBoundaries = [&](auto passOption) {
      passOption.bufferizeFunctionBoundaries = true;
      return passOption;
    };

    auto setLWP = [&](auto passOption) {
      passOption.disableLWPLoop = disableLWPLoop;
      passOption.LWPloopDepth = LWPloopDepth;
      passOption.instrumentHexKLPhases = instrumentLWPHexKLPhases;
      return passOption;
    };

    auto setAlpsLedger = [&](auto passOption, StringRef phase) {
      passOption.phase = phase.str();
      passOption.pageBytes = alpsLedgerPageBytes;
      passOption.vtcmBudgetBytes = alpsLedgerVtcmBudgetBytes;
      return passOption;
    };
    auto setAlpsDischargePhase = [&](auto passOption, StringRef phase) {
      passOption.phase = phase.str();
      passOption.analyzeInputs = enableAlpsRepresentationSupplyAnalysis &&
                                 phase == "post-bufferization";
      return passOption;
    };
    auto setOmniFetchVDAE = [&](auto passOption) {
      passOption.enableAdaptive = enableOmniFetchAdaptive;
      return passOption;
    };

    auto setDeviceType = [&](auto passOption) {
      passOption.device_type = device_type;
      return passOption;
    };

    // Set ConvTiling flags
    auto setConvTiling = [&](auto passOption) {
      passOption.convTileSizes = convTileSizes;
      return passOption;
    };

    auto setHexKLMode = [&](auto passOption) {
      passOption.mode = hexKLMode;
      passOption.consumerF16Epilogue = enableAlpsHmxF16EpilogueFormation;
      passOption.consumerF16BiasEpilogue =
          enableAlpsHmxF16BiasEpilogueFormation;
      return passOption;
    };

    PassManager pm(&getContext(), moduleOp.getOperationName());

    // RequestCWrappersPass adds an attribute to a function if it has a return
    // value which would generate a c-wrapper function during the
    // FuncToLLVMPass. See here for more information:
    // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
    //
    // For example given the following function definition:
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32>
    //
    // After the RequestCWrappersPass it is converted to,
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32> attributes {llvm.emit_c_interface}
    //
    // After FuncToLLVMPass we get an additional function,
    // llvm.func @_mlir_ciface_foobar(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
    // attributes {llvm.emit_c_interface} {
    //   %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //   %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x
    //   i64>, array<2 x i64>)>
    //   ...
    //   %8 = llvm.call @foobar(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr,
    //   !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //
    //   llvm.store %8, %arg0 :
    //   !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>,
    //   !llvm.ptr
    //
    //   llvm.return
    // }
    //
    if (doesFuncReturnValue(moduleOp))
      pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());

    pm.addNestedPass<func::FuncOp>(createLowerTTXPass());
    pm.addPass(createLowerLibdevicePass());
    pm.addNestedPass<func::FuncOp>(createLowerTPtrPass());
    HexagonLowerTmTensorOptions lowerTmOpts{};
    // Lower attention first, but infer item-7 K/V streams only after fusion.
    // Attaching semantic attributes here forces fusion either to preserve
    // every rewrite manually or to disable useful optimization globally.
    lowerTmOpts.emitKvCacheMetadata = false;
    pm.addNestedPass<func::FuncOp>(createHexagonLowerTmTensorPass(lowerTmOpts));

    // P2b must run at the first stable tensor-Linalg boundary.  The initial
    // model IR still exposes projection bias-add -> expand -> transpose as a
    // proven, single-use chain; subsequent rank reduction/canonicalization can
    // legally fold that producer shape and make the opportunity invisible.
    // Rewriting only the add's result layout does not alter the upstream named
    // matmul, so the later HexKL conversion retains the same eligibility.
    if (enableAlpsProducerDirectAttention) {
      pm.addNestedPass<func::FuncOp>(createAlpsProducerDirectAttentionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }

    // P5i recognizes the patch-convolution -> truncate -> token-layout chain
    // before named convolution generalization destroys the consumer contract.
    // The rank-2 resource fold makes the weight permutation compile-time only;
    // no full im2col or runtime weight transpose is introduced.
    if (enableAlpsPatchConvFormation) {
      pm.addNestedPass<func::FuncOp>(createAlpsPatchConvFormationPass());
      pm.addPass(mlir::hexagon::createFoldResourceTransposePass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }

    // P2e consumes explicit transpose results while their tensor-level
    // producer and terminal consumer contracts are both still visible.  It is
    // independent from the attention-only P2a/P2b patterns and remains
    // default-off for matched ablation.
    if (enableAlpsConsumerDrivenLayout) {
      AlpsConsumerDrivenLayoutOptions p2eOptions{};
      p2eOptions.propagateCodegenContract = enableAlpsConsumerLayoutPropagation;
      p2eOptions.emitDischargeContracts = alpsP2eContracts;
      p2eOptions.allowInnermostLoopInterchange =
          enableAlpsLoopInterchangedDirectFormation;
      p2eOptions.allowRegisterTileFormation = enableAlpsRegisterTileFormation;
      p2eOptions.registerTileDemandBegin = alpsRegisterTileDemandBegin;
      p2eOptions.registerTileDemandEnd = alpsRegisterTileDemandEnd;
      pm.addNestedPass<func::FuncOp>(
          createAlpsConsumerDrivenLayoutPass(p2eOptions));
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }

    pm.addNestedPass<func::FuncOp>(createReduceContractionRankPass());
    if (enableAlpsRegisterTileFormation)
      pm.addPass(std::make_unique<AlpsAwareFoldUnitExtentDimsPass>());
    else
      pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (puntBuffer)
      pm.addNestedPass<func::FuncOp>(createHexagonPuntBufferPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung allocs

    // Keep the legacy precision-conversion option working, while exposing the
    // same lowering as an independent ALPS experiment.  The pass itself keeps
    // matmul, explicit reductions and division out of the downcast policy;
    // eligible convolution and elementwise islands become genuine fp16 before
    // HVX tiling/vectorization rather than widening every lane to fp32.
    if (enableConversionToFp16 || enableAlpsFP16HVXArithmetic)
      pm.addNestedPass<func::FuncOp>(createConversionToFp16Pass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createOptimizeExtfTruncfOpPass());

    // Optimize division to multiplication in linalg.generic
    pm.addNestedPass<func::FuncOp>(createDivToMulOptimizationPass());

    // Quantization related passes in this block
    // Lower quant.qcast and quant.dcast ops to arith dialect
    pm.addNestedPass<func::FuncOp>(quant::createLowerQuantOps());
    // Convert arith ops to linalg elementwise ops
    pm.addPass(createConvertElementwiseToLinalgPass());
    // Remove quant.scast ops
    pm.addPass(createCSEPass());
    if (enableHexKL) {
      pm.addNestedPass<func::FuncOp>(
          mlir::hexagon::createFoldResourceTransposePass());
      pm.addNestedPass<func::FuncOp>(
          mlir::hexagon::createFoldCastsIntoMatmulPass());
      pm.addNestedPass<func::FuncOp>(
          createMatmulToHexKLPass(setHexKLMode(MatmulToHexKLOptions{})));
      if (hexKLMode == "macro") {
        pm.addNestedPass<func::FuncOp>(
            hexagon::createPreprocessWeightsForHMXPass());
      }
      pm.addPass(createCanonicalizerPass());
    }

    // P2a runs after HexKL has claimed every eligible named contraction. The
    // remaining matched QK batch matmuls are HVX-bound attention shapes, so
    // absorbing their head-layout transposes cannot reduce HMX coverage.
    if (enableAlpsZeroCopyAttention) {
      pm.addNestedPass<func::FuncOp>(createAlpsZeroCopyAttentionPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }

    // enableMatmulToConv and enableSeedLayoutConversions are supposed to be set
    // for unit test only. They are not supposed to run on Full models
    if (enableMatmulToConv && enableSeedLayoutConversions) {
      pm.addNestedPass<func::FuncOp>(createMatmulToConvPass());
      pm.addNestedPass<func::FuncOp>(createSeedLayoutConversionsPass());
      pm.addNestedPass<func::FuncOp>(createHexagonExtendPackPass(
          setExtendPack(HexagonExtendPackOptions{})));
      pm.addPass(createCSEPass());
    }

    if (enableConvTiling) {
      pm.addNestedPass<func::FuncOp>(
          createConvTilingPass(setConvTiling(ConvTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    if (enableSeedLayoutConversions) {
      pm.addNestedPass<func::FuncOp>(createPreprocessTiledConv2DPass());
    }

    pm.addNestedPass<func::FuncOp>(createConvertLayoutPass());
    // Recover semantic K/V identity while eager attention is still expressed
    // as named batch_matmul + explicit transpose. ScheduleMatmulForHVX copies
    // these attributes to its replacement generic op.
    if (alpsKvSemanticTracking) {
      HexagonLowerTmTensorOptions kvMetadataOpts{};
      kvMetadataOpts.emitKvCacheMetadata = true;
      kvMetadataOpts.emitKvFusionBoundary = alpsKvFusionPolicy;
      kvMetadataOpts.emitKvElementwiseFusionBoundary =
          enableAlpsKvElementwiseFusionPolicy;
      kvMetadataOpts.emitKvMultiUseFusionBoundary =
          enableAlpsKvMultiUseFusionPolicy;
      kvMetadataOpts.emitKvSplitReductionBoundary =
          enableAlpsKvSplitReductionPolicy;
      pm.addNestedPass<func::FuncOp>(
          createHexagonLowerTmTensorPass(kvMetadataOpts));
    }
    ScheduleMatmulForHVXOptions scheduleOpts{};
    // MatmulToHexKLPass (above) already converted every HexKL-eligible matmul
    // into a hexkl-dialect op, which ScheduleMatmulForHVXPass never matches.
    // The only linalg.matmul ops still present are the ones HexKL declined
    // (non-32-aligned / attention-shaped) and are therefore HVX-bound, so the
    // weight-stationary / activation-multicast reducers are safe to run even
    // when enableHexKL is set.  Gating them on !enableHexKL previously stripped
    // the best HVX reducers on zero-HMX-coverage models (e.g. DINOv2 with 0
    // HexKL rewrites), leaving hexkl-items17 no faster than plain hvx-vector.
    scheduleOpts.enableWeightStationary =
        enableOmniFetchWeightStationary && enableVectorization;
    scheduleOpts.enableActivationMulticast =
        enableOmniFetchActivationMulticast && enableVectorization;
    pm.addNestedPass<func::FuncOp>(
        createScheduleMatmulForHVXPass(scheduleOpts));
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizePass());
    // The generic Linalg generalizer rebuilds named contractions and does not
    // preserve arbitrary attributes. Re-identify K/V immediately, while the
    // unfused attention dataflow and indexing maps are still available.
    if (alpsKvSemanticTracking) {
      HexagonLowerTmTensorOptions kvMetadataOpts{};
      kvMetadataOpts.emitKvCacheMetadata = true;
      kvMetadataOpts.emitKvFusionBoundary = alpsKvFusionPolicy;
      kvMetadataOpts.emitKvElementwiseFusionBoundary =
          enableAlpsKvElementwiseFusionPolicy;
      kvMetadataOpts.emitKvMultiUseFusionBoundary =
          enableAlpsKvMultiUseFusionPolicy;
      kvMetadataOpts.emitKvSplitReductionBoundary =
          enableAlpsKvSplitReductionPolicy;
      pm.addNestedPass<func::FuncOp>(
          createHexagonLowerTmTensorPass(kvMetadataOpts));
    }

    if (returnValueOptimization)
      pm.addNestedPass<func::FuncOp>(createHexagonRVOPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung re-interprets
    pm.addPass(createCSEPass());

    if (enableSCFThreading) {
      assert(!enableMultiThreading && !enableVTCMTiling && scratch == 0 &&
             "currently scf-threading can be enabled only if"
             " linalg multi-threading and vtcm tiling are off");
      pm.addNestedPass<func::FuncOp>(createFormSCFThreadsPass());
    }

    if (enableAlpsMovementLedger)
      pm.addNestedPass<func::FuncOp>(createAlpsMovementLedgerPass(
          setAlpsLedger(AlpsMovementLedgerOptions{}, "pre-fusion")));
    if (alpsContractLedger)
      pm.addNestedPass<func::FuncOp>(
          createAlpsContractDischargeLedgerPass(setAlpsDischargePhase(
              AlpsContractDischargeLedgerOptions{}, "pre-fusion")));

    if (fusion)
      pm.addNestedPass<func::FuncOp>(
          createHexagonFusionPass(setFusion(HexagonFusionOptions{})));
    pm.addPass(createEraseUnusedLinalgOperands());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (alpsContractLedger)
      pm.addNestedPass<func::FuncOp>(
          createAlpsContractDischargeLedgerPass(setAlpsDischargePhase(
              AlpsContractDischargeLedgerOptions{}, "post-fusion")));

    // Recover K/V identity from the final, fused contraction shapes.  This
    // keeps the normal HVX fusion pipeline intact while providing stable
    // metadata to the later bufferized prefetch insertion pass.
    if (alpsKvSemanticTracking) {
      HexagonLowerTmTensorOptions kvMetadataOpts{};
      kvMetadataOpts.emitKvCacheMetadata = true;
      kvMetadataOpts.emitKvFusionBoundary = alpsKvFusionPolicy;
      kvMetadataOpts.emitKvElementwiseFusionBoundary =
          enableAlpsKvElementwiseFusionPolicy;
      kvMetadataOpts.emitKvMultiUseFusionBoundary =
          enableAlpsKvMultiUseFusionPolicy;
      kvMetadataOpts.emitKvSplitReductionBoundary =
          enableAlpsKvSplitReductionPolicy;
      pm.addNestedPass<func::FuncOp>(
          createHexagonLowerTmTensorPass(kvMetadataOpts));
    }

    if (enableAlpsMovementLedger)
      pm.addNestedPass<func::FuncOp>(createAlpsMovementLedgerPass(
          setAlpsLedger(AlpsMovementLedgerOptions{}, "post-fusion")));

    // Full-size attention slicing currently rebuilds the contraction without
    // copying semantic K/V attributes. Keep the marked boundary intact for
    // item-7; ordinary HVX/HexKL configurations retain the existing slicing.
    if (enableSlicing && !alpsKvSlicingPolicy)
      pm.addPass(createHexagonSlicingPass(
          setOpSlicingFactor(HexagonSlicingOptions{})));

    pm.addNestedPass<func::FuncOp>(createDecomposeTensorConcatPass());
    if (forceHVXCroutonization) {
      pm.addNestedPass<func::FuncOp>(createForceHVXCroutonPass());
      pm.addNestedPass<func::FuncOp>(
          createHexagonExtendPackPass(setExtendPack(HexagonExtendPackOptions{
              .upperFrontier = false,
          })));
    }

    pm.addNestedPass<func::FuncOp>(createLowerPackPass());
    pm.addPass(createCSEPass());

    // Slicing may rebuild the contraction and drop semantic attributes.
    // Recover them at the last tensor-Linalg boundary so Hexagon tiling can
    // propagate K/V identity into the vector transfer reads.
    if (alpsKvSemanticTracking) {
      HexagonLowerTmTensorOptions kvMetadataOpts{};
      kvMetadataOpts.emitKvCacheMetadata = true;
      kvMetadataOpts.emitKvFusionBoundary = alpsKvFusionPolicy;
      kvMetadataOpts.emitKvElementwiseFusionBoundary =
          enableAlpsKvElementwiseFusionPolicy;
      kvMetadataOpts.emitKvMultiUseFusionBoundary =
          enableAlpsKvMultiUseFusionPolicy;
      kvMetadataOpts.emitKvSplitReductionBoundary =
          enableAlpsKvSplitReductionPolicy;
      pm.addNestedPass<func::FuncOp>(
          createHexagonLowerTmTensorPass(kvMetadataOpts));
    }

    // Validate the dependencies of the independently selectable components.
    if (enableOmniFetchLayoutAware && !enablePrefetch &&
        !enableAlpsFusedTransformTransfer)
      moduleOp->emitWarning(
          "[OmniFetch] layout-aware mode requires prefetch; no-op");
    if (enableOmniFetchVDAE && !enablePrefetch)
      moduleOp->emitWarning(
          "[OmniFetch] V-DAE requires prefetch operations; no-op");

    // VTCMTilingPass must run when scratch > 0 to create the VTCM allocs
    // that MemoryOffsetsPass will replace with views into the scratch buffer.
    // When scratch > 0, pass it as vtcmBudget so tile sizes respect the
    // per-instance budget rather than the hardcoded 2 MB default.
    if (enableVTCMTiling || scratch > 0) {
      pm.addNestedPass<func::FuncOp>(
          createVTCMTilingPass(setVTCMTiling(VTCMTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    // split linalg.reduce into [parallel,reduce] followed by smaller [reduce].
    if (enableSplitReduceGeneric) {
      pm.addNestedPass<func::FuncOp>(createSplitReduceGenericPass());
    }

    if (enableMultiThreading) {
      pm.addNestedPass<func::FuncOp>(
          createFormVirtualThreadsPass(FormVirtualThreadsOptions{}));
    }

    pm.addPass(removeMLProgramPass());
    if (enableAlpsRegisterTileFormation)
      pm.addPass(std::make_unique<AlpsAwareFoldUnitExtentDimsPass>());
    else
      pm.addPass(createLinalgFoldUnitExtentDimsPass());
    if (enableVectorization) {
      pm.addPass(
          createHexagonTilingPass(setsplitTilingRange(setuseInterchangeVector(
              setenableSplitReduction(HexagonTilingOptions{})))));
    }

    if (enableAlpsRegisterTileFormation)
      pm.addPass(std::make_unique<AlpsAwareFoldUnitExtentDimsPass>());
    else
      pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (alpsContractLedger)
      pm.addNestedPass<func::FuncOp>(
          createAlpsContractDischargeLedgerPass(setAlpsDischargePhase(
              AlpsContractDischargeLedgerOptions{}, "post-tiling")));
    pm.addPass(
        createSmallExponentToMultiplyPass(SmallExponentToMultiplyOptions{}));
    // ===== STEP 1: HOIST SCALAR OPS =====
    // Run before vectorization to expose scalar invariants
    pm.addNestedPass<func::FuncOp>(createHoistScalarOpsPass());
    pm.addPass(createEraseUnusedLinalgOperands());
    pm.addPass(createCSEPass());

    // ===== STEP 1.5: LOOP INVARIANT CODE MOTION =====
    // Move hoisted scalars further up the loop nest
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Run LICM again after canonicalization to catch newly exposed
    // opportunities
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());

    // ===== STEP 2: VECTORIZATION =====
    // Vectorizer now sees cleaner IR with hoisted scalars
    if (enableVectorization) {
      pm.addPass(createHexagonVectorizationPass());
    }
    if (enableAlpsRegisterTileFormation)
      pm.addNestedPass<func::FuncOp>(
          std::make_unique<AlpsRegisterTileMarkerBridgePass>(false));
    pm.addPass(createRewriteUBPoisonToZeroPass());
    pm.addPass(createHexagonVectorLoweringPass());
    pm.addPass(createCanonicalizerPass());

    if (addFastMath) {
      pm.addPass(createHexagonAddFastMathPass());
      pm.addNestedPass<func::FuncOp>(createFoldMulFByZeroPass());
      pm.addPass(createCanonicalizerPass());
    }
    pm.addPass(memref::createResolveShapedTypeResultDimsPass());

    if (enableBufferization) {
      pm.addPass(bufferization::createEmptyTensorEliminationPass());

      // Erase unnecessary vector-to-tensor writeback in loops before
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createEraseVectorToTensorWritebackPass());

      mlir::bufferization::OneShotBufferizePassOptions passOpts;
      passOpts.bufferizeFunctionBoundaries = true;
      passOpts.allowReturnAllocsFromLoops = true;
      pm.addPass(bufferization::createOneShotBufferizePass(passOpts));
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      if (enableAlpsRegisterTileFormation)
        pm.addNestedPass<func::FuncOp>(
            std::make_unique<AlpsRegisterTileMarkerBridgePass>(true));
      if (alpsContractLedger)
        pm.addNestedPass<func::FuncOp>(
            createAlpsContractDischargeLedgerPass(setAlpsDischargePhase(
                AlpsContractDischargeLedgerOptions{}, "post-bufferization")));
      if (enableAlpsContinuityAudit)
        pm.addNestedPass<func::FuncOp>(createAlpsContinuityAuditPass());
      if (enableAlpsCrpSupplyAnalysis)
        pm.addNestedPass<func::FuncOp>(createAlpsCrpSupplyAnalysisPass());
      if (enableAlpsCrpVtcmFormation)
        pm.addNestedPass<func::FuncOp>(createAlpsCrpVtcmFormationPass());
      if (enableAlpsCrpVtcmWindow)
        pm.addNestedPass<func::FuncOp>(createAlpsCrpVtcmWindowPass());
      if (enableAlpsCrpProducerDirectAnalysis ||
          enableAlpsCrpProducerDirectVtcm ||
          enableAlpsCrpProducerDirectHeadMajor ||
          enableAlpsCrpProducerLoopFormation) {
        AlpsCrpProducerDirectAnalysisOptions p5gOptions{};
        p5gOptions.rewriteEpochVtcm = enableAlpsCrpProducerDirectVtcm &&
                                      !enableAlpsCrpProducerDirectHeadMajor &&
                                      !enableAlpsCrpProducerLoopFormation;
        p5gOptions.rewriteEpochHeadMajorVtcm =
            enableAlpsCrpProducerDirectHeadMajor ||
            enableAlpsCrpProducerLoopFormation;
        p5gOptions.rewriteProducerLoopOrder =
            enableAlpsCrpProducerLoopFormation;
        pm.addNestedPass<func::FuncOp>(
            createAlpsCrpProducerDirectAnalysisPass(p5gOptions));
      }
      if (enableAlpsCrpSupplyPrefetch) {
        AlpsCrpSupplyPrefetchOptions p5fOptions{};
        p5fOptions.pageSafeSegmented = enableAlpsCrpSegmentedSupply;
        pm.addNestedPass<func::FuncOp>(
            createAlpsCrpSupplyPrefetchPass(p5fOptions));
      }
      if (enableAlpsLayoutSupplyPrefetch)
        pm.addNestedPass<func::FuncOp>(createAlpsLayoutSupplyPrefetchPass());

      if (enableAlpsAttentionDestinationFormation)
        pm.addNestedPass<func::FuncOp>(
            createAlpsAttentionDestinationFormationPass());

      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS1Pass());
      }

      pm.addNestedPass<func::FuncOp>(
          bufferization::createBufferLoopHoistingPass());

      pm.addNestedPass<func::FuncOp>(createCopyCanonicalizationPass());
      pm.addPass(createCanonicalizerPass());

      bufferization::buildBufferDeallocationPipeline(
          pm, bufferization::BufferDeallocationPipelineOptions{});

      pm.addPass(createCSEPass());
      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS2Pass());
      }

      // SCF Loop Unrolling of innermost loop after vectorization.
      if (enableSCFLoopUnroll) {
        pm.addNestedPass<func::FuncOp>(createSCFLoopUnrollPass());
      }

      pm.addNestedPass<func::FuncOp>(createConvertZeroSizeMemrefPass());
      pm.addPass(createConvertBufferizationToMemRefPass());

      // In a single-basic-block monolithic function (e.g. a full transformer
      // unrolled by torch-mlir), ownership-based deallocation sinks every
      // dealloc to the block terminator, keeping all buffers live at once and
      // exhausting DSP VTLB mappings. Sink each dealloc to right after its
      // buffer's last user to bound the peak concurrent working set.
      // Allocation-lifetime shortening is an OmniFetch memory-planning
      // optimization, not part of either external prefetch baseline.  Keep
      // the native/Prefetch-Kernel-HX/APT-GET-HX rows on the upstream
      // ownership path so their output ABI and latency are not confounded.
      if (enablePrefetch)
        pm.addNestedPass<func::FuncOp>(
            bufferization::createOptimizeAllocationLivenessPass());

      // The Hexagon backend miscompiles the by-value memref (sret) return for
      // large monolithic single-block functions: the return epilogue writes
      // only the descriptor's `allocated` field, leaving aligned/offset/sizes/
      // strides = 0, so the host reads a NULL data pointer. Rewrite memref
      // results into trailing out-param arguments (function returns void) to
      // avoid that return path entirely. hoistStaticAllocs reuses the caller's
      // buffer so no copy is introduced for static output shapes.
      if (enableBufferResultsToOutParams) {
        bufferization::BufferResultsToOutParamsPassOptions outParamsOpts;
        // Full-model entry points are public.  The upstream pass preserves
        // public ABIs unless explicitly requested, while the generated host
        // wrapper already uses caller-owned trailing output descriptors.
        // Transform both sides together to avoid an ABI mismatch on device.
        outParamsOpts.modifyPublicFunctions = true;
        outParamsOpts.hoistStaticAllocs = true;
        pm.addPass(
            bufferization::createBufferResultsToOutParamsPass(outParamsOpts));
      }

      // ALPS C runs only after tensor ownership and function-result ABI have
      // stabilized.  It replaces eligible bufferized mixed F16/F32 NCHW
      // convolutions with an output-width vector schedule; later generic
      // vector/LLVM lowering then sees vector FPEXT rather than scalar helpers.
      if (enableAlpsHVXWideningConv)
        pm.addNestedPass<func::FuncOp>(createAlpsHVXWideningConvPass());

      if (enableAlpsMovementLedger)
        pm.addNestedPass<func::FuncOp>(createAlpsMovementLedgerPass(
            setAlpsLedger(AlpsMovementLedgerOptions{}, "post-bufferization")));
    }

    // P5g-c must be formed only after ownership-based deallocation and copy
    // canonicalization.  Carrying an asynchronous schedule through those
    // passes as ordinary store/copy markers loses the marker attributes and
    // the generic copy-to-DMA path then recreates an immediate start+wait.
    // The late pass emits real dma_start/dma_wait ops and owns its explicit
    // ping/pong VTCM lifetime, so the token can span HVX computation.
    if (enableAlpsCrpVtcmAsyncWindow)
      pm.addNestedPass<func::FuncOp>(createAlpsCrpVtcmWindowPass());

    if (enableConvertToHexagonmem)
      pm.addNestedPass<func::FuncOp>(createConvertToHexagonmemPass());

    // External VTCM scratch mode: inject scratch arg and replace VTCM allocs
    // with views into the per-instance scratch buffer (hexagon.scratch).
    if (scratch > 0) {
      InsertScratchArgOptions scratchOpts;
      scratchOpts.scratch = scratch;
      pm.addNestedPass<func::FuncOp>(createInsertScratchArgPass(scratchOpts));
      pm.addNestedPass<func::FuncOp>(createMemoryOffsetsPass());
    }

    if (enableHexKL) {
      if (hexKLMode == "macro") {
        // Lower to HexKL macro API
        pm.addNestedPass<func::FuncOp>(createLowerHexKLMatmulToMacroPass());
      }
    }

    // Decompose hexkl.matmul to micro ops first so PrefetchInsert can see
    // MicroHMX DDR→VTCM copies (tile_row/tile_col) in innermost loops.
    if (enableHexKL && hexKLMode != "macro") {
      auto decomposeOptions = DecomposeHexKLMatmulOptions{};
      decomposeOptions.enableWeightPrepack = enableOmniFetchWeightPrepack;
      decomposeOptions.enablePersistentVtcm = enableHexKLPersistentVtcm;
      decomposeOptions.enableVtcmLifetimeColoring = enableOmniFetchVtcmColoring;
      // P3b's descriptor-bound producer must stage the incoming RM tile in
      // VTCM before the scout performs the WH transform.  Enabling this at
      // decomposition time also reserves the extra, non-aliasing VTCM bank;
      // merely changing PrefetchInsert would produce an out-of-budget view.
      decomposeOptions.enableDmaToVtcm =
          alpsExactOverlap || enableOmniFetchDmaToVtcm;
      decomposeOptions.enableDirectOutputFormation =
          enableAlpsHmxDirectOutputFormation;
      decomposeOptions.enableF16BiasEpilogueFormation =
          enableAlpsHmxF16BiasEpilogueFormation;
      decomposeOptions.enableAsyncDrainAnalysis =
          enableAlpsHmxAsyncDrainAnalysis;
      decomposeOptions.enableAsyncDrain = enableAlpsHmxAsyncDrain;
      pm.addNestedPass<func::FuncOp>(
          createDecomposeHexKLMatmulPass(decomposeOptions));
    }

    // External baseline: infer safe future affine tile addresses and emit only
    // destination-free L2 hints. Keep this outside the OmniFetch gates so a
    // baseline run cannot inherit shadow buffers, DMA/VTCM staging, in-situ
    // layout elimination, V-DAE, or adaptive OmniFetch policy.
    if (enablePrefetchKernelHX || enableAPTGetHX) {
      auto kernelOptions = PrefetchKernelHXOptions{};
      kernelOptions.distance =
          enableAPTGetHX ? aptGetHxDistance : prefetchKernelHxDistance;
      kernelOptions.maxCommandBytes = prefetchKernelHxMaxCommandBytes;
      kernelOptions.baselineKind =
          enableAPTGetHX ? "apt-get-hx" : "prefetch-kernel-hx";
      kernelOptions.requireManualSafe = enableAPTGetHX;
      kernelOptions.manualCandidateIds =
          enableAPTGetHX ? aptGetHxManualCandidateIds.getValue()
                         : std::string();
      pm.addNestedPass<func::FuncOp>(
          hexagon::createPrefetchKernelHXPass(kernelOptions));
    }

    // ===== Omni-Fetch: Plan-A 3-component architecture =====
    //
    //  Component 1 – Prefetch Insertion   (gate: enablePrefetch)
    //  Component 2 – In-Situ Reshape      (gate: enablePrefetch &&
    //  enableOmniFetchLayoutAware)
    //                Layout Ops Elim      (gate: enablePrefetch &&
    //                enableOmniFetchLayoutAware)
    //  Component 3 – V-DAE Decouple       (gate: enableOmniFetchVDAE)
    //
    // OmniFetchToLLVM lowers all omni_fetch dialect ops.  P5f-b emits its
    // already-admitted CRP hints before this section and therefore requires
    // lowering, but must not activate the broad legacy PrefetchInsert pass.

    // --- Component 1: Prefetch Insertion ---
    if (alpsPrefetchPipeline) {
      // Tiling, vectorization, and one-shot bufferization may replace the
      // contraction operation after the earlier post-fusion annotation.
      // Re-infer item-7 metadata on the final bufferized Linalg operations so
      // the prefetch pass never depends on attributes surviving rewrites.
      if (alpsKvRuntimePrefetch || alpsMinimalStaticAdmission) {
        HexagonLowerTmTensorOptions kvMetadataOpts{};
        kvMetadataOpts.emitKvCacheMetadata = true;
        kvMetadataOpts.emitKvFusionBoundary = alpsKvFusionPolicy;
        kvMetadataOpts.emitKvElementwiseFusionBoundary =
            enableAlpsKvElementwiseFusionPolicy;
        kvMetadataOpts.emitKvMultiUseFusionBoundary =
            enableAlpsKvMultiUseFusionPolicy;
        kvMetadataOpts.emitKvSplitReductionBoundary =
            enableAlpsKvSplitReductionPolicy;
        pm.addNestedPass<func::FuncOp>(
            createHexagonLowerTmTensorPass(kvMetadataOpts));
      }
      if (alpsMinimalStaticAdmission) {
        AlpsMinimalStaticAdmissionOptions admissionOptions{};
        admissionOptions.pageBytes = alpsLedgerPageBytes;
        admissionOptions.vtcmBudgetBytes = alpsLedgerVtcmBudgetBytes;
        if (alpsExactOverlap) {
          admissionOptions.minDmaBytes = 2048;
          admissionOptions.enableP3ExactReadiness = true;
        }
        pm.addNestedPass<func::FuncOp>(
            createAlpsMinimalStaticAdmissionPass(admissionOptions));
      }
      auto prefetchOptions = PrefetchInsertOptions{};
      // P2c establishes the transform-transfer mechanism only. Real
      // lookahead/readiness belongs to P3, so its first implementation is
      // deliberately synchronous and cannot accidentally enter the legacy
      // async/persistent pipelines.
      prefetchOptions.lookahead =
          enableAlpsFusedTransformTransfer ? 0 : omniFetchLookahead;
      // Layout-aware flag controls in-situ reshape during prefetch.
      prefetchOptions.enableLayoutAware = alpsLayoutAware;
      prefetchOptions.enableDmaToVtcm =
          alpsExactOverlap || enableOmniFetchDmaToVtcm;
      prefetchOptions.enableInterLayerPrefetch =
          enableAlpsFusedTransformTransfer ? false
                                           : enableOmniFetchInterLayerPrefetch;
      prefetchOptions.enablePersistentWhCache =
          enableAlpsFusedTransformTransfer ? false
                                           : enableOmniFetchPersistentWhCache;
      prefetchOptions.enableTwoDimPipeline =
          alpsExactOverlap ? true
                           : (enableAlpsFusedTransformTransfer
                                  ? false
                                  : enableOmniFetchTwoDimPipeline);
      prefetchOptions.enableKvCachePrefetch =
          alpsKvRuntimePrefetch || alpsMinimalStaticAdmission;
      // An independently enabled item-7 has no other OmniFetch components.
      // Keep that experiment causal by excluding ordinary loop prefetch;
      // cumulative items 1-7 still execute the complete pipeline.
      prefetchOptions.kvCacheOnly =
          (alpsKvRuntimePrefetch || alpsMinimalStaticAdmission) &&
          !alpsExactOverlap && !enableOmniFetchVDAE &&
          !enableOmniFetchPersistentWhCache && !enableOmniFetchTwoDimPipeline &&
          !enableOmniFetchVtcmColoring;
      prefetchOptions.enableDequantReshape =
          enableAlpsFusedTransformTransfer ? false
                                           : enableOmniFetchDequantReshape;
      prefetchOptions.enableAlpsFusedTransformTransfer =
          enableAlpsFusedTransformTransfer;
      prefetchOptions.requireAlpsAdmission = alpsMinimalStaticAdmission;
      prefetchOptions.enableAlpsExactOverlap = alpsExactOverlap;
      prefetchOptions.kvCachePageTokens = omniFetchKvCachePageTokens;
      pm.addNestedPass<func::FuncOp>(
          hexagon::createPrefetchInsertPass(prefetchOptions));
      pm.addPass(createCanonicalizerPass());
      // PrefetchInsert may emit AS1 memref.alloc for large VTCM tiles after
      // the earlier ConvertToHexagonmem pass; convert those now.
      if (enableConvertToHexagonmem)
        pm.addNestedPass<func::FuncOp>(createConvertToHexagonmemPass());
      if (alpsExactReadiness)
        pm.addNestedPass<func::FuncOp>(createAlpsExactReadinessPass());
    }

    // --- Component 2: Layout Ops Elimination (In-Situ Reshape partner) ---
    // Only meaningful when both Prefetch and Layout-Aware are active.
    if (alpsPrefetchPipeline && alpsLayoutAware) {
      pm.addNestedPass<func::FuncOp>(hexagon::createLayoutOpsEliminationPass());
    }

    // --- Component 3: V-DAE (Virtual Decoupled Access-Execute) ---
    // Adds wait/signal semaphore synchronization around prefetch operations.
    // Requires Component 1 to have already inserted prefetch ops.
    if (enableOmniFetchVDAE) {
      pm.addNestedPass<func::FuncOp>(hexagon::createOmniFetchVDAEInsertPass(
          setOmniFetchVDAE(OmniFetchVDAEInsertOptions{})));
      if (mlir::hexagon::isEnvTrue("DUMP_AFTER_VDAE"))
        pm.addPass(createCanonicalizerPass());
    }

    // Lower linalg ops with library_call attribute set to custom fns.
    pm.addPass(createHexagonReplaceWithLibraryCallsPass());
    if (enableHexagonmemCopyToDMA)
      pm.addNestedPass<func::FuncOp>(createHexmemCpyToDMAPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertLinalgToLoopsPass());

    pm.addNestedPass<func::FuncOp>(createFormAsyncThreadsPass());
    pm.addPass(createAsyncFuncToAsyncRuntimePass());
    pm.addPass(createAsyncToAsyncRuntimePass());

    pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

    if (enableLWP)
      pm.addNestedPass<func::FuncOp>(
          createHexagonLWPPass(setLWP(mlir::hexagon::HexagonLWPPassOptions{})));

    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addNestedPass<func::FuncOp>(createExpandMathOpsPass());

    if (expandBoolVec)
      pm.addNestedPass<func::FuncOp>(createExpandBoolVecPass());

    pm.addPass(createFastInversePass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass(
        setIndexBitwidth(ConvertIndexToLLVMPassOptions{})));

    pm.addPass(createConvertAsyncToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass(ConvertFuncToLLVMPassOptions{}));

    pm.addPass(hexagon::createDMAToLLVMPass());
    pm.addPass(hexagonmem::createHexagonMemToLLVMPass(
        setDeviceType(hexagonmem::HexagonMemToLLVMOptions{})));
    pm.addPass(hexkl::createHexKLToLLVMPass());
    // Lower omni_fetch dialect ops to extern-C runtime calls.
    // Required whenever Prefetch (Component 1) has emitted prefetch_in_situ
    // ops, or when weight prepack emitted prefetch_in_situ copies in
    // DecomposeHexKL.
    if (alpsPrefetchPipeline || enableAlpsCrpSupplyPrefetch ||
        enableAlpsLayoutSupplyPrefetch || enableOmniFetchWeightPrepack ||
        enablePrefetchKernelHX || enableAPTGetHX) {
      mlir::omni_fetch::OmniFetchToLLVMOptions ofOpts{};
      ofOpts.enableDualThreadDae = enableOmniFetchDualThreadDae &&
                                   (enableOmniFetchVDAE || alpsExactOverlap);
      pm.addPass(omni_fetch::createOmniFetchToLLVMPass(ofOpts));
    }

    if (enableCollapseAddressSpace) {
      pm.addPass(createCollapseAddressSpacePass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    }

    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    // Some late conversion patterns legitimately materialize ub.poison after
    // the zero-substitution cleanup.  Convert any residual UB ops to their
    // LLVM dialect equivalents so translateModuleToLLVMIR never sees an
    // untranslated UB dialect operation.
    pm.addPass(createUBToLLVMConversionPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (enableHexagonRoutines)
      pm.addPass(createHexagonLLVMEnableHexagonRoutinesPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLinalgToLLVMPass(const LinalgToLLVMOptions &options) {
  return std::make_unique<LinalgToLLVMPass>(options);
}
