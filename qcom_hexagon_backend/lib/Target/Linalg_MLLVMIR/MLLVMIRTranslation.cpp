//===-- MLLVMIRTranslation.cpp - Linalg to LLVM IR Translation ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg to LLVM IR translation registration for
// Hexagon target.
//===----------------------------------------------------------------------===//

#include "hexagon/Target/Linalg_MLLVMIR/MLLVMIRTranslation.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/InitAllPasses.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include <dlfcn.h>
#include <filesystem>
#include <iterator>
#include <string>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hexagon/Conversion/LinalgToLLVM/LowerConstantsSeparately.h"

void setLinalgToLLVMOptions(
    mlir::hexagon::LinalgToLLVMOptions &options,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {

  // DEBUG (disabled): print all options received from Python
  // llvm::errs() << "\n[setLinalgToLLVMOptions] Received options from
  // Python:\n"; for (const auto &kv : arch_kwargs) {
  //   llvm::errs() << "  " << kv.first << " = " << kv.second << "\n";
  // }
  // llvm::errs() << "\n";

  // Note: seems very counter-intuitive due to the fact that compare() returns 0
  // when the strings are actually equal, which is why we negate it to convert
  // it to a boolean.
  const std::string TRUE("True");
  options.fusion = !arch_kwargs.at("fusion").compare(TRUE);
  options.fusionAllowRecompute =
      !arch_kwargs.at("fusionAllowRecompute").compare(TRUE);
  options.fusionDoMultiUse = !arch_kwargs.at("fusionDoMultiUse").compare(TRUE);
  options.enableDoubleBuffering =
      !arch_kwargs.at("enableDoubleBuffering").compare(TRUE);
  options.enableSCFThreading =
      !arch_kwargs.at("enableSCFThreading").compare(TRUE);
  options.enableMultiThreading =
      !arch_kwargs.at("enableMultiThreading").compare(TRUE);
  options.enableVTCMTiling = !arch_kwargs.at("enableVTCMTiling").compare(TRUE);
  options.scratch = std::stoll(arch_kwargs.at("scratch"));
  options.enableConvertToHexagonmem =
      !arch_kwargs.at("enableConvertToHexagonmem").compare(TRUE);
  options.enableHexagonmemCopyToDMA =
      !arch_kwargs.at("enableHexagonmemCopyToDMA").compare(TRUE);
  options.enableHexKL = !arch_kwargs.at("enableHexKL").compare(TRUE);
  options.hexKLMode = arch_kwargs.at("hexKLMode");
  options.enableCollapseAddressSpace =
      !arch_kwargs.at("enableCollapseAddressSpace").compare(TRUE);
  options.tileSizes = arch_kwargs.at("tileSizes");
  options.lowerConstantsInSeparateSharedObjects =
      !arch_kwargs.at("lowerConstantsInSeparateSharedObjects").compare(TRUE);
  options.enableBufferization =
      !arch_kwargs.at("enableBufferization").compare(TRUE);
  options.enableSeedLayoutConversions =
      !arch_kwargs.at("enableSeedLayoutConversions").compare(TRUE);
  options.enableSplitReduction =
      !arch_kwargs.at("enableSplitReduction").compare(TRUE);
  options.enableConvTiling = !arch_kwargs.at("enableConvTiling").compare(TRUE);
  options.convTileSizes = arch_kwargs.at("convTileSizes");
  options.enableLWP = !arch_kwargs.at("enableLWP").compare(TRUE);
  options.disableLWPLoop = !arch_kwargs.at("disableLWPLoop").compare(TRUE);
  options.instrumentLWPHexKLPhases =
      !arch_kwargs.at("instrumentLWPHexKLPhases").compare(TRUE);
  options.enableVectorization =
      !arch_kwargs.at("enableVectorization").compare(TRUE);
  options.enableSplitReduceGeneric =
      !arch_kwargs.at("enableSplitReduceGeneric").compare(TRUE);
  auto it = arch_kwargs.find("device_type");
  if (it != arch_kwargs.end()) {
    options.device_type = it->second;
  } else {
    options.device_type = "hexagon"; // default value
  }
  options.enableHVXInlining =
      !arch_kwargs.at("enableHVXInlining").compare(TRUE);
  options.enableSCFLoopUnroll =
      !arch_kwargs.at("enableSCFLoopUnroll").compare(TRUE);
  options.enableConversionToFp16 =
      !arch_kwargs.at("enableConversionToFp16").compare(TRUE);
  {
    auto it = arch_kwargs.find("enableAlpsFP16HVXArithmetic");
    options.enableAlpsFP16HVXArithmetic =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsHVXWideningConv");
    options.enableAlpsHVXWideningConv =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableBufferResultsToOutParams");
    options.enableBufferResultsToOutParams =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }

  // Alps options (Plan-A: three independent components)
  {
    auto it = arch_kwargs.find("enablePrefetchKernelHX");
    options.enablePrefetchKernelHX =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("prefetchKernelHxDistance");
    if (it != arch_kwargs.end())
      options.prefetchKernelHxDistance = std::stoi(it->second);
  }
  {
    auto it = arch_kwargs.find("prefetchKernelHxMaxCommandBytes");
    if (it != arch_kwargs.end())
      options.prefetchKernelHxMaxCommandBytes = std::stoi(it->second);
  }
  {
    auto it = arch_kwargs.find("enableAPTGetHX");
    options.enableAPTGetHX =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("aptGetHxDistance");
    if (it != arch_kwargs.end())
      options.aptGetHxDistance = std::stoi(it->second);
  }
  {
    auto it = arch_kwargs.find("aptGetHxManualCandidateIds");
    if (it != arch_kwargs.end())
      options.aptGetHxManualCandidateIds = it->second;
  }
  options.enablePrefetch = !arch_kwargs.at("enablePrefetch").compare(TRUE);
  options.enableAlpsLayoutAware =
      !arch_kwargs.at("enableAlpsLayoutAware").compare(TRUE);
  options.alpsLookahead = std::stoi(arch_kwargs.at("alpsLookahead"));
  {
    auto it = arch_kwargs.find("enableAlpsDmaToVtcm");
    options.enableAlpsDmaToVtcm =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  options.enableAlpsVDAE =
      !arch_kwargs.at("enableAlpsVDAE").compare(TRUE);
  options.enableAlpsAdaptive =
      !arch_kwargs.at("enableAlpsAdaptive").compare(TRUE);
  {
    auto it = arch_kwargs.find("enableAlpsWeightPrepack");
    options.enableAlpsWeightPrepack =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsPersistentWhCache");
    options.enableAlpsPersistentWhCache =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsTwoDimPipeline");
    options.enableAlpsTwoDimPipeline =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsVtcmColoring");
    options.enableAlpsVtcmColoring =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvSemanticTracking");
    options.enableAlpsKvSemanticTracking =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvFusionPolicy");
    options.enableAlpsKvFusionPolicy =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvElementwiseFusionPolicy");
    options.enableAlpsKvElementwiseFusionPolicy =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvMultiUseFusionPolicy");
    options.enableAlpsKvMultiUseFusionPolicy =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvSplitReductionPolicy");
    options.enableAlpsKvSplitReductionPolicy =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvSlicingPolicy");
    options.enableAlpsKvSlicingPolicy =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvRuntimePrefetch");
    options.enableAlpsKvRuntimePrefetch =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsMovementLedger");
    options.enableAlpsMovementLedger =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsZeroCopyAttention");
    options.enableAlpsZeroCopyAttention =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsProducerDirectAttention");
    options.enableAlpsProducerDirectAttention =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsConsumerDrivenLayout");
    options.enableAlpsConsumerDrivenLayout =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsConsumerLayoutPropagation");
    options.enableAlpsConsumerLayoutPropagation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsContinuityAudit");
    options.enableAlpsContinuityAudit =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsLoopInterchangedDirectFormation");
    options.enableAlpsLoopInterchangedDirectFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsRegisterTileFormation");
    options.enableAlpsRegisterTileFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("alpsRegisterTileDemandBegin");
    if (it != arch_kwargs.end())
      options.alpsRegisterTileDemandBegin = std::stoll(it->second);
  }
  {
    auto it = arch_kwargs.find("alpsRegisterTileDemandEnd");
    if (it != arch_kwargs.end())
      options.alpsRegisterTileDemandEnd = std::stoll(it->second);
  }
  {
    auto it = arch_kwargs.find("enableAlpsContractDischargeLedger");
    options.enableAlpsContractDischargeLedger =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsRepresentationSupplyAnalysis");
    options.enableAlpsRepresentationSupplyAnalysis =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsLayoutSupplyPrefetch");
    options.enableAlpsLayoutSupplyPrefetch =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpSupplyAnalysis");
    options.enableAlpsCrpSupplyAnalysis =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpSupplyPrefetch");
    options.enableAlpsCrpSupplyPrefetch =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpSegmentedSupply");
    options.enableAlpsCrpSegmentedSupply =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpVtcmFormation");
    options.enableAlpsCrpVtcmFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpVtcmWindow");
    options.enableAlpsCrpVtcmWindow =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpVtcmAsyncWindow");
    options.enableAlpsCrpVtcmAsyncWindow =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpProducerDirectAnalysis");
    options.enableAlpsCrpProducerDirectAnalysis =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpProducerDirectVtcm");
    options.enableAlpsCrpProducerDirectVtcm =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpProducerDirectHeadMajor");
    options.enableAlpsCrpProducerDirectHeadMajor =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsCrpProducerLoopFormation");
    options.enableAlpsCrpProducerLoopFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsAttentionDestinationFormation");
    options.enableAlpsAttentionDestinationFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsPatchConvFormation");
    options.enableAlpsPatchConvFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsHmxF16EpilogueFormation");
    options.enableAlpsHmxF16EpilogueFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsHmxDirectOutputFormation");
    options.enableAlpsHmxDirectOutputFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsHmxF16BiasEpilogueFormation");
    options.enableAlpsHmxF16BiasEpilogueFormation =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsHmxAsyncDrainAnalysis");
    options.enableAlpsHmxAsyncDrainAnalysis =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsHmxAsyncDrain");
    options.enableAlpsHmxAsyncDrain =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsFusedTransformTransfer");
    options.enableAlpsFusedTransformTransfer =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsMinimalStaticAdmission");
    options.enableAlpsMinimalStaticAdmission =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsExactReadiness");
    options.enableAlpsExactReadiness =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsExactOverlap");
    options.enableAlpsExactOverlap =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("alpsLedgerPageBytes");
    if (it != arch_kwargs.end())
      options.alpsLedgerPageBytes = std::stoll(it->second);
  }
  {
    auto it = arch_kwargs.find("alpsLedgerVtcmBudgetBytes");
    if (it != arch_kwargs.end())
      options.alpsLedgerVtcmBudgetBytes = std::stoll(it->second);
  }
  {
    auto it = arch_kwargs.find("enableAlpsKvCachePrefetch");
    options.enableAlpsKvCachePrefetch =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsWeightStationary");
    options.enableAlpsWeightStationary =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsActivationMulticast");
    options.enableAlpsActivationMulticast =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsDequantReshape");
    options.enableAlpsDequantReshape =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("alpsKvCachePageTokens");
    if (it != arch_kwargs.end())
      options.alpsKvCachePageTokens = std::stoi(it->second);
  }
  {
    auto it = arch_kwargs.find("enableAlpsDualThreadDae");
    options.enableAlpsDualThreadDae =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsInterLayerPrefetch");
    options.enableAlpsInterLayerPrefetch =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsAttentionHmx");
    options.enableAlpsAttentionHmx =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableAlpsMPadHmx");
    options.enableAlpsMPadHmx =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
  {
    auto it = arch_kwargs.find("enableHexKLPersistentVtcm");
    // Default false when key absent (matches Passes.td / HexagonOptions).
    options.enableHexKLPersistentVtcm =
        (it != arch_kwargs.end() && !it->second.compare(TRUE));
  }
}

namespace mlir {
namespace hexagon {

static void configureMLIRPrinting(mlir::PassManager &pm) {
  auto printingFlags = mlir::OpPrintingFlags();
  if (mlir::hexagon::isEnvTrue("MLIR_ELIDE_LARGE_CONST_PRINT")) {
    printingFlags.elideLargeElementsAttrs(1);
    printingFlags.elideLargeResourceString(1);
  } else {
    printingFlags.elideLargeElementsAttrs(16);
  }
  SmallVector<StringRef> selectiveDumpPasses;
  if (const char *value = std::getenv("MLIR_DUMP_ONLY_PASSES"))
    StringRef(value).split(selectiveDumpPasses, ',', /*MaxSplit=*/-1,
                           /*KeepEmpty=*/false);
  // Print the IR after HexagonLWPPass if enabled for debug purpose
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [selectiveDumpPasses](mlir::Pass *pass, mlir::Operation *) {
        StringRef passName = pass->getName();
        StringRef passArgument = pass->getArgument();
        bool selected = llvm::any_of(selectiveDumpPasses, [&](StringRef token) {
          token = token.trim();
          return passName.contains_insensitive(token) ||
                 passArgument.contains_insensitive(token);
        });
        return mlir::hexagon::isEnvTrue("MLIR_ENABLE_DUMP") || selected ||
               passName.contains("HexagonLWP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);
}

mlir::ModuleOp translateLinalgToLLVMMLIR(
    mlir::ModuleOp mod,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {
  mlir::PassManager pm(mod->getContext());
  mlir::registerPassManagerCLOptions();
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "failed to apply pass manager CL options\n";
    return nullptr;
  }
  configureMLIRPrinting(pm);

  // set your enable/disable individual pass options here
  // or funnel to here.
  LinalgToLLVMOptions options;
  setLinalgToLLVMOptions(options, arch_kwargs);
  pm.addPass(createLinalgToLLVMPass(options));

  if (failed(pm.run(mod))) {
    llvm::errs() << "Linalg to Hexagon Pass execution failed";
    return nullptr;
  }
  return mod;
}

// -------------------------------------------------
// --- Translating to multiple LLVM/MLIR modules ---
// -------------------------------------------------

class CustomPassManager : public mlir::PassManager {
public:
  CustomPassManager(mlir::MLIRContext *context, LinalgToLLVMOptions options)
      : mlir::PassManager(context) {
    addPass(createLinalgToLLVMPass(options));

    // Careful: here we are giving ownership on this pass, so we can't access it
    // ourselves anymore directly
    addPass(std::make_unique<LowerConstantsSeparatelyPass>());
  }

  std::vector<ModuleOp> getProducedModules() const {
    LowerConstantsSeparatelyPass *ptr_lowerConstantsPass;
    auto passes = getPasses();

    // Trick to get the LowerConstantsSeparatelyPass since we had given
    // ownership to it. It's just some plain bureaucracy: finding the pass
    // LowerConstantsSeparatelyPass amongst all the passes that were added to
    // the pass manager
    for (auto &pass : passes) {
      if ((ptr_lowerConstantsPass =
               dynamic_cast<LowerConstantsSeparatelyPass *>(&pass))) {
        // We found the LowerConstantsSeparatelyPass pass in the pass manager
        break;
      }
    }
    if (!ptr_lowerConstantsPass) {
      std::cerr << "Error: pass_lower_constants_separately is null!"
                << std::endl;
      return {};
    }

    // Returning the list of modules the pass LowerConstantsSeparatelyPass has
    // produced
    return ptr_lowerConstantsPass->getProducedModules();
  }
};

std::vector<ModuleOp> translateLinalgToMultipleLLVMMLIRModules(
    ModuleOp mod,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {

  LinalgToLLVMOptions options;
  setLinalgToLLVMOptions(options, arch_kwargs);

  CustomPassManager pm(mod->getContext(), options);
  configureMLIRPrinting(pm);
  if (failed(pm.run(mod))) {
    llvm::errs() << "Custom pass manager for producing multiple modules failed";
    return std::vector<ModuleOp>();
  }

  std::vector<ModuleOp> all_modules = pm.getProducedModules();
  // Insert the main module (mutated) at the front of the vector that will be
  // returned
  all_modules.insert(all_modules.begin(), mod);
  return all_modules;
}

} // namespace hexagon
} // namespace mlir
