//===- Transforms.h - Linalg lowering and optimization passes -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_TRANSFORMS_H
#define HEXAGON_TRANSFORMS_TRANSFORMS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {
#define GEN_PASS_DECL
#include "hexagon/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createCollapseAddressSpacePass();

std::unique_ptr<OperationPass<func::FuncOp>> createCopyCanonicalizationPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createConvTilingPass(const ConvTilingOptions &options = ConvTilingOptions());
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLayoutPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertToHexagonmemPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertZeroSizeMemrefPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeTensorConcatPass();

std::unique_ptr<OperationPass<ModuleOp>> createEraseUnusedLinalgOperands();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createExpandBoolVecPass();

std::unique_ptr<OperationPass<func::FuncOp>> createExpandMathOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createFastInversePass();

std::unique_ptr<OperationPass<ModuleOp>> createHexagonAddFastMathPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonDoubleBufferGenericS1Pass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonDoubleBufferGenericS2Pass();

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonLLVMEnableHexagonRoutinesPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonLowerTmTensorPass(
    const HexagonLowerTmTensorOptions &options =
        HexagonLowerTmTensorOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createHexagonLWPPass(
    const HexagonLWPPassOptions &options = HexagonLWPPassOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHexagonPuntBufferPass();

std::unique_ptr<OperationPass<ModuleOp>>
createHexagonReplaceWithLibraryCallsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHexagonRVOPass();

std::unique_ptr<OperationPass<ModuleOp>> createHexagonVectorLoweringPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgGeneralizePass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerLibdevicePass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createLowerTPtrPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createLowerTTXPass();

std::unique_ptr<OperationPass<func::FuncOp>> createMatmulToConvPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMatmulToHexKLPass(
    const MatmulToHexKLOptions &options = MatmulToHexKLOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createDecomposeHexKLMatmulPass(
    const DecomposeHexKLMatmulOptions &options = DecomposeHexKLMatmulOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createInsertScratchArgPass(
    const InsertScratchArgOptions &options = InsertScratchArgOptions());

std::unique_ptr<OperationPass<func::FuncOp>> createMemoryOffsetsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createScheduleMatmulForHVXPass(
    const ScheduleMatmulForHVXOptions &options =
        ScheduleMatmulForHVXOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createSeedLayoutConversionsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createForceHVXCroutonPass();

std::unique_ptr<OperationPass<ModuleOp>> createSmallExponentToMultiplyPass(
    const SmallExponentToMultiplyOptions &options =
        SmallExponentToMultiplyOptions());
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createPreprocessTiledConv2DPass();

std::unique_ptr<OperationPass<ModuleOp>> removeMLProgramPass();

std::unique_ptr<Pass> createReduceContractionRankPass();

std::unique_ptr<Pass> createFoldCastsIntoMatmulPass();

std::unique_ptr<Pass> createHoistScalarOpsPass();

std::unique_ptr<Pass> createFoldMulFByZeroPass();

std::unique_ptr<Pass> createFoldResourceTransposePass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLowerHexKLMatmulToMacroPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPreprocessWeightsForHMXPass();

std::unique_ptr<Pass> createFoldPackUnpackConstantsPass();

std::unique_ptr<Pass> createEliminateRedundantUnpackPackPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDivToMulOptimizationPass();

std::unique_ptr<OperationPass<func::FuncOp>> createSCFLoopUnrollPass(
    const SCFLoopUnrollOptions &options = SCFLoopUnrollOptions());

/// Prefetch: inserts prefetch operations to preload data from DDR to VTCM.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createPrefetchInsertPass(
    const PrefetchInsertOptions &options = PrefetchInsertOptions());

/// Prefetch-Kernel-HX external baseline: reconstructs safe affine future tile
/// addresses and emits fixed-distance L2 hints only.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createPrefetchKernelHXPass(
    const PrefetchKernelHXOptions &options = PrefetchKernelHXOptions());

/// ALPS P1: analysis-only future-representation and physical-movement ledger.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsMovementLedgerPass(
    const AlpsMovementLedgerOptions &options = AlpsMovementLedgerOptions());

/// ALPS P2d: choose one conservative representation-supply action per
/// candidate and record every admission/rejection for later P3 consumption.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsMinimalStaticAdmissionPass(
    const AlpsMinimalStaticAdmissionOptions &options =
        AlpsMinimalStaticAdmissionOptions());

/// ALPS P3a: require exact invocation/version/tile/layout/tier identity before
/// an admitted async action may be consumed by P3b.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsExactReadinessPass();

/// ALPS P2a: absorb provably equivalent attention layout chains into consumer
/// indexing maps, eliminating physical transpose materializations.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsZeroCopyAttentionPass();

/// ALPS P2b: make eligible attention elementwise producers write their final
/// contiguous head-major consumer representation directly.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsProducerDirectAttentionPass();

/// ALPS P2e: infer terminal consumer representation contracts and make a
/// strictly eligible producer form the required contiguous layout directly.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsConsumerDrivenLayoutPass(
    const AlpsConsumerDrivenLayoutOptions &options =
        AlpsConsumerDrivenLayoutOptions());

/// ALPS P5a: classify stable consumer-layout contracts at a compiler phase.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsContractDischargeLedgerPass(
    const AlpsContractDischargeLedgerOptions &options =
        AlpsContractDischargeLedgerOptions());

/// ALPS P5c: prefetch a provably future tile while P2e directly forms the
/// representation demanded by its consumer.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAlpsLayoutSupplyPrefetchPass(
    const AlpsLayoutSupplyPrefetchOptions &options =
        AlpsLayoutSupplyPrefetchOptions());

/// V-DAE: decouples Memory Access and Compute Execution using semaphores.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createOmniFetchVDAEInsertPass(
    const OmniFetchVDAEInsertOptions &options = OmniFetchVDAEInsertOptions());

/// Layout Ops Elimination: eliminates redundant layout ops when in-situ
/// reshape is enabled.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLayoutOpsEliminationPass();

} // namespace hexagon
} // namespace mlir

#endif //  HEXAGON_TRANSFORMS_TRANSFORMS_H
