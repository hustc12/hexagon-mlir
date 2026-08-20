//===- Passes.td - Linalg to LLVM conversion passes  -------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_LINALG_TO_LLVM_CONVERSION_PASSES
#define HEXAGON_LINALG_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// -- The main linalg-to-llvm pass-pipeline --
def LinalgToLLVM : Pass<"linalg-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Linalg to LLVM";
  let constructor = "hexagon::createLinalgToLLVMPass()";
  let options = [
    Option<"puntBuffer", "punt-buffer", "bool",
           /*default=*/"true",
           "Punt input kernel buffer pointer directly to user instead"
           " of making local copy first.">,
    Option<"returnValueOptimization", "rvo", "bool",
           /*default=*/"true",
           "Rewrites the IR so that it is easier for one-shot bufferizer and"
           " eliminate empty-tensor to fold away unnecessary copying from"
           " local buffer to output buffer">,
    Option<"enableMatmulToConv", "enableMatmulToConv", "bool",
            /*default=*/"false",
            "Lower matmul ops to conv ops">,
    Option<"fusion", "fusion", "bool",
           /*default=*/"true",
           "Attempt to fuse linalg generic ops">,
    Option<"fusionAllowRecompute", "far", "bool",
           /*default=*/"false",
           "Allow recomputing values when fusing linalg generic ops">,
    Option<"fusionDoMultiUse", "fdmu", "bool",
           /*default=*/"true",
           "Do multi-use fusion">,
    Option<"useInterchangeVector", "use-interchange-vector", "bool",
           /*default=*/"false",
           "Use interchange vector in the HexagonTiling pass"
           " if default inner loop is not conducive for vectorization.">,
    Option<"enableDoubleBuffering", "enable-double-buffering", "bool",
          /*default=*/"false",
          "Enable double buffering optimization">,
    Option<"enableSCFThreading", "enable-scf-threading", "bool",
          /*default=*/"false",
          "Enable scf-level multi-threading">,
    Option<"enableMultiThreading", "enable-multi-threading", "bool",
          /*default=*/"false",
          "Enable asnyc multi-threading">,
    Option<"addFastMath", "add-fastmath", "bool",
           /*default=*/"true",
           "Add Fast Math Flag to HexagonAddFastMath Pass.">,
    Option<"splitTilingRange", "split-tiling-range", "bool",
          /*default=*/"true",
          "Configure HexagonTiling pass to do op splitting, "
          "instead of padding, when dealing with loops "
          "which are not  perfectly tileable.">,
    Option<"enableSplitReduction", "enable-split-reduction", "bool",
           /*default=*/"true",
           "Configure HexagonTiling pass to enable split reduction, "
           " for reduction loops. ">,
    Option<"enableSplitReduceGeneric", "enable-split-reduce-generic", "bool",
           /*default=*/"false",
           "Enable split-reduce-generic to convert linalg.generic [reduce] {..}"
           "into a linalg.generic [reduce, parallel] {..} for vectorization, "
           "followed by a final single reduction.">,
    Option<"enableSlicing", "enable-slicing", "bool",
           /*default=*/"false",
           "Flag to enable/disable operator slicing pass">,
    Option<"slicingFactor", "slicing-factor", "int",
           /*default=*/"1",
           "Operator slicing factor.">,
    Option<"expandBoolVec", "expand-bool-vec", "bool",
           /*default=*/"true",
           "replace ui1_to_fp with select %cond %vec_1s %vec_0s">,
    Option<"enableHexagonRoutines", "enable-hexagon-routines", "bool",
          /*default=*/"true",
          "Replaces LLVM intrisics with vectorized Hexagon routines.">,
    Option<"enableVTCMTiling", "enable-vtcm-tiling", "bool",
          /*default=*/"true",
          "enable lowering linalg-on-tensors (DDR) to tiled-compute on VTCM">,
    Option<"scratch", "scratch", "int64_t",
          /*default=*/"0",
          "per-instance VTCM scratch size in bytes (0 = disabled). When >0, "
          "enables external VTCM flow: InsertScratchArg injects a "
          "memref<Nxi8,1> {hexagon.scratch} arg, VTCMTilingPass tiles within "
          "this budget, and MemoryOffsetsPass replaces allocs with views.">,
    Option<"tileSizes", "tile-sizes-override", "std::string",
           /*default=*/"",
           "override tile size.">,
    Option<"enableConvertToHexagonmem", "enable-convert-to-hexagonmem", "bool",
          /*default=*/"true",
          "enable conversion from memref to hexagonmem ops">,
    Option<"enableHexagonmemCopyToDMA", "enable-hexagonmem-copy-to-dma", "bool",
          /*default=*/"false",
          "enable conversion from hexagonmem copy to dma ops">,
    Option<"enableHexKL", "enable-hexkl", "bool",
          /*default=*/"false",
          "Enable linalg matmul to hexkl dialect lowering">,
    Option<"hexKLMode", "hexkl-mode", "std::string",
          /*default=*/"\"micro\"",
          "HexKL API mode: 'micro' (default) or 'macro'">,
    // External prefetch baselines. These options deliberately do not enable
    // any OmniFetch transformation or consumer rewiring.
    Option<"enablePrefetchKernelHX", "enable-prefetch-kernel-hx", "bool",
          /*default=*/"false",
          "Enable the isolated Prefetch-Kernel-HX affine address baseline">,
    Option<"prefetchKernelHxDistance", "prefetch-kernel-hx-distance", "int",
          /*default=*/"1",
          "Static future-iteration distance used by Prefetch-Kernel-HX">,
    Option<"prefetchKernelHxMaxCommandBytes", "prefetch-kernel-hx-max-command-bytes", "int",
          /*default=*/"8191",
          "Maximum bytes admitted for one Prefetch-Kernel-HX L2FETCH request">,
    Option<"enableAPTGetHX", "enable-apt-get-hx", "bool",
          /*default=*/"false",
          "Enable APT-GET-HX timing for manually-safe address candidates">,
    Option<"aptGetHxDistance", "apt-get-hx-distance", "int",
          /*default=*/"1",
          "Distance selected offline by scripts/apt_get_hx_select.py">,
    Option<"aptGetHxManualCandidateIds", "apt-get-hx-manual-candidate-ids", "std::string",
          /*default=*/"\"\"",
          "Comma-separated candidate IDs manually qualified for APT-GET-HX">,
    // Omni-Fetch: Plan-A three independent components -----------------------
    // Component 1: Prefetch insertion (base component; required by 2 & 3)
    Option<"enablePrefetch", "enable-prefetch", "bool",
          /*default=*/"false",
          "Insert omni_fetch.prefetch_in_situ ops (base Omni-Fetch component)">,
    // Component 2: In-Situ Reshape (requires enablePrefetch)
    Option<"enableOmniFetchLayoutAware", "enable-omnifetch-layout-aware", "bool",
          /*default=*/"true",
          "Enable layout-aware in-situ prefetch mapping; requires enablePrefetch">,
    Option<"omniFetchLookahead", "omnifetch-lookahead", "int",
          /*default=*/"2",
          "Static look-ahead distance for Omni-Fetch prefetching">,
    Option<"enableOmniFetchDmaToVtcm", "enable-omnifetch-dma-to-vtcm", "bool",
          /*default=*/"false",
          "DMA-pack OmniFetch weight tiles into VTCM staging (vs DDR stage)">,
    // Component 3: V-DAE (requires enablePrefetch)
    Option<"enableOmniFetchVDAE", "enable-omnifetch-vdae", "bool",
          /*default=*/"false",
          "Decouple Access/Execute with semaphores (V-DAE); requires enablePrefetch">,
    Option<"enableOmniFetchAdaptive", "enable-omnifetch-adaptive", "bool",
          /*default=*/"true",
          "Enable adaptive PMU-driven prefetch distance updates">,
    Option<"enableOmniFetchWeightPrepack", "enable-omnifetch-weight-prepack", "bool",
          /*default=*/"false",
          "Hoist HexKL RM->WH out of the M-loop: pack one column's WH tiles into "
          "VTCM once, then stream M rows (no DDR round-trip). Requires enableHexKL">,
    Option<"enableOmniFetchPersistentWhCache", "enable-omnifetch-persistent-wh-cache", "bool",
          /*default=*/"false",
          "Enable generation-safe cross-token caching of transformed WH tiles">,
    Option<"enableOmniFetchTwoDimPipeline", "enable-omnifetch-two-dim-pipeline", "bool",
          /*default=*/"false",
          "Enable the load/reshape/compute pipeline for profitable HexKL weight sites">,
    Option<"enableOmniFetchVtcmColoring", "enable-omnifetch-vtcm-coloring", "bool",
          /*default=*/"false",
          "Enable interference-based lifetime coloring of HexKL VTCM regions">,
    // ALPS P0 separates the four effects historically coupled by item 7.
    // The legacy OmniFetch option below remains an umbrella compatibility
    // switch and enables every ALPS K/V control.
    Option<"enableAlpsKvSemanticTracking", "enable-alps-kv-semantic-tracking", "bool",
          /*default=*/"false",
          "ALPS: identify and propagate semantic attention K/V streams only">,
    Option<"enableAlpsKvFusionPolicy", "enable-alps-kv-fusion-policy", "bool",
          /*default=*/"false",
          "P0 umbrella: enable all ALPS K/V topology policies">,
    Option<"enableAlpsKvElementwiseFusionPolicy", "enable-alps-kv-elementwise-fusion-policy", "bool",
          /*default=*/"false",
          "ALPS: preserve K/V elementwise and reshape fusion boundaries">,
    Option<"enableAlpsKvMultiUseFusionPolicy", "enable-alps-kv-multi-use-fusion-policy", "bool",
          /*default=*/"false",
          "ALPS: suppress function-wide multi-use fusion at K/V boundaries">,
    Option<"enableAlpsKvSplitReductionPolicy", "enable-alps-kv-split-reduction-policy", "bool",
          /*default=*/"false",
          "ALPS: suppress split reduction for semantic K/V contractions">,
    Option<"enableAlpsKvSlicingPolicy", "enable-alps-kv-slicing-policy", "bool",
          /*default=*/"false",
          "ALPS: preserve K/V metadata by disabling legacy attention slicing">,
    Option<"enableAlpsKvRuntimePrefetch", "enable-alps-kv-runtime-prefetch", "bool",
          /*default=*/"false",
          "ALPS: emit page-aware runtime prefetch for eligible persistent K/V streams">,
    Option<"enableAlpsMovementLedger", "enable-alps-movement-ledger", "bool",
          /*default=*/"false",
          "ALPS P1: emit analysis-only representation and physical-movement ledgers">,
    Option<"enableAlpsZeroCopyAttention", "enable-alps-zero-copy-attention", "bool",
          /*default=*/"false",
          "ALPS P2a: absorb proven attention transpose chains into consumer indexing maps">,
    Option<"enableAlpsProducerDirectAttention", "enable-alps-producer-direct-attention", "bool",
          /*default=*/"false",
          "ALPS P2b: make attention producers write contiguous head layouts directly">,
    Option<"enableAlpsFusedTransformTransfer", "enable-alps-fused-transform-transfer", "bool",
          /*default=*/"false",
          "ALPS P2c: replace proven HexKL transfer plus layout chains with synchronous in-situ formation">,
    Option<"enableAlpsMinimalStaticAdmission", "enable-alps-minimal-static-admission", "bool",
          /*default=*/"false",
          "ALPS P2d: conservatively admit one representation-supply action per candidate">,
    Option<"enableAlpsExactReadiness", "enable-alps-exact-readiness", "bool",
          /*default=*/"false",
          "ALPS P3a: audit async actions against exact invocation/tile descriptor identity">,
    Option<"enableAlpsExactOverlap", "enable-alps-exact-overlap", "bool",
          /*default=*/"false",
          "ALPS P3b: execute one-tile descriptor-bound HMX weight DMA/scout overlap">,
    Option<"alpsLedgerPageBytes", "alps-ledger-page-bytes", "int64_t",
          /*default=*/"4096",
          "Page size used for ALPS P1 static page-footprint estimates">,
    Option<"alpsLedgerVtcmBudgetBytes", "alps-ledger-vtcm-budget-bytes", "int64_t",
          /*default=*/"2097152",
          "Per-candidate VTCM-fit threshold used by the ALPS P1 ledger">,
    Option<"enableOmniFetchKvCachePrefetch", "enable-omnifetch-kv-cache-prefetch", "bool",
          /*default=*/"false",
          "Legacy item-7 umbrella: enable all four ALPS K/V controls">,
    Option<"enableOmniFetchWeightStationary", "enable-omnifetch-weight-stationary", "bool",
          /*default=*/"false",
          "Enable N1 phase-specialized weight-stationary HVX projection scheduling">,
    Option<"enableOmniFetchActivationMulticast", "enable-omnifetch-activation-multicast", "bool",
          /*default=*/"false",
          "Enable N2 activation multicast across equal-shape sibling HVX projections">,
    Option<"enableOmniFetchDequantReshape", "enable-omnifetch-dequant-reshape", "bool",
          /*default=*/"false",
          "Fuse W8 compressed-tile load, dequantization, and HexKL WH reshape">,
    Option<"omniFetchKvCachePageTokens", "omnifetch-kv-cache-page-tokens", "int",
          /*default=*/"32",
          "Logical attention K/V cache page size in tokens">,
    Option<"enableOmniFetchDualThreadDae", "enable-omnifetch-dual-thread-dae", "bool",
          /*default=*/"false",
          "Run deferred dma_wait+WH on a scout thread (default off; flag off ≡ "
          "current single-thread V-DAE). Requires enableOmniFetchVDAE">,
    Option<"enableOmniFetchInterLayerPrefetch", "enable-omnifetch-inter-layer-prefetch", "bool",
          /*default=*/"false",
          "Allow PrefetchInsert on outer HexKL loops (next-layer weight prefetch)">,
    Option<"enableOmniFetchAttentionHmx", "enable-omnifetch-attention-hmx", "bool",
          /*default=*/"false",
          "Pad attention-like matmuls (K==M or N==M) so they become HexKL-eligible">,
    Option<"enableOmniFetchMPadHmx", "enable-omnifetch-m-pad-hmx", "bool",
          /*default=*/"false",
          "Pad the M (rows/tokens) dimension up to a multiple of 32 so unaligned-token encoders become HexKL-eligible">,
    Option<"enableHexKLPersistentVtcm", "enable-hexkl-persistent-vtcm", "bool",
          /*default=*/"false",
          "Reuse one max-sized VTCM slab across HexKL matmuls in a function">,
    Option<"enableCollapseAddressSpace", "enable-collapse-address-space", "bool",
          /*default=*/"true",
          "replace llvm.ptr<1> with llvm.ptr at last stage of convert-to-llvm">,
    Option<"lowerConstantsInSeparateSharedObjects", "lower-constants-in-separate-shared-objects", "bool",
          /*default=*/"false",
          "Lowers the constant tensors separately in their own MLIR modules, which then get compiled independently into their own .o, and then independently into various shared objects (.so)">,
    Option<"enableBufferization", "enable-bufferization", "bool",
          /*default=*/"true",
          "enable bufferization passes; set to false for manual buffer management">,
    Option<"enableSeedLayoutConversions", "enable-seed-layout-conversions", "bool",
          /*default=*/"false",
          "wraps HMX applicable convolutions with layout conversions">,
    Option<"enableConvTiling", "enable-conv-tiling", "bool",
          /*default=*/"false",
          "Tile the Convolution Op">,
    Option<"convTileSizes", "conv-tile-sizes", "std::string",
          /*default=*/"",
          "Provide the tile sizes in the following format: '<dim1>:<tilesize1>, <dim2>:<tilesize2>,...' ">,
    Option<"enableLWP", "enable-lwp", "bool",
           /*default=*/"false",
           "Flag to enable/disable lwp instrumentation">,
    Option<"disableLWPLoop", "loop", "bool",
           /*default=*/"false",
           "Turn on or off loop level instrumentation">,
    Option<"LWPloopDepth", "depth", "int",
           /*default=*/"1",
           "Depth level instrumentation">,
    Option<"extendPackUpperFrontier", "upper", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.pack frontiers.">,
    Option<"extendPackLowerFrontier", "lower", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.unpack frontiers.">,
    Option<"extendPackParallelsOnly", "parallels-only", "bool",
           /*default=*/"true",
           "Allow un-staging linalg ops that resolve to reduction "
           "generics even when it is possibly to pack those. This impacts "
           "both upper and lower frontier">,
    Option<"device_type", "device-type", "std::string",
	   /*default=*/"\"hexagon\"",
           "Target device type for backend-specific lowering">,
    Option<"enableVectorization", "enable-vectorization", "bool",
          /*default=*/"true",
          "Enable vectorization for HVX">,
    Option<"enableConversionToFp16", "enable-conversion-to-fp16", "bool",
          /*default=*/"true",
          "Enables the pass which converts fp64/fp32 operations to fp16">,
    Option<"enableHVXInlining", "enable-hvx-inlining", "bool",
          /*default=*/"false",
          "Enable aggressive inlining after linking runtime modules">,
    Option<"enableSCFLoopUnroll", "enable-scf-loop-unroll", "bool",
          /*default=*/"false",
          "Enable SCF loop unrolling optimization">,
    Option<"forceHVXCroutonization", "force-hvx-croutonization", "bool",
          /*default=*/"false",
          "Force HVX croutonization pass">,
    Option<"enableBufferResultsToOutParams", "enable-buffer-results-to-out-params", "bool",
              /*default=*/"false",
              "Convert memref-typed function results into trailing out-param "
              "arguments (function returns void). Avoids the by-value memref "
              "(sret) return that the Hexagon backend miscompiles for large "
              "monolithic functions.">
   ];
}

// --- Supporting passes ---

def FormAsyncThreads : Pass<"form-async-threads", "func::FuncOp"> {
  let summary = "Rewrite scf::forall as async threads";
  let constructor = "hexagon::createFormAsyncThreadsPass()";
}
def FormSCFThreads : Pass<"form-scf-threads", "func::FuncOp"> {
  let summary = "Multi-thread SCF loop";
  let description = [{
    This pass turns scf::for into parallel loops for multi-threading.
    It runs early on in the pipeline and looks for scf:for that is
    over-arching over most of the kernel. The pass will do some
    checks but the onus is currently mostly on the user to guarantee
    parallel conversion correctness.
  }];
  let constructor = "hexagon::createFormSCFThreadsPass()";
}

def FormVirtualThreads : Pass<"form-virtual-threads", "func::FuncOp"> {
  let summary = "Rewrite generics as scf::forall for thread formation"
                 "after further lowering.";
  let constructor = "hexagon::createFormVirtualThreadsPass()";
  let options = [
       Option<"preferredNumThreads", "preferred-num-threads", "int64_t",
              /*default=*/"4",
              "Preferred number of virtual threads to create.">,
       Option<"blockSize", "block-size", "int64_t",
              /*default=*/"0",
              "Block size for testing (user-override)">,
  ];
}

def HexmemCpyToDMA :
    InterfacePass<"hexmem-cpy-to-dma", "mlir::FunctionOpInterface"> {
  let summary = "Pass to rewrite hexagonmem.copy and memref.copy ops to memref.dma ops";

  let constructor =
       "hexagon::createHexmemCpyToDMAPass()";
}

def HexagonExtendPack:
    InterfacePass<"extend-pack-frontier", "mlir::FunctionOpInterface"> {
  let summary = "Extend pack and unpack frontier around linalg ops.";

  let options = [
    Option<"upperFrontier", "upper", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.pack frontiers.">,
    Option<"lowerFrontier", "lower", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.unpack frontiers.">,
    Option<"parallelsOnly", "parallels-only", "bool",
           /*default=*/"true",
           "Allow un-staging linalg ops that resolve to reduction "
           "generics even when it is possibly to pack those. This impacts "
           "both upper and lower frontier">
  ];

  let constructor =
       "hexagon::createHexagonExtendPackPass()";
}

def HexagonFusion :
    InterfacePass<"hexagon-fusion", "mlir::FunctionOpInterface"> {
  let summary = "Pass to fuse linalg generic ops";
  let options = [
    Option<"fusionAllowRecompute", "far", "bool",
           /*default=*/"false",
           "Allow recomputing values when fusing linalg generic ops">,
    Option<"fusionDoMultiUse", "fdmu", "bool",
           /*default=*/"true",
           "Do multi-use fusion">
   ];
  let constructor =
       "hexagon::createHexagonFusionPass()";
}

def SplitReduceGeneric : Pass<"split-reduce-generic", "func::FuncOp"> {
  let summary = "Split generic reduction operations for optimization";
  let description = [{
    This pass splits a linalg.generic [reduce] {..}into a
    linalg.generic [reduce, parallel] {..} that can be more effectively
    vectorized. That is followed by a final horizontal reduction.
  }];
  let constructor = "hexagon::createSplitReduceGenericPass()";
}

def EraseVectorToTensorWriteback : Pass<"erase-vector-to-tensor-writeback", "func::FuncOp"> {
  let summary = "Erase unnecessary vector-to-tensor writeback in loops";
  let description = [{
    This pass optimizes IR where vectors are written back to tensors
    and yielded in scf.for loops, only to be immediately read back
    as vectors after the loop. When the tensor has no other uses,
    the loop can directly yield the vector instead.
  }];
  let constructor = "hexagon::createEraseVectorToTensorWritebackPass()";
}

def HexagonSlicing :
    Pass<"hexagon-slicing", "mlir::ModuleOp"> {
  let summary = "Pass to slice ops into K parts, "
                " where K is identified by slicingFactor";
  let options = [
    Option<"slicingFactor", "slicing-factor", "int",
           /*default=*/"1",
           "Operator slicing factor.">
  ];
  let constructor = "hexagon::createHexagonSlicingPass()";
}

def HexagonTiling :
    Pass<"hexagon-tiling", "mlir::ModuleOp"> {
  let summary = "Pass to tile tensor linalg ops. Presently, "
                " just for vectorization.";

  let options = [
    Option<"useInterchangeVector", "use-interchange-vector", "bool",
           /*default=*/"false",
           "Use interchange vector if default inner loop "
           "is not conducive for vectorization.">,
    Option<"splitTilingRange", "split-tiling-range", "bool",
           /*default=*/"true",
           "Split the not perfectly tileable loop into perfectly tileable "
           " and the remainder parts. If set to false, padding is added "
           "to make it a perfectly tileable shape">,
    Option<"enableSplitReduction", "enable-split-reduction", "bool",
           /*default=*/"false",
           "Splits reduction loop to expose more opportunities for "
           "vectorization, ">
   ];

  let constructor = "hexagon::createHexagonTilingPass()";
}

def HexagonVectorization :
    Pass<"hexagon-vectorization", "mlir::ModuleOp"> {
  let summary = "Pass to vectorize tensor linalg ops.";
  let constructor = "hexagon::createHexagonVectorizationPass()";
}

def LowerPack : Pass<"lower-pack", "func::FuncOp"> {
  let summary = "lower linalg.pack to simpler ops";
  let constructor = "hexagon::createLowerPackPass()";
}

def LowerConstantsSeparately : Pass<"lower-constants-separately", "mlir::ModuleOp"> {
  let summary = "Lowers constants separately from the code, possibly in multiple modules if above a certain threshold";
  let constructor = "hexagon::createLowerConstantsSeparatelyPass()";
}

def RewriteUBPoisonToZero :
    Pass<"rewrite-ub-poison-to-zero", "mlir::ModuleOp"> {
  let summary = "Pass to rewrite ub poison values to zero.";
  let constructor = "hexagon::createRewriteUBPoisonToZeroPass()";
}

def VTCMTiling :
     InterfacePass<"vtcm-tiling", "mlir::FunctionOpInterface"> {
  let summary = "Lower linalg-on-tensors (DDR) to tiled-compute on VTCM";
  let options = [
    Option<"tileSizes", "tile-size-override", "std::string",
           /*default=*/"",
           "override tile size. Only intended for debugging.">,
    Option<"vtcmBudget", "vtcm-budget", "int64_t",
           /*default=*/"0",
           "per-instance VTCM budget in bytes for tile size computation. "
           "0 = use default (vtcmSizeInBytes = 2 MB).">
  ];
  let constructor = "hexagon::createVTCMTilingPass()";
}

def ConversionToFp16 :
    InterfacePass<"conversion-to-fp16", "mlir::FunctionOpInterface"> {
  let summary =
        "Pass to convert intermediate fp32 and fp64 operations to fp16 operations";
  let constructor = "hexagon::createConversionToFp16Pass()";
}

def OptimizeExtfTruncfOp :
    InterfacePass<"optimize-extf-truncf-op", "mlir::FunctionOpInterface"> {
  let summary =
        "Pass to optimize IR and remove redundant extf and truncf op";
  let constructor = "hexagon::createOptimizeExtfTruncfOpPass()";
}

#endif
