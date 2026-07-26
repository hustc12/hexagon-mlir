//===- Passes.td - Linalg lowering and optimization passes -*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_PASSES
#define HEXAGON_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CollapseAddressSpace :
    Pass<"collapse-address-space"> {
  let summary = "Reduce all address spaces to DDR";
  let description = [{
    This is needed for those address spaces (namely TCM)
    that did not get converted to hexagonmem allocs.
  }];

  let constructor = "hexagon::createCollapseAddressSpacePass()";
}

def CopyCanonicalization :
    Pass<"copy-canonicalization", "func::FuncOp"> {
  let summary = "Eliminate intermediate buffer allocations in copy chains";
  let description = [{
    Canonicalizes memory copy operations by eliminating intermediate
    allocations that serve as passthroughs in write-once, read-many patterns.
  }];
  let constructor =
       "hexagon::createCopyCanonicalizationPass()";
}

def ConvertLayout : Pass<"convert-layout", "func::FuncOp"> {
  let summary = "Convert hexagonmem::ConvertLayoutOp to linalg::GenericOp";
  let constructor = "hexagon::createConvertLayoutPass()";
}

def ConvertToHexagonmem :
     InterfacePass<"convert-to-hexagonmem", "mlir::FunctionOpInterface"> {
  let summary = "Lower memref ops to hexagonmem";
  let description = [{
    Converts vtcm allocs and copies, `memref.alloc memref<4x4096, 1>`
    to equivalent hexagonmem opps e.g. `hexagonmem.alloc .. `.
  }];
  let constructor = "hexagon::createConvertToHexagonmemPass()";
}

def ConvertZeroSizeMemref :
    Pass<"convert-zero-size-memref", "func::FuncOp"> {
    let summary = "Convert zero size memref to unit size.";
    let constructor = "hexagon::createConvertZeroSizeMemrefPass()";
}

def DecomposeTensorConcat : Pass<"triton-decompose-tensor-concat", "func::FuncOp"> {
  let summary = "Convert tensor.concat to other tensor ops";
  let constructor = "hexagon::createDecomposeTensorConcatPass()";
}

def EraseUnusedLinalgOperands :
     Pass<"erase-unused-linalg-operands", "mlir::ModuleOp"> {
   let summary = "Erase unused linalg operand and remove dead code.";
   let constructor = "hexagon::createEraseUnusedLinalgOperands()";
}

def ExpandBoolVec :
    InterfacePass<"expand-bool-vec", "mlir::FunctionOpInterface"> {
  let summary = "replace ui1_to_fp with select %cond %vec_1s %vec_0s";
  let constructor = "hexagon::createExpandBoolVecPass()";
}

def ExpandMathOps : Pass<"expand-math-ops", "func::FuncOp"> {
  let summary = "Expand complex math ops to simple math ops";
  let constructor = "hexagon::createExpandMathOpsPass()";
}

def FastInverse :
    Pass<"fast-inverse", "mlir::ModuleOp"> {
  let summary = "Pass to convert element-wise division to multiplication by reciprocal.";
  let constructor = "hexagon::createFastInversePass()";
}

def HexagonAddFastMath :
    Pass<"hexagon-add-fastmath", "mlir::ModuleOp"> {
  let summary = "Add fast math flag to selected ops.";
  let constructor = "hexagon::createHexagonAddFastMathPass()";
}

def HexagonLLVMEnableHexagonRoutines :
    Pass<"llvm-intr-to-hexagon-routines", "mlir::ModuleOp"> {
  let summary = "Replace LLVM intrinsics with vectorized Hexagon routines.";
  let constructor = "hexagon::createHexagonLLVMEnableHexagonRoutinesPass()";
}

def HexagonLWPPass : Pass<"hexagon-lwp-instrumentation", "func::FuncOp"> {
  let summary = "Pass to instrument and profile with hexagon instrinsic";

  let options = [
    Option<"disableLWPLoop", "loop", "bool",
           /*default=*/"false",
           "Turn on or off loop level instrumentation">,
    Option<"LWPloopDepth", "depth", "int",
           /*default=*/"1",
           "Depth level instrumentation">,
   ];

  let constructor = "hexagon::createHexagonLWPPass()";
}

def HexagonPuntBuffer :
    InterfacePass<"hexagon-punt-buffer", "mlir::FunctionOpInterface"> {
  let summary = "erase copies and forward buffer directly to user";
  let constructor =
       "hexagon::createHexagonPuntBufferPass()";
}

def HexagonDoubleBufferGenericS1 :
    InterfacePass<"hexagon-double-buffer-generic-s1",
                  "mlir::FunctionOpInterface"> {
  let summary = "implement double buffering - Stage 1";

  let description = [{
    Implements sotfware pipelining using  double-buffering.
    It takes a tiled IR that originated from a fused
    linalg.generic and rewrites it as a software-pipelined
    loop using double-buffering mechanism.
    This is implemented in two stages:

      S1 : (this pass)
         This pass must run after one-shot-bufferization.
         The double buffers are introduced into the IR but
         data transfers are kept as `memref.copy` because
         de-allocator has not run yet.
      S2 : Please see HexagonDoubleBufferS2 pass.
  }];
  let constructor =
       "hexagon::createHexagonDoubleBufferGenericS1Pass()";
}

def HexagonDoubleBufferGenericS2 :
    InterfacePass<"hexagon-double-buffer-generic-s2",
                  "mlir::FunctionOpInterface"> {
  let summary = "implement double buffering - Stage 2";

  let description = [{
    Implements sotfware pipelining using  double-buffering.
    It has two stages.
      S1 : See HexagonDoubleBufferGenericS1 for Stage 1.

      S2 : (this pass)
         This stage runs after loop-hoisting and deallocation
         passes have run. In this stage  memref.dma_start
         and memref.dma_wait are introduced into the IR.
  }];
  let constructor =
       "hexagon::createHexagonDoubleBufferGenericS2Pass()";
}

def HexagonLowerTmTensor :
    InterfacePass<"hexagon-lower-tm-tensor", "mlir::FunctionOpInterface"> {
  let summary = "Lower lower tm-tensor ops to mlir dialect ops";
  let options = [
    Option<"emitKvCacheMetadata", "emit-kv-cache-metadata", "bool",
           /*default=*/"false",
           "Attach semantic K/V stream metadata for item-7 page-aware prefetch.">,
  ];
  let constructor = "hexagon::createHexagonLowerTmTensorPass()";
}

def HexagonReplaceWithLibraryCalls :
    Pass<"hexagon-replace-with-library-calls", "mlir::ModuleOp"> {
  let summary = "Replaces linalg ops with library_call attr set with "
                "library calls.";
  let constructor =
       "hexagon::createHexagonReplaceWithLibraryCallsPass()";
}

def HexagonRVO :
    InterfacePass<"hexagon-rvo", "mlir::FunctionOpInterface"> {
  let summary = "return value optimization";
  let description = [{
    This pass rewrites the graph so that instead of creating results
    in a temporary buffer and then copying to output buffer, instead
    the materialization of result occurs directly into output buffer;
  }];

  let constructor =
       "hexagon::createHexagonRVOPass()";
}

def HexagonVectorLowering :
    Pass<"hexagon-vector-lowering", "mlir::ModuleOp"> {
  let summary = "Pass to simplify vector multi-reduction and contraction.";
  let constructor = "hexagon::createHexagonVectorLoweringPass()";
}

def LinalgGeneralize :
    InterfacePass<"linalg-generalize", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower linalg named op to generic.";

  let constructor =
       "hexagon::createLinalgGeneralizePass()";
}

def LowerLibdevice :
    Pass<"lower-libdevice", "mlir::ModuleOp"> {
    let summary = "Convert tt.extern_elementwise libdevice ops to func.func";
	let constructor = "hexagon::createLowerLibdevicePass()";
}

def LowerTPtr :
    InterfacePass<"lower-tptr", "mlir::FunctionOpInterface"> {
  let summary = "Lower tptr ops to mlir dialect ops";
  let constructor = "hexagon::createLowerTPtrPass()";
}

def LowerTTX :
    InterfacePass<"lower-ttx", "mlir::FunctionOpInterface"> {
  let summary = "Lower ttx ops to mlir dialect ops";
  let constructor = "hexagon::createLowerTTXPass()";
}

def MatmulToConv : Pass<"matmul-to-conv", "func::FuncOp"> {
  let summary = "Convert linalg.matmul to linalg.conv2d ops";

  let constructor = "hexagon::createMatmulToConvPass()";
}

def MatmulToHexKL :
    InterfacePass<"matmul-to-hexkl", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg.matmul to hexkl ops";
  let options = [
    Option<"enableAttentionHmx", "enable-attention-hmx", "bool",
           /*default=*/"false",
           "Pad attention-like matmuls (K==M or N==M) to break the skip heuristic "
           "and lower them through HexKL (unpad results).">,
  ];
  let constructor = "hexagon::createMatmulToHexKLPass()";
}

def DecomposeHexKLMatmul :
    InterfacePass<"decompose-hexkl-matmul", "mlir::FunctionOpInterface"> {
  let summary = "Decompose hexkl.matmul into hexkl micro HMX operations";
  let description = [{
    Decomposes hexkl::MatmulOp into a sequence of hexkl micro HMX operations
    that perform tiled matrix multiplication. This pass implements the same
    tiling strategy as hexkl_matmul_f16f16_f32 in the runtime library.

    The decomposition includes:
    - VTCM allocation for tile buffers
    - Triple-nested loops for row/column/inner tiling
    - Activation and weight tile loading and layout conversion
    - HMX matrix multiplication with accumulation
    - Result readback and format conversion (f16 to f32)
  }];

  let options = [
    Option<"enableWeightPrepack", "enable-weight-prepack", "bool",
           /*default=*/"false",
           "Hoist the RM->WH weight transform out of the M-loop into a one-time "
           "prologue that pre-packs every (kt,colTile) tile into a DDR WH buffer; "
           "the inner K-loop then only copies the pre-packed tile into VTCM.">,
    Option<"enablePersistentVtcm", "enable-persistent-vtcm", "bool",
           /*default=*/"false",
           "Allocate one function-scoped VTCM slab sized to the max HexKL matmul "
           "requirement and reuse it across matmuls (avoids per-matmul "
           "alloc/dealloc churn). Falls back to per-matmul alloc when any K is "
           "dynamic. Default off: measured gain is noise-level and OmniFetch "
           "interaction needs more validation.">,
    Option<"enableVtcmLifetimeColoring", "enable-vtcm-lifetime-coloring", "bool",
           /*default=*/"false",
           "Assign VTCM offsets from phase lifetimes so scratch, weight, "
           "activation, flat-output, and accumulator regions reuse storage.">,
    Option<"enableDmaToVtcm", "enable-dma-to-vtcm", "bool",
           /*default=*/"false",
           "Reserve a non-aliasing VTCM staging tile for OmniFetch DMA.">,
  ];

  let constructor = "hexagon::createDecomposeHexKLMatmulPass()";
}

def ConvTiling : Pass<"conv-tiling", "func::FuncOp"> {
  let summary = "Tile linalg.conv2d ops";
  let options = [
    Option<"convTilingFactor", "conv-tile-size", "int",
           /*default=*/"-1",
           "override tile size. Only intended for debugging.">,
    Option<"convTileHeightDim", "conv-tile-height-dim", "bool",
           /*default=*/"false",
           "Only tile height dimension. Only intended for debugging">,
    Option<"convTileOCDim", "conv-tile-oc-dim", "bool",
           /*default=*/"false",
           "Only tile OC dim. Only intended for debugging.">

  ];
  let constructor = "hexagon::createConvTilingPass()";
}

def MemoryOffsets : Pass<"memory-offsets", "func::FuncOp"> {
  let summary = "Assigns offsets to buffers.";
  let description = [{
    Assigns offsets to buffers within a function to support memory
    reuse and layout optimization. The basic strategy is to replace multiple
    hexagonmem.alloc operations with views into a single statically allocated
    buffer, enabling memory reuse through offset assignment.
  }];
  let constructor = "hexagon::createMemoryOffsetsPass()";
}

def RemoveMLProgramPass :
    Pass<"remove-ml-program", "mlir::ModuleOp"> {
    let summary = "Remove ml_program operations from the IR";
	let constructor = "hexagon::removeMLProgramPass()";
}

def ReduceContractionRankPass :
  Pass<"reduce-contraction-rank", "mlir::func::FuncOp"> {
  let summary = "Reduce rank of named contraction ops with unit dims";
  let description = [{
    Converts batch_matmul with batch=1 to matmul, and matvec with unit dims to dot.
  }];
  let constructor = "hexagon::createReduceContractionRankPass()";
}

def HoistScalarOps : Pass<"hoist-scalar-ops-in-linalg-generic", "func::FuncOp"> {
  let summary = "Hoist scalar arith/math ops out of linalg.generic regions";
  let description = [{
    This pass identifies loop-invariant scalar arithmetic and math operations
    within linalg.generic regions and hoists them outside the operation.
    The hoisted scalar values are then passed back into the generic operation
    as additional inputs.
  }];
  let constructor = "mlir::hexagon::createHoistScalarOpsPass()";
}

def FoldMulFByZeroPass : Pass<"fold-mulf-by-zero"> {
  let summary = "Add pass to fold x * 0 => 0 when fast math is enabled.";
  let description = [{
    Peephole optimization to fold x * 0 => 0 when appropriate fast math flags
    are enabled.
  }];
  let constructor = "mlir::hexagon::createFoldMulFByZeroPass()";
}

def ScheduleMatmulForHVX :
    InterfacePass<"schedule-matmul-for-hvx", "mlir::FunctionOpInterface"> {
  let summary = "Pass to schedule matmul and its variants";

  let constructor =
       "hexagon::createScheduleMatmulForHVXPass()";
}

def SmallExponentToMultiply :
    Pass<"small-exponent-to-multiply", "mlir::ModuleOp"> {
    let summary = "Convert pow with small integer exponent to multiplications.";
    let constructor = "hexagon::createSmallExponentToMultiplyPass()";
    let options = [
      Option<"maxExponent", "max-exponent", "int",
             /*default=*/"4",
             "Maximum exponent to convert to multiplications.">,
      Option<"allowNegative", "allow-negative", "bool",
             /*default=*/"true",
             "Allow conversion of negative exponents.">,
    ];
}

//===----------------------------------------------------------------------===//
// Omni-Fetch: Prefetch insertion pass
//===----------------------------------------------------------------------===//

def PrefetchInsert :
    InterfacePass<"prefetch-insert", "mlir::FunctionOpInterface"> {
  let summary = "Insert prefetch operations to preload data from DDR to VTCM";
  let description = [{
    This pass inserts prefetch operations to preload data from DDR to VTCM,
    reducing memory latency during compute operations.

    The pass:
    1. Detects loops with HMX compute operations
    2. Identifies DDR inputs that need to be prefetched
    3. Allocates VTCM shadow buffers
    4. Inserts prefetch operations in prologue and loop body

    The prefetch can optionally perform in-situ layout transformation during
    the DDR→VTCM transfer (controlled by enableLayoutAware option).

    This pass is independent of V-DAE. It only inserts prefetch operations
    without any synchronization mechanism. The V-DAE pass (if enabled) will
    later add semaphore synchronization around these prefetch operations.

    Options
    -------
    - `lookahead`       : static prefetch look-ahead distance (default: 2)
    - `enable-layout-aware` : enable in-situ layout transformation during prefetch
  }];

  let options = [
    Option<"lookahead", "lookahead", "int",
           /*default=*/"2",
           "Static prefetch look-ahead distance (number of iterations).">,
    Option<"enableLayoutAware", "enable-layout-aware", "bool",
           /*default=*/"true",
           "Enable in-situ layout-aware prefetch; when false, emit linear prefetches only.">,
    Option<"enableDmaToVtcm", "enable-dma-to-vtcm", "bool",
           /*default=*/"false",
           "DMA-pack weight tiles into VTCM staging (flatOff); when false, pack into DDR stage.">,
    Option<"enableInterLayerPrefetch", "enable-inter-layer-prefetch", "bool",
           /*default=*/"false",
           "Also consider outer (non-innermost) HexKL loops for weight prefetch.">,
    Option<"enablePersistentWhCache", "enable-persistent-wh-cache", "bool",
           /*default=*/"false",
           "Use the generation-safe runtime WH cache for profitable weight sites.">,
    Option<"enableTwoDimPipeline", "enable-two-dim-pipeline", "bool",
           /*default=*/"false",
           "Pipeline DDR load for tile t+1, WH transform for tile t+1, and "
           "HMX compute for tile t without redundantly transforming current tiles.">,
    Option<"enableKvCachePrefetch", "enable-kv-cache-prefetch", "bool",
           /*default=*/"false",
           "Insert page-coalesced L2 hints for compiler-identified attention K/V streams.">,
    Option<"kvCachePageTokens", "kv-cache-page-tokens", "int",
           /*default=*/"32",
           "Logical K/V cache page size in sequence tokens used for accounting and coalescing.">,
  ];

  let constructor = "hexagon::createPrefetchInsertPass()";

  let dependentDialects = [
    "mlir::omni_fetch::OmniFetchDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Omni-Fetch: V-DAE (Virtual Decoupled Access-Execute) pass
//===----------------------------------------------------------------------===//

def OmniFetchVDAEInsert :
    InterfacePass<"omni-fetch-vdae-insert", "mlir::FunctionOpInterface"> {
  let summary = "Decouple Memory Access and Compute Execution using semaphores";
  let description = [{
    Implements the Virtual Decoupled Access-Execute (V-DAE) transformation
    for DNN inference loops on Hexagon NPU.

    **Problem**: VLIW (Very Long Instruction Word) architecture means a single
    memory load stall can halt an entire instruction packet, freezing all
    compute units in the same cycle.

    **Solution**: Decouple Memory Access and Compute Execution using hardware
    semaphores, allowing Access Thread and Execute Thread to run in parallel.

    The pass walks loops that already have `omni_fetch.prefetch_in_situ`
    operations (inserted by PrefetchInsertPass) and adds synchronization:

      Prologue:
        %sem = omni_fetch.create_sem
        <existing prefetch operations>
        omni_fetch.signal %sem

      Loop body (per iteration i):
        omni_fetch.wait %sem           -- ensure tile [i] is ready
        <existing HMX compute on tile[i]>
        <existing prefetch for tile[i + lookahead]>
        omni_fetch.signal %sem         -- notify tile [i + lookahead] ready
        %dist = omni_fetch.adaptive_control(%dist)  -- optional

    **Key design**: This pass does NOT insert prefetch operations. It only
    adds synchronization around existing prefetch operations. The prefetch
    operations should be inserted by PrefetchInsertPass first.

    Options
    -------
    - `enable-adaptive` : emit adaptive_control ops to tune at runtime
  }];

  let options = [
    Option<"enableAdaptive", "enable-adaptive", "bool",
           /*default=*/"true",
           "Emit omni_fetch.adaptive_control ops for runtime PMU tuning.">,
  ];

  let constructor = "hexagon::createOmniFetchVDAEInsertPass()";

  let dependentDialects = [
    "mlir::omni_fetch::OmniFetchDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Omni-Fetch: Layout Ops Elimination pass
//===----------------------------------------------------------------------===//

def LayoutOpsElimination :
    InterfacePass<"layout-ops-elimination", "mlir::FunctionOpInterface"> {
  let summary = "Eliminate redundant layout ops when in-situ reshape is enabled";
  let description = [{
    This pass eliminates redundant layout transformation operations
    (transpose, permute, slice) when in-situ reshape is enabled during
    prefetch operations.

    The pass works by:
    1. Finding all `omni_fetch.prefetch_in_situ` operations with
       `layout_transform != None`
    2. Walking backward from each prefetch's `src` operand to find
       layout ops that match the in-situ transform
    3. Marking those ops as redundant (since the hardware performs
       the transform during prefetch)
    4. Deleting the marked ops

    This pass should run AFTER any pass that inserts prefetch_in_situ
    operations (e.g., V-DAE pass) and BEFORE lowering to LLVM.

    Key insight: This pass detects `prefetch_in_situ` operations, NOT
    loops. It works at function level and is independent of V-DAE.
    Even without V-DAE, if other passes insert prefetch_in_situ operations,
    this pass will eliminate redundant layout ops.

    Design: This pass is bound to in-situ reshape. Whenever layout-aware
    prefetch is enabled, this pass should run to eliminate redundant ops.
  }];

  let constructor = "hexagon::createLayoutOpsEliminationPass()";

  let dependentDialects = [
    "mlir::omni_fetch::OmniFetchDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif
