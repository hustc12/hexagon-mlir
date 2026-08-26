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
           "Attach semantic K/V stream metadata for ALPS analysis/prefetch.">,
    Option<"emitKvFusionBoundary", "emit-kv-fusion-boundary", "bool",
           /*default=*/"false",
           "Compatibility umbrella for every ALPS K/V topology boundary.">,
    Option<"emitKvElementwiseFusionBoundary", "emit-kv-elementwise-fusion-boundary", "bool",
           /*default=*/"false",
           "Protect K/V elementwise and reshape fusion boundaries.">,
    Option<"emitKvMultiUseFusionBoundary", "emit-kv-multi-use-fusion-boundary", "bool",
           /*default=*/"false",
           "Protect K/V boundaries from function-wide multi-use fusion.">,
    Option<"emitKvSplitReductionBoundary", "emit-kv-split-reduction-boundary", "bool",
           /*default=*/"false",
           "Protect K/V contractions from split reduction.">,
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

def MatmulToHexKL : InterfacePass<"matmul-to-hexkl", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg.matmul to hexkl.matmul";

  let description = [{
    Converts linalg.matmul operations with FP16 inputs to hexkl.matmul.

    Modes:
    - macro: High-level API with stricter requirements
      * Outputs f16 only (must upcast to f32 later if needed)
      * Strict dimension limits (M≤1600, N≤5120, K≤76030)
      * Memory allocation handled by macro API
      * Optimized DMA transfers and tiling strategy

    - micro(default): Low-level API with more flexibility
      * Can output f16 or f32 directly
      * Fine grained control over tiling and memory allocation
  }];

  let constructor = "mlir::hexagon::createMatmulToHexKLPass()";

  let options = [
    Option<"mode", "mode", "std::string",
           /*default=*/"\"micro\"",
           "HexKL API mode: 'micro' (default) or 'macro'">
  ];
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
  let description = [{
    Tiles Conv2D operations along specified dimensions.

    Assumes NHWC layout for input/output tensors:
    - Dimension 0: Batch (N)
    - Dimension 1: Height (H)
    - Dimension 2: Width (W)
    - Dimension 3: Channels (C)

    Example: --conv-tile-sizes="1:8,3:32"
    This tiles dimension 1 (height) with size 8 and dimension 3 (channels) with size 32.

    Constraints:
    - Height tile size (dim 1) must be a multiple of 8
    - Channel tile size (dim 3) must be a multiple of 32
  }];
  let options = [
    Option<"convTileSizes", "conv-tile-sizes", "std::string",
           /*default=*/"",
           "Tile sizes for conv2d dimensions in format: dim:size,dim:size,... ">
  ];
  let constructor = "hexagon::createConvTilingPass()";
}

def InsertScratchArg : Pass<"insert-scratch-arg", "func::FuncOp"> {
  let summary = "Inject per-instance VTCM scratch argument";
  let description = [{
    Adds a function argument typed as memref<Nxi8, 1> {hexagon.scratch} before
    the trailing Triton program-info arg pack (num_programs_x/y/z, pid_x/y/z),
    where N is the per-instance VTCM budget in bytes (the `scratch` option).

    The static size enables MemoryOffsetsPass to validate at compile time that
    all VTCM allocs fit within the per-instance budget.

    The wrapper allocates scratch * prod(grid) bytes total and passes each
    instance a non-overlapping slice of size `scratch` bytes.

    Idempotent: if a hexagon.scratch arg already exists, no arg is inserted.
    Skips functions that do not have the trailing 6 i32 program-info ABI pack.
  }];
  let options = [
    Option<"scratch", "scratch", "int64_t",
           /*default=*/"0",
           "Per-instance VTCM scratch size in bytes (must be > 0)">
  ];
  let constructor = "hexagon::createInsertScratchArgPass()";
}

def MemoryOffsets : Pass<"memory-offsets", "func::FuncOp"> {
  let summary = "Assigns offsets to VTCM buffers.";
  let description = [{
    Replaces memref.alloc/hexagonmem.alloc operations in VTCM address space
    (space 1) with memref.view ops into a single VTCM buffer, enabling memory
    reuse through static offset assignment. Dealloc ops are removed only for
    allocations that this pass replaces.

    Two modes of operation:
    1. External VTCM mode: if a function argument is marked with the
       "hexagon.scratch" attribute, that argument is used as the VTCM
       buffer. The argument must be static-sized (memref<Nxi8, 1>); the
       pass validates at compile time that all kernel allocations fit
       within N bytes and fails if they do not.
    2. Fallback mode: if no such argument is present, a single hexagonmem.alloc
       sized to the exact required bytes is created at function entry, subject
       to the configured bufferSize budget cap (default 1 MB).

    In both modes, all eligible memref.alloc/hexagonmem.alloc ops in address
    space 1 are replaced with memref.view ops at statically computed aligned
    offsets.

    Restrictions (see MemoryOffsetsPass.cpp for details):
    - Only static, unconditional allocations are handled. Dynamic-sized allocs
      and allocs inside loops are left untouched.
    - All allocs are laid out simultaneously (no liveness analysis); allocs
      with non-overlapping lifetimes do not share memory.
  }];
  let options = [
    Option<"bufferSize", "buffer-size", "int64_t",
           /*default=*/"1048576",
           "Fallback-mode VTCM budget cap in bytes">,
    Option<"alignment", "alignment", "int64_t",
           /*default=*/"2048",
           "Byte alignment for buffer offsets in VTCM">,
  ];
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


def FoldCastsIntoMatmul : Pass<"fold-casts-into-matmul", "func::FuncOp"> {
  let summary = "Fold FP16 to FP32 casts into linalg.matmul operations";
  let description = [{
    This pass identifies linalg.matmul operations with FP32 inputs that come
    from FP16→FP32 extension casts and folds the casts into the matmul,
    resulting in a matmul with FP16 inputs and FP32 output.

    This enables efficient mixed-precision computation where the matmul can
    be performed in FP16 with FP32 accumulation, avoiding unnecessary type
    conversions.
  }];
  let constructor = "mlir::hexagon::createFoldCastsIntoMatmulPass()";
  let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect",
                           "tensor::TensorDialect"];
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

def FoldResourceTranspose : Pass<"fold-resource-transpose"> {
  let summary = "Fold transpose of f16 dense resource constants";
  let description = [{
    Pass to fold linalg.transpose ops acting on dense_resource<f16> constants
    into a new pre-transposed constant. Used to avoid runtime transpose overhead
    for static weights.
  }];
  let constructor = "mlir::hexagon::createFoldResourceTransposePass()";
  let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect",
                           "tensor::TensorDialect"];
}

def ScheduleMatmulForHVX :
    InterfacePass<"schedule-matmul-for-hvx", "mlir::FunctionOpInterface"> {
  let summary = "Pass to schedule matmul and its variants";
  let options = [
    Option<"enableWeightStationary", "enable-weight-stationary", "bool",
           /*default=*/"false",
           "Enable OmniFetch N1 transpose-equivalent weight-stationary "
           "scheduling for profitable static FP16 projection matmuls">,
    Option<"enableActivationMulticast", "enable-activation-multicast", "bool",
           /*default=*/"false",
           "Enable OmniFetch N2 fusion of equal-shape sibling FP16 "
           "projections so one activation load feeds both consumers">,
  ];

  let constructor =
       "hexagon::createScheduleMatmulForHVXPass()";
}

def PreprocessTiledConv2D :
  InterfacePass<"preprocess-tiled-conv2d", "mlir::FunctionOpInterface"> {
  let summary = "Preprocess tiled linalg.conv2d for hmx.conv2d conversion";
  let constructor = "hexagon::createPreprocessTiledConv2DPass()";
}

def SeedLayoutConversions :
  InterfacePass<"seed-layout-conversions", "mlir::FunctionOpInterface"> {
  let summary = "wrap conv2d ops with packs and conversion cast";
  let description = [{
    Wrap hmx applicable conv2d ops with pack/unpack ops.
    To ensure it is valid IR after pack insertions, this pass also inserts
    conversion casts around the conv2d.
  For example, given :
  ```mlir
     %0 = linalg.conv_2d_nhwc_fhwc
                    ins(%in1, %in2 : tensor<1x15x7x59xf16>,  tensor<59x1x1x59xf16>)
                     outs(%result: tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>
  after running the pass:

  %cst = arith.constant dense<[2, 32, 1, 1, 2, 16, 2]> : tensor<7xindex>
  %cst_0 = arith.constant 0.000000e+00 : f16

  // pack_1 of activation
  %0 = tensor.empty() : tensor<1x2x2x2x8x4x32xf16>
  %pack = linalg.pack %arg0 padding_value(%cst_0 : f16)
            inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32]
            into %0 : tensor<1x15x7x59xf16> -> tensor<1x2x2x2x8x4x32xf16>

  // pack_2 of activation
  %1 = tensor.empty() : tensor<1x2x2x2x8x2x32x2xf16>
  %pack_1 = linalg.pack %pack
            inner_dims_pos = [5] inner_tiles = [2]
            into %1 : tensor<1x2x2x2x8x4x32xf16> -> tensor<1x2x2x2x8x2x32x2xf16>

  // wrap into conversion_cast
  %2 = builtin.unrealized_conversion_cast %pack_1
           : tensor<1x2x2x2x8x2x32x2xf16> to tensor<1x15x7x59xf16>

  // pad, reshape and transpose filter
  %padded = tensor.pad %arg1 low[0, 0, 0, 0] high[5, 0, 0, 5] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst_0 : f16
  } : tensor<59x1x1x59xf16> to tensor<64x1x1x64xf16>
  %reshape = tensor.reshape %padded(%cst)
           : (tensor<64x1x1x64xf16>, tensor<7xindex>) -> tensor<2x32x1x1x2x16x2xf16>
  %3 = tensor.empty() : tensor<2x2x1x1x16x32x2xf16>
  %transposed = linalg.transpose
           ins(%reshape : tensor<2x32x1x1x2x16x2xf16>)
           outs(%3 : tensor<2x2x1x1x16x32x2xf16>) permutation = [0, 4, 2, 3, 5, 1, 6]

  %4 = builtin.unrealized_conversion_cast %transposed : tensor<2x2x1x1x16x32x2xf16> to tensor<59x1x1x59xf16>
  %5 = tensor.empty() : tensor<1x15x7x59xf16>
  %6 = linalg.conv_2d_nhwc_fhwc ins(%2, %4 : tensor<1x15x7x59xf16>, tensor<59x1x1x59xf16>) outs(%5 : tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>
  %7 = builtin.unrealized_conversion_cast %6 : tensor<1x15x7x59xf16> to tensor<1x2x2x2x8x2x32x2xf16>

  // unpack
  %unpack = linalg.unpack %7 inner_dims_pos = [5] inner_tiles = [2] into %0 : tensor<1x2x2x2x8x2x32x2xf16> -> tensor<1x2x2x2x8x4x32xf16>
  %unpack_2 = linalg.unpack %unpack inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %5 : tensor<1x2x2x2x8x4x32xf16> -> tensor<1x15x7x59xf16>
  }];

  let constructor =
       "hexagon::createSeedLayoutConversionsPass()";

}

def LowerHexKLMatmulToMacro : InterfacePass<"lower-hexkl-matmul-to-macro", "mlir::FunctionOpInterface"> {
  let summary = "Lower hexkl.matmul to optimized hexkl_macro_mm_f16 call";
  let description = [{
    This pass lowers hexkl::MatmulOp to a single call to the optimized
    hexkl_macro_mm_f16 library function, handling layout transformations
    for activations (RM→AH) and output (AH→RM). Assumes weights are
    already in WH layout (pre-transformed by PreprocessWeightsForHMX pass).
  }];
  let constructor = "mlir::hexagon::createLowerHexKLMatmulToMacroPass()";
}

def PreprocessWeightsForHMX :   InterfacePass<"preprocess-weights-hmx", "mlir::FunctionOpInterface"> {
  let summary = "Transform constant weights to HMX WH layout at compile time";
  let description = [{
    This pass identifies constant weight tensors used as RHS in matmul operations
    and transforms them from row-major to WH (HMX weight) layout by calling
    hexkl_macro_rm_to_wh_f16_inplace at compile time.
  }];
  let constructor = "mlir::hexagon::createPreprocessWeightsForHMXPass()";
}

def ForceHVXCrouton :
    InterfacePass<"force-hvx-crouton", "mlir::FunctionOpInterface"> {
  let summary = "Force HVX Crouton mode by annotating functions";
  let constructor = "hexagon::createForceHVXCroutonPass()";
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

def FoldPackUnpackConstants :
    Pass<"fold-pack-unpack-constants"> {
  let summary = "Fold pack/unpack/transpose/reshape/pad operations on constant tensors";
  let description = [{
    This pass performs constant folding for tensor operations:
    1. Evaluates pack/unpack operations on constant tensors at compile time
    2. Evaluates transpose operations on constant tensors at compile time
    3. Evaluates expand_shape/collapse_shape operations on constant tensors
    4. Evaluates reshape operations on constant tensors at compile time
    5. Evaluates pad operations on constant tensors at compile time
    6. Replaces the operations with the computed constant result

    This eliminates runtime overhead for transforming constant weights
    and other constant tensors.

    Example:
      %cst = arith.constant dense<...> : tensor<768xf16>
      %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [32]
                into %dest : tensor<768xf16> -> tensor<24x32xf16>

    Becomes:
      %packed = arith.constant dense<...> : tensor<24x32xf16>
  }];
  let constructor = "hexagon::createFoldPackUnpackConstantsPass()";
}

def EliminateRedundantUnpackPack :
    Pass<"eliminate-redundant-unpack-pack"> {
  let summary = "Eliminate redundant unpack/pack pairs";
  let description = [{
    This pass eliminates redundant unpack/pack pairs when:
    1. An unpack operation's result is consumed by a pack operation
    2. Both operations have the same inner_dims_pos and inner_tiles
    3. Both operations have the same outer_dims_perm
    4. The tensor type of unpack's source matches pack's destination type

    Pattern:
      %unpacked = linalg.unpack %packed inner_dims_pos = [...] inner_tiles = [...]
      %repacked = linalg.pack %unpacked padding_value(%pad) inner_dims_pos = [...]
                  inner_tiles = [...]
    
    The tensor type check (unpack's source type == pack's dest type) is a
    NECESSARY but NOT SUFFICIENT condition for correctness. It ensures structural
    compatibility but does NOT guarantee that the padding values added by pack
    match the values discarded by unpack.

    IMPORTANT LIMITATION:
    This pass currently does NOT verify that the padding value used by pack
    matches the values being discarded by unpack. This verification requires
    data flow analysis to prove that the padded values discarded by unpack
    are the same as the pack's padding value.

    Users of this pass must ensure this property holds in their use case, or
    accept that the optimization may be unsound in general. Future improvements
    will add data flow analysis to automatically verify this property.
  }];
  let constructor = "hexagon::createEliminateRedundantUnpackPackPass()";
}

def DivToMulOptimization : Pass<"div-to-mul-optimization", "func::FuncOp"> {
  let summary = "Optimize division by broadcasted values to multiplication";
  let description = [{
    This pass optimizes division operations in linalg.generic when the divisor
    is previously broadcasted (via linalg.fill or linalg.generic broadcast),
    by converting them to multiplications with the reciprocal.

    Transforms:
    - x / fill(y) -> x * fill(1/y)
    - x / broadcast(y) -> x * broadcast(1/y)

    This optimization is beneficial as multiplication is typically faster than
    division on hardware accelerators.
  }];
  let constructor = "hexagon::createDivToMulOptimizationPass()";
}

def SCFLoopUnroll : Pass<"scf-loop-unroll", "func::FuncOp"> {
  let summary = "Unroll SCF for loops with constant bounds";
  let description = [{
    This pass unrolls the innermost scf.for loops which satisfy
    checks and constraints. This is complementary to LLVM level
    loop unrolling. The pass creates a new loop with a larger
    step size (original_step * unroll_factor)
    and clones the loop body unroll_factor times.
  }];
  let options = [
    Option<"unrollFactor", "unroll-factor", "int64_t",
           /*default=*/"2",
           "Factor by which to unroll loops">
  ];
  let constructor = "hexagon::createSCFLoopUnrollPass()";
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
    Option<"kvCacheOnly", "kv-cache-only", "bool",
           /*default=*/"false",
           "Insert only attention K/V prefetch and skip generic/HexKL loop prefetch.">,
    Option<"enableDequantReshape", "enable-dequant-reshape", "bool",
           /*default=*/"false",
           "Use the W8 compressed cache and fuse dequantization with WH production.">,
    Option<"enableAlpsFusedTransformTransfer", "enable-alps-fused-transform-transfer", "bool",
           /*default=*/"false",
           "ALPS P2c strict mode: process HexKL micro loops only and annotate synchronous fused transform-transfer sites.">,
    Option<"requireAlpsAdmission", "require-alps-admission", "bool",
           /*default=*/"false",
           "Emit a K/V L2 hint only when P2d selected alps.p2d.action=l2_hint.">,
    Option<"enableAlpsExactOverlap", "enable-alps-exact-overlap", "bool",
           /*default=*/"false",
           "ALPS P3b: emit descriptor-bound one-tile HMX weight DMA/scout overlap.">,
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
// External prefetching baseline: Prefetch-Kernel-HX
//===----------------------------------------------------------------------===//

def PrefetchKernelHX :
    InterfacePass<"prefetch-kernel-hx", "mlir::FunctionOpInterface"> {
  let summary = "Insert fixed-distance Hexagon L2 hints from safe static address kernels";
  let description = [{
    Implements the load-free affine/2-D subset of the software prefetcher from
    "Classifying Memory Access Patterns for Prefetching".  This is an external
    baseline, not an OmniFetch cumulative component: it reconstructs a future
    memref.subview from the loop induction value and emits only
    omni_fetch.l2_hint.  It never allocates shadow storage, rewires consumers,
    performs layout conversion, uses DMA/VTCM, or enables V-DAE.
  }];
  let options = [
    Option<"distance", "distance", "int",
           /*default=*/"1",
           "Fixed future-iteration distance for the address-only baseline.">,
    Option<"maxCommandBytes", "max-command-bytes", "int",
           /*default=*/"8191",
           "Maximum logical bytes admitted for one V73 L2FETCH request.">,
    Option<"enableTwoDimensional", "enable-two-dimensional", "bool",
           /*default=*/"true",
           "Admit statically strided two-dimensional tile kernels.">,
    Option<"baselineKind", "baseline-kind", "std::string",
           /*default=*/"\"prefetch-kernel-hx\"",
           "Ledger/hint identity: prefetch-kernel-hx or apt-get-hx.">,
    Option<"requireManualSafe", "require-manual-safe", "bool",
           /*default=*/"false",
           "Require prefetch_baseline.manual_safe=true on the view or loop.">,
    Option<"manualCandidateIds", "manual-candidate-ids", "std::string",
           /*default=*/"\"\"",
           "Comma-separated stable candidate IDs admitted by APT-GET-HX.">,
  ];
  let constructor = "hexagon::createPrefetchKernelHXPass()";
  let dependentDialects = [
    "mlir::omni_fetch::OmniFetchDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ALPS P1: analysis-only representation and physical-movement ledger
//===----------------------------------------------------------------------===//

def AlpsMovementLedger :
    InterfacePass<"alps-movement-ledger", "mlir::FunctionOpInterface"> {
  let summary = "Report representation demands and physical data movement without changing IR";
  let description = [{
    Emits machine-readable ALPS P1 ledger records for shaped values, K/V
    representation candidates, descriptor-only views, layout transforms,
    allocations, and copies.  The pass is analysis-only: it never adds
    attributes, rewrites values, or changes code generation.  Running it at
    pre-fusion, post-fusion, and post-bufferization provides a causal view of
    which logical representation demands become physical movement.
  }];
  let options = [
    Option<"phase", "phase", "std::string",
           /*default=*/"\"unspecified\"",
           "Pipeline observation point written into every ledger record.">,
    Option<"pageBytes", "page-bytes", "int64_t",
           /*default=*/"4096",
           "Page size used only for static page-footprint estimates.">,
    Option<"vtcmBudgetBytes", "vtcm-budget-bytes", "int64_t",
           /*default=*/"2097152",
           "Conservative per-candidate VTCM-fit threshold for legal-action reporting.">,
  ];
  let constructor = "hexagon::createAlpsMovementLedgerPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ALPS P2d: conservative static representation admission
//===----------------------------------------------------------------------===//

def AlpsMinimalStaticAdmission :
    InterfacePass<"alps-minimal-static-admission", "mlir::FunctionOpInterface"> {
  let summary = "Select a conservative movement action and emit an auditable decision ledger";
  let description = [{
    Consumes the representation metadata established by ALPS P0/P1/P2 and
    selects only among native/no-op, page-safe L2 hint, synchronous in-situ,
    and DMA/VTCM asynchronous actions.  L2 hints may be materialized
    immediately.  Synchronous in-situ requires proven eliminated physical
    bytes, while DMA/VTCM remains rejected until P3 provides exact readiness
    and slot ownership.  The pass never performs a global search and records
    every admission/rejection as stable operation attributes plus a ledger.
  }];
  let options = [
    Option<"pageBytes", "page-bytes", "int64_t",
           /*default=*/"4096",
           "Page size used for page-footprint and safe L2 request accounting.">,
    Option<"vtcmBudgetBytes", "vtcm-budget-bytes", "int64_t",
           /*default=*/"2097152",
           "Conservative per-function VTCM capacity available to admitted tiles.">,
    Option<"minL2Bytes", "min-l2-bytes", "int64_t",
           /*default=*/"4096",
           "Reject L2 hints smaller than this many logical bytes.">,
    Option<"minDmaBytes", "min-dma-bytes", "int64_t",
           /*default=*/"4096",
           "Reject DMA/VTCM candidates smaller than this many bytes.">,
    Option<"minOverlapOps", "min-overlap-ops", "int64_t",
           /*default=*/"8",
           "Minimum static producer-to-consumer/loop window for an early action.">,
    Option<"enableP3ExactReadiness", "enable-p3-exact-readiness", "bool",
           /*default=*/"false",
           "Allow a legal HMX weight candidate to select DMA/VTCM async after P3a.">,
  ];
  let constructor = "hexagon::createAlpsMinimalStaticAdmissionPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ALPS P3a: exact-readiness descriptor contract
//===----------------------------------------------------------------------===//

def AlpsExactReadiness :
    InterfacePass<"alps-exact-readiness", "mlir::FunctionOpInterface"> {
  let summary = "Audit admitted async prefetches for exact descriptor identity";
  let description = [{
    Establishes the compiler half of the P3a contract.  Only a P2d-admitted
    DMA/VTCM prefetch with an enclosing tile loop and explicit tile operands
    receives exact-readiness metadata.  Runtime state and lowering are
    generation-safe; this pass does not start DMA, overlap work, or silently
    fall back to the legacy process-global descriptor ring.
  }];
  let constructor = "hexagon::createAlpsExactReadinessPass()";
  let dependentDialects = [
    "mlir::omni_fetch::OmniFetchDialect",
    "mlir::scf::SCFDialect",
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

//===----------------------------------------------------------------------===//
// ALPS P2a: strict zero-copy attention layout absorption
//===----------------------------------------------------------------------===//

def AlpsZeroCopyAttention :
    InterfacePass<"alps-zero-copy-attention", "mlir::FunctionOpInterface"> {
  let summary = "Absorb proven attention transpose chains into consumer maps";
  let description = [{
    Rewrites only statically proven batch-one QK attention chains of the form
    expand -> transpose -> collapse -> batch_matmul into an equivalent
    linalg.generic contraction whose affine input maps directly address the
    producer layout. Descriptor-only views are not counted as saved bytes; only
    the two eliminated physical transpose results are reported.
  }];
  let constructor = "hexagon::createAlpsZeroCopyAttentionPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def AlpsProducerDirectAttention :
    InterfacePass<"alps-producer-direct-attention", "mlir::FunctionOpInterface"> {
  let summary = "Write proven attention head layouts directly from producers";
  let description = [{
    Fuses a sole-use projection bias-add with a statically proven
    expand/transpose chain. The replacement producer writes contiguous BHMD or
    BHDM directly, eliminating the canonical BMC materialization while keeping
    consumer access contiguous.
  }];
  let constructor = "hexagon::createAlpsProducerDirectAttentionPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def AlpsConsumerDrivenLayout :
    InterfacePass<"alps-consumer-driven-layout", "mlir::FunctionOpInterface"> {
  let summary = "Form a consumer-required layout directly in a proven producer";
  let description = [{
    Builds an auditable future-representation contract for physical transpose
    results and their terminal HVX/HMX consumers.  The first conservative
    rewrite retargets a sole-use, all-parallel tensor producer directly to the
    consumer-required permutation only when the innermost contiguous dimension
    is preserved.  Unsupported, mixed-engine, dynamic, or ambiguous cases are
    retained as native layout materializations.
  }];
  let constructor = "hexagon::createAlpsConsumerDrivenLayoutPass()";
  let options = [
    Option<"propagateCodegenContract", "propagate-codegen-contract", "bool",
           /*default=*/"false",
           "ALPS P2f: preserve admitted consumer layout contracts through "
           "HVX tiling and vectorization">,
    Option<"emitDischargeContracts", "emit-discharge-contracts", "bool",
           /*default=*/"false",
           "ALPS P5a: assign stable IDs and origin fingerprints to admitted "
           "consumer-layout contracts">,
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def AlpsContractDischargeLedger :
    InterfacePass<"alps-contract-discharge-ledger", "mlir::FunctionOpInterface"> {
  let summary = "Audit ALPS layout-contract survival at a pipeline boundary";
  let description = [{
    Reads the stable P5a contract table recorded on a function and classifies
    each contract as explicitly surviving, represented by an operation whose
    fused location still contains the origin, or no longer traceable at the
    selected compiler boundary. This pass is analysis-only.
  }];
  let constructor = "hexagon::createAlpsContractDischargeLedgerPass()";
  let options = [
    Option<"phase", "phase", "std::string", /*default=*/"\"unknown\"",
           "Pipeline boundary reported by the ledger">,
    Option<"analyzeInputs", "analyze-inputs", "bool", /*default=*/"false",
           "ALPS P5b: analyze supply candidates feeding surviving contracts">,
    Option<"minLeadOps", "min-lead-ops", "int64_t", /*default=*/"4",
           "Minimum producer-to-consumer operation distance for supply">,
    Option<"minBytes", "min-bytes", "int64_t", /*default=*/"128",
           "Minimum static contiguous input bytes for supply">,
  ];
}

def AlpsLayoutSupplyPrefetch :
    InterfacePass<"alps-layout-supply-prefetch", "mlir::FunctionOpInterface"> {
  let summary = "Prefetch the next tile feeding a discharged layout contract";
  let description = [{
    ALPS P5c combines consumer-driven in-situ layout formation with a strictly
    admitted loop-local supply action. It emits a destination-free L2 hint for
    the next read-only, contiguous tile only when the final vector read can be
    traced to a static subview of storage invariant in the selected loop. The
    current tile remains directly formed in the consumer representation; this
    pass never recreates a canonical intermediate or a transpose.
  }];
  let constructor = "hexagon::createAlpsLayoutSupplyPrefetchPass()";
  let options = [
    Option<"distance", "distance", "int64_t", /*default=*/"1",
           "Number of loop iterations to prefetch ahead">,
    Option<"maxBytes", "max-bytes", "int64_t", /*default=*/"65535",
           "Maximum bytes in one admitted tile hint">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::omni_fetch::OmniFetchDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
}

#endif
