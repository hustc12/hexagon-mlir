//===- OmniFetchToLLVMPass.cpp - omni_fetch → LLVM lowering  --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Lowers all omni_fetch dialect ops to extern-C calls into the Hexagon device
// runtime (OmniFetchRuntime.c), keeping the compiler and runtime decoupled.
//
// Each pattern follows the same shape as HexKLToLLVMPass:
//   1. Obtain (or insert) the LLVM function declaration via lookupOrCreateFn.
//   2. Extract aligned pointers from MemRef descriptors where needed.
//   3. Replace the dialect op with an llvm.call.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/OmniFetchToLLVM/OmniFetchExternalFnNames.h"
#include "hexagon/Conversion/OmniFetchToLLVM/OmniFetchToLLVM.h"
#include "hexagon/Dialect/OmniFetch/IR/OmniFetchDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "omni-fetch-to-llvm"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::omni_fetch;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/OmniFetchToLLVM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Extract the aligned data pointer from a lowered MemRef descriptor.
static Value alignedPtr(ConversionPatternRewriter &rewriter, Location loc,
                        Value memrefDesc) {
  MemRefDescriptor desc(memrefDesc);
  return desc.alignedPtr(rewriter, loc);
}

/// Get or insert `i32 __omni_fetch_create_sem()`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertCreateSem(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module, getCreateSemFnName(),
                                {}, i32Ty);
}

/// Get or insert `void __omni_fetch_signal(i32)`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertSignal(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  auto voidTy = LLVM::LLVMVoidType::get(module.getContext());
  return LLVM::lookupOrCreateFn(rewriter, module, getSignalFnName(),
                                {i32Ty}, voidTy);
}

/// Get or insert `void __omni_fetch_wait(i32)`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertWait(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  auto voidTy = LLVM::LLVMVoidType::get(module.getContext());
  return LLVM::lookupOrCreateFn(rewriter, module, getWaitFnName(),
                                {i32Ty}, voidTy);
}

/// Get or insert the `__omni_fetch_prefetch_insitu` declaration.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertPrefetchInSitu(ModuleOp module,
                          ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module.getContext();
  auto ptrTy  = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty  = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  // (src_ptr, dest_ptr, elem_bytes, num_elems, layout_kind, lookahead,
  //  index_map_ptr)
  SmallVector<Type, 7> argTys = {ptrTy, ptrTy, i32Ty, i32Ty,
                                 i32Ty, i32Ty, ptrTy};
  return LLVM::lookupOrCreateFn(rewriter, module, getPrefetchInSituFnName(),
                                argTys, voidTy);
}

/// Get or insert `i32 __omni_fetch_update_distance(i32)`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertUpdateDistance(ModuleOp module,
                          ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module, getUpdateDistanceFnName(),
                                {i32Ty}, i32Ty);
}

//===----------------------------------------------------------------------===//
// CreateSemOp  →  i32 __omni_fetch_create_sem()
//===----------------------------------------------------------------------===//
struct LowerCreateSem : public ConvertOpToLLVMPattern<CreateSemOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CreateSemOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fnOrErr = getOrInsertCreateSem(module, rewriter);
    if (failed(fnOrErr))
      return failure();

    // Call the runtime; result is i32 semaphore index.
    auto call = rewriter.create<LLVM::CallOp>(
        loc, *fnOrErr, ValueRange{});
    Value semI32 = call.getResult();

    // The dialect op returns `index`; on Hexagon index is i64.
    // Simply sign-extend i32 → i64 (which is the LLVM representation of index).
    Value semIdx = rewriter.create<LLVM::SExtOp>(
        loc, typeConverter->convertType(rewriter.getIndexType()), semI32);

    rewriter.replaceOp(op, semIdx);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SignalOp  →  void __omni_fetch_signal(i32 sem)
//===----------------------------------------------------------------------===//
struct LowerSignal : public ConvertOpToLLVMPattern<SignalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fnOrErr = getOrInsertSignal(module, rewriter);
    if (failed(fnOrErr))
      return failure();

    // Convert index sem_handle → i32
    // After LLVM lowering, index is already i64, so no IndexCastOp needed.
    Value semI64 = adaptor.getSemHandle();
    Value semI32 = rewriter.create<LLVM::TruncOp>(
        loc, rewriter.getI32Type(), semI64);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, *fnOrErr,
                                              ValueRange{semI32});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WaitOp  →  void __omni_fetch_wait(i32 sem)
//===----------------------------------------------------------------------===//
struct LowerWait : public ConvertOpToLLVMPattern<WaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fnOrErr = getOrInsertWait(module, rewriter);
    if (failed(fnOrErr))
      return failure();

    Value semI64 = adaptor.getSemHandle();
    Value semI32 = rewriter.create<LLVM::TruncOp>(
        loc, rewriter.getI32Type(), semI64);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, *fnOrErr,
                                              ValueRange{semI32});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PrefetchInSituOp  →  void __omni_fetch_prefetch_insitu(…)
//
// The runtime signature:
//   void __omni_fetch_prefetch_insitu(
//       const void *src, void *dest,
//       int32_t elem_bytes, int32_t num_elems,
//       int32_t layout_kind, int32_t lookahead,
//       const int32_t *index_map);  // NULL for non-Custom
//===----------------------------------------------------------------------===//
struct LowerPrefetchInSitu
    : public ConvertOpToLLVMPattern<PrefetchInSituOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PrefetchInSituOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fnOrErr = getOrInsertPrefetchInSitu(module, rewriter);
    if (failed(fnOrErr))
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    auto i32Ty  = IntegerType::get(ctx, 32);
    auto ptrTy  = LLVM::LLVMPointerType::get(ctx);

    // --- src / dest aligned pointers ---
    // Extract raw pointers
    Value srcPtr  = alignedPtr(rewriter, loc, adaptor.getSrc());
    Value destPtr = alignedPtr(rewriter, loc, adaptor.getDest());
    
    // Cast to generic address space (0) if needed using proper LLVM addrspacecast
    if (srcPtr.getType() != ptrTy) {
      srcPtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, srcPtr);
    }
    if (destPtr.getType() != ptrTy) {
      destPtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, destPtr);
    }

    // --- element byte-size (from dest element type) ---
    auto destMemref = cast<MemRefType>(op.getDest().getType());
    int64_t elemBytes =
        destMemref.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemBytesVal =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(elemBytes)));

    // --- total element count ---
    int64_t numElems = 1;
    for (auto d : destMemref.getShape())
      numElems *= d;
    Value numElemsVal =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(numElems)));

    // --- layout kind ---
    Value layoutKindVal =
        rewriter.create<LLVM::ConstantOp>(
            loc, i32Ty,
            rewriter.getI32IntegerAttr(
                static_cast<int32_t>(op.getLayoutTransform())));

    // --- lookahead ---
    Value lookaheadVal =
        rewriter.create<LLVM::ConstantOp>(
            loc, i32Ty,
            rewriter.getI32IntegerAttr(op.getLookahead()));

    // --- index_map pointer (null unless Custom) ---
    Value indexMapPtr;
    if (auto idxMap = op.getIndexMap()) {
      // For now, pass NULL and let the runtime use default mapping
      // TODO: Implement proper index map passing via global constant
      // The issue is that LLVM::GlobalOp requires specific attribute format
      // that's not compatible with DenseI32ArrayAttr
      indexMapPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);
    } else {
      // Pass NULL for non-custom layouts.
      indexMapPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, *fnOrErr,
        ValueRange{srcPtr, destPtr, elemBytesVal, numElemsVal,
                   layoutKindVal, lookaheadVal, indexMapPtr});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AdaptiveControlOp  →  i32 __omni_fetch_update_distance(i32)
//===----------------------------------------------------------------------===//
struct LowerAdaptiveControl
    : public ConvertOpToLLVMPattern<AdaptiveControlOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AdaptiveControlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fnOrErr = getOrInsertUpdateDistance(module, rewriter);
    if (failed(fnOrErr))
      return failure();

    auto call = rewriter.create<LLVM::CallOp>(
        loc, *fnOrErr, ValueRange{adaptor.getCurrentDistance()});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//
struct OmniFetchToLLVMPass
    : public OmniFetchToLLVMBase<OmniFetchToLLVMPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    LLVMTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);

    patterns.add<LowerCreateSem, LowerSignal, LowerWait, LowerPrefetchInSitu,
                 LowerAdaptiveControl>(typeConverter);

    LLVMConversionTarget target(*ctx);
    target.addIllegalDialect<OmniFetchDialect>();
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public factory
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> mlir::omni_fetch::createOmniFetchToLLVMPass() {
  return std::make_unique<OmniFetchToLLVMPass>();
}
