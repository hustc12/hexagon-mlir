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

#define GEN_PASS_DEF_OMNIFETCHTOLLVM
#include "hexagon/Conversion/OmniFetchToLLVM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Extract the aligned data pointer from a lowered MemRef descriptor,
/// advanced by the descriptor's offset (in elements).  Subviews produced by
/// PrefetchInsert are offset into a parent buffer; ignoring the offset made
/// the runtime read/write the wrong address (DSP Bad VA / exit 13).
static Value alignedPtrWithOffset(ConversionPatternRewriter &rewriter,
                                  Location loc, Value memrefDesc,
                                  Type elementType) {
  MemRefDescriptor desc(memrefDesc);
  Value ptr = desc.alignedPtr(rewriter, loc);
  Value offset = desc.offset(rewriter, loc);
  // GEP by element offset.
  return rewriter.create<LLVM::GEPOp>(
      loc, ptr.getType(), elementType, ptr, ValueRange{offset});
}

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
  //  index_map_ptr, tile_row, tile_col, src_cols, act_off, scr_off, src_rows)
  SmallVector<Type, 13> argTys = {ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty,
                                  ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty,
                                  i32Ty};
  return LLVM::lookupOrCreateFn(rewriter, module, getPrefetchInSituFnName(),
                                argTys, voidTy);
}

/// Get or insert `void __omni_fetch_l2_hint_2d(ptr, i32, i32, i32)`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertL2Hint2D(ModuleOp module, ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module, getL2Hint2DFnName(),
                                {ptrTy, i32Ty, i32Ty, i32Ty}, voidTy);
}

/// Get or insert `void __omni_fetch_copy2d(...)`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertCopy2D(ModuleOp module, ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module.getContext();
  auto ptrTy  = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty  = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  // (src, dest, elem_bytes, rows, cols, src_row_stride, dst_row_stride)
  SmallVector<Type, 7> argTys = {ptrTy, ptrTy, i32Ty, i32Ty,
                                 i32Ty, i32Ty, i32Ty};
  return LLVM::lookupOrCreateFn(rewriter, module, getCopy2DFnName(), argTys,
                                voidTy);
}

/// Get or insert `i32 __omni_fetch_update_distance(i32)`.
static FailureOr<LLVM::LLVMFuncOp>
getOrInsertUpdateDistance(ModuleOp module,
                          ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module, getUpdateDistanceFnName(),
                                {i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertInvocationBegin(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module, getInvocationBeginFnName(),
                                {}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertInvocationEnd(ModuleOp module,
                         ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module, getInvocationEndFnName(),
                                {i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertDescriptorAcquire(ModuleOp module,
                             ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  auto i64Ty = IntegerType::get(module.getContext(), 64);
  return LLVM::lookupOrCreateFn(
      rewriter, module, getDescriptorAcquireFnName(),
      {i32Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertDescriptorTransition(ModuleOp module,
                                ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                getDescriptorTransitionFnName(),
                                {i32Ty, i32Ty, i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertDescriptorConsume(ModuleOp module,
                             ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  auto i64Ty = IntegerType::get(module.getContext(), 64);
  return LLVM::lookupOrCreateFn(
      rewriter, module, getDescriptorConsumeFnName(),
      {i32Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertDescriptorRelease(ModuleOp module,
                             ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                getDescriptorReleaseFnName(), {i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertExactWeightKick(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, getExactWeightKickFnName(),
      {i32Ty, i64Ty, ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty},
      i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertExactWeightConsume(ModuleOp module,
                              ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  auto i64Ty = IntegerType::get(module.getContext(), 64);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                getExactWeightConsumeFnName(),
                                {i32Ty, i64Ty, i32Ty, i32Ty}, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getOrInsertExactWeightRelease(ModuleOp module,
                              ConversionPatternRewriter &rewriter) {
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  auto i64Ty = IntegerType::get(module.getContext(), 64);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                getExactWeightReleaseFnName(),
                                {i32Ty, i64Ty, i32Ty, i32Ty}, i32Ty);
}

static Value integerCast(ConversionPatternRewriter &rewriter, Location loc,
                         Value value, unsigned width) {
  auto source = cast<IntegerType>(value.getType());
  auto target = rewriter.getIntegerType(width);
  if (source.getWidth() == width)
    return value;
  if (source.getWidth() > width)
    return rewriter.create<LLVM::TruncOp>(loc, target, value);
  return rewriter.create<LLVM::SExtOp>(loc, target, value);
}

static Value runtimeSuccess(ConversionPatternRewriter &rewriter, Location loc,
                            Value result) {
  Value zero = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
  return rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, result,
                                       zero);
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

struct LowerInvocationBegin
    : public ConvertOpToLLVMPattern<InvocationBeginOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(InvocationBeginOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn = getOrInsertInvocationBegin(op->getParentOfType<ModuleOp>(),
                                         rewriter);
    if (failed(fn))
      return failure();
    Value handle =
        rewriter.create<LLVM::CallOp>(op.getLoc(), *fn, ValueRange{})
            .getResult();
    Type indexTy = typeConverter->convertType(rewriter.getIndexType());
    auto width = cast<IntegerType>(indexTy).getWidth();
    rewriter.replaceOp(op, integerCast(rewriter, op.getLoc(), handle, width));
    return success();
  }
};

struct LowerInvocationEnd : public ConvertOpToLLVMPattern<InvocationEndOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(InvocationEndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn =
        getOrInsertInvocationEnd(op->getParentOfType<ModuleOp>(), rewriter);
    if (failed(fn))
      return failure();
    Value handle = integerCast(rewriter, op.getLoc(), adaptor.getContext(), 32);
    Value result = rewriter
                       .create<LLVM::CallOp>(op.getLoc(), *fn,
                                             ValueRange{handle})
                       .getResult();
    rewriter.replaceOp(op, runtimeSuccess(rewriter, op.getLoc(), result));
    return success();
  }
};

struct LowerDescriptorAcquire
    : public ConvertOpToLLVMPattern<DescriptorAcquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DescriptorAcquireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn = getOrInsertDescriptorAcquire(op->getParentOfType<ModuleOp>(),
                                           rewriter);
    if (failed(fn))
      return failure();
    SmallVector<Value> args = {
        integerCast(rewriter, op.getLoc(), adaptor.getContext(), 32),
        integerCast(rewriter, op.getLoc(), adaptor.getValueVersion(), 64),
        integerCast(rewriter, op.getLoc(), adaptor.getTile(), 64),
        adaptor.getLayout(), adaptor.getSourceTier(),
        adaptor.getDestinationTier()};
    Value handle =
        rewriter.create<LLVM::CallOp>(op.getLoc(), *fn, args).getResult();
    Type indexTy = typeConverter->convertType(rewriter.getIndexType());
    auto width = cast<IntegerType>(indexTy).getWidth();
    rewriter.replaceOp(op, integerCast(rewriter, op.getLoc(), handle, width));
    return success();
  }
};

struct LowerDescriptorTransition
    : public ConvertOpToLLVMPattern<DescriptorTransitionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DescriptorTransitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn = getOrInsertDescriptorTransition(op->getParentOfType<ModuleOp>(),
                                              rewriter);
    if (failed(fn))
      return failure();
    Value result = rewriter
                       .create<LLVM::CallOp>(
                           op.getLoc(), *fn,
                           ValueRange{integerCast(rewriter, op.getLoc(),
                                                  adaptor.getDescriptor(), 32),
                                      adaptor.getExpectedState(),
                                      adaptor.getNextState()})
                       .getResult();
    rewriter.replaceOp(op, runtimeSuccess(rewriter, op.getLoc(), result));
    return success();
  }
};

struct LowerDescriptorConsume
    : public ConvertOpToLLVMPattern<DescriptorConsumeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DescriptorConsumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn = getOrInsertDescriptorConsume(op->getParentOfType<ModuleOp>(),
                                           rewriter);
    if (failed(fn))
      return failure();
    SmallVector<Value> args = {
        integerCast(rewriter, op.getLoc(), adaptor.getDescriptor(), 32),
        integerCast(rewriter, op.getLoc(), adaptor.getValueVersion(), 64),
        integerCast(rewriter, op.getLoc(), adaptor.getTile(), 64),
        adaptor.getLayout(), adaptor.getSourceTier(),
        adaptor.getDestinationTier()};
    Value result =
        rewriter.create<LLVM::CallOp>(op.getLoc(), *fn, args).getResult();
    rewriter.replaceOp(op, runtimeSuccess(rewriter, op.getLoc(), result));
    return success();
  }
};

struct LowerDescriptorRelease
    : public ConvertOpToLLVMPattern<DescriptorReleaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(DescriptorReleaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn = getOrInsertDescriptorRelease(op->getParentOfType<ModuleOp>(),
                                           rewriter);
    if (failed(fn))
      return failure();
    Value handle =
        integerCast(rewriter, op.getLoc(), adaptor.getDescriptor(), 32);
    Value result = rewriter
                       .create<LLVM::CallOp>(op.getLoc(), *fn,
                                             ValueRange{handle})
                       .getResult();
    rewriter.replaceOp(op, runtimeSuccess(rewriter, op.getLoc(), result));
    return success();
  }
};

struct LowerExactWeightKick
    : public ConvertOpToLLVMPattern<ExactWeightKickOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ExactWeightKickOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fn = getOrInsertExactWeightKick(op->getParentOfType<ModuleOp>(),
                                         rewriter);
    if (failed(fn))
      return failure();
    auto srcTy = cast<MemRefType>(op.getSrc().getType());
    auto destTy = cast<MemRefType>(op.getDest().getType());
    Value src = alignedPtrWithOffset(rewriter, op.getLoc(), adaptor.getSrc(),
                                     srcTy.getElementType());
    Value dest = alignedPtrWithOffset(rewriter, op.getLoc(), adaptor.getDest(),
                                      destTy.getElementType());
    // The C ABI accepts generic pointers.  VTCM memrefs lower to AS1, so the
    // descriptor-bound path needs the same explicit cast as PrefetchInSitu.
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    if (src.getType() != ptrTy)
      src = LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), ptrTy, src);
    if (dest.getType() != ptrTy)
      dest = LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), ptrTy, dest);
    SmallVector<Value> args = {
        integerCast(rewriter, op.getLoc(), adaptor.getContext(), 32),
        integerCast(rewriter, op.getLoc(), adaptor.getValueVersion(), 64),
        src, dest, adaptor.getTileRow(), adaptor.getTileCol(),
        adaptor.getSourceCols(), adaptor.getWeightOffset(),
        adaptor.getStageOffset()};
    Value result =
        rewriter.create<LLVM::CallOp>(op.getLoc(), *fn, args).getResult();
    rewriter.replaceOp(op, runtimeSuccess(rewriter, op.getLoc(), result));
    return success();
  }
};

template <typename OpTy, bool IsRelease>
struct LowerExactWeightTerminal : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LLVM::LLVMFuncOp> fn =
        IsRelease
            ? getOrInsertExactWeightRelease(op->template getParentOfType<ModuleOp>(),
                                            rewriter)
            : getOrInsertExactWeightConsume(op->template getParentOfType<ModuleOp>(),
                                            rewriter);
    if (failed(fn))
      return failure();
    SmallVector<Value> args = {
        integerCast(rewriter, op.getLoc(), adaptor.getContext(), 32),
        integerCast(rewriter, op.getLoc(), adaptor.getValueVersion(), 64),
        adaptor.getTileRow(), adaptor.getTileCol()};
    Value result =
        rewriter.create<LLVM::CallOp>(op.getLoc(), *fn, args).getResult();
    rewriter.replaceOp(op, runtimeSuccess(rewriter, op.getLoc(), result));
    return success();
  }
};

using LowerExactWeightConsume =
    LowerExactWeightTerminal<ExactWeightConsumeOp, false>;
using LowerExactWeightRelease =
    LowerExactWeightTerminal<ExactWeightReleaseOp, true>;

//===----------------------------------------------------------------------===//
// PrefetchInSituOp  →  void __omni_fetch_prefetch_insitu(…)
//                 or   void __omni_fetch_copy2d(…) for rank-2 LAYOUT_NONE
//
// The runtime signature:
//   void __omni_fetch_prefetch_insitu(
//       const void *src, void *dest,
//       int32_t elem_bytes, int32_t num_elems,
//       int32_t layout_kind, int32_t lookahead,
//       const int32_t *index_map,  // NULL for non-Custom
//       int32_t tile_row, int32_t tile_col, int32_t src_cols,
//       int32_t act_off, int32_t scr_off, int32_t src_rows);
//===----------------------------------------------------------------------===//
struct LowerPrefetchInSitu
    : public ConvertOpToLLVMPattern<PrefetchInSituOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PrefetchInSituOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    MLIRContext *ctx = rewriter.getContext();
    auto i32Ty  = IntegerType::get(ctx, 32);
    auto ptrTy  = LLVM::LLVMPointerType::get(ctx);

    // --- src / dest pointers (aligned + memref offset) ---
    auto srcMemrefTy = cast<MemRefType>(op.getSrc().getType());
    auto destMemref = cast<MemRefType>(op.getDest().getType());
    // HexKL HMX* fusion writes into the full i8 VTCM slab with absolute byte
    // offsets — match HexKLToLLVM (alignedPtr, no descriptor offset).
    const bool hexklActFusion =
        op.getLayoutTransform() == LayoutTransform::HMXActivation &&
        op.getTileParams().size() >= 6;
    const bool hexklWeightSlab =
        (op.getLayoutTransform() == LayoutTransform::HMXWeight ||
         op.getLayoutTransform() == LayoutTransform::HMXWeightDequantI8) &&
        op.getTileParams().size() >= 4;
    Value srcPtr;
    Value destPtr;
    if (hexklActFusion || hexklWeightSlab) {
      srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());
      destPtr = alignedPtr(rewriter, loc, adaptor.getDest());
    } else {
      srcPtr = alignedPtrWithOffset(rewriter, loc, adaptor.getSrc(),
                                    srcMemrefTy.getElementType());
      destPtr = alignedPtrWithOffset(rewriter, loc, adaptor.getDest(),
                                     destMemref.getElementType());
    }

    // Cast to generic address space (0) if needed using proper LLVM addrspacecast
    if (srcPtr.getType() != ptrTy) {
      srcPtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, srcPtr);
    }
    if (destPtr.getType() != ptrTy) {
      destPtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, destPtr);
    }

    // --- element byte-size (from dest element type) ---
    int64_t elemBytes =
        destMemref.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemBytesVal =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(elemBytes)));

    auto cI32 = [&](int32_t v) {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(v));
    };

    // Rank-2 LAYOUT_NONE: use stride-aware copy2d.  Inner-dim tiles produce
    // strided src subviews; a flat num_elems memcpy would OOB / Bad VA.
    if (op.getLayoutTransform() == LayoutTransform::None &&
        srcMemrefTy.getRank() == 2 && destMemref.getRank() == 2 &&
        srcMemrefTy.hasStaticShape() && destMemref.hasStaticShape()) {
      int64_t rows = srcMemrefTy.getShape()[0];
      int64_t cols = srcMemrefTy.getShape()[1];
      if (destMemref.getShape()[0] != rows ||
          destMemref.getShape()[1] != cols) {
        return rewriter.notifyMatchFailure(op, "src/dest tile shape mismatch");
      }

      auto rowStrideOr = [](MemRefType t) -> std::optional<int64_t> {
        int64_t offset;
        SmallVector<int64_t> strides;
        if (failed(t.getStridesAndOffset(strides, offset)))
          return std::nullopt;
        if (strides.size() != 2 || strides[1] != 1)
          return std::nullopt;
        return strides[0];
      };
      auto srcStride = rowStrideOr(srcMemrefTy);
      auto dstStride = rowStrideOr(destMemref);
      if (!srcStride || !dstStride)
        return rewriter.notifyMatchFailure(
            op, "expected row-major rank-2 memrefs with unit inner stride");

      auto fnOrErr = getOrInsertCopy2D(module, rewriter);
      if (failed(fnOrErr))
        return failure();

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, *fnOrErr,
          ValueRange{srcPtr, destPtr, elemBytesVal, cI32((int32_t)rows),
                     cI32((int32_t)cols), cI32((int32_t)*srcStride),
                     cI32((int32_t)*dstStride)});
      return success();
    }

    auto fnOrErr = getOrInsertPrefetchInSitu(module, rewriter);
    if (failed(fnOrErr))
      return failure();

    // --- total element count: min(src, dest) static volumes ---
    auto staticVolume = [](MemRefType t) -> int64_t {
      if (!t.hasStaticShape())
        return -1;
      int64_t n = 1;
      for (auto d : t.getShape())
        n *= d;
      return n;
    };
    int64_t srcElems = staticVolume(srcMemrefTy);
    int64_t dstElems = staticVolume(destMemref);
    int64_t numElems = dstElems;
    if (srcElems > 0 && (numElems < 0 || srcElems < numElems))
      numElems = srcElems;
    if (numElems < 0)
      numElems = 0;
    // HexKL HMXWeight with tile_params uses the full matrix as src; volume is
    // one 32×32 f16 tile (not the whole matrix / i8 VTCM slab).
    if ((op.getLayoutTransform() == LayoutTransform::HMXWeight ||
         op.getLayoutTransform() == LayoutTransform::HMXWeightDequantI8) &&
        op.getTileParams().size() >= 3) {
      numElems = 1024;
      elemBytesVal = cI32(2);
    }
    // HexKL HMXActivation writes into the i8 VTCM slab; force one 32×32 tile.
    if (op.getLayoutTransform() == LayoutTransform::HMXActivation &&
        op.getTileParams().size() >= 6) {
      numElems = 1024;
      elemBytesVal = cI32(2);
    }
    Value numElemsVal = cI32(static_cast<int32_t>(numElems));

    // --- layout kind ---
    Value layoutKindVal =
        cI32(static_cast<int32_t>(op.getLayoutTransform()));

    // --- lookahead ---
    Value lookaheadVal = cI32(op.getLookahead());

    // --- index_map pointer (null unless Custom) ---
    Value indexMapPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);
    (void)op.getIndexMap();

    // --- HexKL tile params (default -1 = unused) ---
    Value tileRow = cI32(-1);
    Value tileCol = cI32(-1);
    Value srcCols = cI32(-1);
    Value actOff = cI32(-1);
    Value scrOff = cI32(-1);
    Value srcRows = cI32(-1);
    auto tileParams = adaptor.getTileParams();
    if (tileParams.size() >= 3) {
      tileRow = tileParams[0];
      tileCol = tileParams[1];
      srcCols = tileParams[2];
    }
    // Weight slab form: [3]=weight_off, optional [4]=VTCM stage_off for DMA.
    if (tileParams.size() == 4 || tileParams.size() == 5) {
      actOff = tileParams[3];
      if (tileParams.size() == 5)
        scrOff = tileParams[4];
    }
    if (tileParams.size() >= 6) {
      actOff = tileParams[3];
      scrOff = tileParams[4];
      srcRows = tileParams[5];
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, *fnOrErr,
        ValueRange{srcPtr, destPtr, elemBytesVal, numElemsVal, layoutKindVal,
                   lookaheadVal, indexMapPtr, tileRow, tileCol, srcCols, actOff,
                   scrOff, srcRows});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// L2HintOp → read-only __omni_fetch_l2_hint_2d(...)
//===----------------------------------------------------------------------===//
struct LowerL2Hint : public ConvertOpToLLVMPattern<L2HintOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(L2HintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto srcType = cast<MemRefType>(op.getSrc().getType());
    if (!srcType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static source shape");

    Value srcPtr = alignedPtrWithOffset(rewriter, loc, adaptor.getSrc(),
                                        srcType.getElementType());
    if (srcPtr.getType() != ptrTy)
      srcPtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, srcPtr);

    auto cI32 = [&](int32_t value) {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(value));
    };
    int64_t elemBytes =
        srcType.getElementType().getIntOrFloatBitWidth() / 8;
    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset = 0;
    if (failed(srcType.getStridesAndOffset(sourceStrides, sourceOffset)) ||
        sourceStrides.empty() || sourceStrides.back() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected a statically strided source with contiguous inner dim");

    int64_t widthBytes = srcType.getShape().back() * elemBytes;
    int64_t height = 1;
    int64_t strideBytes = widthBytes;
    // A stream subview is [1,...,S,...,1,D]. Select the closest non-unit
    // dimension outside D as the hardware 2-D height and preserve its physical
    // stride. This covers [B,H,S,D] and fused [B,S,H,D] layouts.
    for (int64_t dim = srcType.getRank() - 2; dim >= 0; --dim) {
      if (srcType.getDimSize(dim) > 1) {
        height = srcType.getDimSize(dim);
        strideBytes = sourceStrides[dim] * elemBytes;
        break;
      }
    }

    auto fnOrErr = getOrInsertL2Hint2D(module, rewriter);
    if (failed(fnOrErr))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, *fnOrErr,
        ValueRange{srcPtr, cI32(static_cast<int32_t>(widthBytes)),
                   cI32(static_cast<int32_t>(height)),
                   cI32(static_cast<int32_t>(strideBytes))});
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
    : public ::impl::OmniFetchToLLVMBase<OmniFetchToLLVMPass> {

  using OmniFetchToLLVMBase::OmniFetchToLLVMBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    LLVMTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);

    patterns.add<LowerCreateSem, LowerSignal, LowerWait, LowerInvocationBegin,
                 LowerInvocationEnd, LowerDescriptorAcquire,
                 LowerDescriptorTransition, LowerDescriptorConsume,
                 LowerDescriptorRelease, LowerExactWeightKick,
                 LowerExactWeightConsume, LowerExactWeightRelease,
                 LowerPrefetchInSitu, LowerL2Hint, LowerAdaptiveControl>(
        typeConverter);

    LLVMConversionTarget target(*ctx);
    target.addIllegalDialect<OmniFetchDialect>();
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // After lowering: optionally arm the dual-thread scout for functions that
    // actually use OmniFetch wait/signal.
    if (!enableDualThreadDae)
      return;

    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isDeclaration() || func.empty())
        continue;
      bool usesOmni = false;
      func.walk([&](LLVM::CallOp call) {
        if (auto callee = call.getCallee()) {
          if (callee->starts_with("__omni_fetch_"))
            usesOmni = true;
        }
      });
      if (!usesOmni)
        continue;

      OpBuilder b(&func.front(), func.front().begin());
      Location loc = func.getLoc();
      FailureOr<LLVM::LLVMFuncOp> setFn = LLVM::lookupOrCreateFn(
          b, module, getSetDualThreadDaeFnName(), {i32Ty}, voidTy);
      if (failed(setFn))
        continue;
      Value one = b.create<LLVM::ConstantOp>(loc, i32Ty, b.getI32IntegerAttr(1));
      b.create<LLVM::CallOp>(loc, *setFn, ValueRange{one});
      break; // once per module is enough (global runtime flag)
    }
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public factory
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> mlir::omni_fetch::createOmniFetchToLLVMPass(
    const OmniFetchToLLVMOptions &options) {
  return std::make_unique<OmniFetchToLLVMPass>(options);
}
