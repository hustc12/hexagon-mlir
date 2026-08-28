//===- FoldResourceTransposePass.cpp - Fold Transpose of Resources --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

struct FoldResourceTranspose
    : public mlir::OpRewritePattern<mlir::linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value input = op.getDpsInputOperand(0)->get();

    // A compile-time linear collapse does not move data.  P5i intentionally
    // flattens FCHW weights to [OC, IC*KH*KW] before transposing them, so look
    // through exactly that static view and use its rank-2 shape for the fold.
    mlir::Value resourceInput = input;
    if (auto collapse = input.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
      auto collapsedType =
          mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      auto sourceType =
          mlir::dyn_cast<mlir::RankedTensorType>(collapse.getSrcType());
      if (!collapsedType || collapsedType.getRank() != 2 ||
          !collapsedType.hasStaticShape() || !sourceType ||
          !sourceType.hasStaticShape() ||
          collapsedType.getNumElements() != sourceType.getNumElements())
        return mlir::failure();
      resourceInput = collapse.getSrc();
    }

    auto constOp = resourceInput.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp)
      return mlir::failure();

    auto resourceAttr =
        mlir::dyn_cast<mlir::DenseResourceElementsAttr>(constOp.getValue());
    if (!resourceAttr)
      return mlir::failure();

    // Validate Type Rank 2, F16
    auto resourceType =
        mlir::cast<mlir::RankedTensorType>(resourceAttr.getType());
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!type || type.getRank() != 2 || !type.getElementType().isF16() ||
        !resourceType.getElementType().isF16() ||
        type.getNumElements() != resourceType.getNumElements())
      return mlir::failure();

    auto blob = resourceAttr.getRawHandle().getBlob();
    if (!blob)
      return mlir::failure();

    llvm::ArrayRef<char> rawData = blob->getData();
    if (rawData.size() != type.getNumElements() * 2) // 2 bytes per f16
      return mlir::failure();

    // Perform the Transpose
    int64_t rows = type.getDimSize(0);
    int64_t cols = type.getDimSize(1);

    std::vector<uint16_t> transposedData(type.getNumElements());

    const uint16_t *src = reinterpret_cast<const uint16_t *>(rawData.data());
    uint16_t *dst = transposedData.data();

    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        dst[c * rows + r] = src[r * cols + c];
      }
    }

    // Create a new Resource Blob and replace const op
    size_t byteSize = transposedData.size() * sizeof(uint16_t);
    size_t alignment = alignof(uint16_t);

    char *buffer = static_cast<char *>(std::malloc(byteSize));
    if (!buffer)
      return mlir::failure();
    std::memcpy(buffer, transposedData.data(), byteSize);

    auto deleter = [](void *ptr, size_t, size_t) { std::free(ptr); };

    auto newBlob = mlir::AsmResourceBlob(llvm::ArrayRef<char>(buffer, byteSize),
                                         alignment, deleter,
                                         /*dataIsMutable=*/false);

    llvm::SmallVector<int64_t> newShape = {cols, rows};
    auto newType = mlir::RankedTensorType::get(newShape, type.getElementType());

    std::string newHandleName =
        (resourceAttr.getRawHandle().getKey() + "_transposed").str();
    auto newResourceAttr = mlir::DenseResourceElementsAttr::get(
        newType, newHandleName, std::move(newBlob));

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, newResourceAttr);

    return mlir::success();
  }
};

struct FoldResourceTransposePass
    : public mlir::PassWrapper<FoldResourceTransposePass,
                               mlir::OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldResourceTransposePass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldResourceTranspose>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const final {
    return "fold-resource-transpose";
  }

  mlir::StringRef getDescription() const final {
    return "Fold transpose of f16 dense resource constants";
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createFoldResourceTransposePass() {
  return std::make_unique<FoldResourceTransposePass>();
}

} // namespace hexagon
} // namespace mlir
