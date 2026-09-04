//===- LayoutAwareMapping.cpp - Compile-time layout index computation -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Computes the static index map used by alps.prefetch_in_situ to perform
// in-situ layout reshaping during the DDR→VTCM transfer.
//
// Background
// ----------
// Hexagon HMX (the on-chip matrix accelerator) requires tensors to be stored
// in a tightly packed "deep-interleaved" format where elements from different
// channels are interleaved within each 1024-bit HVX vector register.
// Specifically, for weights of shape [K_out, K_in]:
//
//   HMX physical layout (32 elements / vector):
//     element[vec_idx] = logical[row * 32 + col] where:
//       row = vec_idx / tile_width
//       col = vec_idx % tile_width
//
// When the source tensor in DDR is stored in standard row-major order (NCHW
// or KxC), computing the index map tells the runtime EXACTLY which DDR byte
// to read for each position in the VTCM tile.  The runtime then uses a
// `vgather` loop to fill the VTCM tile in one pass, performing the reshape
// during the transfer rather than in a separate step.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Alps/IR/AlpsDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "alps-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::alps;

namespace {

//===----------------------------------------------------------------------===//
// Helper: row-major flat offset for a multi-dimensional index
//===----------------------------------------------------------------------===//
static int64_t flatOffset(ArrayRef<int64_t> idx, ArrayRef<int64_t> strides) {
  assert(idx.size() == strides.size());
  int64_t off = 0;
  for (size_t i = 0; i < idx.size(); ++i)
    off += idx[i] * strides[i];
  return off;
}

static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

//===----------------------------------------------------------------------===//
// HMX weight tile layout:  [K_out/32][K_in][32]  (deep-interleaved)
//
// The HMX accumulator is fed with 32-element slices along K_out.
// For a weight tile of shape [M, K] (M = #output channels = multiple of 32,
// K = inner dim), the HMX physical order reads:
//
//   for tile_m in range(M // 32):
//     for k in range(K):
//       for m in range(32):
//         dest[tile_m * K * 32 + k * 32 + m]
//           = src[m + tile_m*32][k]   (row-major src)
//
// The index map entry at position `dest_flat` is the corresponding `src_flat`.
//===----------------------------------------------------------------------===//
static SmallVector<int32_t> computeHMXWeightMap(ArrayRef<int64_t> srcShape) {
  assert(srcShape.size() == 2 && "HMX weight map expects 2-D tensor");
  int64_t M = srcShape[0]; // output channels
  int64_t K = srcShape[1]; // inner / input channels

  SmallVector<int32_t> idxMap;
  idxMap.reserve(M * K);

  int64_t tileSize = 32;
  int64_t numTiles = (M + tileSize - 1) / tileSize;

  for (int64_t tile = 0; tile < numTiles; ++tile) {
    for (int64_t k = 0; k < K; ++k) {
      for (int64_t m = 0; m < tileSize; ++m) {
        int64_t srcRow = tile * tileSize + m;
        // Clamp out-of-bounds to last valid row (padding for non-multiple-of-32)
        if (srcRow >= M)
          srcRow = M - 1;
        int32_t srcFlat = static_cast<int32_t>(srcRow * K + k);
        idxMap.push_back(srcFlat);
      }
    }
  }

  DBG("HMX weight map: M=" << M << " K=" << K
                            << " mapSize=" << idxMap.size());
  return idxMap;
}

//===----------------------------------------------------------------------===//
// HMX activation tile layout:  [N][C/32][H][W][32]  (NHWC32)
//
// Hexagon HVX processes 32 channels at a time; activations must be stored
// in channel-last, 32-channel-interleaved format (similar to NHWC with
// inner-channel vectorisation).
//
// Mapping from NCHW [N][C][H][W] src to NHWC32 [N][C/32][H][W][32] dest:
//   src_flat = n*C*H*W + c*H*W + h*W + w
//   dest_flat = n*(C/32)*H*W*32 + (c/32)*H*W*32 + h*W*32 + w*32 + (c%32)
//
// We record the src_flat for every dest_flat position.
//===----------------------------------------------------------------------===//
static SmallVector<int32_t> computeHMXActivationMap(ArrayRef<int64_t> srcShape) {
  // srcShape is [N, C, H, W]; handle common 2-D [batch, channel] as well
  if (srcShape.size() == 2) {
    // Treat as [batch, C] → [batch, C/32, 32]
    int64_t N = srcShape[0];
    int64_t C = srcShape[1];
    int64_t vec = 32;
    SmallVector<int32_t> idxMap;
    idxMap.reserve(N * C);
    for (int64_t n = 0; n < N; ++n)
      for (int64_t cg = 0; cg < (C + vec - 1) / vec; ++cg)
        for (int64_t cv = 0; cv < vec; ++cv) {
          int64_t c = cg * vec + cv;
          int32_t srcFlat =
              static_cast<int32_t>((c < C) ? (n * C + c) : (n * C + C - 1));
          idxMap.push_back(srcFlat);
        }
    return idxMap;
  }

  assert(srcShape.size() == 4 && "HMX activation map expects NCHW 4-D tensor");
  int64_t N = srcShape[0], C = srcShape[1];
  int64_t H = srcShape[2], W = srcShape[3];
  int64_t vec = 32;
  int64_t C32 = (C + vec - 1) / vec;

  SmallVector<int32_t> idxMap;
  idxMap.reserve(N * C * H * W);

  for (int64_t n = 0; n < N; ++n)
    for (int64_t cg = 0; cg < C32; ++cg)
      for (int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w)
          for (int64_t cv = 0; cv < vec; ++cv) {
            int64_t c = cg * vec + cv;
            // Clamp OOB channels
            int32_t srcFlat;
            if (c < C)
              srcFlat = static_cast<int32_t>(n * C * H * W + c * H * W +
                                             h * W + w);
            else
              srcFlat = static_cast<int32_t>(n * C * H * W +
                                             (C - 1) * H * W + h * W + w);
            idxMap.push_back(srcFlat);
          }

  DBG("HMX activation map: N=" << N << " C=" << C << " H=" << H << " W=" << W
                                << " mapSize=" << idxMap.size());
  return idxMap;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//
SmallVector<int32_t>
mlir::alps::computeHMXIndexMap(MLIRContext * /*ctx*/,
                                     MemRefType srcType, MemRefType dstType,
                                     LayoutTransform transform) {
  // CRITICAL FIX: Use dstType (VTCM tile) shape for index_map size,
  // not srcType (full DDR tensor) shape!
  // The index_map must have one entry per element in the destination tile.
  ArrayRef<int64_t> dstShape = dstType.getShape();
  
  DBG("computeHMXIndexMap: transform=" << (int)transform);
  DBG("  srcShape: rank=" << srcType.getRank());
  DBG("  dstShape: rank=" << dstType.getRank());

  switch (transform) {
  case LayoutTransform::None: {
    // Identity map: dest[i] = src[i]
    int64_t n = 1;
    for (auto d : dstShape)
      n *= d;
    SmallVector<int32_t> id;
    id.reserve(n);
    for (int32_t i = 0; i < (int32_t)n; ++i)
      id.push_back(i);
    DBG("  Identity map size: " << id.size());
    return id;
  }
  case LayoutTransform::HMXWeight:
  case LayoutTransform::HMXWeightDequantI8:
    // Use destination tile shape for map computation
    return computeHMXWeightMap(dstShape);
  case LayoutTransform::HMXActivation:
    // Use destination tile shape for map computation
    return computeHMXActivationMap(dstShape);
  case LayoutTransform::Custom:
    // For Custom, the caller supplies the index_map directly.
    return {};
  case LayoutTransform::L2Hint:
    // Cache hint only — no gather map.
    return {};
  }
  llvm_unreachable("unhandled LayoutTransform");
}
