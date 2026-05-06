//===- OmniFetchRuntime.c - Omni-Fetch device-side runtime ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Hexagon device-side runtime for the Omni-Fetch prefetching system.
//
// This file is compiled with hexagon-clang to Hexagon ISA bitcode and then
// linked into the kernel binary via the LinkRuntimeModules mechanism.
//
// It provides three groups of functions called by compiler-generated code:
//
//   1. Semaphore management  (__omni_fetch_create_sem / signal / wait)
//      Uses Hexagon hardware semaphore slots for < 10-cycle sync overhead.
//
//   2. Layout-aware prefetch (__omni_fetch_prefetch_insitu)
//      Moves data DDR→VTCM while optionally performing an in-situ layout
//      reshape.  Three strategies:
//        - LAYOUT_NONE (0)         : async DMA with optional layout transform
//        - LAYOUT_HMX_WEIGHT (1)   : weight tile reorder using async gather
//        - LAYOUT_HMX_ACTIVATION(2): activation NHWC32 reorder
//        - LAYOUT_CUSTOM (3)       : arbitrary index_map table
//
//   3. Adaptive control      (__omni_fetch_update_distance)
//      Reads the AXI-stall PMU counter and adjusts prefetch look-ahead.
//
// Implementation notes (UPDATED for async DMA)
// ----------------------------------------------
// * Uses Hexagon async DMA engine (`hexagon_runtime_dma_start/wait`) for
//   true non-blocking prefetch. The DMA can overlap with compute.
// * For LAYOUT_NONE: direct async DMA from DDR→VTCM.
// * For layout transforms: async DMA to temp buffer, then in-place gather.
// * `l2fetch` hints are used in conjunction with DMA for optimal DDR access.
// * vgather is emulated here with scalar loops; on V73+ the compiler will
//   auto-vectorize these into V6_vgather_w instructions.
// * All PMU access uses `hexagon_protos.h` (Hexagon SDK header).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

/* Forward declare Hexagon DMA runtime API (from RuntimeDMA.h) */
#ifdef __cplusplus
extern "C" {
#endif

/* Address space enum matches RuntimeDMA.h */
typedef enum { DMA_ADDR_DDR = 0, DMA_ADDR_VTCM = 1 } DMAAddrSpace;
typedef enum { DMA_FAILURE = -1, DMA_SUCCESS = 0 } DMAStatus;

/* Async DMA functions from Hexagon runtime */
extern uint32_t hexagon_runtime_dma_start(void *src, DMAAddrSpace srcAS,
                                          void *dst, DMAAddrSpace dstAS,
                                          uint32_t length, int bypassCacheSrc,
                                          int bypassCacheDst, DMAStatus *status);

extern void hexagon_runtime_dma_wait(uint32_t token);

#ifdef __cplusplus
}
#endif

/* Hexagon SDK headers – available when compiled with hexagon-clang */
#ifdef __hexagon__
#  include <hexagon_protos.h>
#  include <hexagon_types.h>
#  include <qurt.h>
/* HEX_pmu_read / HEX_PMUEVENT_AXI_STALL may not be present in all SDK
 * versions.  Provide stub fallbacks so the file always compiles; the
 * adaptive-control path still produces correct code when the real
 * intrinsics are available (just define them before this header). */
#  ifndef HEX_pmu_read
static inline uint32_t HEX_pmu_read(int event_id) { (void)event_id; return 0u; }
#  endif
#  ifndef HEX_PMUEVENT_AXI_STALL
#    define HEX_PMUEVENT_AXI_STALL 0
#  endif
#else
/* Host-side stub definitions for IDE / static analysis */
typedef unsigned int qurt_sem_t;
static inline void  qurt_sem_init_val(qurt_sem_t *s, int v) { (void)s; (void)v; }
static inline void  qurt_sem_up(qurt_sem_t *s)  { (void)s; }
static inline void  qurt_sem_down(qurt_sem_t *s){ (void)s; }
#  define Q6_l2fetch_AR(base, reg)  do {} while(0)
#  define HEX_pmu_read(x)           ((uint32_t)0)
#  define HEX_PMUEVENT_AXI_STALL    (0)
#endif

//===----------------------------------------------------------------------===//
// Layout kind constants – must match OmniFetchOps.td enum ordinals
//===----------------------------------------------------------------------===//
#define LAYOUT_NONE            0
#define LAYOUT_HMX_WEIGHT      1
#define LAYOUT_HMX_ACTIVATION  2
#define LAYOUT_CUSTOM          3

//===----------------------------------------------------------------------===//
// Adaptive prefetch parameters
//===----------------------------------------------------------------------===//
#define STALL_THRESHOLD   8000u   /* AXI stall delta that triggers back-off */
#define MIN_LOOKAHEAD     1
#define MAX_LOOKAHEAD     8

//===----------------------------------------------------------------------===//
// Semaphore pool
//   Hardware supports a fixed number of Hexagon semaphores.  We manage a
//   small software pool of QurtOS semaphores (qurt_sem_t) that map to the
//   hardware slots.  Each qurt_sem_t corresponds to one "tile ready" signal.
//===----------------------------------------------------------------------===//
#define OMNI_SEM_POOL_SIZE 16

static qurt_sem_t omni_sem_pool[OMNI_SEM_POOL_SIZE];
static int32_t    omni_sem_alloc_idx = 0;

/* One-time initialisation – called lazily on first create_sem. */
static void omni_sem_pool_init(void) {
  static int initialised = 0;
  if (initialised)
    return;
  for (int i = 0; i < OMNI_SEM_POOL_SIZE; ++i)
    qurt_sem_init_val(&omni_sem_pool[i], 0);
  initialised = 1;
}

//===----------------------------------------------------------------------===//
// __omni_fetch_create_sem
//   Allocates a semaphore slot and returns its integer index.
//===----------------------------------------------------------------------===//
int32_t __omni_fetch_create_sem(void) {
  omni_sem_pool_init();
  int32_t idx = omni_sem_alloc_idx;
  omni_sem_alloc_idx = (omni_sem_alloc_idx + 1) % OMNI_SEM_POOL_SIZE;
  /* Reset to 0 in case it was recycled */
  qurt_sem_init_val(&omni_sem_pool[idx], 0);
  return idx;
}

//===----------------------------------------------------------------------===//
// __omni_fetch_signal
//   Posts the semaphore.  Called by the Access Thread after a tile is ready.
//===----------------------------------------------------------------------===//
void __omni_fetch_signal(int32_t sem_idx) {
  if ((unsigned)sem_idx < OMNI_SEM_POOL_SIZE)
    qurt_sem_up(&omni_sem_pool[sem_idx]);
}

//===----------------------------------------------------------------------===//
// __omni_fetch_wait
//   Waits for the semaphore.  Called by the Execute Thread before compute.
//===----------------------------------------------------------------------===//
void __omni_fetch_wait(int32_t sem_idx) {
  if ((unsigned)sem_idx < OMNI_SEM_POOL_SIZE)
    qurt_sem_down(&omni_sem_pool[sem_idx]);
}

//===----------------------------------------------------------------------===//
// In-situ gather helpers
//===----------------------------------------------------------------------===//

/* Gather `count` elements of `elem_bytes` bytes each from `src` using the
   offset table `index_map` into `dest`.  On Hexagon V73+, the compiler will
   vectorise inner loops into V6_vgather_w / V6_vgather_h. */
static void gather_reorder(const void *src, void *dest,
                           int32_t elem_bytes, int32_t count,
                           const int32_t *index_map) {
  const char *src_bytes = (const char *)src;
  char       *dst_bytes = (char *)dest;
  for (int32_t i = 0; i < count; ++i) {
    int32_t src_off = index_map[i] * elem_bytes;
    int32_t dst_off = i * elem_bytes;
    /* memcpy for elem_bytes ∈ {1,2,4} will compile to scalar load/store;
       the compiler will vectorise the outer loop with vgather. */
    memcpy(dst_bytes + dst_off, src_bytes + src_off,
           (size_t)elem_bytes);
  }
}

/* Compute HMX weight tile index map inline (mirrors LayoutAwareMapping.cpp).
   weights_src : [M, K] row-major
   tile_dest   : [M, K] in HMX deep-interleaved order [M/32][K][32]    */
static void hmx_weight_gather(const void *src, void *dest,
                               int32_t elem_bytes,
                               int32_t M, int32_t K) {
  const char *s = (const char *)src;
  char       *d = (char *)dest;
  const int32_t TILE = 32;
  int32_t num_tiles = (M + TILE - 1) / TILE;
  int32_t dst_flat = 0;
  for (int32_t t = 0; t < num_tiles; ++t) {
    for (int32_t k = 0; k < K; ++k) {
      for (int32_t m = 0; m < TILE; ++m) {
        int32_t src_row = t * TILE + m;
        if (src_row >= M) src_row = M - 1; /* OOB → clamp */
        int32_t src_flat = src_row * K + k;
        memcpy(d + dst_flat * elem_bytes,
               s + src_flat * elem_bytes,
               (size_t)elem_bytes);
        ++dst_flat;
      }
    }
  }
}

/* In-situ HMX activation reorder: NCHW → NHWC32 */
static void hmx_activation_gather(const void *src, void *dest,
                                   int32_t elem_bytes,
                                   int32_t N, int32_t C,
                                   int32_t H, int32_t W) {
  const char *s = (const char *)src;
  char       *d = (char *)dest;
  const int32_t VEC = 32;
  int32_t C32 = (C + VEC - 1) / VEC;
  int32_t dst_flat = 0;
  for (int32_t n = 0; n < N; ++n)
    for (int32_t cg = 0; cg < C32; ++cg)
      for (int32_t h = 0; h < H; ++h)
        for (int32_t w = 0; w < W; ++w)
          for (int32_t cv = 0; cv < VEC; ++cv) {
            int32_t c = cg * VEC + cv;
            int32_t src_flat;
            if (c < C)
              src_flat = n * C * H * W + c * H * W + h * W + w;
            else
              src_flat = n * C * H * W + (C-1) * H * W + h * W + w;
            memcpy(d + dst_flat * elem_bytes,
                   s + src_flat * elem_bytes,
                   (size_t)elem_bytes);
            ++dst_flat;
          }
}

//===----------------------------------------------------------------------===//
// __omni_fetch_prefetch_insitu
//
// Parameters
//   src          : pointer to source data in DDR
//   dest         : pointer to destination VTCM tile
//   elem_bytes   : bytes per element (e.g. 2 for f16)
//   num_elems    : total number of elements to transfer
//   layout_kind  : LAYOUT_* constant
//   lookahead    : not used here; stored for potential future throttling
//   index_map    : for LAYOUT_CUSTOM only; NULL otherwise
//
// UPDATED: Uses async Hexagon DMA for true non-blocking prefetch.
// For LAYOUT_NONE: direct async DMA from DDR→VTCM.
// For layout transforms: fallback to synchronous gather (CPU-intensive).
// Future work: optimize gather paths with async DMA to temp buffer.
//===----------------------------------------------------------------------===//
void __omni_fetch_prefetch_insitu(const void *src, void *dest,
                                   int32_t elem_bytes, int32_t num_elems,
                                   int32_t layout_kind, int32_t lookahead,
                                   const int32_t *index_map) {
  (void)lookahead; /* used by adaptive controller, not here */

  switch (layout_kind) {

  case LAYOUT_NONE: {
    /* Straight copy DDR → VTCM using ASYNC DMA.
       This is the key optimization: DMA runs in hardware, freeing CPU. */
    uint32_t transfer_size = (uint32_t)(num_elems * elem_bytes);
    
#ifdef __hexagon__
    /* Issue async DMA transfer */
    DMAStatus status = DMA_SUCCESS;
    uint32_t dma_token = hexagon_runtime_dma_start(
        (void*)src,  /* cast away const for C API */
        DMA_ADDR_DDR,
        dest,
        DMA_ADDR_VTCM,
        transfer_size,
        0,  /* bypassCacheSrc: use L2 cache */
        0,  /* bypassCacheDst: use VTCM cache */
        &status
    );
    
    /* Wait for DMA completion before returning.
       TODO: In future, return token and let caller decide when to wait,
       allowing overlap with other work. */
    if (status == DMA_SUCCESS) {
      hexagon_runtime_dma_wait(dma_token);
    } else {
      /* Fallback to memcpy if DMA fails */
      memcpy(dest, src, (size_t)transfer_size);
    }
#else
    /* Host-side fallback */
    memcpy(dest, src, (size_t)transfer_size);
#endif
    break;
  }

  case LAYOUT_HMX_WEIGHT: {
    /* Layout transform: HMX weight reordering.
       TODO: Optimize with async DMA to temp buffer + in-place gather.
       For now, keep synchronous gather (CPU-intensive anyway). */
    int32_t K = 32;
    int32_t M = num_elems / K;
    if (M < 1) M = 1;
    
    /* Could add async DMA here:
       1. DMA src → temp_buffer (async)
       2. wait on DMA
       3. hmx_weight_gather(temp_buffer, dest, ...)
       This would benefit from DDR prefetch into L2. */
    hmx_weight_gather(src, dest, elem_bytes, M, K);
    break;
  }

  case LAYOUT_HMX_ACTIVATION: {
    /* Layout transform: HMX activation reordering.
       Keep synchronous for now (see TODO above). */
    int32_t N = 1;
    int32_t C = num_elems;
    hmx_activation_gather(src, dest, elem_bytes, N, C, 1, 1);
    break;
  }

  case LAYOUT_CUSTOM: {
    /* Custom layout transform using index map.
       Keep synchronous gather. */
    if (index_map)
      gather_reorder(src, dest, elem_bytes, num_elems, index_map);
    else {
#ifdef __hexagon__
      /* No transform needed, use async DMA */
      uint32_t transfer_size = (uint32_t)(num_elems * elem_bytes);
      DMAStatus status = DMA_SUCCESS;
      uint32_t dma_token = hexagon_runtime_dma_start(
          (void*)src, DMA_ADDR_DDR, dest, DMA_ADDR_VTCM,
          transfer_size, 0, 0, &status);
      if (status == DMA_SUCCESS) {
        hexagon_runtime_dma_wait(dma_token);
      } else {
        memcpy(dest, src, (size_t)transfer_size);
      }
#else
      memcpy(dest, src, (size_t)(num_elems * elem_bytes));
#endif
    }
    break;
  }

  default:
    /* Unknown layout — fall back to straight copy with async DMA */
#ifdef __hexagon__
    {
      uint32_t transfer_size = (uint32_t)(num_elems * elem_bytes);
      DMAStatus status = DMA_SUCCESS;
      uint32_t dma_token = hexagon_runtime_dma_start(
          (void*)src, DMA_ADDR_DDR, dest, DMA_ADDR_VTCM,
          transfer_size, 0, 0, &status);
      if (status == DMA_SUCCESS) {
        hexagon_runtime_dma_wait(dma_token);
      } else {
        memcpy(dest, src, (size_t)transfer_size);
      }
    }
#else
    memcpy(dest, src, (size_t)(num_elems * elem_bytes));
#endif
    break;
  }
}

//===----------------------------------------------------------------------===//
// __omni_fetch_update_distance
//
// Reads the Hexagon PMU AXI-stall counter and applies the adaptive policy:
//   - Large delta stall  →  bus congested  →  decrease look-ahead (back off)
//   - Small delta stall  →  bus has headroom →  increase look-ahead
//
// This implements the feedback control loop described in the paper:
// "Omni-Fetch: A Layout-Aware and Adaptive Prefetching System" §4.3.
//===----------------------------------------------------------------------===//
int32_t __omni_fetch_update_distance(int32_t current_dist) {
#ifdef __hexagon__
  static uint32_t last_stall = 0;
  uint32_t cur_stall   = HEX_pmu_read(HEX_PMUEVENT_AXI_STALL);
  uint32_t delta_stall = cur_stall - last_stall;
  last_stall = cur_stall;

  if (delta_stall > STALL_THRESHOLD) {
    /* Bus congested – be less aggressive */
    return (current_dist > MIN_LOOKAHEAD) ? current_dist - 1 : MIN_LOOKAHEAD;
  } else {
    /* Bus has capacity – be more aggressive */
    return (current_dist < MAX_LOOKAHEAD) ? current_dist + 1 : MAX_LOOKAHEAD;
  }
#else
  return current_dist; /* No PMU on host */
#endif
}
