//===- OmniFetchRuntime.c - Omni-Fetch device-side runtime ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
// Hexagon device-side runtime for the Omni-Fetch prefetching system.
// Compiled to bitcode via hexagon-clang and linked by LinkRuntimeModules.
//
// Design constraints:
//   - Must NOT use qurt.h  (qurt_sem_* unavailable in Unsigned PD)
//   - Must NOT use stdatomic.h  (_Assert symbol missing on DSP)
//   - Must NOT use hexagon_protos.h  (pulls in _Assert)
//   - Must NOT use assert()
//   - memcpy/memset are fine (provided by libc++.so.1 on device)
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Layout kind constants – must match OmniFetchOps.td enum ordinals
 * ------------------------------------------------------------------------- */
#define LAYOUT_NONE            0
#define LAYOUT_HMX_WEIGHT      1
#define LAYOUT_HMX_ACTIVATION  2
#define LAYOUT_CUSTOM          3

/* -------------------------------------------------------------------------
 * Adaptive prefetch parameters
 * ------------------------------------------------------------------------- */
#define MIN_LOOKAHEAD  1
#define MAX_LOOKAHEAD  8
#define STALL_THRESHOLD 8000u

/* -------------------------------------------------------------------------
 * Semaphore – volatile counter + proper spin-wait.
 *
 * V-DAE execution model
 * ---------------------
 * V-DAE assumes Access Thread and Execute Thread run CONCURRENTLY.  On a
 * single Hexagon hardware thread this is realised through software
 * pipelining: the Access Thread role is played by the *previous* loop
 * iteration (which issued the prefetch for the current tile K iterations
 * ahead), while the Execute Thread role is played by the *current*
 * iteration (which computes on the tile).
 *
 * Semaphore semantics:
 *   signal(sem) – issued AFTER a prefetch_insitu transfer completes,
 *                 indicating the VTCM tile is ready for consumption.
 *   wait(sem)   – issued BEFORE the HMX compute, ensuring the tile is
 *                 valid.  Must NOT return until the counter is > 0.
 *
 * With synchronous prefetch_insitu (current default), the signal is always
 * posted before the corresponding wait, so wait() returns immediately.
 * With async DMA prefetch (future), the spin provides the necessary ordering
 * guarantee without any OS dependency.
 *
 * Atomicity note: on a single-threaded DSP the volatile read/write is
 * sufficient.  On a multi-threaded DSP (QuRT multi-PD) the memw_locked /
 * memw_store_locked intrinsics would be required; the spin body below is
 * structured to facilitate that upgrade.
 * ------------------------------------------------------------------------- */
#define OMNI_SEM_POOL_SIZE 16

/* Maximum spin iterations before giving up (prevents infinite hang on
 * incorrect usage; should never be reached in a correct program). */
#define OMNI_SEM_MAX_SPIN  0x100000

typedef volatile int omni_sem_t;
static omni_sem_t omni_sem_pool[OMNI_SEM_POOL_SIZE];
static int omni_sem_alloc_idx = 0;

static void omni_sem_pool_init(void) {
  static int initialised = 0;
  if (initialised) return;
  for (int i = 0; i < OMNI_SEM_POOL_SIZE; ++i)
    omni_sem_pool[i] = 0;
  initialised = 1;
}

int32_t __omni_fetch_create_sem(void) {
  omni_sem_pool_init();
  int32_t idx = omni_sem_alloc_idx;
  omni_sem_alloc_idx = (omni_sem_alloc_idx + 1) % OMNI_SEM_POOL_SIZE;
  omni_sem_pool[idx] = 0;
  return idx;
}

void __omni_fetch_signal(int32_t sem_idx) {
  if ((unsigned)sem_idx >= OMNI_SEM_POOL_SIZE) return;
  /* Write is visible to the same hardware thread (single-threaded model)
   * and to a second HW thread via the Hexagon memory model.  A compiler
   * barrier suffices here; the volatile qualifier on omni_sem_t prevents
   * the store from being reordered across the preceding prefetch. */
  omni_sem_pool[sem_idx]++;
}

void __omni_fetch_wait(int32_t sem_idx) {
  if ((unsigned)sem_idx >= OMNI_SEM_POOL_SIZE) return;
  /* Spin until the Access Thread signals that the VTCM tile is ready.
   * In the current synchronous-prefetch model this loop body executes
   * zero times (signal was already posted before wait is reached).
   * With async DMA the spin provides the necessary ordering fence. */
  int spins = 0;
  while (omni_sem_pool[sem_idx] <= 0) {
    if (++spins >= OMNI_SEM_MAX_SPIN)
      break;  /* Safety valve: avoid infinite hang on mis-use. */
#ifdef __hexagon__
    /* Hexagon pause instruction: yield the pipeline for one cycle.
     * Reduces power and bus contention during the spin loop. */
    __asm__ volatile("pause(#255)");
#endif
  }
  omni_sem_pool[sem_idx]--;
}

/* -------------------------------------------------------------------------
 * In-situ gather helpers (scalar; compiler auto-vectorises on V73+)
 * ------------------------------------------------------------------------- */
static void gather_reorder(const void *src, void *dest,
                           int32_t elem_bytes, int32_t count,
                           const int32_t *index_map) {
  const char *s = (const char *)src;
  char       *d = (char *)dest;
  for (int32_t i = 0; i < count; ++i)
    memcpy(d + i * elem_bytes, s + index_map[i] * elem_bytes,
           (size_t)elem_bytes);
}

static void hmx_weight_gather(const void *src, void *dest,
                               int32_t elem_bytes, int32_t M, int32_t K) {
  const char *s = (const char *)src;
  char       *d = (char *)dest;
  const int32_t TILE = 32;
  int32_t num_tiles = (M + TILE - 1) / TILE;
  int32_t dst_flat = 0;
  for (int32_t t = 0; t < num_tiles; ++t)
    for (int32_t k = 0; k < K; ++k)
      for (int32_t m = 0; m < TILE; ++m) {
        int32_t src_row = t * TILE + m;
        if (src_row >= M) src_row = M - 1;
        memcpy(d + dst_flat * elem_bytes,
               s + (src_row * K + k) * elem_bytes,
               (size_t)elem_bytes);
        ++dst_flat;
      }
}

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
              src_flat = n * C * H * W + (C - 1) * H * W + h * W + w;
            memcpy(d + dst_flat * elem_bytes,
                   s + src_flat * elem_bytes,
                   (size_t)elem_bytes);
            ++dst_flat;
          }
}

/* -------------------------------------------------------------------------
 * L2 fetch helpers
 *
 * l2fetch is an asynchronous cache-hint instruction on Hexagon (V62+).
 * It initiates a line fetch from DDR → L2 without stalling the pipeline.
 * We use it as a low-overhead "warm-up" hint before the blocking memcpy so
 * that by the time memcpy reads the source data it is already in L2 cache.
 *
 * l2fetch encoding:  l2fetch(Rtt, Rs)
 *   Rtt[63:32] = stride (bytes between rows)
 *   Rtt[31:16] = width  (bytes per row)
 *   Rtt[15:0]  = height (number of rows)
 * For a flat 1-D buffer we use stride=width=total_bytes, height=1.
 * Maximum single l2fetch = 64 kB; split into chunks if larger.
 * ------------------------------------------------------------------------- */
#ifdef __hexagon__
static void omni_l2fetch(const void *ptr, uint32_t total_bytes) {
  const char *p = (const char *)ptr;
  const uint32_t kChunk = 0x8000u;  /* 32 kB per l2fetch call */
  while (total_bytes > 0) {
    uint32_t chunk = total_bytes < kChunk ? total_bytes : kChunk;
    /* Pack the l2fetch descriptor: stride=chunk, width=chunk, height=1 */
    uint64_t spec = ((uint64_t)chunk << 32) | ((uint64_t)chunk << 16) | 1ULL;
    __asm__ volatile("l2fetch(%0, %1)" : : "r"(p), "r"(spec) : "memory");
    p += chunk;
    total_bytes -= chunk;
  }
}
#endif

/* -------------------------------------------------------------------------
 * __omni_fetch_prefetch_insitu
 *
 * Execution model
 * ---------------
 * This function implements the "Access Thread" role of V-DAE.  It runs
 * BEFORE the HMX compute (Execute Thread) for the SAME iteration's tile by
 * being issued K iterations ahead (K = lookahead).
 *
 * Phase 1 (async hint): emit l2fetch to begin warming the source data into
 *   L2 cache while the pipeline continues.  This overlaps with preceding
 *   compute and reduces the effective DDR latency seen by Phase 2.
 *
 * Phase 2 (synchronous copy): perform the actual layout-aware gather from
 *   DDR/L2 into the VTCM shadow buffer.  Because Phase 1 pre-warmed L2,
 *   Phase 2 accesses L2-resident data (fast) rather than DDR (slow).
 *
 * After this function returns, the caller (V-DAE pass) issues signal(sem)
 * to mark the VTCM tile as ready for the Execute Thread.
 *
 * NOTE: When the Hexagon DMA engine (v66+) is available via a supported
 *   API, Phase 2 can be replaced with an async DMA kick followed by a
 *   DMA-completion poll inside wait().  The semaphore infrastructure is
 *   already structured for that upgrade.
 * ------------------------------------------------------------------------- */
void __omni_fetch_prefetch_insitu(const void *src, void *dest,
                                   int32_t elem_bytes, int32_t num_elems,
                                   int32_t layout_kind, int32_t lookahead,
                                   const int32_t *index_map) {
  (void)lookahead;

  if (elem_bytes <= 0 || num_elems <= 0 || !src || !dest)
    return;

  uint32_t total_bytes = (uint32_t)(num_elems * elem_bytes);

#ifdef __hexagon__
  /* l2fetch disabled for now: bad pointers fault the DSP (exit 13).
   * Re-enable once pointer/offset lowering is fully validated.
   * omni_l2fetch(src, total_bytes);
   */
  (void)total_bytes;
#endif

  switch (layout_kind) {

  case LAYOUT_NONE: {
    /* Phase 2: linear copy DDR → VTCM shadow buffer. */
    memcpy(dest, src, (size_t)total_bytes);
    break;
  }

  case LAYOUT_HMX_WEIGHT: {
    int32_t K = 32;
    int32_t M = (num_elems > 0) ? num_elems / K : 1;
    hmx_weight_gather(src, dest, elem_bytes, M, K);
    break;
  }

  case LAYOUT_HMX_ACTIVATION: {
    hmx_activation_gather(src, dest, elem_bytes, 1, num_elems, 1, 1);
    break;
  }

  case LAYOUT_CUSTOM: {
    if (index_map)
      gather_reorder(src, dest, elem_bytes, num_elems, index_map);
    else
      memcpy(dest, src, (size_t)total_bytes);
    break;
  }

  default:
    memcpy(dest, src, (size_t)total_bytes);
    break;
  }
}

/* Rank-2 tile copy that respects row strides.  Required when PrefetchInsert
 * tiles an inner dimension: the src subview is strided, so a flat memcpy of
 * rows*cols elements would read the wrong (contiguous) bytes. */
void __omni_fetch_copy2d(const void *src, void *dest, int32_t elem_bytes,
                         int32_t rows, int32_t cols,
                         int32_t src_row_stride_elems,
                         int32_t dst_row_stride_elems) {
  if (elem_bytes <= 0 || rows <= 0 || cols <= 0 || !src || !dest)
    return;
  if (src_row_stride_elems < cols || dst_row_stride_elems < cols)
    return;

  const char *s = (const char *)src;
  char *d = (char *)dest;
  const size_t row_bytes = (size_t)cols * (size_t)elem_bytes;
  const size_t src_pitch = (size_t)src_row_stride_elems * (size_t)elem_bytes;
  const size_t dst_pitch = (size_t)dst_row_stride_elems * (size_t)elem_bytes;

  /* Fast path: both sides contiguous. */
  if (src_row_stride_elems == cols && dst_row_stride_elems == cols) {
    memcpy(d, s, row_bytes * (size_t)rows);
    return;
  }

  for (int32_t r = 0; r < rows; ++r)
    memcpy(d + (size_t)r * dst_pitch, s + (size_t)r * src_pitch, row_bytes);
}

/* -------------------------------------------------------------------------
 * __omni_fetch_update_distance  (adaptive lookahead control)
 * ------------------------------------------------------------------------- */
int32_t __omni_fetch_update_distance(int32_t current_dist) {
  /* PMU access via hexagon_protos.h is avoided to prevent _Assert dependency.
     Return current distance unchanged – adaptive control is a no-op for now. */
  (void)current_dist;
  return current_dist;
}
