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

#ifdef __hexagon__
/* HexKL micro API — linked via -lhexkl_micro on the final device .so. */
int hexkl_micro_hmx_rm_to_wh_f16(uint8_t *vtcm_base, uint32_t weight_offset,
                                 const _Float16 *wt_old, uint32_t row_tile,
                                 uint32_t col_tile, uint32_t wt_cols);
#endif

/* -------------------------------------------------------------------------
 * Layout kind constants – must match OmniFetchOps.td enum ordinals
 * ------------------------------------------------------------------------- */
#define LAYOUT_NONE            0
#define LAYOUT_HMX_WEIGHT      1
#define LAYOUT_HMX_ACTIVATION  2
#define LAYOUT_CUSTOM          3
#define LAYOUT_L2_HINT         4

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

/* -------------------------------------------------------------------------
 * Async DMA support (Phase 2b)
 *
 * When lookahead > 0, LAYOUT_NONE / HMX* kick a UserDMA transfer of the raw
 * DDR tile into a staging buffer and return immediately.  __omni_fetch_wait
 * completes the DMA and (for HMX*) finishes the in-situ gather into dest.
 * That lets Mm overlap with the DDR→staging transfer on a single HW thread.
 * ------------------------------------------------------------------------- */
enum { OMNI_DDR = 0, OMNI_VTCM = 1 };
enum { OMNI_DMA_OK = 0 };

extern uint32_t hexagon_runtime_dma_start(void *src, int srcAS, void *dst,
                                          int dstAS, uint32_t length,
                                          int bypassCacheSrc,
                                          int bypassCacheDst, int *status);
extern void hexagon_runtime_dma_wait(uint32_t token);

#define OMNI_STAGE_ELEMS (32 * 32)
#define OMNI_STAGE_SLOTS 4

static uint16_t omni_stage[OMNI_STAGE_SLOTS][OMNI_STAGE_ELEMS];
static int omni_stage_slot = 0;

typedef struct {
  int active;
  uint32_t token;
  void *dest;
  int32_t elem_bytes;
  int32_t num_elems;
  int32_t layout_kind;
  int stage_slot;
} OmniAsyncJob;

static OmniAsyncJob omni_async_job;

/* Forward decls – defined below. */
static void hmx_weight_gather(const void *src, void *dest, int32_t elem_bytes,
                              int32_t M, int32_t K);
static void hmx_activation_gather(const void *src, void *dest,
                                  int32_t elem_bytes, int32_t N, int32_t C,
                                  int32_t H, int32_t W);

static void omni_async_complete(void) {
  if (!omni_async_job.active)
    return;
#ifdef __hexagon__
  if (omni_async_job.token != 0)
    hexagon_runtime_dma_wait(omni_async_job.token);
#endif
  const void *staged = omni_stage[omni_async_job.stage_slot];
  void *dest = omni_async_job.dest;
  int32_t eb = omni_async_job.elem_bytes;
  int32_t ne = omni_async_job.num_elems;
  switch (omni_async_job.layout_kind) {
  case LAYOUT_NONE:
    memcpy(dest, staged, (size_t)ne * (size_t)eb);
    break;
  case LAYOUT_HMX_WEIGHT: {
    int32_t K = 32;
    int32_t M = (ne > 0) ? ne / K : 1;
    hmx_weight_gather(staged, dest, eb, M, K);
    break;
  }
  case LAYOUT_HMX_ACTIVATION:
    hmx_activation_gather(staged, dest, eb, 1, ne, 1, 1);
    break;
  default:
    memcpy(dest, staged, (size_t)ne * (size_t)eb);
    break;
  }
  omni_async_job.active = 0;
}

static int omni_async_kick(const void *src, void *dest, int32_t elem_bytes,
                           int32_t num_elems, int32_t layout_kind) {
  uint32_t bytes = (uint32_t)elem_bytes * (uint32_t)num_elems;
  if (bytes == 0 || bytes > sizeof(omni_stage[0]) || !src || !dest)
    return 0;
  /* Complete any prior job before reusing the staging ring. */
  omni_async_complete();

  int slot = omni_stage_slot;
  omni_stage_slot = (omni_stage_slot + 1) % OMNI_STAGE_SLOTS;
#ifdef __hexagon__
  int status = OMNI_DMA_OK;
  /* Staging lives in DDR (.bss); both ends are DDR address space. */
  uint32_t tok = hexagon_runtime_dma_start(
      (void *)src, OMNI_DDR, omni_stage[slot], OMNI_DDR, bytes,
      /*bypassSrc=*/0, /*bypassDst=*/0, &status);
  if (status != OMNI_DMA_OK) {
    memcpy(omni_stage[slot], src, bytes);
    tok = 0;
  }
  omni_async_job.token = tok;
#else
  memcpy(omni_stage[slot], src, bytes);
  omni_async_job.token = 0;
#endif
  omni_async_job.active = 1;
  omni_async_job.dest = dest;
  omni_async_job.elem_bytes = elem_bytes;
  omni_async_job.num_elems = num_elems;
  omni_async_job.layout_kind = layout_kind;
  omni_async_job.stage_slot = slot;
  return 1;
}

void __omni_fetch_wait(int32_t sem_idx) {
  /* Drain async DMA + finish in-situ gather before Execute consumes the tile. */
  omni_async_complete();

  if ((unsigned)sem_idx >= OMNI_SEM_POOL_SIZE) return;
  int spins = 0;
  while (omni_sem_pool[sem_idx] <= 0) {
    if (++spins >= OMNI_SEM_MAX_SPIN)
      break;
#ifdef __hexagon__
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
                                   const int32_t *index_map,
                                   int32_t tile_row, int32_t tile_col,
                                   int32_t src_cols) {
  if (elem_bytes <= 0 || num_elems <= 0 || !src)
    return;
  /* L2 hints may pass dest==src; real copies require a distinct dest. */
  if (layout_kind != LAYOUT_L2_HINT && !dest)
    return;

  /* HexKL-accurate weight path: same transform as MicroHMXRmToWhF16.
   * src is the full DDR matrix; dest is the VTCM weight slot (offset 0).
   * Do not use the generic HMXWeight gather — it assumes a contiguous
   * 32-wide tile and corrupts strided subviews. */
  if (layout_kind == LAYOUT_HMX_WEIGHT && src_cols > 0 && tile_row >= 0 &&
      tile_col >= 0) {
#ifdef __hexagon__
    (void)elem_bytes;
    (void)num_elems;
    (void)lookahead;
    (void)index_map;
    hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)dest, /*weight_offset=*/0,
                                 (const _Float16 *)src, (uint32_t)tile_row,
                                 (uint32_t)tile_col, (uint32_t)src_cols);
#else
    (void)tile_row;
    (void)tile_col;
    (void)src_cols;
    (void)lookahead;
    (void)index_map;
    /* Host stub: keep old gather for unit tests without HexKL. */
    {
      int32_t K = 32;
      int32_t M = (num_elems > 0) ? num_elems / K : 1;
      hmx_weight_gather(src, dest, elem_bytes, M, K);
    }
#endif
    return;
  }

  uint32_t total_bytes = (uint32_t)(num_elems * elem_bytes);

  /* Phase 2b: async DMA kick when lookahead requests overlap. */
  if (lookahead > 0 &&
      (layout_kind == LAYOUT_NONE || layout_kind == LAYOUT_HMX_WEIGHT ||
       layout_kind == LAYOUT_HMX_ACTIVATION)) {
    if (omni_async_kick(src, dest, elem_bytes, num_elems, layout_kind))
      return;
    /* Fall through to synchronous path if kick refused. */
  }

  switch (layout_kind) {

  case LAYOUT_L2_HINT: {
    /* Cache-warmup only: no memcpy, no compute rewire.  Used for tiny HVX
     * vector tiles where a synchronous DDR→shadow copy is pure overhead. */
#ifdef __hexagon__
    omni_l2fetch(src, total_bytes);
#else
    (void)total_bytes;
#endif
    break;
  }

  case LAYOUT_NONE: {
#ifdef __hexagon__
    /* Warm L2 before the blocking copy when destination is real VTCM/DDR. */
    omni_l2fetch(src, total_bytes);
#endif
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
