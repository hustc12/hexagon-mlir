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
#include <stddef.h>

#ifdef __hexagon__
/* HexKL micro API — linked via -lhexkl_micro on the final device .so. */
int hexkl_micro_hmx_rm_to_wh_f16(uint8_t *vtcm_base, uint32_t weight_offset,
                                 const _Float16 *wt_old, uint32_t row_tile,
                                 uint32_t col_tile, uint32_t wt_cols);
int hexkl_micro_hmx_rm_to_ah_f16(uint8_t *vtcm_base,
                                 uint32_t activation_out_offset,
                                 uint32_t flat_in_offset);
int hexkl_micro_hmx_copy_submatrix_to_f16(uint8_t *vtcm_base,
                                          uint32_t out_offset,
                                          const _Float16 *input_matrix,
                                          uint32_t tile_row, uint32_t tile_col,
                                          uint32_t input_rows,
                                          uint32_t input_cols);
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

#define OMNI_ERROR_SEM_TIMEOUT       (1u << 0)
#define OMNI_ERROR_DESCRIPTOR_FULL   (1u << 1)
static unsigned omni_error_flags = 0;

uint32_t __omni_fetch_get_and_clear_errors(void) {
  return __atomic_exchange_n(&omni_error_flags, 0, __ATOMIC_ACQ_REL);
}

/* -------------------------------------------------------------------------
 * Cross-token WH cache.
 *
 * Entries live in runtime-owned DDR so they survive VTCM arena teardown at
 * the end of one model invocation.  A client-provided (context,generation)
 * pair prevents a reused source address from observing stale transformed
 * weights after model replacement.  lookahead == -1 selects this path.
 * ------------------------------------------------------------------------- */
#define OMNI_WH_CACHE_SLOTS 2048
#define OMNI_WH_CACHE_PROBES 8
#define OMNI_WH_TILE_BYTES 4096

typedef struct {
  uint64_t context;
  uint32_t generation;
  uint32_t epoch;
  const void *source;
  int32_t tile_row;
  int32_t tile_col;
  int32_t source_cols;
  int32_t site_id;
  volatile int valid;
  unsigned char data[OMNI_WH_TILE_BYTES] __attribute__((aligned(128)));
} OmniWhCacheEntry;

static OmniWhCacheEntry omni_wh_cache[OMNI_WH_CACHE_SLOTS];
static uint64_t omni_wh_context = 0;
static uint32_t omni_wh_generation = 0;
static uint32_t omni_wh_epoch = 1;
static unsigned omni_wh_hits = 0;
static unsigned omni_wh_misses = 0;

static unsigned
omni_wh_cache_hash(const void *source, int32_t tile_row, int32_t tile_col,
                   int32_t source_cols, int32_t site_id, uint64_t context,
                   uint32_t generation) {
  uint64_t x = ((uint64_t)(uintptr_t)source >> 7) ^ context;
  x ^= (uint64_t)generation * UINT64_C(0x9e3779b185ebca87);
  x ^= (uint64_t)(uint32_t)site_id * UINT64_C(0xc2b2ae3d27d4eb4f);
  x ^= (uint64_t)(uint32_t)tile_row * UINT64_C(0x165667b19e3779f9);
  x ^= (uint64_t)(uint32_t)tile_col * UINT64_C(0x85ebca77c2b2ae63);
  x ^= (uint64_t)(uint32_t)source_cols * UINT64_C(0x27d4eb2f165667c5);
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  return (unsigned)x & (OMNI_WH_CACHE_SLOTS - 1);
}

void __omni_fetch_wh_cache_set_context(uint64_t context,
                                       uint32_t generation) {
  uint64_t old_context =
      __atomic_load_n(&omni_wh_context, __ATOMIC_ACQUIRE);
  uint32_t old_generation =
      __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
  if (old_context != context || old_generation != generation)
    __atomic_add_fetch(&omni_wh_epoch, 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&omni_wh_context, context, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_wh_generation, generation, __ATOMIC_RELEASE);
}

void __omni_fetch_wh_cache_invalidate(uint64_t context,
                                      uint32_t generation) {
  __atomic_add_fetch(&omni_wh_epoch, 1, __ATOMIC_ACQ_REL);
  for (int i = 0; i < OMNI_WH_CACHE_SLOTS; ++i) {
    if (__atomic_load_n(&omni_wh_cache[i].valid, __ATOMIC_ACQUIRE) &&
        omni_wh_cache[i].context == context &&
        omni_wh_cache[i].generation == generation)
      __atomic_store_n(&omni_wh_cache[i].valid, 0, __ATOMIC_RELEASE);
  }
}

uint64_t __omni_fetch_wh_cache_stats(void) {
  uint64_t hits = __atomic_load_n(&omni_wh_hits, __ATOMIC_RELAXED);
  uint64_t misses = __atomic_load_n(&omni_wh_misses, __ATOMIC_RELAXED);
  return (hits << 32) | (misses & 0xffffffffu);
}

static OmniWhCacheEntry *
omni_wh_cache_lookup(const void *source, int32_t tile_row, int32_t tile_col,
                     int32_t source_cols, int32_t site_id) {
  uint64_t context = __atomic_load_n(&omni_wh_context, __ATOMIC_ACQUIRE);
  uint32_t generation =
      __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
  uint32_t epoch = __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE);
  unsigned base = omni_wh_cache_hash(source, tile_row, tile_col, source_cols,
                                     site_id, context, generation);
  for (unsigned i = 0; i < OMNI_WH_CACHE_PROBES; ++i) {
    OmniWhCacheEntry *entry =
        &omni_wh_cache[(base + i) & (OMNI_WH_CACHE_SLOTS - 1)];
    if (__atomic_load_n(&entry->valid, __ATOMIC_ACQUIRE) &&
        entry->context == context && entry->generation == generation &&
        entry->epoch == epoch &&
        entry->source == source && entry->tile_row == tile_row &&
        entry->tile_col == tile_col && entry->source_cols == source_cols &&
        entry->site_id == site_id)
      return entry;
  }
  return 0;
}

static OmniWhCacheEntry *
omni_wh_cache_reserve(const void *source, int32_t tile_row, int32_t tile_col,
                      int32_t source_cols, int32_t site_id) {
  uint64_t context = __atomic_load_n(&omni_wh_context, __ATOMIC_ACQUIRE);
  uint32_t generation =
      __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
  uint32_t epoch = __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE);
  unsigned base = omni_wh_cache_hash(source, tile_row, tile_col, source_cols,
                                     site_id, context, generation);
  OmniWhCacheEntry *entry = &omni_wh_cache[base];
  for (unsigned i = 0; i < OMNI_WH_CACHE_PROBES; ++i) {
    OmniWhCacheEntry *candidate =
        &omni_wh_cache[(base + i) & (OMNI_WH_CACHE_SLOTS - 1)];
    if (!__atomic_load_n(&candidate->valid, __ATOMIC_ACQUIRE) ||
        candidate->epoch != epoch) {
      entry = candidate;
      break;
    }
  }
  __atomic_store_n(&entry->valid, 0, __ATOMIC_RELEASE);
  entry->context = context;
  entry->generation = generation;
  entry->epoch = epoch;
  entry->source = source;
  entry->tile_row = tile_row;
  entry->tile_col = tile_col;
  entry->source_cols = source_cols;
  entry->site_id = site_id;
  return entry;
}

/* Adaptive controller state (real closed loop; not a no-op).
 *   omni_stall_accum  – sum of spin-wait iterations observed in wait()
 *   omni_stall_events – number of wait() calls contributing to the sum
 *   omni_eff_lookahead– current effective prefetch distance in [MIN,MAX];
 *                       adjusted by __omni_fetch_update_distance from the
 *                       measured average stall and consumed by the prefetch
 *                       path (async-gate + L2 prefetch-ahead depth). */
static unsigned omni_stall_accum = 0;
static int      omni_stall_events = 0;
static int               omni_eff_lookahead = MAX_LOOKAHEAD;

/* Dual-thread DAE scout (Phase 2).  Default off — identical to single-thread
 * software-pipelined V-DAE.  When on, signal() enqueues dma_wait+WH onto a
 * scout worker and returns; wait() only spins on the semaphore. */
static volatile int omni_dual_thread_dae = 0;

enum {
  OMNI_SLOT_IDLE = 0,
  OMNI_SLOT_PENDING = 1,
  OMNI_SLOT_READY = 2
};
static volatile int omni_scout_slot_state = OMNI_SLOT_IDLE;

/* Strong symbol from multithreading/OmniFetchScout.cpp overrides this weak
 * sync fallback when the device .so links hexagon_mlir_async_runtime. */
__attribute__((weak)) void hexagon_runtime_scout_enqueue(void (*fn)(void *),
                                                         void *arg) {
  if (fn)
    fn(arg);
}

void __omni_fetch_set_dual_thread_dae(int32_t enable) {
  __atomic_store_n(&omni_dual_thread_dae, enable ? 1 : 0, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_scout_slot_state, OMNI_SLOT_IDLE, __ATOMIC_RELEASE);
}

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

typedef int omni_sem_t;
static omni_sem_t omni_sem_pool[OMNI_SEM_POOL_SIZE];
static unsigned omni_sem_generation[OMNI_SEM_POOL_SIZE];
static int omni_sem_alloc_idx = 0;
/* Number of published async descriptors; declared here because signal() uses
 * it before the descriptor type and ring are defined below. */
static int omni_async_count = 0;

/* Defined with the async-DMA section below; signal() finishes deferred WH. */
static void omni_async_complete(void);

static void omni_sem_pool_init(void) {
  static int initialised = 0;
  int expected = 0;
  if (__atomic_compare_exchange_n(&initialised, &expected, 1, 0,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    for (int i = 0; i < OMNI_SEM_POOL_SIZE; ++i)
      __atomic_store_n(&omni_sem_pool[i], 0, __ATOMIC_RELAXED);
    for (int i = 0; i < OMNI_SEM_POOL_SIZE; ++i)
      __atomic_store_n(&omni_sem_generation[i], 0, __ATOMIC_RELAXED);
    __atomic_store_n(&initialised, 2, __ATOMIC_RELEASE);
    return;
  }
  while (__atomic_load_n(&initialised, __ATOMIC_ACQUIRE) != 2) {
#ifdef __hexagon__
    __asm__ volatile("pause(#64)");
#endif
  }
}

static void omni_scout_complete_and_ready(void *sem_idx_as_ptr) {
  __atomic_store_n(&omni_scout_slot_state, OMNI_SLOT_PENDING, __ATOMIC_RELEASE);
  omni_async_complete();
  __atomic_store_n(&omni_scout_slot_state, OMNI_SLOT_READY, __ATOMIC_RELEASE);
  int32_t sem_handle = (int32_t)(intptr_t)sem_idx_as_ptr;
  unsigned sem_idx = (unsigned)sem_handle & (OMNI_SEM_POOL_SIZE - 1);
  unsigned generation = (unsigned)sem_handle / OMNI_SEM_POOL_SIZE;
  if (sem_idx < OMNI_SEM_POOL_SIZE &&
      __atomic_load_n(&omni_sem_generation[sem_idx], __ATOMIC_ACQUIRE) ==
          generation)
    __atomic_fetch_add(&omni_sem_pool[sem_idx], 1, __ATOMIC_RELEASE);
}

int32_t __omni_fetch_create_sem(void) {
  omni_sem_pool_init();
  int32_t idx =
      __atomic_fetch_add(&omni_sem_alloc_idx, 1, __ATOMIC_RELAXED) %
      OMNI_SEM_POOL_SIZE;
  unsigned generation =
      __atomic_add_fetch(&omni_sem_generation[idx], 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&omni_sem_pool[idx], 0, __ATOMIC_RELEASE);
  return (int32_t)(generation * OMNI_SEM_POOL_SIZE + (unsigned)idx);
}

void __omni_fetch_signal(int32_t sem_handle) {
  unsigned sem_idx = (unsigned)sem_handle & (OMNI_SEM_POOL_SIZE - 1);
  unsigned generation = (unsigned)sem_handle / OMNI_SEM_POOL_SIZE;
  if (sem_idx >= OMNI_SEM_POOL_SIZE ||
      __atomic_load_n(&omni_sem_generation[sem_idx], __ATOMIC_ACQUIRE) !=
          generation)
    return;
  /* After Mm: finish any deferred HexKL WH (dma_wait + layout) into the idle
   * ping-pong slot so the next iteration's wait only synchronizes the sem. */
  if (__atomic_load_n(&omni_dual_thread_dae, __ATOMIC_ACQUIRE) &&
      __atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) > 0) {
    /* Kick completion onto the scout; do NOT dma_wait+WH on the compute
     * thread.  Scout posts the semaphore when the tile is READY. */
    hexagon_runtime_scout_enqueue(omni_scout_complete_and_ready,
                                  (void *)(intptr_t)sem_handle);
    return;
  }
  omni_async_complete();
  __atomic_fetch_add(&omni_sem_pool[sem_idx], 1, __ATOMIC_RELEASE);
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
extern uint32_t hexagon_runtime_dma2d_start(
    void *src, int srcAS, void *dst, int dstAS, uint32_t width,
    uint32_t height, uint32_t srcStride, uint32_t dstStride, int bypassCacheSrc,
    int bypassCacheDst, int isOrdered, uint32_t cacheAllocationPolicy,
    int *status);
extern void hexagon_runtime_dma_wait(uint32_t token);

#define OMNI_STAGE_ELEMS (32 * 32)
#define OMNI_STAGE_SLOTS 4

static uint16_t omni_stage[OMNI_STAGE_SLOTS][OMNI_STAGE_ELEMS];

typedef struct {
  int active;
  int phase;
  uint32_t token;
  void *dest; /* full HexKL VTCM i8 slab when hexkl_deferred */
  const void *src; /* full DDR matrix when hexkl_deferred */
  int32_t elem_bytes;
  int32_t num_elems;
  int32_t layout_kind;
  int stage_slot;
  /* HexKL weight tile metadata (valid when hexkl_deferred != 0). */
  int hexkl_deferred;
  int32_t tile_row;
  int32_t tile_col;
  int32_t src_cols;
  int32_t weight_off; /* absolute byte offset into dest VTCM slab */
  int32_t vtcm_stage_off; /* >=0: packed tile lives at dest+off (DMA→VTCM) */
  int32_t site_id; /* >=0: item-4 cache identity for item-5 hybrid mode */
} OmniAsyncJob;

enum {
  OMNI_JOB_IDLE = 0,
  OMNI_JOB_LOAD_PENDING = 1,
  OMNI_JOB_LOAD_READY = 2,
  OMNI_JOB_TRANSFORM_READY = 3
};

/* Single-producer/single-consumer descriptor ring. The compiler emits at most
 * one async weight request per loop iteration; signal() consumes the oldest
 * request. Release/acquire publication lets the optional scout consume a
 * descriptor without draining unrelated younger requests. */
static OmniAsyncJob omni_async_jobs[OMNI_STAGE_SLOTS];
static int omni_async_head = 0;
static int omni_async_tail = 0;
static int omni_async_consumer_lock = 0;

/* Forward decls – defined below. */
static void hmx_weight_gather(const void *src, void *dest, int32_t elem_bytes,
                              int32_t M, int32_t K);
static void hmx_activation_gather(const void *src, void *dest,
                                  int32_t elem_bytes, int32_t N, int32_t C,
                                  int32_t H, int32_t W);
#ifdef __hexagon__
static void omni_l2fetch(const void *ptr, uint32_t total_bytes);
#endif

/* Declared below; used as DMA fallback for strided weight tiles. */
void __omni_fetch_copy2d(const void *src, void *dest, int32_t elem_bytes,
                         int32_t rows, int32_t cols,
                         int32_t src_row_stride_elems,
                         int32_t dst_row_stride_elems);

static void omni_pack_weight_tile_to_stage(const _Float16 *src, int32_t tile_row,
                                           int32_t tile_col, int32_t src_cols,
                                           void *stage) {
  const _Float16 *row0 =
      src + (size_t)tile_row * 32 * (size_t)src_cols + (size_t)tile_col * 32;
  __omni_fetch_copy2d(row0, stage, /*elem_bytes=*/2, /*rows=*/32, /*cols=*/32,
                      /*src_row_stride_elems=*/src_cols,
                      /*dst_row_stride_elems=*/32);
}

static void omni_async_complete(void) {
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) <= 0)
    return;
  while (__atomic_exchange_n(&omni_async_consumer_lock, 1,
                             __ATOMIC_ACQUIRE)) {
#ifdef __hexagon__
    __asm__ volatile("pause(#64)");
#endif
  }
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) <= 0) {
    __atomic_store_n(&omni_async_consumer_lock, 0, __ATOMIC_RELEASE);
    return;
  }
  int job_idx = __atomic_load_n(&omni_async_head, __ATOMIC_RELAXED);
  OmniAsyncJob *job = &omni_async_jobs[job_idx];
  if (!__atomic_load_n(&job->active, __ATOMIC_ACQUIRE)) {
    __atomic_store_n(&omni_async_consumer_lock, 0, __ATOMIC_RELEASE);
    return;
  }
#ifdef __hexagon__
  if (job->token != 0)
    hexagon_runtime_dma_wait(job->token);
#endif
  __atomic_store_n(&job->phase, OMNI_JOB_LOAD_READY, __ATOMIC_RELEASE);
  void *dest = job->dest;
  int32_t eb = job->elem_bytes;
  int32_t ne = job->num_elems;

  if (job->hexkl_deferred) {
#ifdef __hexagon__
    /* Finish WH into VTCM after Mm (invoked from signal). */
    if (dest && job->weight_off >= 0) {
      const _Float16 *whSrc = NULL;
      if (job->vtcm_stage_off >= 0) {
        whSrc = (const _Float16 *)((char *)dest + job->vtcm_stage_off);
      } else if (job->stage_slot >= 0) {
        whSrc = (const _Float16 *)omni_stage[job->stage_slot];
      }
      if (whSrc) {
        hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)dest, (uint32_t)job->weight_off,
                                     whSrc,
                                     0, 0, 32);
      } else if (job->src) {
        hexkl_micro_hmx_rm_to_wh_f16(
            (uint8_t *)dest, (uint32_t)job->weight_off,
            (const _Float16 *)job->src, (uint32_t)job->tile_row,
            (uint32_t)job->tile_col, (uint32_t)job->src_cols);
      }
      if (job->site_id >= 0) {
        OmniWhCacheEntry *entry = omni_wh_cache_reserve(
            job->src, job->tile_row, job->tile_col, job->src_cols,
            job->site_id);
        memcpy(entry->data, (char *)dest + job->weight_off,
               OMNI_WH_TILE_BYTES);
        if (entry->epoch ==
            __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE))
          __atomic_store_n(&entry->valid, 1, __ATOMIC_RELEASE);
      }
    }
#else
    (void)dest;
    (void)eb;
    (void)ne;
#endif
    __atomic_store_n(&job->phase, OMNI_JOB_TRANSFORM_READY,
                     __ATOMIC_RELEASE);
    __atomic_store_n(&job->active, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&omni_async_head,
                     (job_idx + 1) % OMNI_STAGE_SLOTS, __ATOMIC_RELAXED);
    __atomic_fetch_sub(&omni_async_count, 1, __ATOMIC_RELEASE);
    __atomic_store_n(&omni_async_consumer_lock, 0, __ATOMIC_RELEASE);
    return;
  }

  const void *staged = omni_stage[job->stage_slot];
  switch (job->layout_kind) {
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
  __atomic_store_n(&job->active, 0, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_async_head, (job_idx + 1) % OMNI_STAGE_SLOTS,
                   __ATOMIC_RELAXED);
  __atomic_fetch_sub(&omni_async_count, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_async_consumer_lock, 0, __ATOMIC_RELEASE);
}

#ifdef __hexagon__
static void omni_l2fetch_weight_tile(const _Float16 *src, int32_t tile_row,
                                     int32_t tile_col, int32_t src_cols) {
  const _Float16 *row0 =
      src + (size_t)tile_row * 32 * (size_t)src_cols + (size_t)tile_col * 32;
  for (int32_t r = 0; r < 32; ++r)
    omni_l2fetch(row0 + (size_t)r * (size_t)src_cols, 32u * 2u);
}
#endif

/* Phase 2b HexKL kick: dma2d-pack the next strided weight tile into VTCM
 * staging (flatOff) when stage_off>=0, else DDR omni_stage.  Returns so Mm
 * overlaps the transfer; __omni_fetch_signal drains DMA and WHs into the idle
 * ping-pong weight slot (slab+weight_off). */
static int omni_async_kick_hexkl_weight(const void *src, void *dest,
                                        int32_t tile_row, int32_t tile_col,
                                        int32_t src_cols, int32_t weight_off,
                                        int32_t stage_off, int32_t site_id) {
  if (!src || !dest || src_cols <= 0 || tile_row < 0 || tile_col < 0 ||
      weight_off < 0)
    return 0;

  /* Backpressure only when every descriptor/staging slot is occupied. */
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >=
      OMNI_STAGE_SLOTS) {
    int spins = 0;
    while (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >=
               OMNI_STAGE_SLOTS &&
           spins < OMNI_SEM_MAX_SPIN) {
      ++spins;
#ifdef __hexagon__
      __asm__ volatile("pause(#255)");
#endif
    }
    if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >=
        OMNI_STAGE_SLOTS) {
      omni_async_complete();
      __atomic_fetch_or(&omni_error_flags, OMNI_ERROR_DESCRIPTOR_FULL,
                        __ATOMIC_RELAXED);
    }
  }

  int job_idx = __atomic_load_n(&omni_async_tail, __ATOMIC_RELAXED);
  OmniAsyncJob *job = &omni_async_jobs[job_idx];

  const _Float16 *row0 =
      (const _Float16 *)src +
      (size_t)tile_row * 32 * (size_t)src_cols + (size_t)tile_col * 32;
  const uint32_t width = 32u * 2u;
  const uint32_t height = 32u;
  const uint32_t srcStride = (uint32_t)src_cols * 2u;
  const uint32_t dstStride = 64u;

  uint32_t tok = 0;
  int slot = -1;
  int vtcmStage = -1;

#ifdef __hexagon__
  omni_l2fetch_weight_tile((const _Float16 *)src, tile_row, tile_col, src_cols);
  /* Do NOT l2fetch tile_row+ah unconditionally: omni_eff_lookahead can be up to
   * MAX_LOOKAHEAD while K-tiles are far fewer (e.g. GPT-2 K=768 → 24 tiles).
   * Unbounded ahead fetch Bad-VA'd the DSP (adb exit 13).  Bounded ahead
   * belongs in IR (PrefetchInsert already clamps nextKt) or needs an explicit
   * max_tile_row in the ABI. */
  int status = OMNI_DMA_OK;
  if (stage_off >= 0) {
    void *vtcmDst = (char *)dest + stage_off;
    tok = hexagon_runtime_dma2d_start(
        (void *)row0, OMNI_DDR, vtcmDst, OMNI_VTCM, width, height, srcStride,
        dstStride, /*bypassSrc=*/0, /*bypassDst=*/0, /*isOrdered=*/0,
        /*cacheAllocationPolicy=*/0, &status);
    if (status != OMNI_DMA_OK) {
      omni_pack_weight_tile_to_stage((const _Float16 *)src, tile_row, tile_col,
                                     src_cols, vtcmDst);
      tok = 0;
    }
    vtcmStage = stage_off;
  } else {
    slot = job_idx;
    tok = hexagon_runtime_dma2d_start(
        (void *)row0, OMNI_DDR, omni_stage[slot], OMNI_DDR, width, height,
        srcStride, dstStride, /*bypassSrc=*/0, /*bypassDst=*/0, /*isOrdered=*/0,
        /*cacheAllocationPolicy=*/0, &status);
    if (status != OMNI_DMA_OK) {
      omni_pack_weight_tile_to_stage((const _Float16 *)src, tile_row, tile_col,
                                     src_cols, omni_stage[slot]);
      tok = 0;
    }
  }
#else
  (void)stage_off;
  slot = job_idx;
  omni_pack_weight_tile_to_stage((const _Float16 *)src, tile_row, tile_col,
                                 src_cols, omni_stage[slot]);
#endif

  job->token = tok;
  job->dest = dest;
  job->src = src;
  job->elem_bytes = 2;
  job->num_elems = OMNI_STAGE_ELEMS;
  job->layout_kind = LAYOUT_HMX_WEIGHT;
  job->stage_slot = slot;
  job->hexkl_deferred = 1;
  job->tile_row = tile_row;
  job->tile_col = tile_col;
  job->src_cols = src_cols;
  job->weight_off = weight_off;
  job->vtcm_stage_off = vtcmStage;
  job->site_id = site_id;
  __atomic_store_n(&job->phase, OMNI_JOB_LOAD_PENDING, __ATOMIC_RELEASE);
  __atomic_store_n(&job->active, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_async_tail, (job_idx + 1) % OMNI_STAGE_SLOTS,
                   __ATOMIC_RELAXED);
  __atomic_fetch_add(&omni_async_count, 1, __ATOMIC_RELEASE);
  return 1;
}

static int omni_async_kick(const void *src, void *dest, int32_t elem_bytes,
                           int32_t num_elems, int32_t layout_kind) {
  uint32_t bytes = (uint32_t)elem_bytes * (uint32_t)num_elems;
  if (bytes == 0 || bytes > sizeof(omni_stage[0]) || !src || !dest)
    return 0;
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >=
      OMNI_STAGE_SLOTS)
    omni_async_complete();
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >=
      OMNI_STAGE_SLOTS)
    return 0;

  int job_idx = __atomic_load_n(&omni_async_tail, __ATOMIC_RELAXED);
  OmniAsyncJob *job = &omni_async_jobs[job_idx];
  int slot = job_idx;
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
  job->token = tok;
#else
  memcpy(omni_stage[slot], src, bytes);
  job->token = 0;
#endif
  job->dest = dest;
  job->src = src;
  job->elem_bytes = elem_bytes;
  job->num_elems = num_elems;
  job->layout_kind = layout_kind;
  job->stage_slot = slot;
  job->hexkl_deferred = 0;
  job->weight_off = -1;
  job->vtcm_stage_off = -1;
  job->site_id = -1;
  __atomic_store_n(&job->phase, OMNI_JOB_LOAD_PENDING, __ATOMIC_RELEASE);
  __atomic_store_n(&job->active, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_async_tail, (job_idx + 1) % OMNI_STAGE_SLOTS,
                   __ATOMIC_RELAXED);
  __atomic_fetch_add(&omni_async_count, 1, __ATOMIC_RELEASE);
  return 1;
}

void __omni_fetch_wait(int32_t sem_handle) {
  /* Sem only.  HexKL deferred WH is finished in signal() after Mm so the
   * DDR→stage DMA can overlap compute; completing in wait() corrupted results. */
  unsigned sem_idx = (unsigned)sem_handle & (OMNI_SEM_POOL_SIZE - 1);
  unsigned generation = (unsigned)sem_handle / OMNI_SEM_POOL_SIZE;
  if (sem_idx >= OMNI_SEM_POOL_SIZE ||
      __atomic_load_n(&omni_sem_generation[sem_idx], __ATOMIC_ACQUIRE) !=
          generation)
    return;
  int spins = 0;
  while (__atomic_load_n(&omni_sem_pool[sem_idx], __ATOMIC_ACQUIRE) <= 0) {
    if (__atomic_load_n(&omni_sem_generation[sem_idx], __ATOMIC_ACQUIRE) !=
        generation)
      return;
    if (++spins >= OMNI_SEM_MAX_SPIN) {
      __atomic_fetch_or(&omni_error_flags, OMNI_ERROR_SEM_TIMEOUT,
                        __ATOMIC_RELAXED);
      break;
    }
#ifdef __hexagon__
    __asm__ volatile("pause(#255)");
#endif
  }
  /* Feed the adaptive controller a real stall signal: spins reflects how long
   * the compute thread waited for the prefetched tile to become ready. */
  __atomic_fetch_add(&omni_stall_accum, (unsigned)spins, __ATOMIC_RELAXED);
  __atomic_fetch_add(&omni_stall_events, 1, __ATOMIC_RELAXED);
  int observed = __atomic_load_n(&omni_sem_pool[sem_idx], __ATOMIC_ACQUIRE);
  while (observed > 0 &&
         !__atomic_compare_exchange_n(&omni_sem_pool[sem_idx], &observed,
                                      observed - 1, 0, __ATOMIC_ACQ_REL,
                                      __ATOMIC_ACQUIRE)) {
  }
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
                                   int32_t src_cols, int32_t act_off,
                                   int32_t scr_off, int32_t src_rows) {
  if (elem_bytes <= 0 || num_elems <= 0 || !src)
    return;
  /* L2 hints may pass dest==src; real copies require a distinct dest. */
  if (layout_kind != LAYOUT_L2_HINT && !dest)
    return;

  /* HexKL-accurate activation path: CopySubmatrix + RmToAh into VTCM.
   * dest is the full HexKL i8 VTCM slab; act_off/scr_off are byte offsets. */
  if (layout_kind == LAYOUT_HMX_ACTIVATION && src_cols > 0 && tile_row >= 0 &&
      tile_col >= 0 && act_off >= 0 && scr_off >= 0 && src_rows > 0) {
#ifdef __hexagon__
    (void)elem_bytes;
    (void)num_elems;
    (void)index_map;
    (void)lookahead;
    hexkl_micro_hmx_copy_submatrix_to_f16(
        (uint8_t *)dest, (uint32_t)scr_off, (const _Float16 *)src,
        (uint32_t)tile_row, (uint32_t)tile_col, (uint32_t)src_rows,
        (uint32_t)src_cols);
    hexkl_micro_hmx_rm_to_ah_f16((uint8_t *)dest, (uint32_t)act_off,
                                 (uint32_t)scr_off);
#else
    (void)tile_row;
    (void)tile_col;
    (void)src_cols;
    (void)act_off;
    (void)scr_off;
    (void)src_rows;
    (void)index_map;
    (void)lookahead;
    /* Host stub: flat gather into dest+act_off is not HexKL-accurate; tests
     * that exercise this path are device-only. */
    hmx_activation_gather(src, (char *)dest + act_off, elem_bytes, 1,
                          num_elems > 0 ? num_elems : 1024, 1, 1);
#endif
    return;
  }

  /* HexKL-accurate weight path: same transform as MicroHMXRmToWhF16.
   * dest is the full VTCM i8 slab; weight_off (act_off ABI slot) is the
   * absolute byte offset.  stage_off (scr_off) >=0 → DMA pack into VTCM.
   * lookahead>0 → async dma2d + WH in signal(). */
  if (layout_kind == LAYOUT_HMX_WEIGHT && src_cols > 0 && tile_row >= 0 &&
      tile_col >= 0) {
    const int32_t weight_off = act_off;
    const int32_t stage_off = scr_off;
    void *wh_dest =
        weight_off >= 0 ? (char *)dest + weight_off : dest;
    if (lookahead == -1) {
      OmniWhCacheEntry *cached =
          omni_wh_cache_lookup(src, tile_row, tile_col, src_cols, src_rows);
      if (cached) {
        memcpy(wh_dest, cached->data, OMNI_WH_TILE_BYTES);
        __atomic_fetch_add(&omni_wh_hits, 1, __ATOMIC_RELAXED);
        return;
      }
      __atomic_fetch_add(&omni_wh_misses, 1, __ATOMIC_RELAXED);
    }
    if (lookahead > 0 && weight_off >= 0) {
      /* Item 5 + item 4 hybrid: a warm persistent tile is already WH-ready,
       * so publish it directly into the idle ping-pong slot. A miss enters
       * the explicit LOAD_PENDING -> LOAD_READY -> TRANSFORM_READY pipeline
       * and is inserted into the same generation-safe cache on completion. */
      if (src_rows >= 0) {
        OmniWhCacheEntry *cached =
            omni_wh_cache_lookup(src, tile_row, tile_col, src_cols, src_rows);
        if (cached) {
          memcpy(wh_dest, cached->data, OMNI_WH_TILE_BYTES);
          __atomic_fetch_add(&omni_wh_hits, 1, __ATOMIC_RELAXED);
          return;
        }
        __atomic_fetch_add(&omni_wh_misses, 1, __ATOMIC_RELAXED);
      }
      if (omni_async_kick_hexkl_weight(src, dest, tile_row, tile_col, src_cols,
                                       weight_off, stage_off, src_rows))
        return;
    }
#ifdef __hexagon__
    (void)elem_bytes;
    (void)num_elems;
    (void)index_map;
    (void)src_rows;
    if (weight_off >= 0) {
      hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)dest, (uint32_t)weight_off,
                                   (const _Float16 *)src, (uint32_t)tile_row,
                                   (uint32_t)tile_col, (uint32_t)src_cols);
    } else {
      /* Legacy view-pointer form (tile_params size 3): dest already at tile. */
      hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)dest, /*weight_offset=*/0,
                                   (const _Float16 *)src, (uint32_t)tile_row,
                                   (uint32_t)tile_col, (uint32_t)src_cols);
    }
#else
    (void)tile_row;
    (void)tile_col;
    (void)src_cols;
    (void)index_map;
    (void)act_off;
    (void)scr_off;
    (void)src_rows;
    {
      int32_t K = 32;
      int32_t M = (num_elems > 0) ? num_elems / K : 1;
      hmx_weight_gather(src, wh_dest, elem_bytes, M, K);
    }
#endif
    if (lookahead == -1) {
      OmniWhCacheEntry *entry =
          omni_wh_cache_reserve(src, tile_row, tile_col, src_cols, src_rows);
      memcpy(entry->data, wh_dest, OMNI_WH_TILE_BYTES);
      if (entry->epoch ==
          __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE))
        __atomic_store_n(&entry->valid, 1, __ATOMIC_RELEASE);
    }
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
    /* Plain memcpy only.  Do NOT l2fetch: prepack copies VTCM→DDR and
     * l2fetch on a VTCM address is undefined (corrupts GPT-2 WH cache while
     * small GEMMs can still look "ok" under loose atol). */
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
  /* Real adaptive control derived from software-measured stalls (no PMU /
   * hexagon_protos.h dependency).  Uses the spin counts accumulated in
   * __omni_fetch_wait since the previous update as the timeliness signal. */
  int events = __atomic_exchange_n(&omni_stall_events, 0, __ATOMIC_ACQ_REL);
  unsigned accum =
      __atomic_exchange_n(&omni_stall_accum, 0, __ATOMIC_ACQ_REL);

  int dist = (current_dist >= MIN_LOOKAHEAD && current_dist <= MAX_LOOKAHEAD)
                 ? current_dist
                 : __atomic_load_n(&omni_eff_lookahead, __ATOMIC_ACQUIRE);

  if (events > 0) {
    unsigned avg = accum / (unsigned)events;
    if (avg > STALL_THRESHOLD) {
      /* Tiles arriving too late – reach further ahead. */
      if (dist < MAX_LOOKAHEAD)
        ++dist;
    } else if (avg < STALL_THRESHOLD / 4u) {
      /* Comfortably ahead – pull back to reduce DMA/L2 pressure. */
      if (dist > MIN_LOOKAHEAD)
        --dist;
    }
  }

  __atomic_store_n(&omni_eff_lookahead, dist, __ATOMIC_RELEASE);
  return dist;
}
