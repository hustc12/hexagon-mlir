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
 * Semaphore – volatile counter + spin-wait.
 *
 * On Hexagon the DSP runs a single hardware thread per PD in the common
 * case, so a volatile counter is sufficient for the Access/Execute ordering
 * we need.  For multi-threaded DSP use, the memw_locked / store_locked
 * intrinsics provide the necessary atomicity without any OS dependency.
 * ------------------------------------------------------------------------- */
#define OMNI_SEM_POOL_SIZE 16

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
  omni_sem_pool[sem_idx]++;
}

void __omni_fetch_wait(int32_t sem_idx) {
  if ((unsigned)sem_idx >= OMNI_SEM_POOL_SIZE) return;
  /* NOTE: In the current single-threaded execution model, prefetch_insitu
   * runs synchronously (DMA wait is called inside prefetch_insitu before
   * returning). So by the time wait() is called, the data is already ready.
   * We simply decrement the counter without spinning. */
  if (omni_sem_pool[sem_idx] > 0)
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
 * __omni_fetch_prefetch_insitu
 * ------------------------------------------------------------------------- */
void __omni_fetch_prefetch_insitu(const void *src, void *dest,
                                   int32_t elem_bytes, int32_t num_elems,
                                   int32_t layout_kind, int32_t lookahead,
                                   const int32_t *index_map) {
  (void)lookahead;

  switch (layout_kind) {

  case LAYOUT_NONE: {
    uint32_t sz = (uint32_t)(num_elems * elem_bytes);
    /* Use memcpy: dest is DDR (heap), not VTCM, so DMA is not applicable.
     * When true VTCM allocation is wired in, switch back to async DMA. */
    memcpy(dest, src, (size_t)sz);
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
      memcpy(dest, src, (size_t)(num_elems * elem_bytes));
    break;
  }

  default:
    memcpy(dest, src, (size_t)(num_elems * elem_bytes));
    break;
  }
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
