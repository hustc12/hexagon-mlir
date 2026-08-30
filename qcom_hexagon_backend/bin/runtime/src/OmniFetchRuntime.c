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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __hexagon__
#include "HAP_user_pmu.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
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
int hexkl_micro_hmx_copy_f16_to_submatrix(
    uint8_t *vtcm_base, uint32_t in_offset, _Float16 *output_matrix,
    uint32_t tile_row, uint32_t tile_col, uint32_t output_rows,
    uint32_t output_cols);
#endif

/* -------------------------------------------------------------------------
 * Layout kind constants – must match OmniFetchOps.td enum ordinals
 * ------------------------------------------------------------------------- */
#define LAYOUT_NONE 0
#define LAYOUT_HMX_WEIGHT 1
#define LAYOUT_HMX_ACTIVATION 2
#define LAYOUT_CUSTOM 3
#define LAYOUT_L2_HINT 4
#define LAYOUT_HMX_WEIGHT_DEQUANT_I8 5

/* P5l consumer-driven HMX drain.  Clang vectorizes the full-width inner loop
 * for HVX; boundary tiles retain the same clipped semantics as HexKL's native
 * drain.  Fusing here avoids a DDR-resident matmul intermediate and a second
 * whole-result bias traversal. */
void alps_hmx_copy_f16_bias_to_submatrix(
    uint8_t *vtcm_base, uint32_t in_offset, const _Float16 *bias,
    _Float16 *dst, int32_t tile_row, int32_t tile_col, int32_t output_rows,
    int32_t output_cols) {
  const _Float16 *src = (const _Float16 *)(vtcm_base + in_offset);
  int32_t row0 = tile_row * 32;
  int32_t col0 = tile_col * 32;
  int32_t rows = output_rows - row0;
  int32_t cols = output_cols - col0;
  if (rows > 32)
    rows = 32;
  if (cols > 32)
    cols = 32;
  if (rows <= 0 || cols <= 0)
    return;
#ifdef __hexagon__
  if (cols == 32) {
    _Float16 bias_pair[64] __attribute__((aligned(128)));
    _Float16 sum_pair[64] __attribute__((aligned(128)));
    for (int32_t c = 0; c < 32; ++c) {
      bias_pair[c] = bias[col0 + c];
      bias_pair[32 + c] = bias[col0 + c];
    }
    HVX_Vector vbias = *(const HVX_Vector *)bias_pair;
    int32_t r = 0;
    for (; r + 1 < rows; r += 2) {
      /* Each pair of flat 32-element VTCM rows is exactly one 128-byte HVX
       * vector.  The final rows may be separated by the matrix stride in DDR,
       * so use two bounded 64-byte copies after the vector add. */
      HVX_Vector values = *(const HVX_Vector *)(src + r * 32);
      *(HVX_Vector *)sum_pair = Q6_Vhf_vadd_VhfVhf(values, vbias);
      memcpy(dst + (row0 + r) * output_cols + col0, sum_pair, 64);
      memcpy(dst + (row0 + r + 1) * output_cols + col0, sum_pair + 32, 64);
    }
    if (r < rows)
      for (int32_t c = 0; c < 32; ++c)
        dst[(row0 + r) * output_cols + col0 + c] =
            src[r * 32 + c] + bias[col0 + c];
    return;
  }
#endif
  for (int32_t r = 0; r < rows; ++r) {
    for (int32_t c = 0; c < cols; ++c)
      dst[(row0 + r) * output_cols + col0 + c] =
          src[r * 32 + c] + bias[col0 + c];
  }
}

/* -------------------------------------------------------------------------
 * Adaptive prefetch parameters
 * ------------------------------------------------------------------------- */
#define MIN_LOOKAHEAD 1
#define MAX_LOOKAHEAD 8
#define STALL_THRESHOLD 8000u

#define OMNI_ERROR_SEM_TIMEOUT (1u << 0)
#define OMNI_ERROR_DESCRIPTOR_FULL (1u << 1)
#define OMNI_ERROR_DESCRIPTOR_STALE (1u << 2)
#define OMNI_ERROR_DESCRIPTOR_STATE (1u << 3)
#define OMNI_ERROR_CONTEXT_BUSY (1u << 4)
static unsigned omni_error_flags = 0;

uint32_t __omni_fetch_get_and_clear_errors(void) {
  return __atomic_exchange_n(&omni_error_flags, 0, __ATOMIC_ACQ_REL);
}

/* -------------------------------------------------------------------------
 * ALPS P3a exact-readiness contexts and descriptors.
 *
 * Storage is bounded and runtime-owned, but ownership is never implicit:
 * every handle carries a generation and every descriptor records the exact
 * invocation/value-version/tile/layout/tier tuple.  This is deliberately
 * separate from the legacy process-global async ring.  P3b will attach the
 * UserDMA token and scout execution to these descriptors.
 * ------------------------------------------------------------------------- */
#define ALPS_CONTEXT_SLOTS 4
#define ALPS_DESCRIPTORS_PER_CONTEXT 8
#define ALPS_DESCRIPTOR_SLOTS                                                  \
  (ALPS_CONTEXT_SLOTS * ALPS_DESCRIPTORS_PER_CONTEXT)

enum {
  ALPS_DESC_FREE = 0,
  ALPS_DESC_LOAD_PENDING = 1,
  ALPS_DESC_LAYOUT_PENDING = 2,
  ALPS_DESC_READY = 3,
  ALPS_DESC_CONSUMING = 4,
  ALPS_DESC_FAILED = 5
};

typedef struct {
  volatile int state;
  uint32_t generation;
  uint32_t context_generation;
  int64_t value_version;
  int64_t tile;
  int32_t layout;
  int32_t source_tier;
  int32_t destination_tier;
  uint32_t dma_token;
  int32_t dma_active;
  void *dest;
  const void *src;
  int32_t tile_row;
  int32_t tile_col;
  int32_t source_cols;
  int32_t weight_offset;
  int32_t stage_offset;
  int32_t stage_slot;
  int32_t dma_credit_owned;
} AlpsExactDescriptor;

typedef struct {
  volatile int in_use;
  uint32_t generation;
  AlpsExactDescriptor descriptors[ALPS_DESCRIPTORS_PER_CONTEXT];
} AlpsInvocationContext;

static AlpsInvocationContext alps_contexts[ALPS_CONTEXT_SLOTS];
static unsigned alps_context_cursor = 0;
static uint64_t alps_descriptor_acquired = 0;
static uint64_t alps_descriptor_consumed = 0;
static uint64_t alps_descriptor_released = 0;
static uint64_t alps_descriptor_failures = 0;
static uint64_t alps_exact_dma_kicks = 0;
static uint64_t alps_exact_dma_completed = 0;
static uint64_t alps_exact_scout_completed = 0;
static uint64_t alps_exact_sync_fallbacks = 0;
static uint64_t alps_exact_consume_waits = 0;
static uint64_t alps_exact_credit_fallbacks = 0;
static uint64_t alps_exact_dma_timeouts = 0;
/* P4A is separately configured by the generated launcher.  It never changes
 * legality or representation; it only throttles the already-legal P3b DMA
 * path for subsequent windows. */
#define ALPS_P4A_WINDOW_COMPLETIONS 64u
#define ALPS_P4A_PMU_UNAVAILABLE 0u
#define ALPS_P4A_PMU_AVAILABLE 1u
#define ALPS_P4A_PMU_READ_FAILED 2u
static int alps_p4a_enabled = 0;
static int alps_p4a_dma_allowed = 1;
static uint64_t alps_p4a_windows = 0;
static uint64_t alps_p4a_throttle_decisions = 0;
static uint64_t alps_p4a_hold_decisions = 0;
static uint64_t alps_p4a_dma_suppressed = 0;
static uint64_t alps_p4a_window_completions = 0;
static uint64_t alps_p4a_window_poll_retries = 0;
static uint64_t alps_p4a_total_poll_retries = 0;
static uint64_t alps_p4a_issue_cycles = 0;
static uint64_t alps_p4a_poll_cycles = 0;
static uint32_t alps_p4a_pmu_status = ALPS_P4A_PMU_UNAVAILABLE;
static uint32_t alps_p4a_pmu_reads = 0;
static uint32_t alps_p4a_pmu_delta[4];
#ifdef __hexagon__
static HAP_pmu_group_config_t alps_p4a_pmu_group;
static uint32_t alps_p4a_pmu_previous[4];
#endif

static uint64_t alps_read_pcycles(void) {
#ifdef __hexagon__
  uint64_t value;
  __asm__ volatile("%[value] = C15:14" : [value] "=r"(value));
  return value;
#else
  return 0;
#endif
}

void __omni_fetch_p4a_configure(int32_t enable) {
  __atomic_store_n(&alps_p4a_enabled, enable ? 1 : 0, __ATOMIC_RELEASE);
  __atomic_store_n(&alps_p4a_dma_allowed, 1, __ATOMIC_RELEASE);
  alps_p4a_windows = 0;
  alps_p4a_throttle_decisions = 0;
  alps_p4a_hold_decisions = 0;
  alps_p4a_dma_suppressed = 0;
  alps_p4a_window_completions = 0;
  alps_p4a_window_poll_retries = 0;
  alps_p4a_total_poll_retries = 0;
  alps_p4a_issue_cycles = 0;
  alps_p4a_poll_cycles = 0;
  alps_p4a_pmu_status = ALPS_P4A_PMU_UNAVAILABLE;
  alps_p4a_pmu_reads = 0;
  memset(alps_p4a_pmu_delta, 0, sizeof(alps_p4a_pmu_delta));
#ifdef __hexagon__
  memset(&alps_p4a_pmu_group, 0, sizeof(alps_p4a_pmu_group));
  memset(alps_p4a_pmu_previous, 0, sizeof(alps_p4a_pmu_previous));
  if (!enable)
    return;
  /* V73 public events: UDMA active, DMPoll cycles, coherent-read stall, and
   * VTCM-write stall.  The SDK explicitly denies HAP user PMU in Unsigned PD;
   * failure is therefore an expected, reportable state, not silently faked. */
  alps_p4a_pmu_group.num_events = 4;
  alps_p4a_pmu_group.pmu_events[0] = 0x812f;
  alps_p4a_pmu_group.pmu_events[1] = 0x8133;
  alps_p4a_pmu_group.pmu_events[2] = 0x814b;
  alps_p4a_pmu_group.pmu_events[3] = 0x8150;
  if (HAP_register_pmu_group(&alps_p4a_pmu_group) == 0 &&
      HAP_read_pmu_group(&alps_p4a_pmu_group) == 0) {
    alps_p4a_pmu_status = ALPS_P4A_PMU_AVAILABLE;
    for (unsigned i = 0; i < 4; ++i)
      alps_p4a_pmu_previous[i] = alps_p4a_pmu_group.pmu_value[i];
  }
#else
  (void)enable;
#endif
}

static void alps_p4a_observe_dma_completion(unsigned poll_retries) {
  if (!__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE))
    return;
  __atomic_fetch_add(&alps_p4a_total_poll_retries, poll_retries,
                     __ATOMIC_RELAXED);
  alps_p4a_window_poll_retries += poll_retries;
  if (++alps_p4a_window_completions < ALPS_P4A_WINDOW_COMPLETIONS)
    return;
  ++alps_p4a_windows;
  uint32_t pmu_window_delta[4] = {0, 0, 0, 0};
#ifdef __hexagon__
  if (alps_p4a_pmu_status == ALPS_P4A_PMU_AVAILABLE) {
    if (HAP_read_pmu_group(&alps_p4a_pmu_group) == 0) {
      ++alps_p4a_pmu_reads;
      for (unsigned i = 0; i < 4; ++i) {
        uint32_t current = alps_p4a_pmu_group.pmu_value[i];
        pmu_window_delta[i] = current - alps_p4a_pmu_previous[i];
        alps_p4a_pmu_delta[i] += pmu_window_delta[i];
        alps_p4a_pmu_previous[i] = current;
      }
    } else {
      alps_p4a_pmu_status = ALPS_P4A_PMU_READ_FAILED;
    }
  }
#endif
  /* A zero-retry completion is the desired fully-hidden prefetch case.  Only
   * sustained late arrival (>=4 polls/completion) is rejected.  When HAP user
   * PMU is available, DMPoll is one of the observed events above; Unsigned PD
   * reports PMU unavailable and uses this exact wait-slot signal rather than
   * inventing counter values.  The threshold is fixed across models. */
  uint64_t average_retries =
      alps_p4a_window_poll_retries / ALPS_P4A_WINDOW_COMPLETIONS;
  uint64_t average_pmu_dmpoll =
      pmu_window_delta[1] / ALPS_P4A_WINDOW_COMPLETIONS;
  if (average_retries >= 4 || average_pmu_dmpoll >= 4) {
    __atomic_store_n(&alps_p4a_dma_allowed, 0, __ATOMIC_RELEASE);
    ++alps_p4a_throttle_decisions;
  } else {
    __atomic_store_n(&alps_p4a_dma_allowed, 1, __ATOMIC_RELEASE);
    ++alps_p4a_hold_decisions;
  }
  alps_p4a_window_completions = 0;
  alps_p4a_window_poll_retries = 0;
}
/* P3b deliberately starts with one retained UserDMA descriptor.  The upstream
 * UserDMA ring recycles completed hardware descriptors without a consumer
 * acknowledgement, so allowing multiple exact tokens would make delayed
 * polls observe a reused slot. */
static int alps_exact_dma_credit = 0;

static void alps_descriptor_fail(unsigned error) {
  __atomic_fetch_add(&alps_descriptor_failures, 1, __ATOMIC_RELAXED);
  __atomic_fetch_or(&omni_error_flags, error, __ATOMIC_RELAXED);
}

static int alps_decode_context(int32_t handle, unsigned *slot,
                               uint32_t *generation) {
  if (handle < 0)
    return 0;
  *slot = (unsigned)handle % ALPS_CONTEXT_SLOTS;
  *generation = (uint32_t)handle / ALPS_CONTEXT_SLOTS;
  AlpsInvocationContext *context = &alps_contexts[*slot];
  return __atomic_load_n(&context->in_use, __ATOMIC_ACQUIRE) &&
         __atomic_load_n(&context->generation, __ATOMIC_ACQUIRE) == *generation;
}

static int alps_decode_descriptor(int32_t handle,
                                  AlpsInvocationContext **context,
                                  AlpsExactDescriptor **descriptor) {
  if (handle < 0)
    return 0;
  unsigned flat = (unsigned)handle % ALPS_DESCRIPTOR_SLOTS;
  uint32_t generation = (uint32_t)handle / ALPS_DESCRIPTOR_SLOTS;
  unsigned context_slot = flat / ALPS_DESCRIPTORS_PER_CONTEXT;
  unsigned descriptor_slot = flat % ALPS_DESCRIPTORS_PER_CONTEXT;
  AlpsInvocationContext *candidate_context = &alps_contexts[context_slot];
  AlpsExactDescriptor *candidate_descriptor =
      &candidate_context->descriptors[descriptor_slot];
  if (!__atomic_load_n(&candidate_context->in_use, __ATOMIC_ACQUIRE) ||
      __atomic_load_n(&candidate_descriptor->generation, __ATOMIC_ACQUIRE) !=
          generation ||
      candidate_descriptor->context_generation !=
          __atomic_load_n(&candidate_context->generation, __ATOMIC_ACQUIRE))
    return 0;
  *context = candidate_context;
  *descriptor = candidate_descriptor;
  return 1;
}

int32_t __omni_fetch_invocation_begin(void) {
  unsigned start =
      __atomic_fetch_add(&alps_context_cursor, 1, __ATOMIC_RELAXED);
  for (unsigned probe = 0; probe < ALPS_CONTEXT_SLOTS; ++probe) {
    unsigned slot = (start + probe) % ALPS_CONTEXT_SLOTS;
    AlpsInvocationContext *context = &alps_contexts[slot];
    int expected = 0;
    if (!__atomic_compare_exchange_n(&context->in_use, &expected, 1, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      continue;
    uint32_t generation =
        (__atomic_add_fetch(&context->generation, 1, __ATOMIC_ACQ_REL) &
         UINT32_C(0x1ffffff));
    if (generation == 0) {
      generation = 1;
      __atomic_store_n(&context->generation, generation, __ATOMIC_RELEASE);
    }
    for (unsigned i = 0; i < ALPS_DESCRIPTORS_PER_CONTEXT; ++i)
      __atomic_store_n(&context->descriptors[i].state, ALPS_DESC_FREE,
                       __ATOMIC_RELEASE);
    return (int32_t)(generation * ALPS_CONTEXT_SLOTS + slot);
  }
  alps_descriptor_fail(OMNI_ERROR_CONTEXT_BUSY);
  return -1;
}

int32_t __omni_fetch_invocation_end(int32_t context_handle) {
  unsigned slot;
  uint32_t generation;
  if (!alps_decode_context(context_handle, &slot, &generation)) {
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STALE);
    return 0;
  }
  AlpsInvocationContext *context = &alps_contexts[slot];
  for (unsigned i = 0; i < ALPS_DESCRIPTORS_PER_CONTEXT; ++i) {
    if (__atomic_load_n(&context->descriptors[i].state, __ATOMIC_ACQUIRE) !=
        ALPS_DESC_FREE) {
      alps_descriptor_fail(OMNI_ERROR_CONTEXT_BUSY);
      return 0;
    }
  }
  __atomic_store_n(&context->in_use, 0, __ATOMIC_RELEASE);
  return 1;
}

int32_t __omni_fetch_descriptor_acquire(int32_t context_handle,
                                        int64_t value_version, int64_t tile,
                                        int32_t layout, int32_t source_tier,
                                        int32_t destination_tier) {
  unsigned context_slot;
  uint32_t context_generation;
  if (!alps_decode_context(context_handle, &context_slot,
                           &context_generation)) {
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STALE);
    return -1;
  }
  AlpsInvocationContext *context = &alps_contexts[context_slot];
  for (unsigned slot = 0; slot < ALPS_DESCRIPTORS_PER_CONTEXT; ++slot) {
    AlpsExactDescriptor *descriptor = &context->descriptors[slot];
    int expected = ALPS_DESC_FREE;
    if (!__atomic_compare_exchange_n(&descriptor->state, &expected,
                                     ALPS_DESC_LOAD_PENDING, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      continue;
    uint32_t generation =
        (__atomic_add_fetch(&descriptor->generation, 1, __ATOMIC_ACQ_REL) &
         UINT32_C(0x3ffffff));
    if (generation == 0) {
      generation = 1;
      __atomic_store_n(&descriptor->generation, generation, __ATOMIC_RELEASE);
    }
    descriptor->context_generation = context_generation;
    descriptor->value_version = value_version;
    descriptor->tile = tile;
    descriptor->layout = layout;
    descriptor->source_tier = source_tier;
    descriptor->destination_tier = destination_tier;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_fetch_add(&alps_descriptor_acquired, 1, __ATOMIC_RELAXED);
    unsigned flat = context_slot * ALPS_DESCRIPTORS_PER_CONTEXT + slot;
    return (int32_t)(generation * ALPS_DESCRIPTOR_SLOTS + flat);
  }
  alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_FULL);
  return -1;
}

int32_t __omni_fetch_descriptor_transition(int32_t descriptor_handle,
                                           int32_t expected_state,
                                           int32_t next_state) {
  AlpsInvocationContext *context;
  AlpsExactDescriptor *descriptor;
  if (!alps_decode_descriptor(descriptor_handle, &context, &descriptor)) {
    (void)context;
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STALE);
    return 0;
  }
  int legal = (expected_state == ALPS_DESC_LOAD_PENDING &&
               next_state == ALPS_DESC_LAYOUT_PENDING) ||
              (expected_state == ALPS_DESC_LAYOUT_PENDING &&
               next_state == ALPS_DESC_READY);
  int expected = (int)expected_state;
  if (!legal || !__atomic_compare_exchange_n(
                    &descriptor->state, &expected, (int)next_state, 0,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
    return 0;
  }
  return 1;
}

int32_t __omni_fetch_descriptor_consume(int32_t descriptor_handle,
                                        int64_t value_version, int64_t tile,
                                        int32_t layout, int32_t source_tier,
                                        int32_t destination_tier) {
  AlpsInvocationContext *context;
  AlpsExactDescriptor *descriptor;
  if (!alps_decode_descriptor(descriptor_handle, &context, &descriptor)) {
    (void)context;
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STALE);
    return 0;
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  if (descriptor->value_version != value_version || descriptor->tile != tile ||
      descriptor->layout != layout || descriptor->source_tier != source_tier ||
      descriptor->destination_tier != destination_tier) {
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STALE);
    return 0;
  }
  int expected = ALPS_DESC_READY;
  if (!__atomic_compare_exchange_n(&descriptor->state, &expected,
                                   ALPS_DESC_CONSUMING, 0, __ATOMIC_ACQ_REL,
                                   __ATOMIC_ACQUIRE)) {
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
    return 0;
  }
  __atomic_fetch_add(&alps_descriptor_consumed, 1, __ATOMIC_RELAXED);
  return 1;
}

int32_t __omni_fetch_descriptor_release(int32_t descriptor_handle) {
  AlpsInvocationContext *context;
  AlpsExactDescriptor *descriptor;
  if (!alps_decode_descriptor(descriptor_handle, &context, &descriptor)) {
    (void)context;
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STALE);
    return 0;
  }
  int expected = ALPS_DESC_CONSUMING;
  if (!__atomic_compare_exchange_n(&descriptor->state, &expected,
                                   ALPS_DESC_FREE, 0, __ATOMIC_ACQ_REL,
                                   __ATOMIC_ACQUIRE)) {
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
    return 0;
  }
  __atomic_fetch_add(&alps_descriptor_released, 1, __ATOMIC_RELAXED);
  return 1;
}

uint64_t __omni_fetch_descriptor_counts(void) {
  uint64_t acquired =
      __atomic_load_n(&alps_descriptor_acquired, __ATOMIC_RELAXED);
  uint64_t consumed =
      __atomic_load_n(&alps_descriptor_consumed, __ATOMIC_RELAXED);
  return (acquired << 32) | (consumed & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_descriptor_release_failures(void) {
  uint64_t released =
      __atomic_load_n(&alps_descriptor_released, __ATOMIC_RELAXED);
  uint64_t failures =
      __atomic_load_n(&alps_descriptor_failures, __ATOMIC_RELAXED);
  return (released << 32) | (failures & UINT64_C(0xffffffff));
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

/* M1: V73-aware L2 prefetch scheduler statistics.  V73 can terminate an
 * active l2fetch when a younger command is issued, and silently drops
 * generated addresses that leave the start page.  Keep those decisions
 * visible in model benchmark logs rather than treating l2fetch as an
 * unobservable hint. */
static unsigned omni_l2_issued = 0;
static unsigned omni_l2_busy_suppressed = 0;
static unsigned omni_l2_page_clipped = 0;
static unsigned omni_l2_unsupported = 0;
static uint64_t omni_l2_requested_bytes = 0;
static uint64_t omni_l2_issued_bytes = 0;
static unsigned omni_l2_budget_suppressed = 0;
static unsigned omni_l2_duplicate_suppressed = 0;
static unsigned omni_l2_max_commands = 0;
static uint64_t omni_l2_max_bytes = 0;
#define OMNI_L2_RECENT_REQUESTS 64
#define OMNI_L2_SEGMENTED_SITES 256
static uint64_t omni_l2_recent_requests[OMNI_L2_RECENT_REQUESTS];
static unsigned omni_l2_recent_count = 0;
static unsigned omni_l2_recent_cursor = 0;
static unsigned omni_l2_segmented_cursor[OMNI_L2_SEGMENTED_SITES];

/* Configure a per-invocation traffic envelope for OmniFetch. Zero limits keep
 * the legacy/unbounded behavior used by external prefetch baselines. */
void __omni_fetch_l2_configure(uint32_t max_commands, uint64_t max_bytes,
                               uint32_t recent_requests) {
  omni_l2_max_commands = max_commands;
  omni_l2_max_bytes = max_bytes;
  omni_l2_recent_count = recent_requests < OMNI_L2_RECENT_REQUESTS
                             ? recent_requests
                             : OMNI_L2_RECENT_REQUESTS;
  omni_l2_recent_cursor = 0;
  for (unsigned i = 0; i < OMNI_L2_RECENT_REQUESTS; ++i)
    omni_l2_recent_requests[i] = 0;
  for (unsigned i = 0; i < OMNI_L2_SEGMENTED_SITES; ++i)
    omni_l2_segmented_cursor[i] = 0;
  __atomic_store_n(&omni_l2_issued, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_busy_suppressed, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_page_clipped, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_unsupported, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_requested_bytes, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_issued_bytes, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_budget_suppressed, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&omni_l2_duplicate_suppressed, 0, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_l2_scheduler_counts(void) {
  uint64_t issued = __atomic_load_n(&omni_l2_issued, __ATOMIC_RELAXED);
  uint64_t busy = __atomic_load_n(&omni_l2_busy_suppressed, __ATOMIC_RELAXED);
  return (issued << 32) | (busy & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_l2_scheduler_limits(void) {
  uint64_t clipped = __atomic_load_n(&omni_l2_page_clipped, __ATOMIC_RELAXED);
  uint64_t unsupported =
      __atomic_load_n(&omni_l2_unsupported, __ATOMIC_RELAXED);
  return (clipped << 32) | (unsupported & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_l2_requested_bytes(void) {
  return __atomic_load_n(&omni_l2_requested_bytes, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_l2_issued_bytes(void) {
  return __atomic_load_n(&omni_l2_issued_bytes, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_l2_policy_suppressed(void) {
  uint64_t budget =
      __atomic_load_n(&omni_l2_budget_suppressed, __ATOMIC_RELAXED);
  uint64_t duplicate =
      __atomic_load_n(&omni_l2_duplicate_suppressed, __ATOMIC_RELAXED);
  return (budget << 32) | (duplicate & UINT64_C(0xffffffff));
}

/* Item 8: generation-safe compressed weight stream.  Each entry keeps one
 * 32x32 symmetric W8 tile (1 KiB) plus its scale instead of a 4 KiB WH tile.
 * A miss quantizes the immutable FP16 source once.  A hit reads the compressed
 * tile, dequantizes into a short-lived FP16 tile, and immediately performs
 * RM->WH, avoiding a separately materialized dequantized matrix. */
#define OMNI_W8_CACHE_SLOTS 512
#define OMNI_W8_TILE_ELEMS 1024
#define OMNI_W8_GROUP_ROWS 8
#define OMNI_W8_GROUPS (32 / OMNI_W8_GROUP_ROWS)
typedef struct {
  uint64_t context;
  uint32_t generation;
  uint32_t epoch;
  const void *source;
  int32_t tile_row;
  int32_t tile_col;
  int32_t source_cols;
  int32_t site_id;
  /* Group-wise symmetric scales: four K groups per output column.  The
   * compressed representation is 1024 B weights + 512 B scales, still 25%
   * smaller than the 2048 B FP16 tile while avoiding tile-wide outliers. */
  float scales[OMNI_W8_GROUPS * 32];
  volatile int valid;
  int8_t data[OMNI_W8_TILE_ELEMS] __attribute__((aligned(128)));
} OmniW8CacheEntry;

static OmniW8CacheEntry omni_w8_cache[OMNI_W8_CACHE_SLOTS];
static unsigned omni_w8_hits = 0;
static unsigned omni_w8_misses = 0;
static void hmx_weight_gather(const void *src, void *dest, int32_t elem_bytes,
                              int32_t M, int32_t K);

uint64_t __omni_fetch_w8_cache_stats(void) {
  uint64_t hits = __atomic_load_n(&omni_w8_hits, __ATOMIC_RELAXED);
  uint64_t misses = __atomic_load_n(&omni_w8_misses, __ATOMIC_RELAXED);
  return (hits << 32) | (misses & 0xffffffffu);
}

static unsigned omni_w8_hash(const void *source, int32_t tile_row,
                             int32_t tile_col, int32_t source_cols,
                             int32_t site_id, uint64_t context,
                             uint32_t generation) {
  /* The lowered source is often a function-local materialization whose
   * address changes between invocations.  Site ID + generation identify the
   * immutable logical weight; keeping the transient pointer in the key turns
   * every allocator address change into a false miss. */
  (void)source;
  uint64_t x = context;
  x ^= (uint64_t)generation * UINT64_C(0x9e3779b185ebca87);
  x ^= (uint64_t)(uint32_t)tile_row * UINT64_C(0x165667b19e3779f9);
  x ^= (uint64_t)(uint32_t)tile_col * UINT64_C(0x85ebca77c2b2ae63);
  x ^= (uint64_t)(uint32_t)source_cols * UINT64_C(0x27d4eb2f165667c5);
  x ^= (uint64_t)(uint32_t)site_id * UINT64_C(0xc2b2ae3d27d4eb4f);
  x ^= x >> 33;
  return (unsigned)x & (OMNI_W8_CACHE_SLOTS - 1);
}

static OmniW8CacheEntry *omni_w8_lookup_or_quantize(const _Float16 *source,
                                                    int32_t tile_row,
                                                    int32_t tile_col,
                                                    int32_t source_cols,
                                                    int32_t site_id) {
  uint64_t context = __atomic_load_n(&omni_wh_context, __ATOMIC_ACQUIRE);
  uint32_t generation = __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
  uint32_t epoch = __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE);
  unsigned slot = omni_w8_hash(source, tile_row, tile_col, source_cols, site_id,
                               context, generation);
  OmniW8CacheEntry *entry = &omni_w8_cache[slot];
  if (__atomic_load_n(&entry->valid, __ATOMIC_ACQUIRE) &&
      entry->context == context && entry->generation == generation &&
      entry->epoch == epoch && entry->tile_row == tile_row &&
      entry->tile_col == tile_col && entry->source_cols == source_cols &&
      entry->site_id == site_id) {
    __atomic_fetch_add(&omni_w8_hits, 1, __ATOMIC_RELAXED);
    return entry;
  }

  __atomic_store_n(&entry->valid, 0, __ATOMIC_RELEASE);
  const _Float16 *row0 = source + (size_t)tile_row * 32 * (size_t)source_cols +
                         (size_t)tile_col * 32;
  for (int group = 0; group < OMNI_W8_GROUPS; ++group)
    for (int c = 0; c < 32; ++c) {
      float max_abs = 0.0f;
      int row_begin = group * OMNI_W8_GROUP_ROWS;
      for (int rr = 0; rr < OMNI_W8_GROUP_ROWS; ++rr) {
        float value =
            (float)row0[(size_t)(row_begin + rr) * (size_t)source_cols + c];
        float abs_value = value < 0.0f ? -value : value;
        if (abs_value > max_abs)
          max_abs = abs_value;
      }
      float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
      entry->scales[group * 32 + c] = scale;
      float inv_scale = 1.0f / scale;
      for (int rr = 0; rr < OMNI_W8_GROUP_ROWS; ++rr) {
        int r = row_begin + rr;
        float scaled =
            (float)row0[(size_t)r * (size_t)source_cols + c] * inv_scale;
        int q = (int)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        if (q > 127)
          q = 127;
        if (q < -127)
          q = -127;
        entry->data[r * 32 + c] = (int8_t)q;
      }
    }
  entry->context = context;
  entry->generation = generation;
  entry->epoch = epoch;
  entry->source = source;
  entry->tile_row = tile_row;
  entry->tile_col = tile_col;
  entry->source_cols = source_cols;
  entry->site_id = site_id;
  __atomic_store_n(&entry->valid, 1, __ATOMIC_RELEASE);
  __atomic_fetch_add(&omni_w8_misses, 1, __ATOMIC_RELAXED);
  return entry;
}

static void omni_w8_dequant_to_wh(const _Float16 *source, void *dest,
                                  int32_t weight_off, int32_t tile_row,
                                  int32_t tile_col, int32_t source_cols,
                                  int32_t site_id) {
  OmniW8CacheEntry *entry = omni_w8_lookup_or_quantize(
      source, tile_row, tile_col, source_cols, site_id);
  _Float16 dequant[OMNI_W8_TILE_ELEMS] __attribute__((aligned(128)));
  for (int r = 0; r < 32; ++r)
    for (int c = 0; c < 32; ++c) {
      float scale = entry->scales[(r / OMNI_W8_GROUP_ROWS) * 32 + c];
      dequant[r * 32 + c] = (_Float16)((float)entry->data[r * 32 + c] * scale);
    }
#ifdef __hexagon__
  hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)dest, (uint32_t)weight_off, dequant,
                               0, 0, 32);
#else
  hmx_weight_gather(dequant, (char *)dest + weight_off, 2, 32, 32);
#endif
}

static unsigned omni_wh_cache_hash(const void *source, int32_t tile_row,
                                   int32_t tile_col, int32_t source_cols,
                                   int32_t site_id, uint64_t context,
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

void __omni_fetch_wh_cache_set_context(uint64_t context, uint32_t generation) {
  uint64_t old_context = __atomic_load_n(&omni_wh_context, __ATOMIC_ACQUIRE);
  uint32_t old_generation =
      __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
  if (old_context != context || old_generation != generation)
    __atomic_add_fetch(&omni_wh_epoch, 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&omni_wh_context, context, __ATOMIC_RELEASE);
  __atomic_store_n(&omni_wh_generation, generation, __ATOMIC_RELEASE);
}

void __omni_fetch_wh_cache_invalidate(uint64_t context, uint32_t generation) {
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
  uint32_t generation = __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
  uint32_t epoch = __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE);
  unsigned base = omni_wh_cache_hash(source, tile_row, tile_col, source_cols,
                                     site_id, context, generation);
  for (unsigned i = 0; i < OMNI_WH_CACHE_PROBES; ++i) {
    OmniWhCacheEntry *entry =
        &omni_wh_cache[(base + i) & (OMNI_WH_CACHE_SLOTS - 1)];
    if (__atomic_load_n(&entry->valid, __ATOMIC_ACQUIRE) &&
        entry->context == context && entry->generation == generation &&
        entry->epoch == epoch && entry->source == source &&
        entry->tile_row == tile_row && entry->tile_col == tile_col &&
        entry->source_cols == source_cols && entry->site_id == site_id)
      return entry;
  }
  return 0;
}

static OmniWhCacheEntry *
omni_wh_cache_reserve(const void *source, int32_t tile_row, int32_t tile_col,
                      int32_t source_cols, int32_t site_id) {
  uint64_t context = __atomic_load_n(&omni_wh_context, __ATOMIC_ACQUIRE);
  uint32_t generation = __atomic_load_n(&omni_wh_generation, __ATOMIC_ACQUIRE);
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
static int omni_stall_events = 0;
static int omni_eff_lookahead = MAX_LOOKAHEAD;

/* Dual-thread DAE scout (Phase 2).  Default off — identical to single-thread
 * software-pipelined V-DAE.  When on, signal() enqueues dma_wait+WH onto a
 * scout worker and returns; wait() only spins on the semaphore. */
static volatile int omni_dual_thread_dae = 0;

enum { OMNI_SLOT_IDLE = 0, OMNI_SLOT_PENDING = 1, OMNI_SLOT_READY = 2 };
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
#define OMNI_SEM_MAX_SPIN 0x100000

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
  int32_t idx = __atomic_fetch_add(&omni_sem_alloc_idx, 1, __ATOMIC_RELAXED) %
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
extern uint32_t
hexagon_runtime_dma2d_start(void *src, int srcAS, void *dst, int dstAS,
                            uint32_t width, uint32_t height, uint32_t srcStride,
                            uint32_t dstStride, int bypassCacheSrc,
                            int bypassCacheDst, int isOrdered,
                            uint32_t cacheAllocationPolicy, int *status);
extern void hexagon_runtime_dma_wait(uint32_t token);
extern int32_t hexagon_runtime_dma_poll(uint32_t token);

/* ALPS P5n HMX result evacuation.  Two 2 KiB RM tiles occupy one independent
 * 4 KiB VTCM allocation unit.  Waiting happens before HMX overwrites a slot;
 * start returns immediately so the descriptor can overlap the next HMX tile.
 * Boundary-row fallback remains in compiler IR and never enters this path. */
static uint32_t alps_hmx_drain_token[2];
static unsigned char alps_hmx_drain_active[2];
static uint64_t alps_hmx_drain_issued;
static uint64_t alps_hmx_drain_completed;
static uint64_t alps_hmx_drain_issued_bytes;
static uint64_t alps_hmx_drain_sync_fallbacks;

void alps_hmx_async_drain_wait_slot(int32_t slot) {
  if ((unsigned)slot >= 2u)
    return;
  if (alps_hmx_drain_active[slot]) {
#ifdef __hexagon__
    unsigned polls = 0;
    uint64_t poll_start = __atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE)
                              ? alps_read_pcycles()
                              : 0;
    while (!hexagon_runtime_dma_poll(alps_hmx_drain_token[slot])) {
      if (++polls >= OMNI_SEM_MAX_SPIN) {
        /* The descriptor is already in flight and its VTCM slot cannot be
         * reused.  Retire it with the runtime's blocking fallback, then make
         * the saturated window visible to traffic control. */
        hexagon_runtime_dma_wait(alps_hmx_drain_token[slot]);
        break;
      }
      __asm__ volatile("pause(#64)");
    }
    if (__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE)) {
      __atomic_fetch_add(&alps_p4a_poll_cycles,
                         alps_read_pcycles() - poll_start,
                         __ATOMIC_RELAXED);
      alps_p4a_observe_dma_completion(polls);
    }
    __atomic_fetch_add(&alps_hmx_drain_completed, 1, __ATOMIC_RELAXED);
#endif
    alps_hmx_drain_active[slot] = 0;
  }
}

static void alps_hmx_sync_drain_f16(void *vtcm, int32_t in_offset,
                                    _Float16 *dst, int32_t tile_row,
                                    int32_t tile_col, int32_t output_rows,
                                    int32_t output_cols, const _Float16 *src,
                                    _Float16 *out, uint32_t columns) {
#ifdef __hexagon__
  (void)src;
  (void)out;
  (void)columns;
  hexkl_micro_hmx_copy_f16_to_submatrix(
      (uint8_t *)vtcm, (uint32_t)in_offset, dst, (uint32_t)tile_row,
      (uint32_t)tile_col, (uint32_t)output_rows, (uint32_t)output_cols);
#else
  (void)vtcm;
  (void)in_offset;
  (void)dst;
  (void)tile_row;
  (void)tile_col;
  (void)output_rows;
  for (int32_t r = 0; r < 32; ++r)
    for (uint32_t c = 0; c < columns; ++c)
      out[(size_t)r * (size_t)output_cols + c] = src[(size_t)r * 32u + c];
#endif
}

void alps_hmx_async_drain_start_f16(void *vtcm, int32_t in_offset,
                                    _Float16 *dst, int32_t tile_row,
                                    int32_t tile_col, int32_t output_rows,
                                    int32_t output_cols, int32_t slot) {
  if (!vtcm || !dst || (unsigned)slot >= 2u || in_offset < 0 ||
      tile_row < 0 || tile_col < 0 || output_rows <= 0 || output_cols <= 0)
    return;

  int32_t row = tile_row * 32;
  int32_t col = tile_col * 32;
  if (row < 0 || row + 32 > output_rows || col < 0 || col >= output_cols)
    return;
  uint32_t columns = (uint32_t)(output_cols - col);
  if (columns > 32u)
    columns = 32u;

  _Float16 *src = (_Float16 *)((char *)vtcm + in_offset);
  _Float16 *out = dst + (size_t)row * (size_t)output_cols + (size_t)col;

  /* R never changes the HMX result representation.  Once the observed
   * window rejects more DMA traffic, execute the identical bounded drain
   * synchronously and leave the slot inactive. */
  if (__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE) &&
      !__atomic_load_n(&alps_p4a_dma_allowed, __ATOMIC_ACQUIRE)) {
    alps_hmx_sync_drain_f16(vtcm, in_offset, dst, tile_row, tile_col,
                            output_rows, output_cols, src, out, columns);
    __atomic_fetch_add(&alps_p4a_dma_suppressed, 1, __ATOMIC_RELAXED);
    return;
  }
#ifdef __hexagon__
  int status = OMNI_DMA_OK;
  uint64_t issue_start = __atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE)
                             ? alps_read_pcycles()
                             : 0;
  uint32_t token = hexagon_runtime_dma2d_start(
      src, OMNI_VTCM, out, OMNI_DDR, columns * 2u, 32u,
      /*srcStride=*/64u, /*dstStride=*/(uint32_t)output_cols * 2u,
      /*bypassCacheSrc=*/0, /*bypassCacheDst=*/0, /*isOrdered=*/0,
      /*cacheAllocationPolicy=*/0, &status);
  if (__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE))
    __atomic_fetch_add(&alps_p4a_issue_cycles,
                       alps_read_pcycles() - issue_start, __ATOMIC_RELAXED);
  if (status == OMNI_DMA_OK) {
    alps_hmx_drain_token[slot] = token;
    alps_hmx_drain_active[slot] = 1;
    __atomic_fetch_add(&alps_hmx_drain_issued, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&alps_hmx_drain_issued_bytes,
                       (uint64_t)columns * 2u * 32u, __ATOMIC_RELAXED);
    return;
  }
#endif
  /* Recoverable admission fallback: preserve correctness if the UserDMA ring
   * cannot accept the descriptor. */
  alps_hmx_sync_drain_f16(vtcm, in_offset, dst, tile_row, tile_col,
                          output_rows, output_cols, src, out, columns);
  __atomic_fetch_add(&alps_hmx_drain_sync_fallbacks, 1, __ATOMIC_RELAXED);
}

void alps_hmx_async_drain_flush(void) {
  alps_hmx_async_drain_wait_slot(0);
  alps_hmx_async_drain_wait_slot(1);
}

uint64_t alps_hmx_async_drain_counts(void) {
  uint64_t issued =
      __atomic_load_n(&alps_hmx_drain_issued, __ATOMIC_RELAXED);
  uint64_t completed =
      __atomic_load_n(&alps_hmx_drain_completed, __ATOMIC_RELAXED);
  return (issued << 32) | (completed & UINT64_C(0xffffffff));
}

uint64_t alps_hmx_async_drain_issued_bytes(void) {
  return __atomic_load_n(&alps_hmx_drain_issued_bytes, __ATOMIC_RELAXED);
}

uint64_t alps_hmx_async_drain_sync_fallbacks(void) {
  return __atomic_load_n(&alps_hmx_drain_sync_fallbacks, __ATOMIC_RELAXED);
}

#define OMNI_STAGE_ELEMS (32 * 32)
#define OMNI_STAGE_SLOTS 4

static uint16_t omni_stage[OMNI_STAGE_SLOTS][OMNI_STAGE_ELEMS];

typedef struct {
  int active;
  int phase;
  uint32_t token;
  void *dest;      /* full HexKL VTCM i8 slab when hexkl_deferred */
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
  int32_t weight_off;     /* absolute byte offset into dest VTCM slab */
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

static void omni_pack_weight_tile_to_stage(const _Float16 *src,
                                           int32_t tile_row, int32_t tile_col,
                                           int32_t src_cols, void *stage) {
  const _Float16 *row0 =
      src + (size_t)tile_row * 32 * (size_t)src_cols + (size_t)tile_col * 32;
  __omni_fetch_copy2d(row0, stage, /*elem_bytes=*/2, /*rows=*/32, /*cols=*/32,
                      /*src_row_stride_elems=*/src_cols,
                      /*dst_row_stride_elems=*/32);
}

/* P3b descriptor-owned staging.  Unlike omni_stage, a slot is addressed by
 * the exact descriptor and cannot be consumed through a process-global FIFO. */
static uint16_t alps_exact_stage[ALPS_DESCRIPTOR_SLOTS][OMNI_STAGE_ELEMS];

static int64_t alps_pack_tile_identity(int32_t tile_row, int32_t tile_col) {
  return (int64_t)(((uint64_t)(uint32_t)tile_row << 32) | (uint32_t)tile_col);
}

static int32_t alps_exact_find(int32_t context_handle, int64_t value_version,
                               int32_t tile_row, int32_t tile_col) {
  unsigned context_slot;
  uint32_t context_generation;
  if (!alps_decode_context(context_handle, &context_slot, &context_generation))
    return -1;
  AlpsInvocationContext *context = &alps_contexts[context_slot];
  int64_t tile = alps_pack_tile_identity(tile_row, tile_col);
  for (unsigned slot = 0; slot < ALPS_DESCRIPTORS_PER_CONTEXT; ++slot) {
    AlpsExactDescriptor *descriptor = &context->descriptors[slot];
    int state = __atomic_load_n(&descriptor->state, __ATOMIC_ACQUIRE);
    if (state == ALPS_DESC_FREE ||
        descriptor->context_generation != context_generation ||
        descriptor->value_version != value_version ||
        descriptor->tile != tile || descriptor->layout != LAYOUT_HMX_WEIGHT ||
        descriptor->source_tier != OMNI_DDR ||
        descriptor->destination_tier != OMNI_VTCM)
      continue;
    uint32_t generation =
        __atomic_load_n(&descriptor->generation, __ATOMIC_ACQUIRE);
    unsigned flat = context_slot * ALPS_DESCRIPTORS_PER_CONTEXT + slot;
    return (int32_t)(generation * ALPS_DESCRIPTOR_SLOTS + flat);
  }
  return -1;
}

static void alps_exact_sync_weight(const void *src, void *dest,
                                   int32_t tile_row, int32_t tile_col,
                                   int32_t src_cols, int32_t weight_offset) {
  if (!src || !dest || tile_row < 0 || tile_col < 0 || src_cols <= 0 ||
      weight_offset < 0)
    return;
#ifdef __hexagon__
  hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)dest, (uint32_t)weight_offset,
                               (const _Float16 *)src, (uint32_t)tile_row,
                               (uint32_t)tile_col, (uint32_t)src_cols);
#else
  uint16_t stage[OMNI_STAGE_ELEMS];
  omni_pack_weight_tile_to_stage((const _Float16 *)src, tile_row, tile_col,
                                 src_cols, stage);
  hmx_weight_gather(stage, (char *)dest + weight_offset, 2, 32, 32);
#endif
}

static void alps_exact_complete_impl(void *descriptor_handle_as_ptr,
                                     int completed_by_scout) {
  int32_t descriptor_handle = (int32_t)(intptr_t)descriptor_handle_as_ptr;
  AlpsInvocationContext *context;
  AlpsExactDescriptor *descriptor;
  if (!alps_decode_descriptor(descriptor_handle, &context, &descriptor))
    return;
  (void)context;
  int expected = ALPS_DESC_LOAD_PENDING;
  if (!__atomic_compare_exchange_n(&descriptor->state, &expected,
                                   ALPS_DESC_LAYOUT_PENDING, 0,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return;
#ifdef __hexagon__
  if (descriptor->dma_active) {
    int polls = 0;
    uint64_t poll_start = __atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE)
                              ? alps_read_pcycles()
                              : 0;
    while (!hexagon_runtime_dma_poll(descriptor->dma_token)) {
      if (++polls >= OMNI_SEM_MAX_SPIN) {
        __atomic_store_n(&descriptor->state, ALPS_DESC_FAILED,
                         __ATOMIC_RELEASE);
        __atomic_fetch_add(&alps_exact_dma_timeouts, 1, __ATOMIC_RELAXED);
        alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
        /* Keep the credit retained: the hardware descriptor and VTCM target
         * may still be live and therefore cannot be safely reused. */
        return;
      }
      __asm__ volatile("pause(#64)");
    }
    if (__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE)) {
      uint64_t poll_end = alps_read_pcycles();
      __atomic_fetch_add(&alps_p4a_poll_cycles, poll_end - poll_start,
                         __ATOMIC_RELAXED);
      alps_p4a_observe_dma_completion((unsigned)polls);
    }
    descriptor->dma_active = 0;
  }
#endif
  const _Float16 *stage = NULL;
  if (descriptor->stage_offset >= 0)
    stage =
        (const _Float16 *)((char *)descriptor->dest + descriptor->stage_offset);
  else if (descriptor->stage_slot >= 0)
    stage = (const _Float16 *)alps_exact_stage[descriptor->stage_slot];
#ifdef __hexagon__
  if (stage)
    hexkl_micro_hmx_rm_to_wh_f16((uint8_t *)descriptor->dest,
                                 (uint32_t)descriptor->weight_offset, stage, 0,
                                 0, 32);
  else
    alps_exact_sync_weight(descriptor->src, descriptor->dest,
                           descriptor->tile_row, descriptor->tile_col,
                           descriptor->source_cols, descriptor->weight_offset);
#else
  if (stage)
    hmx_weight_gather(
        stage, (char *)descriptor->dest + descriptor->weight_offset, 2, 32, 32);
#endif
  expected = ALPS_DESC_LAYOUT_PENDING;
  if (!__atomic_compare_exchange_n(&descriptor->state, &expected,
                                   ALPS_DESC_READY, 0, __ATOMIC_RELEASE,
                                   __ATOMIC_ACQUIRE)) {
    if (descriptor->dma_credit_owned) {
      descriptor->dma_credit_owned = 0;
      __atomic_store_n(&alps_exact_dma_credit, 0, __ATOMIC_RELEASE);
    }
    alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
    return;
  }
  if (descriptor->dma_credit_owned) {
    descriptor->dma_credit_owned = 0;
    __atomic_store_n(&alps_exact_dma_credit, 0, __ATOMIC_RELEASE);
  }
  __atomic_fetch_add(&alps_exact_dma_completed, 1, __ATOMIC_RELAXED);
  if (completed_by_scout)
    __atomic_fetch_add(&alps_exact_scout_completed, 1, __ATOMIC_RELAXED);
}

static void alps_exact_complete(void *descriptor_handle_as_ptr) {
  alps_exact_complete_impl(descriptor_handle_as_ptr, /*completed_by_scout=*/1);
}

int32_t __omni_fetch_exact_weight_kick(int32_t context_handle,
                                       int64_t value_version, const void *src,
                                       void *dest, int32_t tile_row,
                                       int32_t tile_col, int32_t src_cols,
                                       int32_t weight_offset,
                                       int32_t stage_offset) {
  if (!src || !dest || tile_row < 0 || tile_col < 0 || src_cols <= 0 ||
      weight_offset < 0)
    return 0;
  int64_t tile = alps_pack_tile_identity(tile_row, tile_col);
  int32_t descriptor_handle =
      __omni_fetch_descriptor_acquire(context_handle, value_version, tile,
                                      LAYOUT_HMX_WEIGHT, OMNI_DDR, OMNI_VTCM);
  AlpsInvocationContext *context;
  AlpsExactDescriptor *descriptor;
  if (descriptor_handle < 0 ||
      !alps_decode_descriptor(descriptor_handle, &context, &descriptor)) {
    alps_exact_sync_weight(src, dest, tile_row, tile_col, src_cols,
                           weight_offset);
    __atomic_fetch_add(&alps_exact_sync_fallbacks, 1, __ATOMIC_RELAXED);
    return 0;
  }
  (void)context;
  unsigned flat = (unsigned)descriptor_handle % ALPS_DESCRIPTOR_SLOTS;
  descriptor->dest = dest;
  descriptor->src = src;
  descriptor->tile_row = tile_row;
  descriptor->tile_col = tile_col;
  descriptor->source_cols = src_cols;
  descriptor->weight_offset = weight_offset;
  descriptor->stage_offset = stage_offset;
  descriptor->stage_slot = stage_offset >= 0 ? -1 : (int32_t)flat;
  descriptor->dma_token = 0;
  descriptor->dma_active = 0;
  descriptor->dma_credit_owned = 0;

  if (__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE) &&
      !__atomic_load_n(&alps_p4a_dma_allowed, __ATOMIC_ACQUIRE)) {
    alps_exact_sync_weight(src, dest, tile_row, tile_col, src_cols,
                           weight_offset);
    __omni_fetch_descriptor_transition(
        descriptor_handle, ALPS_DESC_LOAD_PENDING, ALPS_DESC_LAYOUT_PENDING);
    __omni_fetch_descriptor_transition(
        descriptor_handle, ALPS_DESC_LAYOUT_PENDING, ALPS_DESC_READY);
    __atomic_fetch_add(&alps_exact_sync_fallbacks, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&alps_p4a_dma_suppressed, 1, __ATOMIC_RELAXED);
    return 1;
  }

  int expected_credit = 0;
  if (!__atomic_compare_exchange_n(&alps_exact_dma_credit, &expected_credit, 1,
                                   0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    /* Preserve the exact descriptor lifecycle even when the one DMA credit is
     * busy.  The fallback produces the demanded WH tile synchronously and
     * publishes READY, so consume/release counters still close exactly. */
    alps_exact_sync_weight(src, dest, tile_row, tile_col, src_cols,
                           weight_offset);
    __omni_fetch_descriptor_transition(
        descriptor_handle, ALPS_DESC_LOAD_PENDING, ALPS_DESC_LAYOUT_PENDING);
    __omni_fetch_descriptor_transition(
        descriptor_handle, ALPS_DESC_LAYOUT_PENDING, ALPS_DESC_READY);
    __atomic_fetch_add(&alps_exact_sync_fallbacks, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&alps_exact_credit_fallbacks, 1, __ATOMIC_RELAXED);
    return 1;
  }
  descriptor->dma_credit_owned = 1;

  const _Float16 *row0 = (const _Float16 *)src +
                         (size_t)tile_row * 32 * (size_t)src_cols +
                         (size_t)tile_col * 32;
#ifdef __hexagon__
  int status = OMNI_DMA_OK;
  void *stage = stage_offset >= 0 ? (void *)((char *)dest + stage_offset)
                                  : (void *)alps_exact_stage[flat];
  int dstAS = stage_offset >= 0 ? OMNI_VTCM : OMNI_DDR;
  uint64_t issue_start = __atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE)
                             ? alps_read_pcycles()
                             : 0;
  descriptor->dma_token = hexagon_runtime_dma2d_start(
      (void *)row0, OMNI_DDR, stage, dstAS, /*width=*/64, /*height=*/32,
      (uint32_t)src_cols * 2u, /*dstStride=*/64, /*bypassSrc=*/0,
      /*bypassDst=*/0, /*isOrdered=*/0, /*cacheAllocationPolicy=*/0, &status);
  if (__atomic_load_n(&alps_p4a_enabled, __ATOMIC_ACQUIRE))
    __atomic_fetch_add(&alps_p4a_issue_cycles,
                       alps_read_pcycles() - issue_start, __ATOMIC_RELAXED);
  if (status != OMNI_DMA_OK) {
    descriptor->dma_credit_owned = 0;
    __atomic_store_n(&alps_exact_dma_credit, 0, __ATOMIC_RELEASE);
    alps_exact_sync_weight(src, dest, tile_row, tile_col, src_cols,
                           weight_offset);
    descriptor->dma_token = 0;
    __omni_fetch_descriptor_transition(
        descriptor_handle, ALPS_DESC_LOAD_PENDING, ALPS_DESC_LAYOUT_PENDING);
    __omni_fetch_descriptor_transition(
        descriptor_handle, ALPS_DESC_LAYOUT_PENDING, ALPS_DESC_READY);
    __atomic_fetch_add(&alps_exact_sync_fallbacks, 1, __ATOMIC_RELAXED);
    return 1;
  }
  /* Token zero is the first valid UserDMA ring token; status, not the numeric
   * token value, distinguishes a successfully issued transfer. */
  descriptor->dma_active = 1;
#else
  (void)row0;
  omni_pack_weight_tile_to_stage((const _Float16 *)src, tile_row, tile_col,
                                 src_cols, alps_exact_stage[flat]);
#endif
  __atomic_thread_fence(__ATOMIC_RELEASE);
  __atomic_fetch_add(&alps_exact_dma_kicks, 1, __ATOMIC_RELAXED);
  if (__atomic_load_n(&omni_dual_thread_dae, __ATOMIC_ACQUIRE))
    hexagon_runtime_scout_enqueue(alps_exact_complete,
                                  (void *)(intptr_t)descriptor_handle);
  return 1;
}

int32_t __omni_fetch_exact_weight_consume(int32_t context_handle,
                                          int64_t value_version,
                                          int32_t tile_row, int32_t tile_col) {
  int32_t descriptor_handle =
      alps_exact_find(context_handle, value_version, tile_row, tile_col);
  if (descriptor_handle < 0)
    return 0;
  AlpsInvocationContext *context;
  AlpsExactDescriptor *descriptor;
  if (!alps_decode_descriptor(descriptor_handle, &context, &descriptor))
    return 0;
  (void)context;
  int spins = 0;
  for (;;) {
    int state = __atomic_load_n(&descriptor->state, __ATOMIC_ACQUIRE);
    if (state == ALPS_DESC_READY)
      break;
    if (state == ALPS_DESC_FAILED) {
#ifdef __hexagon__
      __builtin_trap();
#endif
      return 0;
    }
    if (state == ALPS_DESC_LOAD_PENDING) {
      // If the scout has not claimed this descriptor by demand time, steal
      // the completion work on the consumer.  The LOAD_PENDING CAS in the
      // implementation makes this race-safe with the queued scout callback.
      alps_exact_complete_impl((void *)(intptr_t)descriptor_handle,
                               /*completed_by_scout=*/0);
      continue;
    }
    if (state != ALPS_DESC_LAYOUT_PENDING) {
      alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
      return 0;
    }
    if (++spins >= OMNI_SEM_MAX_SPIN) {
      // Never let an experimental P3b run occupy the DSP indefinitely.  A
      // timed-out layout owner cannot be stolen safely because it may still
      // write the destination.  Fail loudly on device rather than allowing
      // HMX to consume a non-READY tile; host contracts report false.
      alps_descriptor_fail(OMNI_ERROR_DESCRIPTOR_STATE);
#ifdef __hexagon__
      __builtin_trap();
#endif
      return 0;
    }
#ifdef __hexagon__
    __asm__ volatile("pause(#64)");
#endif
  }
  __atomic_fetch_add(&alps_exact_consume_waits, (uint64_t)spins,
                     __ATOMIC_RELAXED);
  return __omni_fetch_descriptor_consume(
      descriptor_handle, value_version,
      alps_pack_tile_identity(tile_row, tile_col), LAYOUT_HMX_WEIGHT, OMNI_DDR,
      OMNI_VTCM);
}

int32_t __omni_fetch_exact_weight_release(int32_t context_handle,
                                          int64_t value_version,
                                          int32_t tile_row, int32_t tile_col) {
  int32_t descriptor_handle =
      alps_exact_find(context_handle, value_version, tile_row, tile_col);
  if (descriptor_handle < 0)
    return 0;
  return __omni_fetch_descriptor_release(descriptor_handle);
}

uint64_t __omni_fetch_exact_dma_counts(void) {
  uint64_t kicks = __atomic_load_n(&alps_exact_dma_kicks, __ATOMIC_RELAXED);
  uint64_t completed =
      __atomic_load_n(&alps_exact_dma_completed, __ATOMIC_RELAXED);
  return (kicks << 32) | (completed & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_exact_overlap_counts(void) {
  uint64_t scout =
      __atomic_load_n(&alps_exact_scout_completed, __ATOMIC_RELAXED);
  uint64_t fallback =
      __atomic_load_n(&alps_exact_sync_fallbacks, __ATOMIC_RELAXED);
  return (scout << 32) | (fallback & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_exact_consume_waits(void) {
  return __atomic_load_n(&alps_exact_consume_waits, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_exact_control_counts(void) {
  uint64_t credit =
      __atomic_load_n(&alps_exact_credit_fallbacks, __ATOMIC_RELAXED);
  uint64_t timeouts =
      __atomic_load_n(&alps_exact_dma_timeouts, __ATOMIC_RELAXED);
  return (credit << 32) | (timeouts & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_p4a_window_counts(void) {
  uint64_t windows = __atomic_load_n(&alps_p4a_windows, __ATOMIC_RELAXED);
  uint64_t suppressed =
      __atomic_load_n(&alps_p4a_dma_suppressed, __ATOMIC_RELAXED);
  return (windows << 32) | (suppressed & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_p4a_decision_counts(void) {
  uint64_t throttle =
      __atomic_load_n(&alps_p4a_throttle_decisions, __ATOMIC_RELAXED);
  uint64_t hold = __atomic_load_n(&alps_p4a_hold_decisions, __ATOMIC_RELAXED);
  return (throttle << 32) | (hold & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_p4a_pmu_status_counts(void) {
  uint64_t status = __atomic_load_n(&alps_p4a_pmu_status, __ATOMIC_RELAXED);
  uint64_t reads = __atomic_load_n(&alps_p4a_pmu_reads, __ATOMIC_RELAXED);
  return (status << 32) | (reads & UINT64_C(0xffffffff));
}

uint64_t __omni_fetch_p4a_issue_cycles(void) {
  return __atomic_load_n(&alps_p4a_issue_cycles, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_p4a_poll_cycles(void) {
  return __atomic_load_n(&alps_p4a_poll_cycles, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_p4a_poll_retries(void) {
  return __atomic_load_n(&alps_p4a_total_poll_retries, __ATOMIC_RELAXED);
}

uint64_t __omni_fetch_p4a_pmu_values01(void) {
  return ((uint64_t)alps_p4a_pmu_delta[0] << 32) |
         (uint64_t)alps_p4a_pmu_delta[1];
}

uint64_t __omni_fetch_p4a_pmu_values23(void) {
  return ((uint64_t)alps_p4a_pmu_delta[2] << 32) |
         (uint64_t)alps_p4a_pmu_delta[3];
}

static void omni_async_complete(void) {
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) <= 0)
    return;
  while (__atomic_exchange_n(&omni_async_consumer_lock, 1, __ATOMIC_ACQUIRE)) {
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
                                     whSrc, 0, 0, 32);
      } else if (job->src) {
        hexkl_micro_hmx_rm_to_wh_f16(
            (uint8_t *)dest, (uint32_t)job->weight_off,
            (const _Float16 *)job->src, (uint32_t)job->tile_row,
            (uint32_t)job->tile_col, (uint32_t)job->src_cols);
      }
      if (job->site_id >= 0) {
        OmniWhCacheEntry *entry =
            omni_wh_cache_reserve(job->src, job->tile_row, job->tile_col,
                                  job->src_cols, job->site_id);
        memcpy(entry->data, (char *)dest + job->weight_off, OMNI_WH_TILE_BYTES);
        if (entry->epoch == __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE))
          __atomic_store_n(&entry->valid, 1, __ATOMIC_RELEASE);
      }
    }
#else
    (void)dest;
    (void)eb;
    (void)ne;
#endif
    __atomic_store_n(&job->phase, OMNI_JOB_TRANSFORM_READY, __ATOMIC_RELEASE);
    __atomic_store_n(&job->active, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&omni_async_head, (job_idx + 1) % OMNI_STAGE_SLOTS,
                     __ATOMIC_RELAXED);
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
static void omni_l2fetch_2d(const void *ptr, uint32_t width, uint32_t height,
                            uint32_t stride);

static void omni_l2fetch_weight_tile(const _Float16 *src, int32_t tile_row,
                                     int32_t tile_col, int32_t src_cols) {
  const _Float16 *row0 =
      src + (size_t)tile_row * 32 * (size_t)src_cols + (size_t)tile_col * 32;
  /* One V73 2-D command replaces the old 32-command row storm.  The helper
   * clips height at the first virtual-page boundary and suppresses the issue
   * if an older useful command is still active. */
  omni_l2fetch_2d(row0, /*width=*/32u * 2u, /*height=*/32u,
                  /*stride=*/(uint32_t)src_cols * 2u);
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

  const _Float16 *row0 = (const _Float16 *)src +
                         (size_t)tile_row * 32 * (size_t)src_cols +
                         (size_t)tile_col * 32;
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
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >= OMNI_STAGE_SLOTS)
    omni_async_complete();
  if (__atomic_load_n(&omni_async_count, __ATOMIC_ACQUIRE) >= OMNI_STAGE_SLOTS)
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
   * DDR→stage DMA can overlap compute; completing in wait() corrupted results.
   */
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
  while (observed > 0 && !__atomic_compare_exchange_n(
                             &omni_sem_pool[sem_idx], &observed, observed - 1,
                             0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
  }
}

/* -------------------------------------------------------------------------
 * In-situ gather helpers (scalar; compiler auto-vectorises on V73+)
 * ------------------------------------------------------------------------- */
static void gather_reorder(const void *src, void *dest, int32_t elem_bytes,
                           int32_t count, const int32_t *index_map) {
  const char *s = (const char *)src;
  char *d = (char *)dest;
  for (int32_t i = 0; i < count; ++i)
    memcpy(d + i * elem_bytes, s + index_map[i] * elem_bytes,
           (size_t)elem_bytes);
}

static void hmx_weight_gather(const void *src, void *dest, int32_t elem_bytes,
                              int32_t M, int32_t K) {
  const char *s = (const char *)src;
  char *d = (char *)dest;
  const int32_t TILE = 32;
  int32_t num_tiles = (M + TILE - 1) / TILE;
  int32_t dst_flat = 0;
  for (int32_t t = 0; t < num_tiles; ++t)
    for (int32_t k = 0; k < K; ++k)
      for (int32_t m = 0; m < TILE; ++m) {
        int32_t src_row = t * TILE + m;
        if (src_row >= M)
          src_row = M - 1;
        memcpy(d + dst_flat * elem_bytes, s + (src_row * K + k) * elem_bytes,
               (size_t)elem_bytes);
        ++dst_flat;
      }
}

static void hmx_activation_gather(const void *src, void *dest,
                                  int32_t elem_bytes, int32_t N, int32_t C,
                                  int32_t H, int32_t W) {
  const char *s = (const char *)src;
  char *d = (char *)dest;
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
            memcpy(d + dst_flat * elem_bytes, s + src_flat * elem_bytes,
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
 * V73 extended l2fetch encoding:
 *   Rtt[63:48] = direction (zero for forward)
 *   Rtt[47:32] = stride (bytes between rows)
 *   Rtt[31:16] = width  (bytes per row)
 *   Rtt[15:0]  = height (number of rows)
 *
 * M1 treats the engine as single-flight, caps a normal command below 8 KiB
 * per the HVX guide, and never programs an address outside the start page.
 * ------------------------------------------------------------------------- */
#ifdef __hexagon__
enum {
  OMNI_L2_PAGE_BYTES = 4096,
  OMNI_L2_RECOMMENDED_MAX_BYTES = 8191,
  /* V73 Programmer's Reference Manual, USR register: PFA is bit 31.
   * The old bit-3 test never observed an active request, so a segmented CRP
   * hint could overrun the hardware's three-entry pending-command queue. */
  OMNI_USR_PFA_BIT = 31
};

static unsigned omni_l2fetch_active(void) {
  uint32_t usr;
  __asm__ volatile("%0 = usr" : "=r"(usr));
  return (usr >> OMNI_USR_PFA_BIT) & 1u;
}

static void omni_l2fetch_2d(const void *ptr, uint32_t width, uint32_t height,
                            uint32_t stride) {
  uint64_t requested = (uint64_t)width * (uint64_t)height;
  __atomic_fetch_add(&omni_l2_requested_bytes, requested, __ATOMIC_RELAXED);
  if (!ptr || width == 0 || height == 0 || stride == 0 || width > UINT16_MAX ||
      height > UINT16_MAX || stride > UINT16_MAX) {
    __atomic_fetch_add(&omni_l2_unsupported, 1, __ATOMIC_RELAXED);
    return;
  }

  uintptr_t start = (uintptr_t)ptr;
  uintptr_t startPage = start & ~(uintptr_t)(OMNI_L2_PAGE_BYTES - 1);
  uint32_t pageRoom = OMNI_L2_PAGE_BYTES - (uint32_t)(start - startPage);
  uint32_t safeWidth = width < pageRoom ? width : pageRoom;
  uint32_t maxRowsByBytes = OMNI_L2_RECOMMENDED_MAX_BYTES / safeWidth;
  uint32_t safeHeight = height < maxRowsByBytes ? height : maxRowsByBytes;

  while (safeHeight > 1) {
    uintptr_t lastStart = start + (uintptr_t)(safeHeight - 1) * stride;
    uintptr_t lastEnd = lastStart + safeWidth - 1;
    if ((lastStart & ~(uintptr_t)(OMNI_L2_PAGE_BYTES - 1)) == startPage &&
        (lastEnd & ~(uintptr_t)(OMNI_L2_PAGE_BYTES - 1)) == startPage)
      break;
    --safeHeight;
  }
  if (safeWidth != width || safeHeight != height)
    __atomic_fetch_add(&omni_l2_page_clipped, 1, __ATOMIC_RELAXED);
  if (safeWidth == 0 || safeHeight == 0) {
    __atomic_fetch_add(&omni_l2_unsupported, 1, __ATOMIC_RELAXED);
    return;
  }

  uint64_t safeBytes = (uint64_t)safeWidth * (uint64_t)safeHeight;
  unsigned issued = __atomic_load_n(&omni_l2_issued, __ATOMIC_RELAXED);
  uint64_t issuedBytes =
      __atomic_load_n(&omni_l2_issued_bytes, __ATOMIC_RELAXED);
  uint64_t remainingBytes =
      issuedBytes < omni_l2_max_bytes ? omni_l2_max_bytes - issuedBytes : 0;
  if ((omni_l2_max_commands && issued >= omni_l2_max_commands) ||
      (omni_l2_max_bytes && safeBytes > remainingBytes)) {
    __atomic_fetch_add(&omni_l2_budget_suppressed, 1, __ATOMIC_RELAXED);
    return;
  }

  /* Coalesce only identical recent hardware requests. Distinct offsets on one
   * page may cover distinct demand lines and therefore remain independent. */
  if (omni_l2_recent_count) {
    uint64_t requestKey =
        ((uint64_t)((uintptr_t)ptr >> 7) * UINT64_C(0x9e3779b97f4a7c15)) ^
        ((uint64_t)safeWidth << 48) ^ ((uint64_t)safeHeight << 32) ^
        (uint64_t)stride;
    if (requestKey == 0)
      requestKey = 1;
    for (unsigned i = 0; i < omni_l2_recent_count; ++i) {
      if (omni_l2_recent_requests[i] == requestKey) {
        __atomic_fetch_add(&omni_l2_duplicate_suppressed, 1, __ATOMIC_RELAXED);
        return;
      }
    }
    omni_l2_recent_requests[omni_l2_recent_cursor] = requestKey;
    omni_l2_recent_cursor = (omni_l2_recent_cursor + 1) % omni_l2_recent_count;
  }

  /* Do not overwrite a useful V73 command.  This is deliberately a
   * nonblocking suppression rather than a compute-thread spin. */
  if (omni_l2fetch_active()) {
    __atomic_fetch_add(&omni_l2_busy_suppressed, 1, __ATOMIC_RELAXED);
    return;
  }

  uint64_t spec = ((uint64_t)stride << 32) | ((uint64_t)safeWidth << 16) |
                  (uint64_t)safeHeight;
  __asm__ volatile("l2fetch(%0, %1)" : : "r"(ptr), "r"(spec) : "memory");
  __atomic_fetch_add(&omni_l2_issued, 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&omni_l2_issued_bytes,
                     (uint64_t)safeWidth * (uint64_t)safeHeight,
                     __ATOMIC_RELAXED);
}

static void omni_l2fetch(const void *ptr, uint32_t total_bytes) {
  if (total_bytes == 0)
    return;
  uint32_t width = total_bytes;
  if (width > OMNI_L2_RECOMMENDED_MAX_BYTES)
    width = OMNI_L2_RECOMMENDED_MAX_BYTES;
  omni_l2fetch_2d(ptr, width, /*height=*/1, /*stride=*/width);
}
#endif

void __omni_fetch_l2_hint_2d(const void *src, int32_t width_bytes,
                             int32_t height, int32_t stride_bytes) {
  if (!src || width_bytes <= 0 || height <= 0 || stride_bytes <= 0)
    return;
#ifdef __hexagon__
  omni_l2fetch_2d(src, (uint32_t)width_bytes, (uint32_t)height,
                  (uint32_t)stride_bytes);
#else
  (void)src;
  (void)width_bytes;
  (void)height;
  (void)stride_bytes;
#endif
}

void __omni_fetch_l2_hint_segmented(const void *src, int32_t width_bytes,
                                    int32_t rows, int32_t stride_bytes,
                                    int32_t site_id) {
  if (!src || width_bytes <= 0 || rows <= 0 || stride_bytes <= 0 || site_id < 0)
    return;
#ifdef __hexagon__
  /* The same CRP source site is revisited many times by an enclosing loop
   * (1440 calls/site in full DINOv2). Rotate the physical row instead of
   * expanding one logical hint into a burst of 16 l2fetch commands. This
   * preserves exact byte addressing, stays single-flight, and converts the
   * measured duplicate traffic into useful cross-row coverage. */
  unsigned slot = (unsigned)site_id % OMNI_L2_SEGMENTED_SITES;
  unsigned cursor =
      __atomic_fetch_add(&omni_l2_segmented_cursor[slot], 1, __ATOMIC_RELAXED);
  unsigned row = cursor % (unsigned)rows;
  const char *row_ptr = (const char *)src + (size_t)row * stride_bytes;
  omni_l2fetch_2d(row_ptr, (uint32_t)width_bytes, /*height=*/1,
                  (uint32_t)width_bytes);
#else
  (void)site_id;
#endif
}

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
                                  const int32_t *index_map, int32_t tile_row,
                                  int32_t tile_col, int32_t src_cols,
                                  int32_t act_off, int32_t scr_off,
                                  int32_t src_rows) {
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
    /* Activation and weight tiles share the same 32x32 FP16 source geometry.
     * Run both through the M1 single-flight/page-safe scheduler; otherwise
     * activation-only model graphs bypass M1 completely and produce a false
     * "OmniFetch enabled" ablation with zero issued requests. */
    omni_l2fetch_weight_tile((const _Float16 *)src, tile_row, tile_col,
                             src_cols);
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

  /* W8A16 item-8 path. The persistent entry is the compressed stream; the
   * dequantized FP16 tile exists only on the stack and feeds WH immediately. */
  if (layout_kind == LAYOUT_HMX_WEIGHT_DEQUANT_I8 && src_cols > 0 &&
      tile_row >= 0 && tile_col >= 0) {
    int32_t weight_off = act_off >= 0 ? act_off : 0;
    omni_w8_dequant_to_wh((const _Float16 *)src, dest, weight_off, tile_row,
                          tile_col, src_cols, src_rows);
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
    void *wh_dest = weight_off >= 0 ? (char *)dest + weight_off : dest;
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
      if (entry->epoch == __atomic_load_n(&omni_wh_epoch, __ATOMIC_ACQUIRE))
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
  unsigned accum = __atomic_exchange_n(&omni_stall_accum, 0, __ATOMIC_ACQ_REL);

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
