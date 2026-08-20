// Host-side contract test for the ALPS P3a exact-readiness state machine.

#include <stdint.h>
#include <stdio.h>

int32_t __omni_fetch_invocation_begin(void);
int32_t __omni_fetch_invocation_end(int32_t context);
int32_t __omni_fetch_descriptor_acquire(int32_t context, int64_t version,
                                        int64_t tile, int32_t layout,
                                        int32_t source_tier,
                                        int32_t destination_tier);
int32_t __omni_fetch_descriptor_transition(int32_t descriptor,
                                           int32_t expected, int32_t next);
int32_t __omni_fetch_descriptor_consume(int32_t descriptor, int64_t version,
                                        int64_t tile, int32_t layout,
                                        int32_t source_tier,
                                        int32_t destination_tier);
int32_t __omni_fetch_descriptor_release(int32_t descriptor);
uint32_t __omni_fetch_get_and_clear_errors(void);
uint64_t __omni_fetch_descriptor_counts(void);
uint64_t __omni_fetch_descriptor_release_failures(void);
int32_t __omni_fetch_exact_weight_kick(
    int32_t context, int64_t version, const void *src, void *dest,
    int32_t tile_row, int32_t tile_col, int32_t source_cols,
    int32_t weight_offset, int32_t stage_offset);
int32_t __omni_fetch_exact_weight_consume(int32_t context, int64_t version,
                                          int32_t tile_row,
                                          int32_t tile_col);
int32_t __omni_fetch_exact_weight_release(int32_t context, int64_t version,
                                          int32_t tile_row,
                                          int32_t tile_col);
uint64_t __omni_fetch_exact_dma_counts(void);
uint64_t __omni_fetch_exact_control_counts(void);
void __omni_fetch_p4a_configure(int32_t enable);
uint64_t __omni_fetch_p4a_window_counts(void);
uint64_t __omni_fetch_p4a_decision_counts(void);
uint64_t __omni_fetch_p4a_pmu_status_counts(void);

int main(void) {
  __omni_fetch_p4a_configure(1);
  if (__omni_fetch_p4a_window_counts() != 0 ||
      __omni_fetch_p4a_decision_counts() != 0 ||
      __omni_fetch_p4a_pmu_status_counts() != 0)
    return 8;
  int32_t context = __omni_fetch_invocation_begin();
  int32_t descriptor =
      __omni_fetch_descriptor_acquire(context, 7, 3, 1, 0, 1);
  if (context < 0 || descriptor < 0)
    return 1;
  if (!__omni_fetch_descriptor_transition(descriptor, 1, 2) ||
      !__omni_fetch_descriptor_transition(descriptor, 2, 3))
    return 2;

  // A different tile must not consume this READY descriptor.
  if (__omni_fetch_descriptor_consume(descriptor, 7, 4, 1, 0, 1))
    return 3;
  if (!__omni_fetch_descriptor_consume(descriptor, 7, 3, 1, 0, 1) ||
      !__omni_fetch_descriptor_release(descriptor) ||
      !__omni_fetch_invocation_end(context))
    return 4;

  // Reusing the context slot changes its generation; the old descriptor is
  // stale even if its numeric slot is reused.
  int32_t next_context = __omni_fetch_invocation_begin();
  if (next_context < 0 ||
      __omni_fetch_descriptor_release(descriptor) ||
      !__omni_fetch_invocation_end(next_context))
    return 5;

  uint16_t weight[64 * 64];
  unsigned char wh[4096];
  for (int i = 0; i < 64 * 64; ++i)
    weight[i] = (uint16_t)(i + 1);
  int32_t exact_context = __omni_fetch_invocation_begin();
  if (!__omni_fetch_exact_weight_kick(exact_context, 19, weight, wh, 1, 0,
                                      64, 0, -1) ||
      !__omni_fetch_exact_weight_consume(exact_context, 19, 1, 0) ||
      !__omni_fetch_exact_weight_release(exact_context, 19, 1, 0) ||
      !__omni_fetch_invocation_end(exact_context))
    return 6;

  uint64_t counts = __omni_fetch_descriptor_counts();
  uint64_t release_failures = __omni_fetch_descriptor_release_failures();
  uint64_t dma = __omni_fetch_exact_dma_counts();
  uint64_t control = __omni_fetch_exact_control_counts();
  if ((uint32_t)(counts >> 32) != 2 || (uint32_t)counts != 2 ||
      (uint32_t)(release_failures >> 32) != 2 ||
      (uint32_t)release_failures < 2 ||
      (uint32_t)(dma >> 32) != 1 || (uint32_t)dma != 1 ||
      control != 0 ||
      __omni_fetch_get_and_clear_errors() == 0)
    return 7;
  puts("ALPS exact-readiness contract: PASS");
  return 0;
}
