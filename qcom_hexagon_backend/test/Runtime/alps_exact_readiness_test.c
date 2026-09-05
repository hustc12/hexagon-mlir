// Host-side contract test for the ALPS P3a exact-readiness state machine.

#include <stdint.h>
#include <stdio.h>

int32_t __alps_invocation_begin(void);
int32_t __alps_invocation_end(int32_t context);
int32_t __alps_descriptor_acquire(int32_t context, int64_t version,
                                        int64_t tile, int32_t layout,
                                        int32_t source_tier,
                                        int32_t destination_tier);
int32_t __alps_descriptor_transition(int32_t descriptor,
                                           int32_t expected, int32_t next);
int32_t __alps_descriptor_consume(int32_t descriptor, int64_t version,
                                        int64_t tile, int32_t layout,
                                        int32_t source_tier,
                                        int32_t destination_tier);
int32_t __alps_descriptor_release(int32_t descriptor);
uint32_t __alps_get_and_clear_errors(void);
uint64_t __alps_descriptor_counts(void);
uint64_t __alps_descriptor_release_failures(void);
int32_t __alps_exact_weight_kick(
    int32_t context, int64_t version, const void *src, void *dest,
    int32_t tile_row, int32_t tile_col, int32_t source_cols,
    int32_t weight_offset, int32_t stage_offset, int32_t panel_tiles);
int32_t __alps_exact_weight_consume(int32_t context, int64_t version,
                                          int32_t tile_row,
                                          int32_t tile_col);
int32_t __alps_exact_weight_release(int32_t context, int64_t version,
                                          int32_t tile_row,
                                          int32_t tile_col);
uint64_t __alps_exact_dma_counts(void);
uint64_t __alps_exact_control_counts(void);
uint64_t __alps_exact_vdae_counts(void);
uint64_t __alps_exact_vdae_ready_bytes(void);
uint64_t __alps_exact_vdae_wait_cycles(void);
uint64_t __alps_exact_vdae_cache_counts(void);
void __alps_set_dual_thread_dae(int32_t enable);
void __alps_p4a_configure(int32_t enable);
void __alps_p4a_configure_policy(int32_t enable, int32_t initial_dma_allowed,
                                 uint32_t window_completions,
                                 uint32_t late_poll_threshold,
                                 uint32_t probe_interval);
int32_t __alps_p4a_request_allowed(void);
void __alps_p4a_observe_dma_completion(unsigned poll_retries);
uint64_t __alps_p4a_window_counts(void);
uint64_t __alps_p4a_decision_counts(void);
uint64_t __alps_p4a_pmu_status_counts(void);
uint64_t __alps_p4a_recovery_counts(void);
uint64_t __alps_p4a_policy_config(void);

int main(void) {
  // A profile may start in cooldown.  Suppression must periodically admit a
  // probe, and an on-time probe must re-open asynchronous movement.
  __alps_p4a_configure_policy(1, 0, 2, 4, 2);
  if (__alps_p4a_request_allowed() || !__alps_p4a_request_allowed())
    return 9;
  __alps_p4a_observe_dma_completion(0);
  uint64_t recovery = __alps_p4a_recovery_counts();
  if (!__alps_p4a_request_allowed() || (uint32_t)(recovery >> 32) != 1 ||
      (uint32_t)recovery != 1 ||
      __alps_p4a_policy_config() != ((uint64_t)2 << 32 | (uint64_t)4 << 16 | 2))
    return 10;

  __alps_p4a_configure(1);
  if (__alps_p4a_window_counts() != 0 ||
      __alps_p4a_decision_counts() != 0 ||
      __alps_p4a_pmu_status_counts() != 0)
    return 8;
  int32_t context = __alps_invocation_begin();
  int32_t descriptor =
      __alps_descriptor_acquire(context, 7, 3, 1, 0, 1);
  if (context < 0 || descriptor < 0)
    return 1;
  if (!__alps_descriptor_transition(descriptor, 1, 2) ||
      !__alps_descriptor_transition(descriptor, 2, 3))
    return 2;

  // A different tile must not consume this READY descriptor.
  if (__alps_descriptor_consume(descriptor, 7, 4, 1, 0, 1))
    return 3;
  if (!__alps_descriptor_consume(descriptor, 7, 3, 1, 0, 1) ||
      !__alps_descriptor_release(descriptor) ||
      !__alps_invocation_end(context))
    return 4;

  // Reusing the context slot changes its generation; the old descriptor is
  // stale even if its numeric slot is reused.
  int32_t next_context = __alps_invocation_begin();
  if (next_context < 0 ||
      __alps_descriptor_release(descriptor) ||
      !__alps_invocation_end(next_context))
    return 5;

  uint16_t weight[160 * 64];
  unsigned char wh[4 * 4096];
  for (int i = 0; i < 160 * 64; ++i)
    weight[i] = (uint16_t)(i + 1);
  int32_t exact_context = __alps_invocation_begin();
  __alps_set_dual_thread_dae(1);
  if (!__alps_exact_weight_kick(exact_context, 19, weight, wh, 1, 0,
                                      64, 0, -1, 1) ||
      !__alps_exact_weight_consume(exact_context, 19, 1, 0) ||
      !__alps_exact_weight_release(exact_context, 19, 1, 0) ||
      !__alps_invocation_end(exact_context))
    return 6;

  // The first scout request forms and retains the consumer-ready WH tile.
  // Reusing the same immutable source/version should supply WH directly.
  int32_t warm_context = __alps_invocation_begin();
  if (!__alps_exact_weight_kick(warm_context, 19, weight, wh, 1, 0, 64, 0,
                                -1, 1) ||
      !__alps_exact_weight_consume(warm_context, 19, 1, 0) ||
      !__alps_exact_weight_release(warm_context, 19, 1, 0) ||
      !__alps_invocation_end(warm_context))
    return 11;

  // One descriptor/token covers the complete four-tile panel, and the second
  // invocation must reuse that exact consumer-ready panel.
  int32_t panel_context = __alps_invocation_begin();
  if (!__alps_exact_weight_kick(panel_context, 23, weight, wh, 1, 0, 64, 0,
                                -1, 4) ||
      !__alps_exact_weight_consume(panel_context, 23, 1, 0) ||
      !__alps_exact_weight_release(panel_context, 23, 1, 0) ||
      !__alps_invocation_end(panel_context))
    return 12;
  int32_t warm_panel_context = __alps_invocation_begin();
  if (!__alps_exact_weight_kick(warm_panel_context, 23, weight, wh, 1, 0, 64,
                                0, -1, 4) ||
      !__alps_exact_weight_consume(warm_panel_context, 23, 1, 0) ||
      !__alps_exact_weight_release(warm_panel_context, 23, 1, 0) ||
      !__alps_invocation_end(warm_panel_context))
    return 13;

  uint64_t counts = __alps_descriptor_counts();
  uint64_t release_failures = __alps_descriptor_release_failures();
  uint64_t dma = __alps_exact_dma_counts();
  uint64_t control = __alps_exact_control_counts();
  uint64_t vdae = __alps_exact_vdae_counts();
  uint64_t cache = __alps_exact_vdae_cache_counts();
  uint32_t errors = __alps_get_and_clear_errors();
  fprintf(stderr,
          "counts=%llu releases=%llu dma=%llu control=%llu vdae=%llu "
          "ready_bytes=%llu wait_cycles=%llu cache=%llu errors=%u\n",
          (unsigned long long)counts, (unsigned long long)release_failures,
          (unsigned long long)dma, (unsigned long long)control,
          (unsigned long long)vdae,
          (unsigned long long)__alps_exact_vdae_ready_bytes(),
          (unsigned long long)__alps_exact_vdae_wait_cycles(),
          (unsigned long long)cache, errors);
  if ((uint32_t)(counts >> 32) != 5 || (uint32_t)counts != 5 ||
      (uint32_t)(release_failures >> 32) != 5 ||
      (uint32_t)release_failures < 2 ||
      (uint32_t)(dma >> 32) != 4 || (uint32_t)dma != 4 ||
      control != 0 || (uint32_t)(vdae >> 32) != 4 ||
      (uint32_t)vdae != 4 || __alps_exact_vdae_ready_bytes() != 20480 ||
      (uint32_t)(cache >> 32) != 2 || (uint32_t)cache != 2 ||
      __alps_exact_vdae_wait_cycles() != 0 || errors == 0)
    return 7;
  puts("ALPS exact-readiness contract: PASS");
  return 0;
}
