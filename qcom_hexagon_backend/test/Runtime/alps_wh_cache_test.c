// Host-side contract test for the generation-safe cross-token WH cache.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

void __alps_wh_cache_set_context(uint64_t context, uint32_t generation);
void __alps_wh_cache_invalidate(uint64_t context, uint32_t generation);
uint64_t __alps_wh_cache_stats(void);
void __alps_prefetch_insitu(
    const void *src, void *dest, int32_t elem_bytes, int32_t num_elems,
    int32_t layout_kind, int32_t lookahead, const int32_t *index_map,
    int32_t tile_row, int32_t tile_col, int32_t src_cols, int32_t act_off,
    int32_t scr_off, int32_t src_rows);

int main(void) {
  uint16_t weight[32 * 32];
  unsigned char first[4096];
  unsigned char second[4096];
  for (int i = 0; i < 32 * 32; ++i)
    weight[i] = (uint16_t)(i * 17 + 3);
  memset(first, 0, sizeof(first));
  memset(second, 0, sizeof(second));

  __alps_wh_cache_set_context(7, 1);
  __alps_prefetch_insitu(weight, first, 2, 1024, 1, -1, 0, 0, 0,
                               32, 0, -1, -1);
  __alps_prefetch_insitu(weight, second, 2, 1024, 1, -1, 0, 0, 0,
                               32, 0, -1, -1);
  uint64_t stats = __alps_wh_cache_stats();
  if ((uint32_t)(stats >> 32) != 1 || (uint32_t)stats != 1 ||
      memcmp(first, second, sizeof(first)) != 0)
    return 1;

  __alps_wh_cache_invalidate(7, 1);
  __alps_prefetch_insitu(weight, second, 2, 1024, 1, -1, 0, 0, 0,
                               32, 0, -1, -1);
  stats = __alps_wh_cache_stats();
  if ((uint32_t)(stats >> 32) != 1 || (uint32_t)stats != 2)
    return 2;

  puts("Alps WH cache contract: PASS");
  return 0;
}
