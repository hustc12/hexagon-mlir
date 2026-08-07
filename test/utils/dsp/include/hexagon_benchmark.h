//===- hexagon_benchmark.h ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

/* Header file used to time a pipeline.
 *  This is meant to be used in device-standalone mode. */

#ifndef HEXAGON_BENCHMARK_H
#define HEXAGON_BENCHMARK_H
#include "HAP_perf.h"
#include "hexagon_types.h"
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <vector>

// Returns the average time, in microseconds, taken to run
// op for the given number of iterations.
template <typename F> uint64_t benchmark_time_us(int iterations, F op) {
  uint64_t start_time = HAP_perf_get_time_us();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_time = HAP_perf_get_time_us();
  return (uint64_t)((end_time - start_time) / iterations);
}

// Returns the average cycles taken to run
// op for the given number of iterations.
template <typename F> uint64_t benchmark_pcycles(int iterations, F op) {
  uint64_t start_cycle = HAP_perf_get_pcycles();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_cycle = HAP_perf_get_pcycles();
  return (uint64_t)((end_cycle - start_cycle) / iterations);
}

// Runs op for the given number of iterations, recording each iteration's
// wall-clock duration (microseconds) into samples. Returns the arithmetic
// mean, byte-identical to benchmark_time_us so existing parsers are unaffected.
template <typename F>
uint64_t benchmark_samples_us(int iterations, std::vector<uint64_t> &samples,
                              F op) {
  samples.clear();
  samples.reserve(iterations);
  uint64_t start_time = HAP_perf_get_time_us();
  for (int i = 0; i < iterations; ++i) {
    uint64_t iter_start = HAP_perf_get_time_us();
    op();
    uint64_t iter_end = HAP_perf_get_time_us();
    samples.push_back(iter_end - iter_start);
  }
  uint64_t end_time = HAP_perf_get_time_us();
  return (uint64_t)((end_time - start_time) / iterations);
}

// Sorts samples and appends percentile summary lines to stdout and, when
// non-null, to the provided perf file. Emitted lines are purely additive
// (PerfP50/PerfP90/PerfP99/PerfMin/PerfSamples) so awk parsers anchored on
// "Perf:" continue to see the unchanged mean.
inline void report_percentiles(const char *name,
                               std::vector<uint64_t> samples,
                               FILE *perf_fp = nullptr) {
  if (samples.empty())
    return;
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  auto pct = [&](double p) -> uint64_t {
    if (n == 1)
      return samples[0];
    double idx = p * (double)(n - 1);
    size_t lo = (size_t)idx;
    size_t hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - (double)lo;
    return (uint64_t)((double)samples[lo] +
                      frac * ((double)samples[hi] - (double)samples[lo]));
  };
  uint64_t p50 = pct(0.50), p90 = pct(0.90), p99 = pct(0.99);
  uint64_t pmin = samples.front();
  printf("\tPerfConfig:%s\n", name ? name : "");
  printf("\tPerfP50:%llu\n", (unsigned long long)p50);
  printf("\tPerfP90:%llu\n", (unsigned long long)p90);
  printf("\tPerfP99:%llu\n", (unsigned long long)p99);
  printf("\tPerfMin:%llu\n", (unsigned long long)pmin);
  printf("\tPerfSamples:%llu\n", (unsigned long long)n);
  if (perf_fp) {
    fprintf(perf_fp, "\tPerfConfig:%s\n", name ? name : "");
    fprintf(perf_fp, "\tPerfP50:%llu\n", (unsigned long long)p50);
    fprintf(perf_fp, "\tPerfP90:%llu\n", (unsigned long long)p90);
    fprintf(perf_fp, "\tPerfP99:%llu\n", (unsigned long long)p99);
    fprintf(perf_fp, "\tPerfMin:%llu\n", (unsigned long long)pmin);
    fprintf(perf_fp, "\tPerfSamples:%llu\n", (unsigned long long)n);
  }
}
#endif
