/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace alltoallv {

using value_type = std::uint32_t;

constexpr value_type kUntouched = 0xa5a5a5a5u;
constexpr std::uint64_t kElementsPerAlignment = 16 / sizeof(value_type);

struct Options {
  std::string pattern = "skewed";
  std::uint64_t bytes_per_rank = 4ull << 20;
  int warmup = 5;
  int iterations = 20;
  int blocks = 16;
  int threads = 256;
  bool help = false;
};

struct alignas(16) DevicePlanEntry {
  std::uint64_t send_count;
  std::uint64_t send_offset;
  std::uint64_t recv_count;
  std::uint64_t recv_offset;
  std::uint64_t remote_recv_offset;
  std::uint64_t reserved;
};

struct Plan {
  int rank = -1;
  int size = 0;
  std::vector<std::uint64_t> send_counts;
  std::vector<std::uint64_t> send_offsets;
  std::vector<std::uint64_t> recv_counts;
  std::vector<std::uint64_t> recv_offsets;
  std::vector<std::uint64_t> remote_recv_offsets;
  std::vector<int> node_roots;
  std::vector<int> local_ranks;
  std::uint64_t send_capacity = 0;
  std::uint64_t recv_capacity = 0;
  std::uint64_t global_send_capacity = 0;
  std::uint64_t global_recv_capacity = 0;
  std::uint64_t max_pair_count = 0;
  std::uint64_t global_payload_elements = 0;
  std::uint64_t global_remote_elements = 0;
  std::uint64_t global_self_elements = 0;
  std::uint64_t global_local_elements = 0;
  std::uint64_t global_network_elements = 0;
  std::uint64_t global_offrail_elements = 0;
  std::uint64_t max_network_send_elements = 0;
  std::uint64_t max_network_recv_elements = 0;
};

inline void mpi_check(int status, const char *call) {
  if (status == MPI_SUCCESS)
    return;
  char message[MPI_MAX_ERROR_STRING];
  int length = 0;
  MPI_Error_string(status, message, &length);
  std::fprintf(stderr, "%s failed: %.*s\n", call, length, message);
  MPI_Abort(MPI_COMM_WORLD, status);
}

inline std::uint64_t parse_bytes(const char *text) {
  char *end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text)
    return 0;
  std::uint64_t scale = 1;
  if (*end == 'k' || *end == 'K') {
    scale = 1ull << 10;
    ++end;
  } else if (*end == 'm' || *end == 'M') {
    scale = 1ull << 20;
    ++end;
  } else if (*end == 'g' || *end == 'G') {
    scale = 1ull << 30;
    ++end;
  }
  return *end == '\0' ? static_cast<std::uint64_t>(value) * scale : 0;
}

inline void print_usage(const char *program) {
  std::printf(
      "Usage: %s [--pattern uniform|offdiagonal|skewed|sparse] "
      "[--bytes-per-rank N[K|M|G]] [--warmup N] [--iters N] "
      "[--blocks N] [--threads N]\n",
      program);
}

inline Options parse_options(int argc, char **argv, int rank) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 < argc)
        return argv[++i];
      if (rank == 0)
        std::fprintf(stderr, "%s requires a value\n", name);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return nullptr;
    };

    if (std::strcmp(argv[i], "--pattern") == 0) {
      options.pattern = need_value("--pattern");
    } else if (std::strcmp(argv[i], "--bytes-per-rank") == 0) {
      options.bytes_per_rank = parse_bytes(need_value("--bytes-per-rank"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      options.warmup = std::atoi(need_value("--warmup"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      options.iterations = std::atoi(need_value("--iters"));
    } else if (std::strcmp(argv[i], "--blocks") == 0) {
      options.blocks = std::atoi(need_value("--blocks"));
    } else if (std::strcmp(argv[i], "--threads") == 0) {
      options.threads = std::atoi(need_value("--threads"));
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      options.help = true;
    } else {
      if (rank == 0)
        std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  bool valid_pattern = options.pattern == "uniform" ||
                       options.pattern == "offdiagonal" ||
                       options.pattern == "skewed" ||
                       options.pattern == "sparse";
  if (!options.help &&
      (!valid_pattern || options.bytes_per_rank < sizeof(value_type) ||
       options.warmup < 0 || options.iterations < 1 || options.blocks < 1 ||
       options.threads < 32 || options.threads > 1024 ||
       options.threads % 32 != 0)) {
    if (rank == 0) {
      std::fprintf(stderr, "Invalid arguments\n");
      print_usage(argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return options;
}

inline void require_matching_collective_options(const Options &options,
                                                int rank) {
  int values[4] = {options.blocks, options.threads, options.warmup,
                   options.iterations};
  int minima[4];
  int maxima[4];
  mpi_check(MPI_Allreduce(values, minima, 4, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(option minimum)");
  mpi_check(MPI_Allreduce(values, maxima, 4, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(option maximum)");
  for (int i = 0; i < 4; ++i) {
    if (minima[i] == maxima[i])
      continue;
    if (rank == 0) {
      std::fprintf(stderr,
                   "--blocks, --threads, --warmup, and --iters must match "
                   "on every rank\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

inline std::uint64_t align_elements(std::uint64_t value) {
  return (value + kElementsPerAlignment - 1) &
         ~(kElementsPerAlignment - 1);
}

inline std::uint64_t make_offsets(const std::vector<std::uint64_t> &counts,
                                  std::vector<std::uint64_t> *offsets) {
  offsets->resize(counts.size());
  std::uint64_t cursor = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    cursor = align_elements(cursor);
    (*offsets)[i] = cursor;
    cursor += counts[i];
  }
  return align_elements(cursor);
}

inline std::vector<std::uint64_t>
make_send_counts(int rank, int size, const Options &options) {
  std::vector<std::uint64_t> weights(size, 1);
  for (int peer = 0; peer < size; ++peer) {
    std::uint64_t hash = static_cast<std::uint64_t>(rank + 1) * 0x9e3779b1u +
                         static_cast<std::uint64_t>(peer + 3) * 0x85ebca77u;
    if (options.pattern == "skewed") {
      weights[peer] = 1 + hash % 7;
      if (peer == (rank * 3 + 1) % size)
        weights[peer] *= 8;
    } else if (options.pattern == "sparse") {
      weights[peer] = hash % 3 == 0 ? 0 : 1 + hash % 5;
    } else if (options.pattern == "offdiagonal" && peer == rank) {
      weights[peer] = 0;
    }
  }

  if (std::all_of(weights.begin(), weights.end(),
                  [](std::uint64_t value) { return value == 0; }))
    weights[rank % size] = 1;

  std::uint64_t budget =
      std::max<std::uint64_t>(1, options.bytes_per_rank / sizeof(value_type));
  std::uint64_t weight_sum = 0;
  for (std::uint64_t weight : weights)
    weight_sum += weight;

  std::vector<std::uint64_t> counts(size, 0);
  std::uint64_t assigned = 0;
  int first_nonzero = -1;
  for (int peer = 0; peer < size; ++peer) {
    if (weights[peer] == 0)
      continue;
    if (first_nonzero < 0)
      first_nonzero = peer;
    counts[peer] = budget * weights[peer] / weight_sum;
    assigned += counts[peer];
  }
  counts[first_nonzero] += budget - assigned;
  return counts;
}

inline Plan make_plan(int rank, int size, const Options &options) {
  Plan plan;
  plan.rank = rank;
  plan.size = size;
  plan.send_counts = make_send_counts(rank, size, options);
  plan.send_capacity = make_offsets(plan.send_counts, &plan.send_offsets);

  plan.recv_counts.resize(size);
  mpi_check(MPI_Alltoall(plan.send_counts.data(), 1, MPI_UINT64_T,
                         plan.recv_counts.data(), 1, MPI_UINT64_T,
                         MPI_COMM_WORLD),
            "MPI_Alltoall(counts)");
  plan.recv_capacity = make_offsets(plan.recv_counts, &plan.recv_offsets);

  plan.remote_recv_offsets.resize(size);
  mpi_check(MPI_Alltoall(plan.recv_offsets.data(), 1, MPI_UINT64_T,
                         plan.remote_recv_offsets.data(), 1, MPI_UINT64_T,
                         MPI_COMM_WORLD),
            "MPI_Alltoall(receive offsets)");

  MPI_Comm local_comm;
  mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_comm),
            "MPI_Comm_split_type(placement)");
  int local_rank = 0;
  int node_root = rank;
  mpi_check(MPI_Comm_rank(local_comm, &local_rank),
            "MPI_Comm_rank(placement)");
  mpi_check(MPI_Allreduce(&rank, &node_root, 1, MPI_INT, MPI_MIN, local_comm),
            "MPI_Allreduce(node root)");
  MPI_Comm_free(&local_comm);
  plan.node_roots.resize(size);
  plan.local_ranks.resize(size);
  mpi_check(MPI_Allgather(&node_root, 1, MPI_INT, plan.node_roots.data(), 1,
                          MPI_INT, MPI_COMM_WORLD),
            "MPI_Allgather(node roots)");
  mpi_check(MPI_Allgather(&local_rank, 1, MPI_INT, plan.local_ranks.data(), 1,
                          MPI_INT, MPI_COMM_WORLD),
            "MPI_Allgather(local ranks)");

  mpi_check(MPI_Allreduce(&plan.send_capacity, &plan.global_send_capacity, 1,
                          MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(send capacity)");
  mpi_check(MPI_Allreduce(&plan.recv_capacity, &plan.global_recv_capacity, 1,
                          MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(receive capacity)");

  std::uint64_t local_max = 0;
  std::uint64_t local_payload = 0;
  std::uint64_t local_remote = 0;
  std::uint64_t local_self = 0;
  std::uint64_t local_direct = 0;
  std::uint64_t local_network = 0;
  std::uint64_t local_offrail = 0;
  std::uint64_t local_network_recv = 0;
  for (int peer = 0; peer < size; ++peer) {
    local_max = std::max(local_max, plan.send_counts[peer]);
    local_payload += plan.send_counts[peer];
    if (peer != rank)
      local_remote += plan.send_counts[peer];
    if (peer == rank) {
      local_self += plan.send_counts[peer];
    } else if (plan.node_roots[peer] == plan.node_roots[rank]) {
      local_direct += plan.send_counts[peer];
    } else {
      local_network += plan.send_counts[peer];
      if (plan.local_ranks[peer] != plan.local_ranks[rank])
        local_offrail += plan.send_counts[peer];
    }
    if (plan.node_roots[peer] != plan.node_roots[rank])
      local_network_recv += plan.recv_counts[peer];
  }
  mpi_check(MPI_Allreduce(&local_max, &plan.max_pair_count, 1, MPI_UINT64_T,
                          MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(max pair count)");
  mpi_check(MPI_Allreduce(&local_payload, &plan.global_payload_elements, 1,
                          MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(payload)");
  mpi_check(MPI_Allreduce(&local_remote, &plan.global_remote_elements, 1,
                          MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(remote payload)");
  std::uint64_t local_classes[4] = {local_self, local_direct, local_network,
                                    local_offrail};
  std::uint64_t global_classes[4] = {};
  mpi_check(MPI_Allreduce(local_classes, global_classes, 4, MPI_UINT64_T,
                          MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(traffic classes)");
  plan.global_self_elements = global_classes[0];
  plan.global_local_elements = global_classes[1];
  plan.global_network_elements = global_classes[2];
  plan.global_offrail_elements = global_classes[3];
  std::uint64_t local_network_maxima[2] = {local_network,
                                           local_network_recv};
  std::uint64_t global_network_maxima[2] = {};
  mpi_check(MPI_Allreduce(local_network_maxima, global_network_maxima, 2,
                          MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(network maxima)");
  plan.max_network_send_elements = global_network_maxima[0];
  plan.max_network_recv_elements = global_network_maxima[1];
  return plan;
}

inline std::vector<DevicePlanEntry> device_entries(const Plan &plan) {
  std::vector<DevicePlanEntry> entries(plan.size);
  for (int peer = 0; peer < plan.size; ++peer) {
    entries[peer] = DevicePlanEntry{plan.send_counts[peer],
                                    plan.send_offsets[peer],
                                    plan.recv_counts[peer],
                                    plan.recv_offsets[peer],
                                    plan.remote_recv_offsets[peer], 0};
  }
  return entries;
}

inline value_type value_for(int source, int destination,
                            std::uint64_t index) {
  std::uint32_t value = 0x9e3779b9u;
  value ^= static_cast<std::uint32_t>(source + 1) * 0x85ebca6bu;
  value ^= static_cast<std::uint32_t>(destination + 1) * 0xc2b2ae35u;
  value ^= static_cast<std::uint32_t>(index) * 0x27d4eb2du;
  return value;
}

inline std::vector<value_type> make_send_buffer(const Plan &plan) {
  std::vector<value_type> buffer(plan.global_send_capacity, kUntouched);
  for (int destination = 0; destination < plan.size; ++destination) {
    for (std::uint64_t i = 0; i < plan.send_counts[destination]; ++i) {
      buffer[plan.send_offsets[destination] + i] =
          value_for(plan.rank, destination, i);
    }
  }
  return buffer;
}

inline std::vector<value_type> make_expected_buffer(const Plan &plan) {
  std::vector<value_type> buffer(plan.global_recv_capacity, kUntouched);
  for (int source = 0; source < plan.size; ++source) {
    for (std::uint64_t i = 0; i < plan.recv_counts[source]; ++i) {
      buffer[plan.recv_offsets[source] + i] =
          value_for(source, plan.rank, i);
    }
  }
  return buffer;
}

inline int validate(const Plan &plan, const std::vector<value_type> &observed,
                    const char *implementation) {
  std::vector<value_type> expected = make_expected_buffer(plan);
  int local_errors = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (observed[i] == expected[i])
      continue;
    if (local_errors < 4) {
      std::fprintf(stderr,
                   "Rank %d: %s mismatch at element %zu: got 0x%08x, "
                   "expected 0x%08x\n",
                   plan.rank, implementation, i, observed[i], expected[i]);
    }
    ++local_errors;
  }

  int total_errors = 0;
  mpi_check(MPI_Allreduce(&local_errors, &total_errors, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(validation)");
  if (plan.rank == 0) {
    if (total_errors == 0)
      std::printf("%s correctness: PASS\n", implementation);
    else
      std::printf("%s correctness: FAIL (%d mismatches)\n", implementation,
                  total_errors);
  }
  return total_errors;
}

inline int local_rank() {
  int rank = 0;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
  MPI_Comm local_comm;
  mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_comm),
            "MPI_Comm_split_type");
  int result = 0;
  mpi_check(MPI_Comm_rank(local_comm, &result), "MPI_Comm_rank(local)");
  MPI_Comm_free(&local_comm);
  return result;
}

inline void print_plan(const Plan &plan, const Options &options,
                       const char *implementation) {
  std::uint64_t local_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t local_max = 0;
  for (std::uint64_t count : plan.send_counts) {
    local_min = std::min(local_min, count);
    local_max = std::max(local_max, count);
  }
  std::uint64_t global_min = 0;
  std::uint64_t global_max = 0;
  mpi_check(MPI_Allreduce(&local_min, &global_min, 1, MPI_UINT64_T, MPI_MIN,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(min count)");
  mpi_check(MPI_Allreduce(&local_max, &global_max, 1, MPI_UINT64_T, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(max count)");
  if (plan.rank == 0) {
    std::printf(
        "%s: %d ranks, pattern=%s, pair sizes=%llu..%llu elements, "
        "payload=%.3f MiB\n",
        implementation, plan.size, options.pattern.c_str(),
        static_cast<unsigned long long>(global_min),
        static_cast<unsigned long long>(global_max),
        static_cast<double>(plan.global_payload_elements * sizeof(value_type)) /
            static_cast<double>(1ull << 20));
    const double bytes_to_mib =
        static_cast<double>(sizeof(value_type)) /
        static_cast<double>(1ull << 20);
    std::printf(
        "Traffic per iteration: self=%.3f MiB, local=%.3f MiB, "
        "network=%.3f MiB, hybrid off-rail scatter=%.3f MiB\n",
        plan.global_self_elements * bytes_to_mib,
        plan.global_local_elements * bytes_to_mib,
        plan.global_network_elements * bytes_to_mib,
        plan.global_offrail_elements * bytes_to_mib);
    std::printf("Hottest rank: network send=%.3f MiB, receive=%.3f MiB\n",
                plan.max_network_send_elements * bytes_to_mib,
                plan.max_network_recv_elements * bytes_to_mib);
  }
}

inline void report_timing(const Plan &plan, const Options &options,
                          const char *implementation, float local_ms) {
  float max_ms = 0.0f;
  mpi_check(MPI_Allreduce(&local_ms, &max_ms, 1, MPI_FLOAT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(timing)");
  if (plan.rank != 0)
    return;
  double average_ms = max_ms / options.iterations;
  double remote_bytes = static_cast<double>(plan.global_remote_elements) *
                        sizeof(value_type);
  double aggregate_gbs = remote_bytes / (average_ms * 1.0e6);
  double local_gbs = static_cast<double>(plan.global_local_elements) *
                     sizeof(value_type) / (average_ms * 1.0e6);
  double network_gbs = static_cast<double>(plan.global_network_elements) *
                       sizeof(value_type) / (average_ms * 1.0e6);
  std::printf(
      "%s performance: %.3f ms/iteration, %.3f GB/s logical non-self\n",
      implementation, average_ms, aggregate_gbs);
  std::printf("%s payload rates: %.3f GB/s local, %.3f GB/s network\n",
              implementation, local_gbs, network_gbs);
}

} // namespace alltoallv
