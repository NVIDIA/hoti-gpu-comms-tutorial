/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/nvshmem_exercise.h"
#include "../alltoallv_common.hpp"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

using alltoallv::value_type;
namespace cg = cooperative_groups;

constexpr std::uint64_t kDefaultDirectChunkBytes = 256ull << 10;
constexpr std::uint64_t kDefaultNetworkChunkBytes = 4ull << 20;
constexpr std::uint64_t kMetadataAlignment = 16;

enum class QpQuietScope : int { thread, warp, block };

__device__ std::uint64_t align_bytes(std::uint64_t value) {
  return (value + kMetadataAlignment - 1) & ~(kMetadataAlignment - 1);
}

__global__ void exchange_plan(const std::uint64_t *send_counts_bytes,
                              std::uint64_t *recv_counts_bytes,
                              std::uint64_t *recv_offsets_bytes,
                              std::uint64_t *remote_recv_offsets_bytes,
                              value_type *recv_buffer,
                              std::uint64_t *send_signal_counts,
                              std::uint64_t *recv_signal_counts, int rank,
                              int nranks, std::uint64_t direct_chunk_bytes,
                              std::uint64_t network_chunk_bytes, int *status) {
  /*
   * TODO: Exchange send_counts_bytes with a block-scoped NVSHMEM all-to-all,
   * build the aligned receive offsets, and exchange those offsets back to the
   * senders. Then classify each outgoing route with nvshmem_ptr, calculate
   * the number of direct or network chunks that route will produce, and
   * exchange those signal counts with a third block-scoped all-to-all. Store
   * the collective status in status.
   */
  (void)send_counts_bytes;
  (void)recv_counts_bytes;
  (void)recv_offsets_bytes;
  (void)remote_recv_offsets_bytes;
  (void)recv_buffer;
  (void)send_signal_counts;
  (void)recv_signal_counts;
  (void)rank;
  (void)nranks;
  (void)direct_chunk_bytes;
  (void)network_chunk_bytes;
  if (threadIdx.x == 0)
    *status = -1;
}

__global__ void nvshmem_alltoallv_kernel(
    const value_type *send_buffer, value_type *recv_buffer,
    const std::uint64_t *send_counts_bytes,
    const std::uint64_t *send_offsets_bytes,
    const std::uint64_t *recv_signal_counts,
    const std::uint64_t *remote_recv_offsets_bytes, std::uint64_t *signals,
    int rank, int nranks, std::uint64_t direct_chunk_bytes,
    std::uint64_t network_chunk_bytes,
    const nvshmemx_qp_handle_t *network_qps, int network_qp_count,
    std::uint64_t max_chunks, QpQuietScope quiet_scope,
    std::uint64_t epoch) {
  const cg::grid_group grid = cg::this_grid();
  // One CTA performs the collective entry handshake; the cooperative grid
  // then holds every producer until all PEs have arrived.
  if (blockIdx.x == 0)
    nvshmemx_barrier_all_block();
  grid.sync();

  /*
   * TODO: Assign rank-rotated destination/chunk pairs to CTAs in chunk-major
   * order. Copy self and directly accessible messages in direct_chunk_bytes
   * chunks. For another PE, use nvshmem_ptr to choose a block-scoped NBI
   * put-with-signal for each direct chunk or a QP-specific thread-scoped NBI
   * put-with-signal for each larger network chunk. Select the network QP from
   * both the destination and chunk index.
   */
  (void)send_buffer;
  (void)recv_buffer;
  (void)send_counts_bytes;
  (void)send_offsets_bytes;
  (void)remote_recv_offsets_bytes;
  (void)network_qps;
  (void)network_qp_count;
  (void)direct_chunk_bytes;
  (void)network_chunk_bytes;
  // Keep the producer/completion phase boundary even in the starter: the
  // one-kernel launch is cooperative so waiting CTAs cannot starve producers.
  grid.sync();

  // The controller CTA owns the QP completion and incoming-signal waits.
  // Keep the phase boundary below: other CTAs wait at the grid sync while the
  // controller finishes the source-side quiet.
  if (blockIdx.x == 0) {
    /*
     * TODO: Cooperatively quiet every default and custom QP using
     * quiet_scope, so this PE can safely reuse its source buffer.
     */
    (void)quiet_scope;
  }
  grid.sync();

  if (blockIdx.x == 0) {
    /*
     * TODO: Wait for the number of source signals supplied by the route-plan
     * exchange. Direct and network peers signal once per route-specific chunk
     * before the kernel may return.
     */
    (void)recv_signal_counts;
    (void)signals;
    (void)rank;
    (void)nranks;
    (void)max_chunks;
    (void)epoch;
  }
  grid.sync();
}

std::uint64_t chunk_bytes_from_environment(const char *name,
                                           std::uint64_t default_bytes,
                                           int rank) {
  const char *text = std::getenv(name);
  std::uint64_t bytes =
      text == nullptr ? default_bytes : alltoallv::parse_bytes(text);
  if (bytes == 0 || bytes % kMetadataAlignment != 0) {
    if (rank == 0)
      std::fprintf(stderr, "%s must be a positive multiple of %llu\n", name,
                   static_cast<unsigned long long>(kMetadataAlignment));
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::uint64_t minimum = 0;
  std::uint64_t maximum = 0;
  alltoallv::mpi_check(
      MPI_Allreduce(&bytes, &minimum, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD),
      "MPI_Allreduce(chunk minimum)");
  alltoallv::mpi_check(
      MPI_Allreduce(&bytes, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
      "MPI_Allreduce(chunk maximum)");
  if (minimum != maximum) {
    if (rank == 0)
      std::fprintf(stderr, "%s must match on every PE\n", name);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return bytes;
}

int network_qp_count_from_environment(int rank, int default_count) {
  const char *name = "HOTI_ALLTOALLV_NETWORK_QPS";
  const char *text = std::getenv(name);
  char *end = nullptr;
  long value = text == nullptr ? default_count : std::strtol(text, &end, 10);
  if (value < 1 || value > std::numeric_limits<int>::max() ||
      (text != nullptr && (end == text || *end != '\0'))) {
    if (rank == 0)
      std::fprintf(stderr, "%s must be a positive integer\n", name);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int count = static_cast<int>(value);
  int minimum = 0;
  int maximum = 0;
  alltoallv::mpi_check(
      MPI_Allreduce(&count, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
      "MPI_Allreduce(QP count minimum)");
  alltoallv::mpi_check(
      MPI_Allreduce(&count, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
      "MPI_Allreduce(QP count maximum)");
  if (minimum != maximum) {
    if (rank == 0)
      std::fprintf(stderr, "%s must match on every PE\n", name);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return count;
}

void require_matching_launch(const alltoallv::Options &options, int rank) {
  alltoallv::require_matching_collective_options(options, rank);
}

void print_route_summary(value_type *recv_buffer, int rank, int nranks) {
  int local_direct = 0;
  int local_network = 0;
  for (int peer = 0; peer < nranks; ++peer) {
    if (peer == rank)
      continue;
    if (nvshmem_ptr(recv_buffer, peer) != nullptr)
      ++local_direct;
    else
      ++local_network;
  }
  int direct = 0;
  int network = 0;
  alltoallv::mpi_check(MPI_Reduce(&local_direct, &direct, 1, MPI_INT, MPI_SUM,
                                  0, MPI_COMM_WORLD),
                       "MPI_Reduce(direct routes)");
  alltoallv::mpi_check(MPI_Reduce(&local_network, &network, 1, MPI_INT, MPI_SUM,
                                  0, MPI_COMM_WORLD),
                       "MPI_Reduce(network routes)");
  if (rank == 0)
    std::printf("NVSHMEM routes: %d direct, %d network\n", direct, network);
}

void classify_direct_routes(alltoallv::Plan *plan, value_type *recv_buffer,
                            int direct_root, int direct_rank) {
  plan->placement_scope = "NVSHMEM direct peer";
  plan->domain_roots.resize(plan->size);
  plan->domain_ranks.resize(plan->size);
  alltoallv::mpi_check(MPI_Allgather(&direct_root, 1, MPI_INT,
                                     plan->domain_roots.data(), 1, MPI_INT,
                                     MPI_COMM_WORLD),
                       "MPI_Allgather(direct roots)");
  alltoallv::mpi_check(MPI_Allgather(&direct_rank, 1, MPI_INT,
                                     plan->domain_ranks.data(), 1, MPI_INT,
                                     MPI_COMM_WORLD),
                       "MPI_Allgather(direct ranks)");

  std::vector<int> outgoing_direct(plan->size, 0);
  std::vector<int> incoming_direct(plan->size, 0);
  for (int peer = 0; peer < plan->size; ++peer)
    outgoing_direct[peer] =
        peer == plan->rank || nvshmem_ptr(recv_buffer, peer) != nullptr;
  alltoallv::mpi_check(MPI_Alltoall(outgoing_direct.data(), 1, MPI_INT,
                                    incoming_direct.data(), 1, MPI_INT,
                                    MPI_COMM_WORLD),
                       "MPI_Alltoall(direct routes)");

  std::uint64_t local_self = 0;
  std::uint64_t local_direct = 0;
  std::uint64_t local_network = 0;
  std::uint64_t local_offrail = 0;
  std::uint64_t local_network_recv = 0;
  for (int peer = 0; peer < plan->size; ++peer) {
    if (peer == plan->rank) {
      local_self += plan->send_counts[peer];
    } else if (outgoing_direct[peer]) {
      local_direct += plan->send_counts[peer];
    } else {
      local_network += plan->send_counts[peer];
      if (plan->domain_ranks[peer] != plan->domain_ranks[plan->rank])
        local_offrail += plan->send_counts[peer];
    }
    if (peer != plan->rank && !incoming_direct[peer])
      local_network_recv += plan->recv_counts[peer];
  }

  std::uint64_t local_classes[4] = {local_self, local_direct, local_network,
                                    local_offrail};
  std::uint64_t global_classes[4] = {};
  alltoallv::mpi_check(MPI_Allreduce(local_classes, global_classes, 4,
                                     MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD),
                       "MPI_Allreduce(direct traffic classes)");
  plan->global_self_elements = global_classes[0];
  plan->global_local_elements = global_classes[1];
  plan->global_network_elements = global_classes[2];
  plan->global_offrail_elements = global_classes[3];

  std::uint64_t local_maxima[2] = {local_network, local_network_recv};
  std::uint64_t global_maxima[2] = {};
  alltoallv::mpi_check(MPI_Allreduce(local_maxima, global_maxima, 2,
                                     MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
                       "MPI_Allreduce(direct route maxima)");
  plan->max_network_send_elements = global_maxima[0];
  plan->max_network_recv_elements = global_maxima[1];
}

template <typename T> T *symmetric_alloc(std::uint64_t count) {
  std::uint64_t allocation_count = std::max<std::uint64_t>(count, 1);
  if (allocation_count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    return nullptr;
  return static_cast<T *>(
      nvshmem_malloc(static_cast<std::size_t>(allocation_count) * sizeof(T)));
}

int launch_plan_exchange(
    const std::uint64_t *send_counts_bytes, std::uint64_t *recv_counts_bytes,
    std::uint64_t *recv_offsets_bytes, std::uint64_t *remote_recv_offsets_bytes,
    value_type *recv_buffer, std::uint64_t *send_signal_counts,
    std::uint64_t *recv_signal_counts, int rank, int nranks,
    std::uint64_t direct_chunk_bytes, std::uint64_t network_chunk_bytes,
    int threads, int *device_status, cudaStream_t stream) {
  void *args[] = {&send_counts_bytes,
                  &recv_counts_bytes,
                  &recv_offsets_bytes,
                  &remote_recv_offsets_bytes,
                  &recv_buffer,
                  &send_signal_counts,
                  &recv_signal_counts,
                  &rank,
                  &nranks,
                  &direct_chunk_bytes,
                  &network_chunk_bytes,
                  &device_status};
  int status =
      nvshmemx_collective_launch(reinterpret_cast<const void *>(exchange_plan),
                                 dim3(1), dim3(threads), args, 0, stream);
  if (status != NVSHMEMX_SUCCESS)
    return status;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int result = 0;
  CUDA_CHECK(cudaMemcpy(&result, device_status, sizeof(result),
                        cudaMemcpyDeviceToHost));
  return result;
}

int validate_device_plan(const alltoallv::Plan &plan,
                         const std::uint64_t *recv_counts_bytes,
                         const std::uint64_t *recv_offsets_bytes,
                         const std::uint64_t *remote_recv_offsets_bytes,
                         const std::uint64_t *send_signal_counts,
                         const std::uint64_t *recv_signal_counts,
                         std::uint64_t max_chunks) {
  std::vector<std::uint64_t> counts(plan.size);
  std::vector<std::uint64_t> offsets(plan.size);
  std::vector<std::uint64_t> remote_offsets(plan.size);
  std::vector<std::uint64_t> send_signal_count(plan.size);
  std::vector<std::uint64_t> recv_signal_count(plan.size);
  std::vector<std::uint64_t> expected_recv_signal_count(plan.size);
  CUDA_CHECK(cudaMemcpy(counts.data(), recv_counts_bytes,
                        plan.size * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(offsets.data(), recv_offsets_bytes,
                        plan.size * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(remote_offsets.data(), remote_recv_offsets_bytes,
                        plan.size * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(send_signal_count.data(), send_signal_counts,
                        plan.size * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(recv_signal_count.data(), recv_signal_counts,
                        plan.size * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost));
  alltoallv::mpi_check(MPI_Alltoall(send_signal_count.data(), 1, MPI_UINT64_T,
                                    expected_recv_signal_count.data(), 1,
                                    MPI_UINT64_T, MPI_COMM_WORLD),
                       "MPI_Alltoall(signal count validation)");

  int local_errors = 0;
  for (int peer = 0; peer < plan.size; ++peer) {
    if (counts[peer] != plan.recv_counts[peer] * sizeof(value_type) ||
        offsets[peer] != plan.recv_offsets[peer] * sizeof(value_type) ||
        remote_offsets[peer] !=
            plan.remote_recv_offsets[peer] * sizeof(value_type) ||
        recv_signal_count[peer] != expected_recv_signal_count[peer] ||
        recv_signal_count[peer] > max_chunks ||
        (peer == plan.rank && recv_signal_count[peer] != 0)) {
      if (local_errors < 4)
        std::fprintf(stderr, "Rank %d: device plan mismatch for peer %d\n",
                     plan.rank, peer);
      ++local_errors;
    }
  }
  int errors = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_errors, &errors, 1, MPI_INT,
                                     MPI_SUM, MPI_COMM_WORLD),
                       "MPI_Allreduce(device plan validation)");
  if (plan.rank == 0)
    std::printf("NVSHMEM device plan: %s\n", errors == 0 ? "PASS" : "FAIL");
  return errors;
}

void launch_alltoallv(const value_type *send_buffer, value_type *recv_buffer,
                      const std::uint64_t *send_counts_bytes,
                      const std::uint64_t *send_offsets_bytes,
                      const std::uint64_t *recv_signal_counts,
                      const std::uint64_t *remote_recv_offsets_bytes,
                      std::uint64_t *signals, int rank, int nranks,
                      std::uint64_t direct_chunk_bytes,
                      std::uint64_t network_chunk_bytes,
                      const nvshmemx_qp_handle_t *network_qps,
                      int network_qp_count, std::uint64_t max_chunks,
                      QpQuietScope quiet_scope, std::uint64_t epoch,
                      const alltoallv::Options &options, cudaStream_t stream) {
  void *args[] = {&send_buffer,
                  &recv_buffer,
                  &send_counts_bytes,
                  &send_offsets_bytes,
                  &recv_signal_counts,
                  &remote_recv_offsets_bytes,
                  &signals,
                  &rank,
                  &nranks,
                  &direct_chunk_bytes,
                  &network_chunk_bytes,
                  &network_qps,
                  &network_qp_count,
                  &max_chunks,
                  &quiet_scope,
                  &epoch};
  int status = nvshmemx_collective_launch(
      reinterpret_cast<const void *>(nvshmem_alltoallv_kernel),
      dim3(options.blocks), dim3(options.threads), args, 0, stream);
  if (status != NVSHMEMX_SUCCESS) {
    std::fprintf(stderr, "Rank %d: NVSHMEM AlltoAllV launch failed: %d\n",
                 rank, status);
    MPI_Abort(MPI_COMM_WORLD, status);
  }
}

void require_cooperative_grid(
    const value_type *send_buffer, value_type *recv_buffer,
    const std::uint64_t *send_counts_bytes,
    const std::uint64_t *send_offsets_bytes,
    const std::uint64_t *recv_signal_counts,
    const std::uint64_t *remote_recv_offsets_bytes, std::uint64_t *signals,
    int rank, int nranks, std::uint64_t direct_chunk_bytes,
    std::uint64_t network_chunk_bytes,
    const nvshmemx_qp_handle_t *network_qps, int network_qp_count,
    std::uint64_t max_chunks, QpQuietScope quiet_scope,
    const alltoallv::Options &options) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int local_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &local_supported, cudaDevAttrCooperativeLaunch, device));
  int all_supported = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_supported, &all_supported, 1,
                                     MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(cooperative launch support)");
  if (!all_supported) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "The single-kernel NVSHMEM implementation requires CUDA "
                   "cooperative launch support on every PE\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::uint64_t epoch = 1;
  void *args[] = {&send_buffer,
                  &recv_buffer,
                  &send_counts_bytes,
                  &send_offsets_bytes,
                  &recv_signal_counts,
                  &remote_recv_offsets_bytes,
                  &signals,
                  &rank,
                  &nranks,
                  &direct_chunk_bytes,
                  &network_chunk_bytes,
                  &network_qps,
                  &network_qp_count,
                  &max_chunks,
                  &quiet_scope,
                  &epoch};
  int local_limit = 0;
  int query_status = nvshmemx_collective_launch_query_gridsize(
      reinterpret_cast<const void *>(nvshmem_alltoallv_kernel),
      dim3(options.threads), args, 0, &local_limit);
  int local_ready = query_status == NVSHMEMX_SUCCESS &&
                    local_limit >= options.blocks;
  int all_ready = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT,
                                     MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(cooperative launch capacity)");
  int collective_limit = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_limit, &collective_limit, 1,
                                     MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                       "MPI_Allreduce(cooperative grid limit)");
  if (all_ready)
    return;
  if (rank == 0) {
    std::fprintf(stderr,
                 "--blocks=%d exceeds the single-kernel cooperative grid "
                 "limit (%d); lower --blocks on every PE\n",
                 options.blocks, collective_limit);
  }
  MPI_Abort(MPI_COMM_WORLD, query_status == NVSHMEMX_SUCCESS ? 1
                                                               : query_status);
}

void free_allocations(value_type *send_buffer, value_type *recv_buffer,
                      std::uint64_t *send_counts_bytes,
                      std::uint64_t *send_offsets_bytes,
                      std::uint64_t *recv_counts_bytes,
                      std::uint64_t *recv_offsets_bytes,
                      std::uint64_t *remote_recv_offsets_bytes,
                      std::uint64_t *send_signal_counts,
                      std::uint64_t *recv_signal_counts, std::uint64_t *signals,
                      int *device_status, nvshmemx_qp_handle_t *network_qps) {
  nvshmem_free(send_buffer);
  nvshmem_free(recv_buffer);
  nvshmem_free(send_counts_bytes);
  nvshmem_free(send_offsets_bytes);
  nvshmem_free(recv_counts_bytes);
  nvshmem_free(recv_offsets_bytes);
  nvshmem_free(remote_recv_offsets_bytes);
  nvshmem_free(send_signal_counts);
  nvshmem_free(recv_signal_counts);
  nvshmem_free(signals);
  CUDA_CHECK(cudaFree(device_status));
  CUDA_CHECK(cudaFree(network_qps));
}

} // namespace

int main(int argc, char **argv) {
  exercise_context_t context;
  exercise_init(&argc, &argv, &context);

  alltoallv::Options options =
      alltoallv::parse_options(argc, argv, context.rank);
  if (options.help) {
    if (context.rank == 0)
      alltoallv::print_usage(argv[0]);
    exercise_finalize(&context);
    return 0;
  }
  require_matching_launch(options, context.rank);

  alltoallv::Plan plan =
      alltoallv::make_plan(context.rank, context.size, options);

  std::uint64_t direct_chunk_bytes = chunk_bytes_from_environment(
      "HOTI_ALLTOALLV_CHUNK_BYTES", kDefaultDirectChunkBytes, context.rank);
  std::uint64_t network_chunk_bytes =
      chunk_bytes_from_environment("HOTI_ALLTOALLV_NETWORK_CHUNK_BYTES",
                                   kDefaultNetworkChunkBytes, context.rank);
  std::uint64_t max_pair_bytes = plan.max_pair_count * sizeof(value_type);
  std::uint64_t smallest_chunk_bytes =
      std::min(direct_chunk_bytes, network_chunk_bytes);
  std::uint64_t max_chunks = std::max<std::uint64_t>(
      1, (max_pair_bytes + smallest_chunk_bytes - 1) / smallest_chunk_bytes);
  if (max_chunks > std::numeric_limits<std::uint64_t>::max() /
                       static_cast<std::uint64_t>(context.size)) {
    if (context.rank == 0)
      std::fprintf(stderr, "The signal table is too large\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::uint64_t signal_count =
      static_cast<std::uint64_t>(context.size) * max_chunks;

  value_type *send_buffer =
      symmetric_alloc<value_type>(plan.global_send_capacity);
  value_type *recv_buffer =
      symmetric_alloc<value_type>(plan.global_recv_capacity);
  std::uint64_t *send_counts_bytes =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *send_offsets_bytes =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *recv_counts_bytes =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *recv_offsets_bytes =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *remote_recv_offsets_bytes =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *send_signal_counts =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *recv_signal_counts =
      symmetric_alloc<std::uint64_t>(context.size);
  std::uint64_t *signals = symmetric_alloc<std::uint64_t>(signal_count);
  int *device_status = nullptr;
  CUDA_CHECK(cudaMalloc(&device_status, sizeof(*device_status)));
  if (send_buffer == nullptr || recv_buffer == nullptr ||
      send_counts_bytes == nullptr || send_offsets_bytes == nullptr ||
      recv_counts_bytes == nullptr || recv_offsets_bytes == nullptr ||
      remote_recv_offsets_bytes == nullptr || send_signal_counts == nullptr ||
      recv_signal_counts == nullptr || signals == nullptr) {
    if (context.rank == 0)
      std::fprintf(stderr, "NVSHMEM symmetric allocation failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int direct_root = context.rank;
  int direct_rank = 0;
  int local_direct_peers = 0;
  int local_network_peers = 0;
  for (int peer = 0; peer < context.size; ++peer) {
    bool direct =
        peer == context.rank || nvshmem_ptr(recv_buffer, peer) != nullptr;
    if (direct) {
      direct_root = std::min(direct_root, peer);
      direct_rank += peer < context.rank;
    }
    if (peer != context.rank) {
      local_direct_peers += direct;
      local_network_peers += !direct;
    }
  }
  int any_direct = 0;
  int any_network = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_direct_peers, &any_direct, 1,
                                     MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                       "MPI_Allreduce(direct routes)");
  alltoallv::mpi_check(MPI_Allreduce(&local_network_peers, &any_network, 1,
                                     MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                       "MPI_Allreduce(network routes)");
  QpQuietScope quiet_scope =
      any_direct && any_network
          ? QpQuietScope::block
          : (any_network ? QpQuietScope::warp : QpQuietScope::thread);
  int default_network_qp_count = any_network ? (any_direct ? 8 : 16) : 1;
  int network_qp_count =
      network_qp_count_from_environment(context.rank, default_network_qp_count);

  nvshmemx_qp_handle_t *host_network_qps = nullptr;
  int qp_status = nvshmemx_qp_create(network_qp_count, &host_network_qps);
  if (qp_status != NVSHMEMX_SUCCESS || host_network_qps == nullptr) {
    if (context.rank == 0)
      std::fprintf(stderr, "NVSHMEM network QP creation failed: %d\n",
                   qp_status);
    MPI_Abort(MPI_COMM_WORLD, qp_status == 0 ? 1 : qp_status);
  }
  nvshmemx_qp_handle_t *network_qps = nullptr;
  CUDA_CHECK(cudaMalloc(&network_qps, network_qp_count * sizeof(*network_qps)));
  CUDA_CHECK(cudaMemcpy(network_qps, host_network_qps,
                        network_qp_count * sizeof(*network_qps),
                        cudaMemcpyHostToDevice));

  classify_direct_routes(&plan, recv_buffer, direct_root, direct_rank);
  alltoallv::print_plan(plan, options, "NVSHMEM AlltoAllV");

  std::vector<value_type> host_send = alltoallv::make_send_buffer(plan);
  std::vector<std::uint64_t> host_send_counts_bytes(context.size);
  std::vector<std::uint64_t> host_send_offsets_bytes(context.size);
  for (int peer = 0; peer < context.size; ++peer) {
    host_send_counts_bytes[peer] = plan.send_counts[peer] * sizeof(value_type);
    host_send_offsets_bytes[peer] =
        plan.send_offsets[peer] * sizeof(value_type);
  }

  CUDA_CHECK(cudaMemcpyAsync(send_buffer, host_send.data(),
                             plan.global_send_capacity * sizeof(value_type),
                             cudaMemcpyHostToDevice, context.stream));
  CUDA_CHECK(cudaMemsetAsync(recv_buffer, 0xa5,
                             plan.global_recv_capacity * sizeof(value_type),
                             context.stream));
  CUDA_CHECK(cudaMemcpyAsync(send_counts_bytes, host_send_counts_bytes.data(),
                             context.size * sizeof(std::uint64_t),
                             cudaMemcpyHostToDevice, context.stream));
  CUDA_CHECK(cudaMemcpyAsync(send_offsets_bytes, host_send_offsets_bytes.data(),
                             context.size * sizeof(std::uint64_t),
                             cudaMemcpyHostToDevice, context.stream));
  CUDA_CHECK(cudaMemsetAsync(recv_counts_bytes, 0,
                             context.size * sizeof(std::uint64_t),
                             context.stream));
  CUDA_CHECK(cudaMemsetAsync(recv_offsets_bytes, 0,
                             context.size * sizeof(std::uint64_t),
                             context.stream));
  CUDA_CHECK(cudaMemsetAsync(remote_recv_offsets_bytes, 0,
                             context.size * sizeof(std::uint64_t),
                             context.stream));
  CUDA_CHECK(cudaMemsetAsync(send_signal_counts, 0,
                             context.size * sizeof(std::uint64_t),
                             context.stream));
  CUDA_CHECK(cudaMemsetAsync(recv_signal_counts, 0,
                             context.size * sizeof(std::uint64_t),
                             context.stream));
  CUDA_CHECK(cudaMemsetAsync(signals, 0, signal_count * sizeof(std::uint64_t),
                             context.stream));
  CUDA_CHECK(cudaMemsetAsync(device_status, 0xff, sizeof(*device_status),
                             context.stream));
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  MPI_Barrier(MPI_COMM_WORLD);

  int plan_status = launch_plan_exchange(
      send_counts_bytes, recv_counts_bytes, recv_offsets_bytes,
      remote_recv_offsets_bytes, recv_buffer, send_signal_counts,
      recv_signal_counts, context.rank, context.size, direct_chunk_bytes,
      network_chunk_bytes, options.threads, device_status, context.stream);
  if (plan_status != 0) {
    if (context.rank == 0)
      std::fprintf(stderr, "NVSHMEM device plan exchange failed: %d\n",
                   plan_status);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  if (validate_device_plan(plan, recv_counts_bytes, recv_offsets_bytes,
                           remote_recv_offsets_bytes, send_signal_counts,
                           recv_signal_counts, max_chunks) != 0) {
    free_allocations(send_buffer, recv_buffer, send_counts_bytes,
                     send_offsets_bytes, recv_counts_bytes, recv_offsets_bytes,
                     remote_recv_offsets_bytes, send_signal_counts,
                     recv_signal_counts, signals, device_status, network_qps);
    exercise_finalize(&context);
    std::free(host_network_qps);
    return 1;
  }
  require_cooperative_grid(
      send_buffer, recv_buffer, send_counts_bytes, send_offsets_bytes,
      recv_signal_counts, remote_recv_offsets_bytes, signals, context.rank,
      context.size, direct_chunk_bytes, network_chunk_bytes, network_qps,
      network_qp_count, max_chunks, quiet_scope, options);

  if (context.rank == 0)
    std::printf("NVSHMEM chunk sizes: direct=%llu bytes, network=%llu bytes\n",
                static_cast<unsigned long long>(direct_chunk_bytes),
                static_cast<unsigned long long>(network_chunk_bytes));
  if (context.rank == 0) {
    int explicit_qps = 0;
    for (int index = 0; index < network_qp_count; ++index)
      explicit_qps += host_network_qps[index] != NVSHMEMX_QP_DEFAULT;
    std::printf("NVSHMEM network QPs: requested=%d, explicit=%d\n",
                network_qp_count, explicit_qps);
    const char *quiet_name =
        quiet_scope == QpQuietScope::block
            ? "block"
            : (quiet_scope == QpQuietScope::warp ? "warp" : "thread");
    std::printf("NVSHMEM completion quiet: %s scoped\n", quiet_name);
  }
  print_route_summary(recv_buffer, context.rank, context.size);

  std::uint64_t epoch = 1;
  launch_alltoallv(send_buffer, recv_buffer, send_counts_bytes,
                   send_offsets_bytes, recv_signal_counts,
                   remote_recv_offsets_bytes, signals, context.rank,
                   context.size, direct_chunk_bytes, network_chunk_bytes,
                   network_qps, network_qp_count, max_chunks, quiet_scope,
                   epoch++, options, context.stream);
  CUDA_CHECK(cudaStreamSynchronize(context.stream));

  std::vector<value_type> observed(plan.global_recv_capacity);
  CUDA_CHECK(cudaMemcpy(observed.data(), recv_buffer,
                        plan.global_recv_capacity * sizeof(value_type),
                        cudaMemcpyDeviceToHost));
  int errors = alltoallv::validate(plan, observed, "NVSHMEM AlltoAllV");
  if (errors != 0) {
    free_allocations(send_buffer, recv_buffer, send_counts_bytes,
                     send_offsets_bytes, recv_counts_bytes, recv_offsets_bytes,
                     remote_recv_offsets_bytes, send_signal_counts,
                     recv_signal_counts, signals, device_status, network_qps);
    exercise_finalize(&context);
    std::free(host_network_qps);
    return 1;
  }

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    launch_alltoallv(send_buffer, recv_buffer, send_counts_bytes,
                     send_offsets_bytes, recv_signal_counts,
                     remote_recv_offsets_bytes, signals, context.rank,
                     context.size, direct_chunk_bytes, network_chunk_bytes,
                     network_qps, network_qp_count, max_chunks, quiet_scope,
                     epoch++, options, context.stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(timing)");

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, context.stream));
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    launch_alltoallv(send_buffer, recv_buffer, send_counts_bytes,
                     send_offsets_bytes, recv_signal_counts,
                     remote_recv_offsets_bytes, signals, context.rank,
                     context.size, direct_chunk_bytes, network_chunk_bytes,
                     network_qps, network_qp_count, max_chunks, quiet_scope,
                     epoch++, options, context.stream);
  }
  CUDA_CHECK(cudaEventRecord(stop, context.stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  alltoallv::report_timing(plan, options, "NVSHMEM AlltoAllV", elapsed_ms);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaMemsetAsync(recv_buffer, 0xa5,
                             plan.global_recv_capacity * sizeof(value_type),
                             context.stream));
  launch_alltoallv(send_buffer, recv_buffer, send_counts_bytes,
                   send_offsets_bytes, recv_signal_counts,
                   remote_recv_offsets_bytes, signals, context.rank,
                   context.size, direct_chunk_bytes, network_chunk_bytes,
                   network_qps, network_qp_count, max_chunks, quiet_scope,
                   epoch++, options, context.stream);
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  CUDA_CHECK(cudaMemcpy(observed.data(), recv_buffer,
                        plan.global_recv_capacity * sizeof(value_type),
                        cudaMemcpyDeviceToHost));
  errors = alltoallv::validate(plan, observed, "NVSHMEM AlltoAllV reuse");
  if (errors != 0) {
    free_allocations(send_buffer, recv_buffer, send_counts_bytes,
                     send_offsets_bytes, recv_counts_bytes, recv_offsets_bytes,
                     remote_recv_offsets_bytes, send_signal_counts,
                     recv_signal_counts, signals, device_status, network_qps);
    exercise_finalize(&context);
    std::free(host_network_qps);
    return 1;
  }

  free_allocations(send_buffer, recv_buffer, send_counts_bytes,
                   send_offsets_bytes, recv_counts_bytes, recv_offsets_bytes,
                   remote_recv_offsets_bytes, send_signal_counts,
                   recv_signal_counts, signals, device_status, network_qps);
  exercise_finalize(&context);
  std::free(host_network_qps);
  return 0;
}
