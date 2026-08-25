/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/nvshmem_exercise.h"
#include "../alltoallv_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

using alltoallv::value_type;

constexpr std::uint64_t kDefaultDirectChunkBytes = 256ull << 10;
constexpr std::uint64_t kDefaultNetworkChunkBytes = 8ull << 20;
constexpr int kDefaultNetworkQpCount = 8;
constexpr std::uint64_t kMetadataAlignment = 16;

__device__ std::uint64_t align_bytes(std::uint64_t value) {
  return (value + kMetadataAlignment - 1) & ~(kMetadataAlignment - 1);
}

__global__ void begin_transfer(int) { nvshmemx_barrier_all_block(); }

__global__ void exchange_plan(const std::uint64_t *send_counts_bytes,
                              std::uint64_t *recv_counts_bytes,
                              std::uint64_t *recv_offsets_bytes,
                              std::uint64_t *remote_recv_offsets_bytes,
                              value_type *recv_buffer,
                              std::uint64_t *send_signal_counts,
                              std::uint64_t *recv_signal_counts, int rank,
                              int nranks, std::uint64_t direct_chunk_bytes,
                              std::uint64_t network_chunk_bytes, int *status) {
  int counts_status = nvshmemx_uint64_alltoall_block(
      NVSHMEM_TEAM_WORLD, recv_counts_bytes, send_counts_bytes, 1);

  if (threadIdx.x == 0) {
    std::uint64_t cursor = 0;
    for (int source = 0; source < nranks; ++source) {
      cursor = align_bytes(cursor);
      recv_offsets_bytes[source] = cursor;
      cursor += recv_counts_bytes[source];
    }
  }
  __syncthreads();

  int offsets_status = nvshmemx_uint64_alltoall_block(
      NVSHMEM_TEAM_WORLD, remote_recv_offsets_bytes, recv_offsets_bytes, 1);

  if (threadIdx.x == 0) {
    for (int destination = 0; destination < nranks; ++destination) {
      std::uint64_t bytes = send_counts_bytes[destination];
      char *destination_address = reinterpret_cast<char *>(recv_buffer) +
                                  remote_recv_offsets_bytes[destination];
      bool direct = destination != rank &&
                    nvshmem_ptr(destination_address, destination) != nullptr;
      std::uint64_t route_chunk_bytes =
          direct ? direct_chunk_bytes : network_chunk_bytes;
      send_signal_counts[destination] =
          destination == rank || bytes == 0
              ? 0
              : (bytes + route_chunk_bytes - 1) / route_chunk_bytes;
    }
  }
  __syncthreads();

  int signals_status = nvshmemx_uint64_alltoall_block(
      NVSHMEM_TEAM_WORLD, recv_signal_counts, send_signal_counts, 1);
  if (threadIdx.x == 0)
    *status = counts_status != 0
                  ? counts_status
                  : (offsets_status != 0 ? offsets_status : signals_status);
}

__global__ void
send_chunks(const value_type *send_buffer, value_type *recv_buffer,
            const std::uint64_t *send_counts_bytes,
            const std::uint64_t *send_offsets_bytes,
            const std::uint64_t *remote_recv_offsets_bytes,
            std::uint64_t *signals, int rank, int nranks,
            std::uint64_t direct_chunk_bytes, std::uint64_t network_chunk_bytes,
            const nvshmemx_qp_handle_t *network_qps, int network_qp_count,
            std::uint64_t max_chunks, std::uint64_t epoch) {
  std::uint64_t task_count = static_cast<std::uint64_t>(nranks) * max_chunks;
  for (std::uint64_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    int destination = static_cast<int>(
        (task % static_cast<std::uint64_t>(nranks) + rank) % nranks);
    std::uint64_t chunk = task / nranks;
    std::uint64_t pair_bytes = send_counts_bytes[destination];
    char *pair_destination = reinterpret_cast<char *>(recv_buffer) +
                             remote_recv_offsets_bytes[destination];
    void *peer_pointer = destination == rank
                             ? pair_destination
                             : nvshmem_ptr(pair_destination, destination);
    std::uint64_t route_chunk_bytes =
        destination == rank || peer_pointer != nullptr ? direct_chunk_bytes
                                                       : network_chunk_bytes;
    std::uint64_t chunk_offset = chunk * route_chunk_bytes;
    if (pair_bytes == 0 || chunk_offset >= pair_bytes)
      continue;

    std::uint64_t remaining = pair_bytes - chunk_offset;
    std::uint64_t bytes =
        route_chunk_bytes < remaining ? route_chunk_bytes : remaining;
    const char *source = reinterpret_cast<const char *>(send_buffer) +
                         send_offsets_bytes[destination] + chunk_offset;
    char *destination_address = pair_destination + chunk_offset;

    if (destination == rank) {
      std::uint64_t vector_count = bytes / sizeof(uint4);
      const uint4 *source_vectors = reinterpret_cast<const uint4 *>(source);
      uint4 *destination_vectors =
          reinterpret_cast<uint4 *>(destination_address);
      for (std::uint64_t vector = threadIdx.x; vector < vector_count;
           vector += blockDim.x)
        destination_vectors[vector] = source_vectors[vector];
      std::uint64_t element_begin =
          vector_count * sizeof(uint4) / sizeof(value_type);
      const value_type *source_elements =
          reinterpret_cast<const value_type *>(source);
      value_type *destination_elements =
          reinterpret_cast<value_type *>(destination_address);
      for (std::uint64_t element = element_begin + threadIdx.x;
           element < bytes / sizeof(value_type); element += blockDim.x)
        destination_elements[element] = source_elements[element];
    } else if (peer_pointer != nullptr) {
      nvshmemx_putmem_signal_nbi_block(
          destination_address, source, bytes,
          signals + static_cast<std::uint64_t>(rank) * max_chunks + chunk,
          epoch, NVSHMEM_SIGNAL_SET, destination);
    } else if (threadIdx.x == 0) {
      nvshmemx_qp_uint_put_signal_nbi(
          reinterpret_cast<value_type *>(destination_address),
          reinterpret_cast<const value_type *>(source),
          bytes / sizeof(value_type),
          signals + static_cast<std::uint64_t>(rank) * max_chunks + chunk,
          epoch, NVSHMEM_SIGNAL_SET, destination,
          network_qps[(static_cast<std::uint64_t>(destination) + chunk) %
                      static_cast<std::uint64_t>(network_qp_count)]);
    }
    __syncthreads();
  }
}

__global__ void wait_for_chunks(const std::uint64_t *recv_signal_counts,
                                std::uint64_t *signals, int rank, int nranks,
                                std::uint64_t max_chunks, std::uint64_t epoch) {
  if (threadIdx.x == 0) {
    nvshmemx_qp_handle_t all_qps = NVSHMEMX_QP_ALL;
    nvshmemx_qp_quiet(NVSHMEMX_PE_ALL, &all_qps, 1);
  }
  std::uint64_t task_count = static_cast<std::uint64_t>(nranks) * max_chunks;
  for (std::uint64_t task = threadIdx.x; task < task_count;
       task += blockDim.x) {
    int source = static_cast<int>(task / max_chunks);
    std::uint64_t chunk = task % max_chunks;
    std::uint64_t expected_chunks = recv_signal_counts[source];
    if (source == rank || chunk >= expected_chunks)
      continue;
    nvshmem_signal_wait_until(
        signals + static_cast<std::uint64_t>(source) * max_chunks + chunk,
        NVSHMEM_CMP_GE, epoch);
  }
  __syncthreads();
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

int network_qp_count_from_environment(int rank) {
  const char *name = "HOTI_ALLTOALLV_NETWORK_QPS";
  const char *text = std::getenv(name);
  char *end = nullptr;
  long value =
      text == nullptr ? kDefaultNetworkQpCount : std::strtol(text, &end, 10);
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
                      std::uint64_t epoch, const alltoallv::Options &options,
                      cudaStream_t stream) {
  int unused = 0;
  void *begin_args[] = {&unused};
  int begin_status = nvshmemx_collective_launch(
      reinterpret_cast<const void *>(begin_transfer), dim3(1),
      dim3(options.threads), begin_args, 0, stream);
  if (begin_status != NVSHMEMX_SUCCESS) {
    std::fprintf(stderr, "Rank %d: NVSHMEM entry launch failed: %d\n", rank,
                 begin_status);
    MPI_Abort(MPI_COMM_WORLD, begin_status);
  }
  send_chunks<<<options.blocks, options.threads, 0, stream>>>(
      send_buffer, recv_buffer, send_counts_bytes, send_offsets_bytes,
      remote_recv_offsets_bytes, signals, rank, nranks, direct_chunk_bytes,
      network_chunk_bytes, network_qps, network_qp_count, max_chunks, epoch);
  CUDA_CHECK(cudaGetLastError());
  void *args[] = {&recv_signal_counts, &signals, &rank, &nranks,
                  &max_chunks,         &epoch};
  int wait_status = nvshmemx_collective_launch(
      reinterpret_cast<const void *>(wait_for_chunks), dim3(1),
      dim3(options.threads), args, 0, stream);
  if (wait_status != NVSHMEMX_SUCCESS) {
    std::fprintf(stderr, "Rank %d: NVSHMEM wait launch failed: %d\n", rank,
                 wait_status);
    MPI_Abort(MPI_COMM_WORLD, wait_status);
  }
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
  alltoallv::print_plan(plan, options, "NVSHMEM AlltoAllV");

  std::uint64_t direct_chunk_bytes = chunk_bytes_from_environment(
      "HOTI_ALLTOALLV_CHUNK_BYTES", kDefaultDirectChunkBytes, context.rank);
  std::uint64_t network_chunk_bytes =
      chunk_bytes_from_environment("HOTI_ALLTOALLV_NETWORK_CHUNK_BYTES",
                                   kDefaultNetworkChunkBytes, context.rank);
  int network_qp_count = network_qp_count_from_environment(context.rank);
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
  }
  print_route_summary(recv_buffer, context.rank, context.size);

  std::uint64_t epoch = 1;
  launch_alltoallv(
      send_buffer, recv_buffer, send_counts_bytes, send_offsets_bytes,
      recv_signal_counts, remote_recv_offsets_bytes, signals, context.rank,
      context.size, direct_chunk_bytes, network_chunk_bytes, network_qps,
      network_qp_count, max_chunks, epoch++, options, context.stream);
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
    launch_alltoallv(
        send_buffer, recv_buffer, send_counts_bytes, send_offsets_bytes,
        recv_signal_counts, remote_recv_offsets_bytes, signals, context.rank,
        context.size, direct_chunk_bytes, network_chunk_bytes, network_qps,
        network_qp_count, max_chunks, epoch++, options, context.stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(timing)");

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, context.stream));
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    launch_alltoallv(
        send_buffer, recv_buffer, send_counts_bytes, send_offsets_bytes,
        recv_signal_counts, remote_recv_offsets_bytes, signals, context.rank,
        context.size, direct_chunk_bytes, network_chunk_bytes, network_qps,
        network_qp_count, max_chunks, epoch++, options, context.stream);
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
  launch_alltoallv(
      send_buffer, recv_buffer, send_counts_bytes, send_offsets_bytes,
      recv_signal_counts, remote_recv_offsets_bytes, signals, context.rank,
      context.size, direct_chunk_bytes, network_chunk_bytes, network_qps,
      network_qp_count, max_chunks, epoch++, options, context.stream);
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
