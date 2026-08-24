/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdio>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 31, 2)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::printf("SKIP: this exercise needs NCCL headers 2.31.2 or newer.\n");
  MPI_Finalize();
  return 0;
}

#else

#include "../nccl_alltoallv_setup.hpp"

namespace {

using alltoallv::DevicePlanEntry;
using alltoallv::value_type;
using alltoallv::nccl_setup::HybridPacketItem;

__device__ std::size_t align_packet_bytes(std::size_t value) {
  return (value + 15) & ~std::size_t{15};
}

__device__ void copy_values(const value_type *source, value_type *destination,
                            std::uint64_t count, int thread, int threads) {
  const std::uint64_t vector_count = count / 4;
  const uint4 *source4 = reinterpret_cast<const uint4 *>(source);
  uint4 *destination4 = reinterpret_cast<uint4 *>(destination);
  for (std::uint64_t i = thread; i < vector_count; i += threads)
    destination4[i] = source4[i];
  for (std::uint64_t i = vector_count * 4 + thread; i < count; i += threads)
    destination[i] = source[i];
}

__device__ std::size_t packet_bytes(const HybridPacketItem *items,
                                    int item_count,
                                    std::size_t header_bytes) {
  std::size_t bytes = header_bytes;
  for (int item = 0; item < item_count; ++item) {
    const std::size_t item_end =
        items[item].payload_offset + items[item].bytes;
    bytes = bytes > item_end ? bytes : item_end;
  }
  return bytes;
}

__global__ void pack_and_deliver_local(
    ncclDevComm dev_comm, ncclWindow_t send_window,
    ncclWindow_t recv_window, ncclWindow_t plan_window,
    ncclWindow_t outbox_window, std::size_t packet_capacity) {
#if __CUDA_ARCH__ >= 700
  ncclBarrierSession<ncclCoopCta> barrier{
      ncclCoopCta(), ncclTeamTagLsa(), dev_comm, blockIdx.x};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire,
               ncclGinFenceLevel::None);

  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);
  const int node_first_rank = dev_comm.rank - lsa.rank;
  const int thread = threadIdx.x + blockIdx.x * blockDim.x;
  const int threads = blockDim.x * gridDim.x;
  const std::size_t header_bytes =
      align_packet_bytes(lsa.nRanks * sizeof(HybridPacketItem));
  const DevicePlanEntry *plan = static_cast<const DevicePlanEntry *>(
      ncclGetLocalPointer(plan_window, 0));

  for (int destination_local = 0; destination_local < lsa.nRanks;
       ++destination_local) {
    const int destination = node_first_rank + destination_local;
    const DevicePlanEntry entry = plan[destination];
    const value_type *source = static_cast<const value_type *>(
        ncclGetLocalPointer(send_window,
                            entry.send_offset * sizeof(value_type)));
    value_type *target = static_cast<value_type *>(ncclGetLsaPointer(
        recv_window, entry.remote_recv_offset * sizeof(value_type),
        destination_local));
    copy_values(source, target, entry.send_count, thread, threads);
  }

  for (int destination_node = 0; destination_node < rail.nRanks;
       ++destination_node) {
    if (destination_node == rail.rank)
      continue;

    char *packet = static_cast<char *>(ncclGetLocalPointer(
        outbox_window, destination_node * packet_capacity));
    HybridPacketItem *items = reinterpret_cast<HybridPacketItem *>(packet);
    const int ingress_rank =
        ncclTeamRankToWorld(dev_comm, rail, destination_node);
    const int destination_first_rank = ingress_rank - lsa.rank;

    std::size_t payload_offset = header_bytes;
    for (int destination_local = 0; destination_local < lsa.nRanks;
         ++destination_local) {
      const int destination = destination_first_rank + destination_local;
      const DevicePlanEntry entry = plan[destination];
      const std::size_t bytes = entry.send_count * sizeof(value_type);
      if (thread == 0) {
        items[destination_local] = HybridPacketItem{
            bytes, entry.remote_recv_offset * sizeof(value_type),
            payload_offset, 0};
      }

      const value_type *source = static_cast<const value_type *>(
          ncclGetLocalPointer(send_window,
                              entry.send_offset * sizeof(value_type)));
      value_type *payload =
          reinterpret_cast<value_type *>(packet + payload_offset);
      copy_values(source, payload, entry.send_count, thread, threads);
      payload_offset = align_packet_bytes(payload_offset + bytes);
    }
  }

  barrier.sync(ncclCoopCta(), cuda::memory_order_release,
               ncclGinFenceLevel::None);
#endif
}

__global__ void exchange_rails_and_scatter(
    ncclDevComm dev_comm, ncclWindow_t recv_window,
    ncclWindow_t outbox_window, ncclWindow_t inbox_window,
    std::size_t packet_capacity, std::uint64_t epoch) {
#if __CUDA_ARCH__ >= 700
  ncclGin gin{dev_comm, 0};
  ncclBarrierSession<ncclCoopCta> barrier{
      ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire,
               ncclGinFenceLevel::None);

  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);
  const std::size_t header_bytes =
      align_packet_bytes(lsa.nRanks * sizeof(HybridPacketItem));

  if (blockIdx.x == 0) {
    for (int destination_node = threadIdx.x;
         destination_node < rail.nRanks;
         destination_node += blockDim.x) {
      if (destination_node == rail.rank)
        continue;
      const std::size_t source_offset =
          destination_node * packet_capacity;
      const HybridPacketItem *items =
          reinterpret_cast<const HybridPacketItem *>(
              static_cast<const char *>(
                  ncclGetLocalPointer(outbox_window, source_offset)));
      gin.put(rail, destination_node, inbox_window,
              rail.rank * packet_capacity, outbox_window, source_offset,
              packet_bytes(items, lsa.nRanks, header_bytes),
              ncclGin_WeakSignalInc{
                  static_cast<unsigned int>(rail.rank)});
    }
  }
  if (blockIdx.x == 0) {
    __syncthreads();
  }

  for (int source_node = 0; source_node < rail.nRanks; ++source_node) {
    if (source_node != rail.rank) {
      gin.waitSignal(ncclCoopCta(), source_node, epoch);
    }
  }

  if (blockIdx.x == 0)
    gin.flush(ncclCoopCta());

  const int thread = threadIdx.x + blockIdx.x * blockDim.x;
  const int threads = blockDim.x * gridDim.x;
  for (int source_node = 0; source_node < rail.nRanks; ++source_node) {
    if (source_node == rail.rank)
      continue;
    const char *packet = static_cast<const char *>(ncclGetLocalPointer(
        inbox_window, source_node * packet_capacity));
    const HybridPacketItem *items =
        reinterpret_cast<const HybridPacketItem *>(packet);
    for (int destination_local = 0; destination_local < lsa.nRanks;
         ++destination_local) {
      const HybridPacketItem item = items[destination_local];
      const value_type *payload =
          reinterpret_cast<const value_type *>(packet + item.payload_offset);
      value_type *destination = static_cast<value_type *>(ncclGetLsaPointer(
          recv_window, item.recv_offset_bytes, destination_local));
      copy_values(payload, destination, item.bytes / sizeof(value_type),
                  thread, threads);
    }
  }

  barrier.sync(ncclCoopCta(), cuda::memory_order_release,
               ncclGinFenceLevel::None);
#endif
}

void launch(const alltoallv::nccl_setup::State &state,
            const alltoallv::Options &options, std::uint64_t epoch) {
  pack_and_deliver_local<<<options.blocks, options.threads, 0, state.stream>>>(
      state.dev_comm, state.send_window, state.recv_window, state.plan_window,
      state.outbox_window, state.packet_capacity);
  ALLTOALLV_CUDA_CHECK(cudaGetLastError());
  exchange_rails_and_scatter<<<options.blocks, options.threads, 0,
                               state.stream>>>(
      state.dev_comm, state.recv_window, state.outbox_window,
      state.inbox_window, state.packet_capacity, epoch);
  ALLTOALLV_CUDA_CHECK(cudaGetLastError());
}

} // namespace

int main(int argc, char **argv) {
  alltoallv::nccl_setup::State state;
  alltoallv::Options options;
  alltoallv::Plan plan;
  if (alltoallv::nccl_setup::prepare(
          &state, &argc, &argv,
          alltoallv::nccl_setup::Backend::HybridRail, &options,
          &plan) != alltoallv::nccl_setup::SetupResult::Ready) {
    alltoallv::nccl_setup::finish(&state);
    return 0;
  }

  alltoallv::print_plan(plan, options, "NCCL LSA + railed GIN AlltoAllV");
  std::uint64_t epoch = 1;
  launch(state, options, epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  int errors = alltoallv::nccl_setup::copy_and_validate(
      &state, plan, "NCCL LSA + railed GIN AlltoAllV");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  for (int i = 0; i < options.warmup; ++i)
    launch(state, options, ++epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD),
                       "MPI_Barrier(benchmark)");

  cudaEvent_t start;
  cudaEvent_t stop;
  ALLTOALLV_CUDA_CHECK(cudaEventCreate(&start));
  ALLTOALLV_CUDA_CHECK(cudaEventCreate(&stop));
  ALLTOALLV_CUDA_CHECK(cudaEventRecord(start, state.stream));
  for (int i = 0; i < options.iterations; ++i)
    launch(state, options, ++epoch);
  ALLTOALLV_CUDA_CHECK(cudaEventRecord(stop, state.stream));
  ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  alltoallv::report_timing(plan, options,
                           "NCCL LSA + railed GIN AlltoAllV", elapsed_ms);
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stop));
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(start));

  ALLTOALLV_CUDA_CHECK(
      cudaMemsetAsync(state.recv, 0xa5, state.recv_bytes, state.stream));
  launch(state, options, ++epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  errors = alltoallv::nccl_setup::copy_and_validate(
      &state, plan, "NCCL LSA + railed GIN AlltoAllV reuse");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  alltoallv::nccl_setup::finish(&state);
  return 0;
}

#endif
