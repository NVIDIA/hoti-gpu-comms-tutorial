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
  const int thread = threadIdx.x + blockIdx.x * blockDim.x;
  const int threads = blockDim.x * gridDim.x;
  const DevicePlanEntry *plan = static_cast<const DevicePlanEntry *>(
      ncclGetLocalPointer(plan_window, 0));
  const std::size_t header_bytes =
      align_packet_bytes(lsa.nRanks * sizeof(HybridPacketItem));

  /* TODO:
   * 1. Copy messages for this LSA domain directly into each local peer's
   *    receive window, rotating the first destination by lsa.rank.
   * 2. For every remote rail rank, write one HybridPacketItem per destination
   *    LSA rank and pack the corresponding payload into that outbox slot.
   *    Rotate the first packed destination by lsa.rank as well.
   */
  (void)rail;
  (void)thread;
  (void)threads;
  (void)plan;
  (void)header_bytes;
  (void)send_window;
  (void)recv_window;
  (void)outbox_window;
  (void)packet_capacity;

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

  /* TODO:
   * 1. Send each packed outbox to the matching GPU on another node with
   *    gin.put(rail, ...) and a weak signal indexed by the source rail rank.
   * 2. Wait until every remote source signal reaches epoch, then flush the
   *    issuing context.
   * 3. Read each received packet and scatter its items through LSA pointers,
   *    rotating the first destination by lsa.rank.
   */
  (void)rail;
  (void)header_bytes;
  (void)recv_window;
  (void)outbox_window;
  (void)inbox_window;
  (void)packet_capacity;
  (void)epoch;

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
