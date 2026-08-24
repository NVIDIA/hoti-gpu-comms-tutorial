/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <nccl.h>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 31, 2)
#error "This exercise requires NCCL 2.31.2 or newer headers"
#endif

#include "../nccl_alltoallv_setup.hpp"

#include <cstdint>

namespace {

using alltoallv::DevicePlanEntry;
using alltoallv::value_type;
using alltoallv::nccl_setup::State;

struct Shard {
  std::uint64_t begin;
  std::uint64_t count;
};

__device__ Shard shard_for(std::uint64_t count, int shard, int shards) {
  const std::uint64_t quotient = count / static_cast<std::uint64_t>(shards);
  const std::uint64_t remainder = count % static_cast<std::uint64_t>(shards);
  const std::uint64_t shard_index = static_cast<std::uint64_t>(shard);
  const std::uint64_t before =
      shard_index < remainder ? shard_index : remainder;
  const std::uint64_t begin = shard_index * quotient + before;
  return Shard{begin, quotient + (shard_index < remainder ? 1 : 0)};
}

__global__ void nccl_gin_alltoallv_kernel(ncclDevComm dev_comm,
                                          ncclWindow_t send_window,
                                          ncclWindow_t recv_window,
                                          ncclWindow_t plan_window) {
#if __CUDA_ARCH__ >= 700
  const ncclTeam world = ncclTeamWorld(dev_comm);
  const DevicePlanEntry *entries =
      static_cast<const DevicePlanEntry *>(ncclGetLocalPointer(plan_window, 0));
  ncclGin gin(dev_comm, 0);
  const ncclGinSignal_t signal = static_cast<ncclGinSignal_t>(blockIdx.x);
  const std::uint64_t signal_before = gin.readSignal(signal);
  ncclGinBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), gin,
                                             ncclTeamTagWorld{}, blockIdx.x};

  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire,
               ncclGinFenceLevel::None);

  if (threadIdx.x == 0) {
    for (int peer = 0; peer < world.nRanks; ++peer) {
      const DevicePlanEntry entry = entries[peer];
      const Shard shard = shard_for(entry.send_count, blockIdx.x, gridDim.x);
      if (shard.count == 0)
        continue;
      gin.put(world, peer, recv_window,
              (entry.remote_recv_offset + shard.begin) * sizeof(value_type),
              send_window,
              (entry.send_offset + shard.begin) * sizeof(value_type),
              shard.count * sizeof(value_type), ncclGin_WeakSignalInc{signal});
    }
  }
  __syncthreads();

  std::uint64_t expected = 0;
  for (int source = 0; source < world.nRanks; ++source) {
    const Shard shard =
        shard_for(entries[source].recv_count, blockIdx.x, gridDim.x);
    expected += shard.count != 0;
  }
  gin.waitSignal(ncclCoopCta(), signal, signal_before + expected);
  gin.flush(ncclCoopCta());

  barrier.sync(ncclCoopCta(), cuda::memory_order_release,
               ncclGinFenceLevel::None);
#endif
}

void require_matching_launch(const alltoallv::Options &options, int rank) {
  int values[2] = {options.blocks, options.threads};
  int minima[2];
  int maxima[2];
  alltoallv::mpi_check(
      MPI_Allreduce(values, minima, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
      "MPI_Allreduce(launch minimum)");
  alltoallv::mpi_check(
      MPI_Allreduce(values, maxima, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
      "MPI_Allreduce(launch maximum)");
  if (minima[0] != maxima[0] || minima[1] != maxima[1]) {
    if (rank == 0)
      std::fprintf(stderr, "--blocks and --threads must match on every rank\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void launch_alltoallv(const State &state, const alltoallv::Options &options) {
  nccl_gin_alltoallv_kernel<<<options.blocks, options.threads, 0,
                              state.stream>>>(
      state.dev_comm, state.send_window, state.recv_window, state.plan_window);
  ALLTOALLV_CUDA_CHECK(cudaGetLastError());
}

} // namespace

int main(int argc, char **argv) {
  State state;
  alltoallv::Options options;
  alltoallv::Plan plan;
  if (alltoallv::nccl_setup::prepare(
          &state, &argc, &argv, alltoallv::nccl_setup::Backend::Gin, &options,
          &plan) != alltoallv::nccl_setup::SetupResult::Ready) {
    alltoallv::nccl_setup::finish(&state);
    return 0;
  }

  require_matching_launch(options, state.rank);
  alltoallv::print_plan(plan, options, "NCCL GIN AlltoAllV");

  launch_alltoallv(state, options);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  int errors = alltoallv::nccl_setup::copy_and_validate(&state, plan,
                                                        "NCCL GIN AlltoAllV");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  for (int iteration = 0; iteration < options.warmup; ++iteration)
    launch_alltoallv(state, options);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(timing)");

  cudaEvent_t start;
  cudaEvent_t stop;
  ALLTOALLV_CUDA_CHECK(cudaEventCreate(&start));
  ALLTOALLV_CUDA_CHECK(cudaEventCreate(&stop));
  ALLTOALLV_CUDA_CHECK(cudaEventRecord(start, state.stream));
  for (int iteration = 0; iteration < options.iterations; ++iteration)
    launch_alltoallv(state, options);
  ALLTOALLV_CUDA_CHECK(cudaEventRecord(stop, state.stream));
  ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  alltoallv::report_timing(plan, options, "NCCL GIN AlltoAllV", elapsed_ms);
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(start));
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stop));

  ALLTOALLV_CUDA_CHECK(
      cudaMemsetAsync(state.recv, 0xa5, state.recv_bytes, state.stream));
  launch_alltoallv(state, options);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  errors = alltoallv::nccl_setup::copy_and_validate(
      &state, plan, "NCCL GIN AlltoAllV reuse");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  alltoallv::nccl_setup::finish(&state);
  return 0;
}
