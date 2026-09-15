/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <nccl.h>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 31, 2)
#error "This exercise requires NCCL 2.31.2 or newer headers"
#endif

#include "../nccl_alltoallv_setup.hpp"

#include <algorithm>
#include <cstdint>

namespace {

using alltoallv::DevicePlanEntry;
using alltoallv::value_type;
using alltoallv::nccl_setup::State;

struct Shard {
  std::uint64_t begin;
  std::uint64_t count;
};

struct RouteWork {
  int peer;
  int source;
  int shard;
  int shards;
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

__device__ RouteWork route_work(ncclTeam world) {
  const int route_count = max(world.nRanks - 1, 1);
  const int rounds = gridDim.x / route_count;
  const int tail = gridDim.x % route_count;
  const int round = blockIdx.x / route_count;
  const int slot = blockIdx.x % route_count;
  const int route = (slot + round) % route_count;
  const int tail_slot =
      (route - (rounds % route_count) + route_count) % route_count;
  return RouteWork{(world.rank + 1 + route) % world.nRanks,
                   (world.rank - 1 - route + world.nRanks) % world.nRanks,
                   round, rounds + (tail_slot < tail ? 1 : 0)};
}

__global__ void nccl_gin_alltoallv_kernel(ncclDevComm dev_comm,
                                          ncclWindow_t send_window,
                                          ncclWindow_t recv_window,
                                          ncclWindow_t plan_window) {
#if __CUDA_ARCH__ >= 700
  const ncclTeam world = ncclTeamWorld(dev_comm);
  const DevicePlanEntry *entries =
      static_cast<const DevicePlanEntry *>(ncclGetLocalPointer(plan_window, 0));
  const RouteWork work = route_work(world);
  const int context_count =
      min(static_cast<int>(gridDim.x),
          static_cast<int>(dev_comm.ginContextCount));
  const int context = static_cast<int>(blockIdx.x) % context_count;
  ncclGin gin(dev_comm, context);
  const ncclGinSignal_t signal = static_cast<ncclGinSignal_t>(blockIdx.x);
  const std::uint64_t signal_before = gin.readSignal(signal);
  ncclGinBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), gin,
                                             ncclTeamTagWorld{}, blockIdx.x};

  // TODO: Enter the world GIN barrier with acquire ordering and no put/get
  // fence. Every receiver must read signal_before before any sender proceeds.

  if (threadIdx.x == 0) {
    const DevicePlanEntry entry = entries[work.peer];
    const Shard shard =
        shard_for(entry.send_count, work.shard, work.shards);
    if (world.nRanks > 1 && shard.count != 0) {
      // TODO: Put this shard from send_window to the peer's recv_window using
      // byte offsets and attach ncclGin_WeakSignalInc for this CTA's signal.
      (void)entry;
    }
  }

  const DevicePlanEntry self = entries[world.rank];
  const Shard self_shard =
      shard_for(self.send_count, blockIdx.x, gridDim.x);
  const value_type *send = static_cast<const value_type *>(
      ncclGetLocalPointer(send_window, 0));
  value_type *recv =
      static_cast<value_type *>(ncclGetLocalPointer(recv_window, 0));
  for (std::uint64_t index = threadIdx.x; index < self_shard.count;
       index += blockDim.x) {
    recv[self.recv_offset + self_shard.begin + index] =
        send[self.send_offset + self_shard.begin + index];
  }
  __syncthreads();

  const Shard incoming =
      shard_for(entries[work.source].recv_count, work.shard, work.shards);
  const std::uint64_t expected =
      world.nRanks > 1 && incoming.count != 0;

  // TODO: Cooperatively wait for signal_before + expected, then flush the
  // CTA's outgoing puts so their source ranges are safe to reuse.

  // TODO: Close the world GIN barrier with release ordering and no put/get
  // fence.
  (void)send_window;
  (void)recv_window;
  (void)signal_before;
  (void)expected;
  (void)barrier;
#endif
}

void require_matching_launch(const alltoallv::Options &options, int rank,
                             int size) {
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
  if (options.blocks < std::max(size - 1, 1)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "--blocks must be at least ranks minus one\n");
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

  require_matching_launch(options, state.rank, state.size);
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

  alltoallv::nccl_setup::clear_recv_for_reuse(&state);
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
