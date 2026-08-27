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

__device__ void copy_segment(const value_type *source, value_type *destination,
                             std::uint64_t count) {
  constexpr std::uint64_t kVectorElements = sizeof(uint4) / sizeof(value_type);
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t global_stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::uint64_t vector_count = count / kVectorElements;

  const uint4 *source_vectors = reinterpret_cast<const uint4 *>(source);
  uint4 *destination_vectors = reinterpret_cast<uint4 *>(destination);
  for (std::uint64_t vector = global_thread; vector < vector_count;
       vector += global_stride)
    destination_vectors[vector] = source_vectors[vector];

  const std::uint64_t tail = vector_count * kVectorElements;
  for (std::uint64_t element = tail + global_thread; element < count;
       element += global_stride)
    destination[element] = source[element];
}

__global__ void nccl_lsa_alltoallv_kernel(ncclDevComm dev_comm,
                                          ncclWindow_t send_window,
                                          ncclWindow_t recv_window,
                                          ncclWindow_t plan_window) {
#if __CUDA_ARCH__ >= 700
  ncclLsaBarrierSession<ncclCoopCta> barrier{
      ncclCoopCta(), dev_comm, ncclTeamTagLsa{}, blockIdx.x, false};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const ncclTeam lsa_team = ncclTeamLsa(dev_comm);
  const DevicePlanEntry *entries =
      static_cast<const DevicePlanEntry *>(ncclGetLocalPointer(plan_window, 0));
  for (int step = 0; step < lsa_team.nRanks; ++step) {
    const int lsa_peer =
        (lsa_team.rank + step + static_cast<int>(blockIdx.x)) %
        lsa_team.nRanks;
    const int world_peer = ncclTeamRankToWorld(dev_comm, lsa_team, lsa_peer);
    const DevicePlanEntry entry = entries[world_peer];
    const value_type *source =
        static_cast<const value_type *>(ncclGetLocalPointer(
            send_window, entry.send_offset * sizeof(value_type)));
    value_type *destination = static_cast<value_type *>(ncclGetLsaPointer(
        recv_window, entry.remote_recv_offset * sizeof(value_type), lsa_peer));
    copy_segment(source, destination, entry.send_count);
  }

  barrier.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
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
  nccl_lsa_alltoallv_kernel<<<options.blocks, options.threads, 0,
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
          &state, &argc, &argv, alltoallv::nccl_setup::Backend::Lsa, &options,
          &plan) != alltoallv::nccl_setup::SetupResult::Ready) {
    alltoallv::nccl_setup::finish(&state);
    return 0;
  }

  require_matching_launch(options, state.rank);
  alltoallv::print_plan(plan, options, "NCCL LSA AlltoAllV");

  launch_alltoallv(state, options);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  int errors = alltoallv::nccl_setup::copy_and_validate(&state, plan,
                                                        "NCCL LSA AlltoAllV");
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
  alltoallv::report_timing(plan, options, "NCCL LSA AlltoAllV", elapsed_ms);
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(start));
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stop));

  ALLTOALLV_CUDA_CHECK(
      cudaMemsetAsync(state.recv, 0xa5, state.recv_bytes, state.stream));
  launch_alltoallv(state, options);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  errors = alltoallv::nccl_setup::copy_and_validate(
      &state, plan, "NCCL LSA AlltoAllV reuse");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  alltoallv::nccl_setup::finish(&state);
  return 0;
}
