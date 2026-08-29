/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdio>
#include <vector>

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

__device__ int route_shard_block(
    int source_domain, int destination_domain, int destination_local,
    int domain_count, int local_count, int shard, int route_shards,
    int blocks) {
  const int domain_delta =
      (destination_domain - source_domain + domain_count) % domain_count;
  const std::uint64_t route =
      static_cast<std::uint64_t>(domain_delta - 1) * local_count +
      destination_local;
  return static_cast<int>((route * route_shards + shard) % blocks);
}

__device__ void shard_slice(std::uint64_t count, int shard,
                            int route_shards, std::uint64_t *offset,
                            std::uint64_t *slice_count) {
  const std::uint64_t vector_count = count / 4;
  const std::uint64_t vector_begin =
      vector_count * shard / route_shards;
  const std::uint64_t vector_end =
      vector_count * (shard + 1) / route_shards;
  *offset = vector_begin * 4;
  const std::uint64_t end =
      shard + 1 == route_shards ? count : vector_end * 4;
  *slice_count = end - *offset;
}

__device__ void scatter_assigned_shards(
    ncclDevComm dev_comm, ncclWindow_t recv_window,
    ncclWindow_t plan_window, ncclWindow_t inbox_window,
    std::size_t slot_bytes, std::size_t domain_bytes,
    int route_shards) {
  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);
  const int task_count = rail.nRanks * lsa.nRanks;
  for (int task = 0; task < task_count; ++task) {
    const int source_domain = task / lsa.nRanks;
    const int destination_local = task % lsa.nRanks;
    if (source_domain == rail.rank)
      continue;
    const int source = ncclTeamRankToWorld(dev_comm, rail, source_domain);
    const DevicePlanEntry *destination_plan =
        static_cast<const DevicePlanEntry *>(ncclGetLsaPointer(
            plan_window, 0, destination_local));
    const DevicePlanEntry entry = destination_plan[source];
    for (int shard = 0; shard < route_shards; ++shard) {
      if (route_shard_block(
              source_domain, rail.rank, destination_local,
              rail.nRanks, lsa.nRanks, shard, route_shards,
              gridDim.x) != blockIdx.x)
        continue;
      std::uint64_t offset = 0;
      std::uint64_t count = 0;
      shard_slice(entry.recv_count, shard, route_shards, &offset, &count);
      if (count == 0)
        continue;
      const value_type *source_values = static_cast<const value_type *>(
          ncclGetLocalPointer(
              inbox_window,
              source_domain * domain_bytes + destination_local * slot_bytes +
                  offset * sizeof(value_type)));
      value_type *destination = static_cast<value_type *>(ncclGetLsaPointer(
          recv_window,
          (entry.recv_offset + offset) * sizeof(value_type),
          destination_local));
      copy_values(source_values, destination, count,
                  threadIdx.x, blockDim.x);
    }
  }
}

__global__ void send_and_deliver_local(
    ncclDevComm dev_comm, ncclWindow_t send_window,
    ncclWindow_t recv_window, ncclWindow_t plan_window,
    ncclWindow_t inbox_window, std::size_t slot_bytes,
    std::size_t domain_bytes, int route_shards, int network_issuers) {
#if __CUDA_ARCH__ >= 700
  const int context_count =
      min(static_cast<int>(gridDim.x),
          static_cast<int>(dev_comm.ginContextCount));
  const int context = static_cast<int>(blockIdx.x) % context_count;
  const ncclGinSignal_t signal =
      static_cast<ncclGinSignal_t>(blockIdx.x);
  ncclGin gin{dev_comm, context};
  ncclBarrierSession<ncclCoopCta> barrier{
      ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire,
               ncclGinFenceLevel::None);

  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);
  const int domain_first_rank = dev_comm.rank - lsa.rank;
  const DevicePlanEntry *plan = static_cast<const DevicePlanEntry *>(
      ncclGetLocalPointer(plan_window, 0));

  for (int step = 0; step < lsa.nRanks; ++step) {
    const int destination_local =
        (lsa.rank + step + blockIdx.x) % lsa.nRanks;
    const int destination = domain_first_rank + destination_local;
    const DevicePlanEntry entry = plan[destination];
    std::uint64_t offset = 0;
    std::uint64_t count = 0;
    shard_slice(entry.send_count, blockIdx.x, gridDim.x, &offset, &count);
    const value_type *source = static_cast<const value_type *>(
        ncclGetLocalPointer(send_window,
                            (entry.send_offset + offset) * sizeof(value_type)));
    value_type *target = static_cast<value_type *>(ncclGetLsaPointer(
        recv_window, (entry.remote_recv_offset + offset) * sizeof(value_type),
        destination_local));

    // TODO: Copy this CTA's local shard from source to target.
    (void)source;
    (void)target;
    (void)count;
  }

  if (threadIdx.x < network_issuers) {
    const int task_count = rail.nRanks * lsa.nRanks;
    for (int task = 0; task < task_count; ++task) {
      const int destination_domain = task / lsa.nRanks;
      const int destination_local = task % lsa.nRanks;
      if (destination_domain == rail.rank)
        continue;
      const int ingress_rank =
          ncclTeamRankToWorld(dev_comm, rail, destination_domain);
      const int destination = ingress_rank - lsa.rank + destination_local;
      const DevicePlanEntry entry = plan[destination];
      for (int shard = 0; shard < route_shards; ++shard) {
        if (route_shard_block(
                rail.rank, destination_domain, destination_local,
                rail.nRanks, lsa.nRanks, shard, route_shards,
                gridDim.x) != blockIdx.x)
          continue;
        std::uint64_t offset = 0;
        std::uint64_t count = 0;
        shard_slice(entry.send_count, shard, route_shards, &offset, &count);
        if (count == 0)
          continue;
        std::uint64_t issuer_offset = 0;
        std::uint64_t issuer_count = 0;
        shard_slice(count, threadIdx.x, network_issuers, &issuer_offset,
                    &issuer_count);
        if (issuer_count == 0)
          continue;
        const std::size_t inbox_offset =
            rail.rank * domain_bytes + destination_local * slot_bytes +
            (offset + issuer_offset) * sizeof(value_type);

        // TODO: Put this issuer slice into inbox_offset on destination_domain.
        // Attach ncclGin_WeakSignalInc for this CTA's signal index.
        (void)inbox_offset;
        (void)issuer_count;
      }
    }
  }
  __syncthreads();
#endif
}

__global__ void wait_and_scatter(
    ncclDevComm dev_comm, ncclWindow_t recv_window,
    ncclWindow_t plan_window, ncclWindow_t inbox_window,
    std::size_t slot_bytes, std::size_t domain_bytes,
    int route_shards, int network_issuers, std::uint64_t epoch) {
#if __CUDA_ARCH__ >= 700
  const int context_count =
      min(static_cast<int>(gridDim.x),
          static_cast<int>(dev_comm.ginContextCount));
  const int context = static_cast<int>(blockIdx.x) % context_count;
  const ncclGinSignal_t signal =
      static_cast<ncclGinSignal_t>(blockIdx.x);
  ncclGin gin{dev_comm, context};
  ncclBarrierSession<ncclCoopCta> barrier{
      ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);

  const int task_count = rail.nRanks * lsa.nRanks;
  __shared__ std::uint64_t expected;
  if (threadIdx.x == 0) {
    expected = 0;
    for (int task = 0; task < task_count; ++task) {
      const int source_domain = task / lsa.nRanks;
      const int destination_local = task % lsa.nRanks;
      if (source_domain == rail.rank)
        continue;
      const int source = ncclTeamRankToWorld(dev_comm, rail, source_domain);
      const DevicePlanEntry *destination_plan =
          static_cast<const DevicePlanEntry *>(ncclGetLsaPointer(
              plan_window, 0, destination_local));
      const DevicePlanEntry entry = destination_plan[source];
      for (int shard = 0; shard < route_shards; ++shard) {
        if (route_shard_block(
                source_domain, rail.rank, destination_local,
                rail.nRanks, lsa.nRanks, shard, route_shards,
                gridDim.x) != blockIdx.x)
          continue;
        std::uint64_t offset = 0;
        std::uint64_t count = 0;
        shard_slice(entry.recv_count, shard, route_shards, &offset, &count);
        for (int issuer = 0; issuer < network_issuers; ++issuer) {
          std::uint64_t issuer_offset = 0;
          std::uint64_t issuer_count = 0;
          shard_slice(count, issuer, network_issuers, &issuer_offset,
                      &issuer_count);
          expected += issuer_count != 0;
        }
      }
    }
  }
  __syncthreads();

  // TODO: If expected is nonzero, wait for this CTA's signal to reach
  // epoch * expected. Then call scatter_assigned_shards for this CTA.
  (void)expected;
  (void)epoch;

  gin.flush(ncclCoopCta());
  barrier.sync(ncclCoopCta(), cuda::memory_order_acq_rel,
               ncclGinFenceLevel::None);
#endif
}

void launch_send_and_deliver_local(const alltoallv::nccl_setup::State &state,
                                   const alltoallv::Options &options,
                                   int route_shards) {
  send_and_deliver_local<<<options.blocks, options.threads, 0,
                           state.stream>>>(
      state.dev_comm, state.send_window, state.recv_window,
      state.plan_window, state.inbox_window, state.hybrid_slot_bytes,
      state.hybrid_domain_bytes, route_shards, options.network_issuers);
  ALLTOALLV_CUDA_CHECK(cudaGetLastError());
}

void launch_wait_and_scatter(const alltoallv::nccl_setup::State &state,
                             const alltoallv::Options &options,
                             int route_shards, std::uint64_t epoch) {
  wait_and_scatter<<<options.blocks, options.threads, 0, state.stream>>>(
      state.dev_comm, state.recv_window, state.plan_window,
      state.inbox_window, state.hybrid_slot_bytes,
      state.hybrid_domain_bytes, route_shards, options.network_issuers,
      epoch);
  ALLTOALLV_CUDA_CHECK(cudaGetLastError());
}

void launch(const alltoallv::nccl_setup::State &state,
            const alltoallv::Options &options, int route_shards,
            std::uint64_t epoch) {
  launch_send_and_deliver_local(state, options, route_shards);
  launch_wait_and_scatter(state, options, route_shards, epoch);
}

struct PhaseTimes {
  float send_and_deliver_ms = 0.0f;
  float wait_and_scatter_ms = 0.0f;
};

PhaseTimes profile_iterations(const alltoallv::nccl_setup::State &state,
                              const alltoallv::Options &options,
                              int route_shards, std::uint64_t *epoch) {
  std::vector<cudaEvent_t> starts(options.iterations);
  std::vector<cudaEvent_t> handoffs(options.iterations);
  std::vector<cudaEvent_t> stops(options.iterations);
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&starts[iteration]));
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&handoffs[iteration]));
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&stops[iteration]));
  }
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(starts[iteration], state.stream));
    launch_send_and_deliver_local(state, options, route_shards);
    ALLTOALLV_CUDA_CHECK(
        cudaEventRecord(handoffs[iteration], state.stream));
    launch_wait_and_scatter(state, options, route_shards, ++*epoch);
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(stops[iteration], state.stream));
  }
  ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stops.back()));

  PhaseTimes result;
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    float elapsed_ms = 0.0f;
    ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms, starts[iteration], handoffs[iteration]));
    result.send_and_deliver_ms += elapsed_ms;
    ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms, handoffs[iteration], stops[iteration]));
    result.wait_and_scatter_ms += elapsed_ms;
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stops[iteration]));
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(handoffs[iteration]));
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(starts[iteration]));
  }
  return result;
}

int choose_route_shards(const alltoallv::nccl_setup::State &state,
                        const alltoallv::Options &options,
                        const alltoallv::Plan &plan) {
  constexpr std::uint64_t kTargetBytes = 1ull << 20;
  const int remote_routes =
      (state.rail_team.nRanks - 1) * state.lsa_team.nRanks;
  std::uint64_t local_max_pair = 0;
  for (int peer = 0; peer < plan.size; ++peer) {
    if (plan.domain_roots[peer] != plan.domain_roots[plan.rank])
      local_max_pair = std::max(local_max_pair, plan.send_counts[peer]);
  }
  std::uint64_t max_pair_count = 0;
  alltoallv::mpi_check(
      MPI_Allreduce(&local_max_pair, &max_pair_count, 1, MPI_UINT64_T,
                    MPI_MAX, MPI_COMM_WORLD),
      "MPI_Allreduce(max network pair count)");
  const bool heavy_route =
      max_pair_count != 0 &&
      max_pair_count >= (plan.max_network_send_elements + 1) / 2;
  const int balanced_capacity =
      std::max(1, options.blocks / remote_routes);
  const int route_capacity =
      heavy_route ? options.blocks : balanced_capacity;
  const std::uint64_t max_pair_bytes = max_pair_count * sizeof(value_type);
  const std::uint64_t size_shards = std::max<std::uint64_t>(
      1, (max_pair_bytes + kTargetBytes - 1) / kTargetBytes);
  return static_cast<int>(std::min<std::uint64_t>(route_capacity,
                                                  size_shards));
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
  const int route_shards = choose_route_shards(state, options, plan);
  if (state.rank == 0) {
    std::printf(
        "NCCL hybrid routing: %d shard(s) per remote route, %d issuer(s) "
        "per shard\n",
        route_shards, options.network_issuers);
  }
  std::uint64_t epoch = 1;
  launch(state, options, route_shards, epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  int errors = alltoallv::nccl_setup::copy_and_validate(
      &state, plan, "NCCL LSA + railed GIN AlltoAllV");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  for (int i = 0; i < options.warmup; ++i)
    launch(state, options, route_shards, ++epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD),
                       "MPI_Barrier(benchmark)");

  float elapsed_ms = 0.0f;
  PhaseTimes phases;
  if (options.profile_phases) {
    phases = profile_iterations(state, options, route_shards, &epoch);
    elapsed_ms = phases.send_and_deliver_ms + phases.wait_and_scatter_ms;
  } else {
    cudaEvent_t start;
    cudaEvent_t stop;
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&start));
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&stop));
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(start, state.stream));
    for (int i = 0; i < options.iterations; ++i)
      launch(state, options, route_shards, ++epoch);
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(stop, state.stream));
    ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stop));
    ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stop));
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(start));
  }
  alltoallv::report_timing(plan, options,
                           "NCCL LSA + railed GIN AlltoAllV", elapsed_ms);
  if (options.profile_phases) {
    alltoallv::report_phase_timing(
        options, "NCCL LSA + railed GIN AlltoAllV",
        phases.send_and_deliver_ms, phases.wait_and_scatter_ms);
  }

  ALLTOALLV_CUDA_CHECK(
      cudaMemsetAsync(state.recv, 0xa5, state.recv_bytes, state.stream));
  launch(state, options, route_shards, ++epoch);
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
