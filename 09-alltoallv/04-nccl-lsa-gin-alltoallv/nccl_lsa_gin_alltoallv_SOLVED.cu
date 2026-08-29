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
    copy_values(source, target, count, threadIdx.x, blockDim.x);
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
        gin.put(rail, destination_domain, inbox_window, inbox_offset,
                send_window,
                (entry.send_offset + offset + issuer_offset) *
                    sizeof(value_type),
                issuer_count * sizeof(value_type),
                ncclGin_WeakSignalInc{signal});
      }
    }
  }
  __syncthreads();
#endif
}

struct CompletionBlockTrace {
  std::uint64_t wait_begin;
  std::uint64_t wait_end;
  std::uint64_t finish_end;
};

struct CompletionWarpTrace {
  std::uint64_t scatter_begin;
  std::uint64_t scatter_end;
};

constexpr int kWarpThreads = 32;

template <bool kProfileCycles>
__global__ void wait_and_scatter(
    ncclDevComm dev_comm, ncclWindow_t recv_window,
    ncclWindow_t plan_window, ncclWindow_t inbox_window,
    std::size_t slot_bytes, std::size_t domain_bytes,
    int route_shards, int network_issuers, std::uint64_t epoch,
    CompletionBlockTrace *block_traces, CompletionWarpTrace *warp_traces) {
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
  if constexpr (kProfileCycles) {
    if (threadIdx.x == 0)
      block_traces[blockIdx.x].wait_begin = clock64();
  }
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
  if (expected != 0) {
    gin.waitSignal(ncclCoopCta(), signal, epoch * expected);
  }
  if constexpr (kProfileCycles) {
    if (threadIdx.x == 0)
      block_traces[blockIdx.x].wait_end = clock64();
    __syncwarp();
    if (threadIdx.x % kWarpThreads == 0) {
      const int warp = threadIdx.x / kWarpThreads;
      warp_traces[blockIdx.x * (blockDim.x / kWarpThreads) + warp]
          .scatter_begin = clock64();
    }
  }
  scatter_assigned_shards(
      dev_comm, recv_window, plan_window, inbox_window, slot_bytes,
      domain_bytes, route_shards);
  if constexpr (kProfileCycles) {
    __syncwarp();
    if (threadIdx.x % kWarpThreads == 0) {
      const int warp = threadIdx.x / kWarpThreads;
      warp_traces[blockIdx.x * (blockDim.x / kWarpThreads) + warp]
          .scatter_end = clock64();
    }
  }
  gin.flush(ncclCoopCta());

  barrier.sync(ncclCoopCta(), cuda::memory_order_acq_rel,
               ncclGinFenceLevel::None);
  if constexpr (kProfileCycles) {
    if (threadIdx.x == 0)
      block_traces[blockIdx.x].finish_end = clock64();
  }
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
  wait_and_scatter<false><<<options.blocks, options.threads, 0, state.stream>>>(
      state.dev_comm, state.recv_window, state.plan_window,
      state.inbox_window, state.hybrid_slot_bytes,
      state.hybrid_domain_bytes, route_shards, options.network_issuers,
      epoch, nullptr, nullptr);
  ALLTOALLV_CUDA_CHECK(cudaGetLastError());
}

void launch_profiled_wait_and_scatter(
    const alltoallv::nccl_setup::State &state,
    const alltoallv::Options &options, int route_shards,
    std::uint64_t epoch, CompletionBlockTrace *block_traces,
    CompletionWarpTrace *warp_traces) {
  wait_and_scatter<true><<<options.blocks, options.threads, 0, state.stream>>>(
      state.dev_comm, state.recv_window, state.plan_window,
      state.inbox_window, state.hybrid_slot_bytes,
      state.hybrid_domain_bytes, route_shards, options.network_issuers,
      epoch, block_traces, warp_traces);
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
  float plan_and_wait_cta_ms = 0.0f;
  float scatter_cta_ms = 0.0f;
  float flush_and_barrier_cta_ms = 0.0f;
};

PhaseTimes profile_iterations(const alltoallv::nccl_setup::State &state,
                              const alltoallv::Options &options,
                              int route_shards, std::uint64_t *epoch) {
  const std::size_t block_traces_per_iteration = options.blocks;
  const std::size_t warp_traces_per_iteration =
      block_traces_per_iteration * options.threads / kWarpThreads;
  CompletionBlockTrace *device_block_traces = nullptr;
  CompletionWarpTrace *device_warp_traces = nullptr;
  std::vector<CompletionBlockTrace> host_block_traces(
      static_cast<std::size_t>(options.iterations) *
      block_traces_per_iteration);
  std::vector<CompletionWarpTrace> host_warp_traces(
      static_cast<std::size_t>(options.iterations) *
      warp_traces_per_iteration);
  ALLTOALLV_CUDA_CHECK(cudaMalloc(
      &device_block_traces,
      host_block_traces.size() * sizeof(*device_block_traces)));
  ALLTOALLV_CUDA_CHECK(cudaMalloc(
      &device_warp_traces,
      host_warp_traces.size() * sizeof(*device_warp_traces)));

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
    launch_profiled_wait_and_scatter(
        state, options, route_shards, ++*epoch,
        device_block_traces + iteration * block_traces_per_iteration,
        device_warp_traces + iteration * warp_traces_per_iteration);
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(stops[iteration], state.stream));
  }
  ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stops.back()));
  ALLTOALLV_CUDA_CHECK(cudaMemcpy(
      host_block_traces.data(), device_block_traces,
      host_block_traces.size() * sizeof(*device_block_traces),
      cudaMemcpyDeviceToHost));
  ALLTOALLV_CUDA_CHECK(cudaMemcpy(
      host_warp_traces.data(), device_warp_traces,
      host_warp_traces.size() * sizeof(*device_warp_traces),
      cudaMemcpyDeviceToHost));
  ALLTOALLV_CUDA_CHECK(cudaFree(device_warp_traces));
  ALLTOALLV_CUDA_CHECK(cudaFree(device_block_traces));

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

  std::uint64_t plan_and_wait_cycles = 0;
  std::uint64_t scatter_cycles = 0;
  std::uint64_t flush_and_barrier_cycles = 0;
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    const CompletionBlockTrace *block_values =
        host_block_traces.data() + iteration * block_traces_per_iteration;
    const CompletionWarpTrace *warp_values =
        host_warp_traces.data() + iteration * warp_traces_per_iteration;
    int slowest_block = 0;
    std::uint64_t slowest_total = 0;
    for (int block = 0; block < options.blocks; ++block) {
      const CompletionBlockTrace &trace = block_values[block];
      const std::uint64_t total = trace.finish_end - trace.wait_begin;
      if (total > slowest_total) {
        slowest_total = total;
        slowest_block = block;
      }
    }
    const CompletionBlockTrace &trace = block_values[slowest_block];
    plan_and_wait_cycles += trace.wait_end - trace.wait_begin;

    std::uint64_t longest_scatter = 0;
    std::uint64_t latest_scatter_end = trace.wait_end;
    for (int warp = 0; warp < options.threads / kWarpThreads; ++warp) {
      const CompletionWarpTrace &warp_trace =
          warp_values[slowest_block * (options.threads / kWarpThreads) +
                      warp];
      longest_scatter = std::max(
          longest_scatter, warp_trace.scatter_end - warp_trace.scatter_begin);
      latest_scatter_end =
          std::max(latest_scatter_end, warp_trace.scatter_end);
    }
    scatter_cycles += longest_scatter;
    flush_and_barrier_cycles +=
        trace.finish_end > latest_scatter_end
            ? trace.finish_end - latest_scatter_end
            : 0;
  }
  const double traced_cycles = static_cast<double>(plan_and_wait_cycles) +
                               scatter_cycles + flush_and_barrier_cycles;
  if (traced_cycles != 0.0) {
    result.plan_and_wait_cta_ms = static_cast<float>(
        result.wait_and_scatter_ms * plan_and_wait_cycles / traced_cycles);
    result.scatter_cta_ms = static_cast<float>(
        result.wait_and_scatter_ms * scatter_cycles / traced_cycles);
    result.flush_and_barrier_cta_ms = static_cast<float>(
        result.wait_and_scatter_ms * flush_and_barrier_cycles / traced_cycles);
  }
  return result;
}

void report_completion_trace_timing(const alltoallv::Options &options,
                                    const char *implementation,
                                    const PhaseTimes &phases) {
  if (!options.profile_phases)
    return;
  int rank = 0;
  alltoallv::mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank),
                       "MPI_Comm_rank(completion trace)");
  struct FloatRank {
    float value;
    int rank;
  };
  FloatRank local{phases.wait_and_scatter_ms, rank};
  FloatRank critical{};
  alltoallv::mpi_check(
      MPI_Allreduce(&local, &critical, 1, MPI_FLOAT_INT, MPI_MAXLOC,
                    MPI_COMM_WORLD),
      "MPI_Allreduce(completion trace critical rank)");
  float critical_phase_ms[3] = {phases.plan_and_wait_cta_ms,
                                phases.scatter_cta_ms,
                                phases.flush_and_barrier_cta_ms};
  alltoallv::mpi_check(
      MPI_Bcast(critical_phase_ms, 3, MPI_FLOAT, critical.rank,
                MPI_COMM_WORLD),
      "MPI_Bcast(completion trace critical phases)");
  if (rank != 0)
    return;

  const float total_ms = critical_phase_ms[0] + critical_phase_ms[1] +
                         critical_phase_ms[2];
  const float plan_fraction =
      total_ms == 0.0f ? 0.0f : 100.0f * critical_phase_ms[0] / total_ms;
  const float scatter_fraction =
      total_ms == 0.0f ? 0.0f : 100.0f * critical_phase_ms[1] / total_ms;
  const float flush_fraction =
      total_ms == 0.0f ? 0.0f : 100.0f * critical_phase_ms[2] / total_ms;
  std::printf(
      "%s completion trace: %.3f ms/iteration plan + signal wait (%.1f%%), "
      "%.3f ms/iteration LSA scatter (%.1f%%), %.3f ms/iteration GIN "
      "flush + barrier (%.1f%%; critical CTA, CUDA-event-scaled)\n",
      implementation, critical_phase_ms[0] / options.iterations,
      plan_fraction, critical_phase_ms[1] / options.iterations,
      scatter_fraction, critical_phase_ms[2] / options.iterations,
      flush_fraction);
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
    report_completion_trace_timing(
        options, "NCCL LSA + railed GIN AlltoAllV", phases);
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
