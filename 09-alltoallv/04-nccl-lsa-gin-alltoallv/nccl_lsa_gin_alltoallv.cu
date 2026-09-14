/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
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
namespace cg = cooperative_groups;

constexpr std::uint64_t kVectorElements = sizeof(uint4) / sizeof(value_type);

// Split count into `shards` aligned, contiguous pieces and return piece
// `shard`. The last piece owns any scalar tail.
__device__ void shard_slice(std::uint64_t count, int shard, int shards,
                            std::uint64_t *offset, std::uint64_t *slice) {
  const std::uint64_t vectors = count / kVectorElements;
  const std::uint64_t begin = vectors * shard / shards;
  const std::uint64_t end = vectors * (shard + 1) / shards;
  *offset = begin * kVectorElements;
  const std::uint64_t limit =
      shard + 1 == shards ? count : end * kVectorElements;
  *slice = limit - *offset;
}

// Copy one contiguous slice with this CTA's own threads.
//
// Unlike the LSA lab, this deliberately keeps ONE store in flight per thread.
// The four-deep version in 02-nccl-lsa-alltoallv holds 16 more registers live,
// which costs occupancy across this much larger fused kernel and measured 20%
// slower here, while the same-domain copy is only about 5% of the hybrid's
// time. Optimise the phase that dominates, not the one that is easy to see.
__device__ void copy_values(const value_type *source, value_type *destination,
                            std::uint64_t count) {
  const std::uint64_t vector_count = count / kVectorElements;
  const uint4 *source_vectors = reinterpret_cast<const uint4 *>(source);
  uint4 *destination_vectors = reinterpret_cast<uint4 *>(destination);
  for (std::uint64_t i = threadIdx.x; i < vector_count; i += blockDim.x)
    destination_vectors[i] = source_vectors[i];
  for (std::uint64_t i = vector_count * kVectorElements + threadIdx.x;
       i < count; i += blockDim.x)
    destination[i] = source[i];
}

// Sender and receiver must agree on the CTA, because the GIN context and the
// signal index both come from it. Using the forward domain distance rather
// than an absolute domain pair gives every source domain the same spread over
// the grid.
__device__ int route_shard_block(int source_domain, int destination_domain,
                                 int destination_local, int domain_count,
                                 int local_count, int shard, int route_shards,
                                 int blocks) {
  const int domain_delta =
      (destination_domain - source_domain + domain_count) % domain_count;
  const std::uint64_t route =
      static_cast<std::uint64_t>(domain_delta - 1) * local_count +
      destination_local;
  return static_cast<int>((route * route_shards + shard) % blocks);
}

// Does this CTA own (source domain -> destination LSA rank, shard)?
__device__ bool owns_route_shard(const ncclTeam &lsa, const ncclTeam &rail,
                                 int source_domain, int destination_domain,
                                 int destination_local, int shard,
                                 int route_shards) {
  return route_shard_block(source_domain, destination_domain, destination_local,
                           rail.nRanks, lsa.nRanks, shard, route_shards,
                           gridDim.x) == static_cast<int>(blockIdx.x);
}

// Five per-CTA timestamps split one collective into the phases that matter:
// issue, wait for the network, scatter, flush, and the completion barrier.
// The stamps compile out of the default kernel, so --profile-phases costs
// nothing when it is off.
struct BlockTrace {
  std::uint64_t issue_begin;
  std::uint64_t wait_begin;
  std::uint64_t wait_end;
  std::uint64_t scatter_end;
  std::uint64_t flush_end;
  std::uint64_t barrier_end;
};

template <bool kProfile>
__device__ void stamp(BlockTrace *trace, std::uint64_t BlockTrace::*field) {
  if constexpr (kProfile) {
    __syncthreads();
    if (threadIdx.x == 0)
      trace->*field = clock64();
  }
}

__device__ void send_and_deliver_local(ncclDevComm dev_comm, ncclGin gin,
                                       ncclWindow_t send_window,
                                       ncclWindow_t recv_window,
                                       ncclWindow_t plan_window,
                                       ncclWindow_t inbox_window,
                                       std::size_t slot_bytes,
                                       std::size_t domain_bytes,
                                       int route_shards) {
#if __CUDA_ARCH__ >= 700
  const ncclGinSignal_t data_signal = static_cast<ncclGinSignal_t>(blockIdx.x);
  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);
  const int domain_first_rank = dev_comm.rank - lsa.rank;
  const DevicePlanEntry *plan = static_cast<const DevicePlanEntry *>(
      ncclGetLocalPointer(plan_window, 0));

  ncclBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), ncclTeamTagWorld(),
                                          gin, blockIdx.x};

  // TODO: Enter the world barrier with acquire ordering, so no rank writes a
  // peer's receive window or inbox before every rank has entered this epoch.
  // Pass ncclGinFenceLevel::None; the flush and exit barrier order the data.
  (void)barrier;

  // Same-domain messages go straight into the peer's receive window. Each CTA
  // owns one contiguous slice and starts its peer loop at a different LSA
  // rank, so the CTAs do not all drive the same NVLink route at once.
  for (int step = 0; step < lsa.nRanks; ++step) {
    const int destination_local = (lsa.rank + step + blockIdx.x) % lsa.nRanks;
    const DevicePlanEntry entry = plan[domain_first_rank + destination_local];
    std::uint64_t offset = 0;
    std::uint64_t count = 0;
    shard_slice(entry.send_count, blockIdx.x, gridDim.x, &offset, &count);

    // TODO: Take the local source with ncclGetLocalPointer(send_window, ...)
    // and the peer target with ncclGetLsaPointer(recv_window, ...,
    // destination_local), both advanced by `offset`, then copy `count`
    // elements with copy_values.
    (void)entry;
    (void)count;
  }

  // Cross-domain messages travel over the rail to the matching LSA rank in the
  // destination domain, landing in that GPU's inbox slot.
  if (threadIdx.x == 0) {
    const int task_count = rail.nRanks * lsa.nRanks;
    for (int task = 0; task < task_count; ++task) {
      const int destination_domain = task / lsa.nRanks;
      const int destination_local = task % lsa.nRanks;
      if (destination_domain == rail.rank)
        continue;
      const int ingress_rank =
          ncclTeamRankToWorld(dev_comm, rail, destination_domain);
      const DevicePlanEntry entry =
          plan[ingress_rank - lsa.rank + destination_local];
      for (int shard = 0; shard < route_shards; ++shard) {
        if (!owns_route_shard(lsa, rail, rail.rank, destination_domain,
                              destination_local, shard, route_shards))
          continue;
        std::uint64_t offset = 0;
        std::uint64_t count = 0;
        shard_slice(entry.send_count, shard, route_shards, &offset, &count);
        if (count == 0)
          continue;

        // TODO: Put this shard into the destination ingress GPU's inbox slot
        // with gin.put(rail, destination_domain, inbox_window, ...). The slot
        // byte offset is
        //   rail.rank * domain_bytes + destination_local * slot_bytes
        // plus this shard's `offset`. Attach ncclGin_WeakSignalInc{data_signal}
        // so the receiver can count arrivals.
        (void)inbox_window;
        (void)domain_bytes;
        (void)slot_bytes;
        (void)data_signal;
      }
    }
  }
  __syncthreads();
#endif
}

template <bool kProfile>
__device__ void wait_and_scatter(ncclDevComm dev_comm, ncclGin gin,
                                 ncclWindow_t recv_window,
                                 ncclWindow_t plan_window,
                                 ncclWindow_t inbox_window,
                                 std::size_t slot_bytes,
                                 std::size_t domain_bytes, int route_shards,
                                 std::uint64_t epoch, BlockTrace *trace) {
#if __CUDA_ARCH__ >= 700
  const ncclGinSignal_t data_signal = static_cast<ncclGinSignal_t>(blockIdx.x);
  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const ncclTeam rail = ncclTeamRail(dev_comm);
  const int task_count = rail.nRanks * lsa.nRanks;

  // Count the inbound shards this CTA owns, then wait for exactly that many
  // weak signals. The count is the same on every epoch, so the running signal
  // total is epoch * expected.
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
          static_cast<const DevicePlanEntry *>(
              ncclGetLsaPointer(plan_window, 0, destination_local));
      const DevicePlanEntry entry = destination_plan[source];
      for (int shard = 0; shard < route_shards; ++shard) {
        if (!owns_route_shard(lsa, rail, source_domain, rail.rank,
                              destination_local, shard, route_shards))
          continue;
        std::uint64_t offset = 0;
        std::uint64_t count = 0;
        shard_slice(entry.recv_count, shard, route_shards, &offset, &count);
        expected += count != 0;
      }
    }
  }
  __syncthreads();

  // TODO: Wait until this CTA's signal has reached epoch * expected. Skip the
  // wait when expected is zero, or the CTA blocks forever on an empty route.
  (void)data_signal;
  (void)epoch;

  stamp<kProfile>(trace, &BlockTrace::wait_end);

  // Deliver each arrived inbox slot to the final GPU inside this domain.
  for (int task = 0; task < task_count; ++task) {
    const int source_domain = task / lsa.nRanks;
    const int destination_local = task % lsa.nRanks;
    if (source_domain == rail.rank)
      continue;
    const int source = ncclTeamRankToWorld(dev_comm, rail, source_domain);
    const DevicePlanEntry *destination_plan =
        static_cast<const DevicePlanEntry *>(
            ncclGetLsaPointer(plan_window, 0, destination_local));
    const DevicePlanEntry entry = destination_plan[source];
    for (int shard = 0; shard < route_shards; ++shard) {
      if (!owns_route_shard(lsa, rail, source_domain, rail.rank,
                            destination_local, shard, route_shards))
        continue;
      std::uint64_t offset = 0;
      std::uint64_t count = 0;
      shard_slice(entry.recv_count, shard, route_shards, &offset, &count);
      if (count == 0)
        continue;
      // TODO: Read the arrived slot with ncclGetLocalPointer(inbox_window,
      // source_domain * domain_bytes + destination_local * slot_bytes + ...)
      // and write it to the final GPU with ncclGetLsaPointer(recv_window,
      // entry.recv_offset + offset, destination_local), then copy_values.
      (void)entry;
      (void)count;
    }
  }
  stamp<kProfile>(trace, &BlockTrace::scatter_end);

  // TODO: One gin.flush(ncclCoopCta()) makes this rank's own puts locally
  // complete, so the send window can be reused. There is no flush per shard.
  stamp<kProfile>(trace, &BlockTrace::flush_end);

  ncclBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), ncclTeamTagWorld(),
                                          gin, blockIdx.x};

  // TODO: Leave the world barrier with acquire-release ordering to publish the
  // collective.
  (void)barrier;
#endif
}

template <bool kProfile>
__global__ void nccl_lsa_gin_alltoallv_kernel(
    ncclDevComm dev_comm, ncclWindow_t send_window, ncclWindow_t recv_window,
    ncclWindow_t plan_window, ncclWindow_t inbox_window,
    std::size_t slot_bytes, std::size_t domain_bytes, int route_shards,
    std::uint64_t epoch, BlockTrace *traces) {
#if __CUDA_ARCH__ >= 700
  const int context_count = min(static_cast<int>(gridDim.x),
                                static_cast<int>(dev_comm.ginContextCount));
  ncclGin gin{dev_comm, static_cast<int>(blockIdx.x) % context_count};
  BlockTrace *trace = kProfile ? &traces[blockIdx.x] : nullptr;

  stamp<kProfile>(trace, &BlockTrace::issue_begin);
  send_and_deliver_local(dev_comm, gin, send_window, recv_window, plan_window,
                         inbox_window, slot_bytes, domain_bytes, route_shards);

  // The launch is cooperative, so every CTA that can produce a matching signal
  // is resident. This grid barrier replaces the old kernel boundary without
  // letting a receiver wait delay a producer.
  cg::this_grid().sync();

  stamp<kProfile>(trace, &BlockTrace::wait_begin);
  wait_and_scatter<kProfile>(dev_comm, gin, recv_window, plan_window,
                             inbox_window, slot_bytes, domain_bytes,
                             route_shards, epoch, trace);
  stamp<kProfile>(trace, &BlockTrace::barrier_end);
#endif
}

template <bool kProfile>
void launch_kernel(const alltoallv::nccl_setup::State &state,
                   const alltoallv::Options &options, int route_shards,
                   std::uint64_t epoch, BlockTrace *traces) {
  ncclDevComm dev_comm = state.dev_comm;
  ncclWindow_t send_window = state.send_window;
  ncclWindow_t recv_window = state.recv_window;
  ncclWindow_t plan_window = state.plan_window;
  ncclWindow_t inbox_window = state.inbox_window;
  std::size_t slot_bytes = state.hybrid_slot_bytes;
  std::size_t domain_bytes = state.hybrid_domain_bytes;
  void *args[] = {&dev_comm,     &send_window,  &recv_window,   &plan_window,
                  &inbox_window, &slot_bytes,   &domain_bytes,  &route_shards,
                  &epoch,        &traces};
  ALLTOALLV_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(
          nccl_lsa_gin_alltoallv_kernel<kProfile>),
      dim3(options.blocks), dim3(options.threads), args, 0, state.stream));
}

void launch_alltoallv(const alltoallv::nccl_setup::State &state,
                      const alltoallv::Options &options, int route_shards,
                      std::uint64_t epoch) {
  launch_kernel<false>(state, options, route_shards, epoch, nullptr);
}

template <bool kProfile>
int cooperative_grid_limit(const alltoallv::Options &options) {
  int device = 0;
  ALLTOALLV_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  ALLTOALLV_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int resident_blocks_per_sm = 0;
  ALLTOALLV_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident_blocks_per_sm,
      reinterpret_cast<const void *>(
          nccl_lsa_gin_alltoallv_kernel<kProfile>),
      options.threads, 0));
  return resident_blocks_per_sm * properties.multiProcessorCount;
}

void require_cooperative_grid(const alltoallv::nccl_setup::State &state,
                              const alltoallv::Options &options) {
  int device = 0;
  ALLTOALLV_CUDA_CHECK(cudaGetDevice(&device));
  int local_supported = 0;
  ALLTOALLV_CUDA_CHECK(cudaDeviceGetAttribute(
      &local_supported, cudaDevAttrCooperativeLaunch, device));
  int all_supported = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_supported, &all_supported, 1,
                                     MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(cooperative launch support)");
  if (!all_supported) {
    if (state.rank == 0) {
      std::fprintf(stderr,
                   "The single-kernel NCCL hybrid implementation requires "
                   "CUDA cooperative launch support on every rank\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // A profiled run launches the other specialization, so honour whichever
  // kernel this run will actually use.
  const int local_limit = options.profile_phases
                              ? cooperative_grid_limit<true>(options)
                              : cooperative_grid_limit<false>(options);
  int collective_limit = 0;
  alltoallv::mpi_check(MPI_Allreduce(&local_limit, &collective_limit, 1,
                                     MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                       "MPI_Allreduce(cooperative grid limit)");
  if (options.blocks <= collective_limit)
    return;
  if (state.rank == 0) {
    std::fprintf(stderr,
                 "--blocks=%d exceeds the single-kernel cooperative grid "
                 "limit (%d); lower --blocks on every rank\n",
                 options.blocks, collective_limit);
  }
  MPI_Abort(MPI_COMM_WORLD, 1);
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
  // One route carrying most of a rank's cross-domain volume may use the whole
  // grid; otherwise reserve a share of the grid for every remote route.
  const bool heavy_route =
      max_pair_count != 0 &&
      max_pair_count >= (plan.max_network_send_elements + 1) / 2;
  const int route_capacity =
      heavy_route ? options.blocks
                  : std::max(1, options.blocks / remote_routes);
  const std::uint64_t max_pair_bytes = max_pair_count * sizeof(value_type);
  const std::uint64_t size_shards = std::max<std::uint64_t>(
      1, (max_pair_bytes + kTargetBytes - 1) / kTargetBytes);
  return static_cast<int>(std::min<std::uint64_t>(route_capacity,
                                                  size_shards));
}

// Run the timed loop with the profiled kernel and split the measured time
// across phases using the slowest CTA's in-kernel clock.
float profile_iterations(const alltoallv::nccl_setup::State &state,
                         const alltoallv::Options &options, int route_shards,
                         std::uint64_t *epoch, double phase_ms[5]) {
  const std::size_t per_iteration = options.blocks;
  BlockTrace *device_traces = nullptr;
  std::vector<BlockTrace> host_traces(
      static_cast<std::size_t>(options.iterations) * per_iteration);
  ALLTOALLV_CUDA_CHECK(
      cudaMalloc(&device_traces, host_traces.size() * sizeof(BlockTrace)));

  cudaEvent_t start;
  cudaEvent_t stop;
  ALLTOALLV_CUDA_CHECK(cudaEventCreate(&start));
  ALLTOALLV_CUDA_CHECK(cudaEventCreate(&stop));
  ALLTOALLV_CUDA_CHECK(cudaEventRecord(start, state.stream));
  for (int i = 0; i < options.iterations; ++i) {
    launch_kernel<true>(state, options, route_shards, ++*epoch,
                        device_traces + i * per_iteration);
  }
  ALLTOALLV_CUDA_CHECK(cudaEventRecord(stop, state.stream));
  ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stop));
  ALLTOALLV_CUDA_CHECK(cudaEventDestroy(start));
  ALLTOALLV_CUDA_CHECK(cudaMemcpy(host_traces.data(), device_traces,
                                  host_traces.size() * sizeof(BlockTrace),
                                  cudaMemcpyDeviceToHost));
  ALLTOALLV_CUDA_CHECK(cudaFree(device_traces));

  double cycles[5] = {};
  for (int i = 0; i < options.iterations; ++i) {
    const BlockTrace *values = host_traces.data() + i * per_iteration;
    const BlockTrace *slowest = values;
    for (int block = 1; block < options.blocks; ++block) {
      if (values[block].barrier_end - values[block].issue_begin >
          slowest->barrier_end - slowest->issue_begin)
        slowest = &values[block];
    }
    cycles[0] += slowest->wait_begin - slowest->issue_begin;
    cycles[1] += slowest->wait_end - slowest->wait_begin;
    cycles[2] += slowest->scatter_end - slowest->wait_end;
    cycles[3] += slowest->flush_end - slowest->scatter_end;
    cycles[4] += slowest->barrier_end - slowest->flush_end;
  }
  const double total = cycles[0] + cycles[1] + cycles[2] + cycles[3] +
                       cycles[4];
  for (int phase = 0; phase < 5; ++phase)
    phase_ms[phase] = total == 0.0 ? 0.0 : elapsed_ms * cycles[phase] / total;
  return elapsed_ms;
}

void report_phase_profile(const alltoallv::Options &options,
                          const char *implementation,
                          const double phase_ms[5]) {
  int rank = 0;
  alltoallv::mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank),
                       "MPI_Comm_rank(phase profile)");
  double local[5];
  double critical[5];
  for (int phase = 0; phase < 5; ++phase)
    local[phase] = phase_ms[phase];
  alltoallv::mpi_check(MPI_Allreduce(local, critical, 5, MPI_DOUBLE, MPI_MAX,
                                     MPI_COMM_WORLD),
                       "MPI_Allreduce(phase profile)");
  if (rank != 0)
    return;
  const double total =
      critical[0] + critical[1] + critical[2] + critical[3] + critical[4];
  static const char *names[5] = {"issue + local delivery",
                                 "wait for inbound shards", "LSA scatter",
                                 "GIN flush", "completion barrier"};
  std::printf("%s phase profile (slowest CTA, CUDA-event-scaled):\n",
              implementation);
  for (int phase = 0; phase < 5; ++phase) {
    std::printf("  %-24s %.3f ms/iteration (%.1f%%)\n", names[phase],
                critical[phase] / options.iterations,
                total == 0.0 ? 0.0 : 100.0 * critical[phase] / total);
  }
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
  require_cooperative_grid(state, options);

  alltoallv::print_plan(plan, options, "NCCL LSA + railed GIN AlltoAllV");
  const int route_shards = choose_route_shards(state, options, plan);
  if (state.rank == 0) {
    std::printf("NCCL hybrid routing: %d shard(s) per remote route\n",
                route_shards);
  }

  std::uint64_t epoch = 1;
  launch_alltoallv(state, options, route_shards, epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  int errors = alltoallv::nccl_setup::copy_and_validate(
      &state, plan, "NCCL LSA + railed GIN AlltoAllV");
  if (errors != 0) {
    alltoallv::nccl_setup::finish(&state);
    return 1;
  }

  for (int i = 0; i < options.warmup; ++i)
    launch_alltoallv(state, options, route_shards, ++epoch);
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD),
                       "MPI_Barrier(benchmark)");

  float elapsed_ms = 0.0f;
  double phase_ms[5] = {};
  if (options.profile_phases) {
    elapsed_ms = profile_iterations(state, options, route_shards, &epoch,
                                    phase_ms);
  } else {
    cudaEvent_t start;
    cudaEvent_t stop;
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&start));
    ALLTOALLV_CUDA_CHECK(cudaEventCreate(&stop));
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(start, state.stream));
    for (int i = 0; i < options.iterations; ++i)
      launch_alltoallv(state, options, route_shards, ++epoch);
    ALLTOALLV_CUDA_CHECK(cudaEventRecord(stop, state.stream));
    ALLTOALLV_CUDA_CHECK(cudaEventSynchronize(stop));
    ALLTOALLV_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(stop));
    ALLTOALLV_CUDA_CHECK(cudaEventDestroy(start));
  }
  alltoallv::report_timing(plan, options,
                           "NCCL LSA + railed GIN AlltoAllV", elapsed_ms);
  if (options.profile_phases) {
    report_phase_profile(options, "NCCL LSA + railed GIN AlltoAllV",
                         phase_ms);
  }

  alltoallv::nccl_setup::clear_recv_for_reuse(&state);
  launch_alltoallv(state, options, route_shards, ++epoch);
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
