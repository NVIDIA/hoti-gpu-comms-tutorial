/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "alltoallv_common.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

#define ALLTOALLV_CUDA_CHECK(call)                                           \
  do {                                                                        \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status));                               \
      MPI_Abort(MPI_COMM_WORLD, status);                                      \
    }                                                                         \
  } while (0)

#define ALLTOALLV_NCCL_CHECK(call)                                           \
  do {                                                                        \
    ncclResult_t status = (call);                                             \
    if (status != ncclSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: NCCL error: %s\n", __FILE__, __LINE__,    \
                   ncclGetErrorString(status));                               \
      MPI_Abort(MPI_COMM_WORLD, status);                                      \
    }                                                                         \
  } while (0)

namespace alltoallv::nccl_setup {

enum class Backend { Lsa, Gin, HybridRail };
enum class SetupResult { Ready, Skipped };

struct State {
  int rank = -1;
  int size = 0;
  bool mpi_initialized = false;
  bool dev_comm_created = false;
  ncclComm_t comm = nullptr;
  ncclDevComm dev_comm{};
  cudaStream_t stream = nullptr;
  ncclTeam_t lsa_team{};
  ncclTeam_t rail_team{};
  ncclGinType_t railed_gin_type = NCCL_GIN_TYPE_NONE;

  void *send = nullptr;
  void *recv = nullptr;
  void *plan = nullptr;
  void *inbox = nullptr;
  ncclWindow_t send_window = nullptr;
  ncclWindow_t recv_window = nullptr;
  ncclWindow_t plan_window = nullptr;
  ncclWindow_t inbox_window = nullptr;

  std::size_t send_bytes = 0;
  std::size_t recv_bytes = 0;
  std::size_t plan_bytes = 0;
  std::size_t hybrid_slot_bytes = 0;
  std::size_t hybrid_domain_bytes = 0;
  std::size_t staging_bytes = 0;
};

inline std::size_t align_bytes(std::size_t value,
                               std::size_t alignment = 16) {
  return (value + alignment - 1) & ~(alignment - 1);
}

inline void print_skip(const State &state, const char *reason) {
  if (state.rank == 0)
    std::printf("SKIP: %s\n", reason);
}

inline const char *gin_type_name(ncclGinType_t type) {
  switch (type) {
  case NCCL_GIN_TYPE_NONE:
    return "none";
  case NCCL_GIN_TYPE_PROXY:
    return "proxy";
  case NCCL_GIN_TYPE_GDAKI:
    return "GDAKI";
  case NCCL_GIN_TYPE_GPI:
    return "GPI";
  case NCCL_GIN_TYPE_EFA_GDA:
    return "EFA GDA";
  default:
    return "unknown";
  }
}

inline void finish(State *state) {
  if (state->dev_comm_created) {
    ALLTOALLV_NCCL_CHECK(ncclDevCommDestroy(state->comm, &state->dev_comm));
    state->dev_comm_created = false;
  }

  auto deregister = [&](ncclWindow_t *window) {
    if (*window != nullptr) {
      ALLTOALLV_NCCL_CHECK(ncclCommWindowDeregister(state->comm, *window));
      *window = nullptr;
    }
  };
  deregister(&state->inbox_window);
  deregister(&state->plan_window);
  deregister(&state->recv_window);
  deregister(&state->send_window);

  auto release = [](void **pointer) {
    if (*pointer != nullptr) {
      ALLTOALLV_NCCL_CHECK(ncclMemFree(*pointer));
      *pointer = nullptr;
    }
  };
  release(&state->inbox);
  release(&state->plan);
  release(&state->recv);
  release(&state->send);

  if (state->stream != nullptr) {
    ALLTOALLV_CUDA_CHECK(cudaStreamDestroy(state->stream));
    state->stream = nullptr;
  }
  if (state->comm != nullptr) {
    ALLTOALLV_NCCL_CHECK(ncclCommDestroy(state->comm));
    state->comm = nullptr;
  }
  if (state->mpi_initialized) {
    MPI_Finalize();
    state->mpi_initialized = false;
  }
}

inline SetupResult prepare(State *state, int *argc, char ***argv,
                           Backend backend, Options *options, Plan *plan) {
  alltoallv::mpi_check(MPI_Init(argc, argv), "MPI_Init");
  state->mpi_initialized = true;
  alltoallv::mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &state->rank),
                       "MPI_Comm_rank");
  alltoallv::mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &state->size),
                       "MPI_Comm_size");

  *options = alltoallv::parse_options(*argc, *argv, state->rank);
  if (options->help) {
    if (state->rank == 0)
      alltoallv::print_usage((*argv)[0]);
    return SetupResult::Skipped;
  }
  alltoallv::require_matching_collective_options(*options, state->rank);
  *plan = alltoallv::make_plan(state->rank, state->size, *options);

  int local = alltoallv::local_rank();
  int device_count = 0;
  ALLTOALLV_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  int has_device = device_count > 0 &&
                   (device_count == 1 || local < device_count);
  int all_have_device = 0;
  alltoallv::mpi_check(MPI_Allreduce(&has_device, &all_have_device, 1, MPI_INT,
                                     MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(device availability)");
  if (!all_have_device) {
    print_skip(*state, "there is not one visible GPU per local MPI rank");
    return SetupResult::Skipped;
  }
  ALLTOALLV_CUDA_CHECK(cudaSetDevice(device_count == 1 ? 0 : local));
  cudaDeviceProp device_properties{};
  ALLTOALLV_CUDA_CHECK(cudaGetDeviceProperties(
      &device_properties, device_count == 1 ? 0 : local));
  int device_api_arch = device_properties.major >= 7;
  int all_device_api_arch = 0;
  alltoallv::mpi_check(MPI_Allreduce(&device_api_arch, &all_device_api_arch, 1,
                                     MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(GPU architecture)");
  if (!all_device_api_arch) {
    print_skip(*state, "the NCCL device API needs compute capability 7.0 or newer");
    return SetupResult::Skipped;
  }
  ALLTOALLV_CUDA_CHECK(
      cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));

  int runtime_version = 0;
  ALLTOALLV_NCCL_CHECK(ncclGetVersion(&runtime_version));
  int local_versions[2] = {NCCL_VERSION_CODE, runtime_version};
  int minimum_versions[2] = {};
  int maximum_versions[2] = {};
  alltoallv::mpi_check(
      MPI_Allreduce(local_versions, minimum_versions, 2, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD),
      "MPI_Allreduce(minimum NCCL versions)");
  alltoallv::mpi_check(
      MPI_Allreduce(local_versions, maximum_versions, 2, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD),
      "MPI_Allreduce(maximum NCCL versions)");
  const bool uniform_versions =
      minimum_versions[0] == maximum_versions[0] &&
      minimum_versions[1] == maximum_versions[1];
  const bool minimum_runtime = runtime_version >= NCCL_VERSION(2, 31, 2);
  // GIN device code is not cross-version compatible.  LSA is compatible with
  // a newer runtime, but never with a runtime older than its headers.
  const bool compatible_runtime = backend == Backend::Lsa
                                      ? runtime_version >= NCCL_VERSION_CODE
                                      : runtime_version == NCCL_VERSION_CODE;
  const int local_version_ready =
      uniform_versions && minimum_runtime && compatible_runtime;
  int all_version_ready = 0;
  alltoallv::mpi_check(
      MPI_Allreduce(&local_version_ready, &all_version_ready, 1, MPI_INT,
                    MPI_LAND, MPI_COMM_WORLD),
      "MPI_Allreduce(NCCL version compatibility)");
  if (!all_version_ready) {
    char reason[256];
    if (!uniform_versions) {
      std::snprintf(reason, sizeof(reason),
                    "NCCL header/runtime versions differ across ranks "
                    "(headers=%d..%d, runtime=%d..%d)",
                    minimum_versions[0], maximum_versions[0],
                    minimum_versions[1], maximum_versions[1]);
    } else if (!minimum_runtime) {
      std::snprintf(reason, sizeof(reason),
                    "the NCCL AlltoAllV labs need runtime 2.31.2 or newer "
                    "(runtime=%d)", runtime_version);
    } else if (backend == Backend::Lsa) {
      std::snprintf(reason, sizeof(reason),
                    "the NCCL LSA runtime (%d) is older than its headers "
                    "(%d)", runtime_version, NCCL_VERSION_CODE);
    } else {
      std::snprintf(reason, sizeof(reason),
                    "GIN requires matching NCCL headers and runtime "
                    "(headers=%d, runtime=%d)", NCCL_VERSION_CODE,
                    runtime_version);
    }
    print_skip(*state, reason);
    return SetupResult::Skipped;
  }

  ncclUniqueId id;
  if (state->rank == 0)
    ALLTOALLV_NCCL_CHECK(ncclGetUniqueId(&id));
  alltoallv::mpi_check(
      MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
      "MPI_Bcast(NCCL unique ID)");
  ALLTOALLV_NCCL_CHECK(
      ncclCommInitRank(&state->comm, state->size, id, state->rank));

  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  ALLTOALLV_NCCL_CHECK(
      ncclCommQueryProperties(state->comm, &properties));
  state->lsa_team = ncclTeamLsa(state->comm);
  state->rail_team = ncclTeamRail(state->comm);
  state->railed_gin_type = properties.railedGinType;
  const int lsa_root =
      ncclTeamRankToWorld(state->comm, state->lsa_team, 0);
  alltoallv::classify_placement(plan, lsa_root, state->lsa_team.rank,
                                "NCCL LSA");

  int min_lsa_size = 0;
  int max_lsa_size = 0;
  int min_rail_size = 0;
  int max_rail_size = 0;
  alltoallv::mpi_check(MPI_Allreduce(&state->lsa_team.nRanks, &min_lsa_size, 1,
                                     MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                       "MPI_Allreduce(minimum LSA size)");
  alltoallv::mpi_check(MPI_Allreduce(&state->lsa_team.nRanks, &max_lsa_size, 1,
                                     MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                       "MPI_Allreduce(maximum LSA size)");
  alltoallv::mpi_check(MPI_Allreduce(&state->rail_team.nRanks, &min_rail_size,
                                     1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                       "MPI_Allreduce(minimum rail size)");
  alltoallv::mpi_check(MPI_Allreduce(&state->rail_team.nRanks, &max_rail_size,
                                     1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                       "MPI_Allreduce(maximum rail size)");

  int capable = properties.deviceApiSupport;
  if (backend == Backend::Lsa)
    capable = capable && state->lsa_team.nRanks == state->size;
  if (backend == Backend::Gin)
    capable = capable && properties.ginType != NCCL_GIN_TYPE_NONE;
  if (backend == Backend::HybridRail) {
    int aligned_layout = 1;
    const int local_first = state->rank - state->lsa_team.rank;
    for (int local_rank = 0; local_rank < state->lsa_team.nRanks;
         ++local_rank) {
      const int world_rank =
          ncclTeamRankToWorld(state->comm, state->lsa_team, local_rank);
      aligned_layout = aligned_layout &&
                       world_rank == local_first + local_rank;
    }
    for (int domain = 0; domain < state->rail_team.nRanks; ++domain) {
      const int ingress =
          ncclTeamRankToWorld(state->comm, state->rail_team, domain);
      if (ingress < 0 || ingress >= state->size) {
        aligned_layout = 0;
        continue;
      }
      aligned_layout =
          aligned_layout &&
          plan->domain_ranks[ingress] == state->lsa_team.rank;
      const int domain_first = ingress - state->lsa_team.rank;
      for (int local_rank = 0; local_rank < state->lsa_team.nRanks;
           ++local_rank) {
        const int world_rank = domain_first + local_rank;
        aligned_layout =
            aligned_layout && world_rank >= 0 && world_rank < state->size;
        if (world_rank < 0 || world_rank >= state->size)
          continue;
        aligned_layout =
            aligned_layout &&
            plan->domain_roots[world_rank] == plan->domain_roots[ingress] &&
            plan->domain_ranks[world_rank] == local_rank;
      }
    }
    capable = capable && properties.railedGinType != NCCL_GIN_TYPE_NONE &&
              state->lsa_team.nRanks > 1 && state->rail_team.nRanks > 1 &&
              state->lsa_team.nRanks * state->rail_team.nRanks == state->size &&
              min_lsa_size == max_lsa_size &&
              min_rail_size == max_rail_size &&
              state->lsa_team.rank == plan->domain_ranks[state->rank] &&
              aligned_layout;
  }
  int all_capable = 0;
  alltoallv::mpi_check(MPI_Allreduce(&capable, &all_capable, 1, MPI_INT,
                                     MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(NCCL capabilities)");
  if (!all_capable) {
    if (backend == Backend::Lsa)
      print_skip(*state, "the communicator is not one LSA domain");
    else if (backend == Backend::Gin)
      print_skip(*state, "full GIN connectivity is unavailable");
    else
      print_skip(*state,
                 "this placement does not provide aligned, contiguous LSA "
                 "teams and railed GIN across at least two LSA domains");
    return SetupResult::Skipped;
  }

  state->send_bytes = std::max<std::size_t>(
      1, plan->global_send_capacity * sizeof(value_type));
  state->recv_bytes = std::max<std::size_t>(
      1, plan->global_recv_capacity * sizeof(value_type));
  state->plan_bytes = std::max<std::size_t>(
      1, plan->size * sizeof(DevicePlanEntry));
  if (backend == Backend::HybridRail) {
    state->hybrid_slot_bytes =
        align_bytes(plan->max_pair_count * sizeof(value_type));
    state->hybrid_domain_bytes =
        state->lsa_team.nRanks * state->hybrid_slot_bytes;
    state->staging_bytes = std::max<std::size_t>(
        1, state->rail_team.nRanks * state->hybrid_domain_bytes);
  }

  ALLTOALLV_NCCL_CHECK(ncclMemAlloc(&state->send, state->send_bytes));
  ALLTOALLV_NCCL_CHECK(ncclMemAlloc(&state->recv, state->recv_bytes));
  ALLTOALLV_NCCL_CHECK(ncclMemAlloc(&state->plan, state->plan_bytes));
  if (backend == Backend::HybridRail) {
    ALLTOALLV_NCCL_CHECK(
        ncclMemAlloc(&state->inbox, state->staging_bytes));
  }

  std::vector<value_type> send_host = alltoallv::make_send_buffer(*plan);
  std::vector<DevicePlanEntry> entries = alltoallv::device_entries(*plan);
  ALLTOALLV_CUDA_CHECK(cudaMemcpyAsync(
      state->send, send_host.data(), state->send_bytes, cudaMemcpyHostToDevice,
      state->stream));
  ALLTOALLV_CUDA_CHECK(
      cudaMemsetAsync(state->recv, 0xa5, state->recv_bytes, state->stream));
  ALLTOALLV_CUDA_CHECK(cudaMemcpyAsync(
      state->plan, entries.data(), state->plan_bytes, cudaMemcpyHostToDevice,
      state->stream));
  if (backend == Backend::HybridRail) {
    ALLTOALLV_CUDA_CHECK(cudaMemsetAsync(state->inbox, 0,
                                        state->staging_bytes, state->stream));
  }
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state->stream));

  ALLTOALLV_NCCL_CHECK(ncclCommWindowRegister(
      state->comm, state->send, state->send_bytes, &state->send_window,
      NCCL_WIN_COLL_SYMMETRIC));
  ALLTOALLV_NCCL_CHECK(ncclCommWindowRegister(
      state->comm, state->recv, state->recv_bytes, &state->recv_window,
      NCCL_WIN_COLL_SYMMETRIC));
  ALLTOALLV_NCCL_CHECK(ncclCommWindowRegister(
      state->comm, state->plan, state->plan_bytes, &state->plan_window,
      NCCL_WIN_COLL_SYMMETRIC));
  if (backend == Backend::HybridRail) {
    ALLTOALLV_NCCL_CHECK(ncclCommWindowRegister(
        state->comm, state->inbox, state->staging_bytes,
        &state->inbox_window, NCCL_WIN_COLL_SYMMETRIC));
  }

  ncclDevCommRequirements requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  const int requested_gin_contexts =
      alltoallv::requested_gin_contexts(*options);
  const int requested_gin_signals = options->blocks;
  if (backend == Backend::Lsa) {
    requirements.lsaBarrierCount = options->blocks;
  } else if (backend == Backend::Gin) {
    requirements.ginContextCount = requested_gin_contexts;
    requirements.worldGinBarrierCount = options->blocks;
    requirements.ginSignalCount = options->blocks;
    requirements.ginQueueDepth = options->gin_queue_depth;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    requirements.ginStrongSignalsRequired = false;
    requirements.ginVaSignalsRequired = false;
  } else {
    requirements.ginContextCount = requested_gin_contexts;
    requirements.barrierCount = options->blocks;
    requirements.ginSignalCount = requested_gin_signals;
    requirements.ginQueueDepth = options->gin_queue_depth;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
    requirements.ginStrongSignalsRequired = false;
    requirements.ginVaSignalsRequired = false;
  }
  ALLTOALLV_NCCL_CHECK(
      ncclDevCommCreate(state->comm, &requirements, &state->dev_comm));
  state->dev_comm_created = true;
  if (backend != Backend::Lsa) {
    int local_resources[2] = {
        static_cast<int>(state->dev_comm.ginContextCount),
        state->dev_comm.ginSignalCount};
    int minimum_resources[2] = {};
    int maximum_resources[2] = {};
    alltoallv::mpi_check(
        MPI_Allreduce(local_resources, minimum_resources, 2, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce(minimum GIN resources)");
    alltoallv::mpi_check(
        MPI_Allreduce(local_resources, maximum_resources, 2, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(maximum GIN resources)");
    const bool uniform_resources =
        minimum_resources[0] == maximum_resources[0] &&
        minimum_resources[1] == maximum_resources[1];
    const bool resources_sufficient = local_resources[0] > 0 &&
                                      local_resources[1] >=
                                          requested_gin_signals;
    const int local_resources_ready =
        uniform_resources && resources_sufficient;
    int all_resources_ready = 0;
    alltoallv::mpi_check(
        MPI_Allreduce(&local_resources_ready, &all_resources_ready, 1,
                      MPI_INT, MPI_LAND, MPI_COMM_WORLD),
        "MPI_Allreduce(GIN resource compatibility)");
    if (!all_resources_ready) {
      char reason[256];
      std::snprintf(reason, sizeof(reason),
                    "GIN resources are unsuitable or differ across ranks "
                    "(contexts=%d..%d, signals=%d..%d; need nonzero "
                    "contexts and at least %d signals)",
                    minimum_resources[0], maximum_resources[0],
                    minimum_resources[1], maximum_resources[1],
                    requested_gin_signals);
      print_skip(*state, reason);
      return SetupResult::Skipped;
    }
  }
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(setup)");

  if (state->rank == 0) {
    std::printf("NCCL topology: world=%d, LSA=%d, rail=%d\n", state->size,
                state->lsa_team.nRanks, state->rail_team.nRanks);
    std::printf(
        "NCCL API: headers=%d, runtime=%d, device=%d, GIN=%s (%d), "
        "railed GIN=%s (%d)\n",
        NCCL_VERSION_CODE, runtime_version,
        static_cast<int>(properties.deviceApiSupport),
        gin_type_name(properties.ginType), static_cast<int>(properties.ginType),
        gin_type_name(properties.railedGinType),
        static_cast<int>(properties.railedGinType));
    if (backend != Backend::Lsa) {
      std::printf(
          "NCCL GIN resources: contexts requested=%d, created=%u; "
          "signals requested=%d, created=%d; queue depth=%d\n",
          requirements.ginContextCount, state->dev_comm.ginContextCount,
          requirements.ginSignalCount, state->dev_comm.ginSignalCount,
          requirements.ginQueueDepth);
    }
  }
  return SetupResult::Ready;
}

inline int copy_buffer_and_validate(State *state, const void *source,
                                    const Plan &plan,
                                    const char *implementation,
                                    value_type bias = 0) {
  std::vector<value_type> observed(plan.global_recv_capacity);
  ALLTOALLV_CUDA_CHECK(cudaMemcpyAsync(
      observed.data(), source, state->recv_bytes, cudaMemcpyDeviceToHost,
      state->stream));
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state->stream));
  return alltoallv::validate(plan, observed, implementation, bias);
}

inline int copy_and_validate(State *state, const Plan &plan,
                             const char *implementation,
                             value_type bias = 0) {
  return copy_buffer_and_validate(state, state->recv, plan, implementation,
                                  bias);
}

// The reuse validation deliberately clears the receive buffer so a missing
// write cannot be hidden by data from a prior launch. Direct LSA writes come
// from peer streams, so all ranks must finish that clear before any rank
// starts the next collective.
inline void clear_recv_for_reuse(State *state) {
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD),
                       "MPI_Barrier(reuse clear begin)");
  ALLTOALLV_CUDA_CHECK(
      cudaMemsetAsync(state->recv, 0xa5, state->recv_bytes, state->stream));
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state->stream));
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD),
                       "MPI_Barrier(reuse clear complete)");
}

} // namespace alltoallv::nccl_setup
