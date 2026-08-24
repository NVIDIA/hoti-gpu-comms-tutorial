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

struct alignas(16) HybridPacketItem {
  std::uint64_t bytes;
  std::uint64_t recv_offset_bytes;
  std::uint64_t payload_offset;
  std::uint64_t reserved;
};

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

  void *send = nullptr;
  void *recv = nullptr;
  void *plan = nullptr;
  void *outbox = nullptr;
  void *inbox = nullptr;
  ncclWindow_t send_window = nullptr;
  ncclWindow_t recv_window = nullptr;
  ncclWindow_t plan_window = nullptr;
  ncclWindow_t outbox_window = nullptr;
  ncclWindow_t inbox_window = nullptr;

  std::size_t send_bytes = 0;
  std::size_t recv_bytes = 0;
  std::size_t plan_bytes = 0;
  std::size_t packet_capacity = 0;
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
  deregister(&state->outbox_window);
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
  release(&state->outbox);
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
  int supported_runtime = runtime_version >= NCCL_VERSION(2, 31, 2);
  int all_supported = 0;
  alltoallv::mpi_check(MPI_Allreduce(&supported_runtime, &all_supported, 1,
                                     MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                       "MPI_Allreduce(NCCL version)");
  if (!all_supported) {
    print_skip(*state, "the NCCL AlltoAllV labs need NCCL 2.31.2 or newer");
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
    capable = capable && properties.railedGinType != NCCL_GIN_TYPE_NONE &&
              state->lsa_team.nRanks > 1 && state->rail_team.nRanks > 1 &&
              state->lsa_team.nRanks * state->rail_team.nRanks == state->size &&
              min_lsa_size == max_lsa_size &&
              min_rail_size == max_rail_size;
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
                 "this placement does not provide uniform LSA teams and "
                 "railed GIN across at least two nodes");
    return SetupResult::Skipped;
  }

  state->send_bytes = std::max<std::size_t>(
      1, plan->global_send_capacity * sizeof(value_type));
  state->recv_bytes = std::max<std::size_t>(
      1, plan->global_recv_capacity * sizeof(value_type));
  state->plan_bytes = std::max<std::size_t>(
      1, plan->size * sizeof(DevicePlanEntry));
  if (backend == Backend::HybridRail) {
    std::size_t header = align_bytes(state->lsa_team.nRanks *
                                     sizeof(HybridPacketItem));
    std::size_t max_message =
        align_bytes(plan->max_pair_count * sizeof(value_type));
    state->packet_capacity =
        header + state->lsa_team.nRanks * max_message;
    state->staging_bytes =
        std::max<std::size_t>(1, state->rail_team.nRanks *
                                    state->packet_capacity);
  }

  ALLTOALLV_NCCL_CHECK(ncclMemAlloc(&state->send, state->send_bytes));
  ALLTOALLV_NCCL_CHECK(ncclMemAlloc(&state->recv, state->recv_bytes));
  ALLTOALLV_NCCL_CHECK(ncclMemAlloc(&state->plan, state->plan_bytes));
  if (backend == Backend::HybridRail) {
    ALLTOALLV_NCCL_CHECK(
        ncclMemAlloc(&state->outbox, state->staging_bytes));
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
    ALLTOALLV_CUDA_CHECK(cudaMemsetAsync(state->outbox, 0,
                                        state->staging_bytes, state->stream));
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
        state->comm, state->outbox, state->staging_bytes,
        &state->outbox_window, NCCL_WIN_COLL_SYMMETRIC));
    ALLTOALLV_NCCL_CHECK(ncclCommWindowRegister(
        state->comm, state->inbox, state->staging_bytes,
        &state->inbox_window, NCCL_WIN_COLL_SYMMETRIC));
  }

  ncclDevCommRequirements requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (backend == Backend::Lsa) {
    requirements.lsaBarrierCount = options->blocks;
  } else if (backend == Backend::Gin) {
    requirements.ginContextCount = 1;
    requirements.worldGinBarrierCount = options->blocks;
    requirements.ginSignalCount = options->blocks;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    requirements.ginStrongSignalsRequired = false;
    requirements.ginVaSignalsRequired = false;
  } else {
    requirements.ginContextCount = 1;
    requirements.barrierCount = options->blocks;
    requirements.ginSignalCount = state->rail_team.nRanks;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
    requirements.ginStrongSignalsRequired = false;
    requirements.ginVaSignalsRequired = false;
  }
  ALLTOALLV_NCCL_CHECK(
      ncclDevCommCreate(state->comm, &requirements, &state->dev_comm));
  state->dev_comm_created = true;
  alltoallv::mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(setup)");

  if (state->rank == 0) {
    std::printf("NCCL topology: world=%d, LSA=%d, rail=%d\n", state->size,
                state->lsa_team.nRanks, state->rail_team.nRanks);
    std::printf(
        "NCCL API: headers=%d, runtime=%d, device=%d, GIN=%d, railed GIN=%d\n",
        NCCL_VERSION_CODE, runtime_version,
        static_cast<int>(properties.deviceApiSupport),
        static_cast<int>(properties.ginType),
        static_cast<int>(properties.railedGinType));
  }
  return SetupResult::Ready;
}

inline int copy_and_validate(State *state, const Plan &plan,
                             const char *implementation) {
  std::vector<value_type> observed(plan.global_recv_capacity);
  ALLTOALLV_CUDA_CHECK(cudaMemcpyAsync(
      observed.data(), state->recv, state->recv_bytes, cudaMemcpyDeviceToHost,
      state->stream));
  ALLTOALLV_CUDA_CHECK(cudaStreamSynchronize(state->stream));
  return alltoallv::validate(plan, observed, implementation);
}

} // namespace alltoallv::nccl_setup
