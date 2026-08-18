/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstddef>
#include <cstdio>

#define DEVICE_API_CUDA_CHECK(call)                                            \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(status));                                     \
      MPI_Abort(MPI_COMM_WORLD, status);                                       \
    }                                                                          \
  } while (0)

#define DEVICE_API_NCCL_CHECK(call)                                            \
  do {                                                                         \
    ncclResult_t status = (call);                                              \
    if (status != ncclSuccess) {                                               \
      fprintf(stderr, "%s:%d: NCCL error: %s\n", __FILE__, __LINE__,           \
              ncclGetErrorString(status));                                     \
      MPI_Abort(MPI_COMM_WORLD, status);                                       \
    }                                                                          \
  } while (0)

namespace device_api {

enum class SetupResult { Ready, Skipped };

struct State {
  int rank = -1;
  int size = 0;
  bool mpi_initialized = false;
  bool dev_comm_created = false;
  ncclComm_t comm = nullptr;
  ncclDevComm dev_comm{};
  cudaStream_t stream = nullptr;
  void *payload = nullptr;
  void *result = nullptr;
  void *signal = nullptr;
  ncclWindow_t payload_window = nullptr;
  ncclWindow_t result_window = nullptr;
  ncclWindow_t signal_window = nullptr;
};

inline void skip(const State &state, const char *reason) {
  if (state.rank == 0)
    printf("SKIP: %s\n", reason);
}

inline void finish(State *state) {
  if (state->dev_comm_created) {
    DEVICE_API_NCCL_CHECK(ncclDevCommDestroy(state->comm, &state->dev_comm));
    state->dev_comm_created = false;
  }
  if (state->signal_window != nullptr) {
    DEVICE_API_NCCL_CHECK(
        ncclCommWindowDeregister(state->comm, state->signal_window));
    state->signal_window = nullptr;
  }
  if (state->result_window != nullptr) {
    DEVICE_API_NCCL_CHECK(
        ncclCommWindowDeregister(state->comm, state->result_window));
    state->result_window = nullptr;
  }
  if (state->payload_window != nullptr) {
    DEVICE_API_NCCL_CHECK(
        ncclCommWindowDeregister(state->comm, state->payload_window));
    state->payload_window = nullptr;
  }
  if (state->signal != nullptr) {
    DEVICE_API_NCCL_CHECK(ncclMemFree(state->signal));
    state->signal = nullptr;
  }
  if (state->payload != nullptr) {
    DEVICE_API_NCCL_CHECK(ncclMemFree(state->payload));
    state->payload = nullptr;
  }
  if (state->result != nullptr) {
    DEVICE_API_NCCL_CHECK(ncclMemFree(state->result));
    state->result = nullptr;
  }
  if (state->stream != nullptr) {
    DEVICE_API_CUDA_CHECK(cudaStreamDestroy(state->stream));
    state->stream = nullptr;
  }
  if (state->comm != nullptr) {
    DEVICE_API_NCCL_CHECK(ncclCommDestroy(state->comm));
    state->comm = nullptr;
  }
  if (state->mpi_initialized) {
    MPI_Finalize();
    state->mpi_initialized = false;
  }
}

inline SetupResult prepare(State *state, int *argc, char ***argv,
                           std::size_t payload_bytes, bool require_lsa,
                           bool require_gin) {
  MPI_Init(argc, argv);
  state->mpi_initialized = true;
  MPI_Comm_rank(MPI_COMM_WORLD, &state->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &state->size);

  if (state->size != 2) {
    skip(*state, "these focused device API labs require exactly two MPI ranks");
    return SetupResult::Skipped;
  }

  int runtime_version;
  DEVICE_API_NCCL_CHECK(ncclGetVersion(&runtime_version));
  int version_supported = runtime_version >= NCCL_VERSION(2, 29, 0);
  int all_versions_supported;
  MPI_Allreduce(&version_supported, &all_versions_supported, 1, MPI_INT,
                MPI_LAND, MPI_COMM_WORLD);
  if (!all_versions_supported) {
    skip(*state, "the linked NCCL runtime is older than 2.29");
    return SetupResult::Skipped;
  }

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state->rank,
                      MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  int device_count;
  DEVICE_API_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  int selected_device = device_count == 1 ? 0 : local_rank;
  int has_local_device =
      device_count > 0 && (device_count == 1 || local_rank < device_count);
  int all_have_local_devices;
  MPI_Allreduce(&has_local_device, &all_have_local_devices, 1, MPI_INT,
                MPI_LAND, MPI_COMM_WORLD);
  if (!all_have_local_devices) {
    skip(*state, "there is not one visible GPU per local MPI rank");
    return SetupResult::Skipped;
  }
  DEVICE_API_CUDA_CHECK(cudaSetDevice(selected_device));

  cudaDeviceProp properties;
  DEVICE_API_CUDA_CHECK(cudaGetDeviceProperties(&properties, selected_device));
  int supported_architecture = properties.major >= 7;
  int all_supported_architectures;
  MPI_Allreduce(&supported_architecture, &all_supported_architectures, 1,
                MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  if (!all_supported_architectures) {
    skip(*state, "the NCCL device API examples need a GPU with compute "
                 "capability 7.0 or newer");
    return SetupResult::Skipped;
  }

  ncclUniqueId id;
  if (state->rank == 0)
    DEVICE_API_NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  DEVICE_API_NCCL_CHECK(
      ncclCommInitRank(&state->comm, state->size, id, state->rank));

  ncclCommProperties_t comm_properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  DEVICE_API_NCCL_CHECK(ncclCommQueryProperties(state->comm, &comm_properties));
  int supports_device_api = comm_properties.deviceApiSupport;
  int all_support_device_api;
  MPI_Allreduce(&supports_device_api, &all_support_device_api, 1, MPI_INT,
                MPI_LAND, MPI_COMM_WORLD);
  if (!all_support_device_api) {
    skip(*state, "this communicator does not support the NCCL device API");
    return SetupResult::Skipped;
  }
  int supports_lsa =
      !require_lsa || comm_properties.nRanks == ncclTeamLsa(state->comm).nRanks;
  int all_support_lsa;
  MPI_Allreduce(&supports_lsa, &all_support_lsa, 1, MPI_INT, MPI_LAND,
                MPI_COMM_WORLD);
  if (!all_support_lsa) {
    skip(*state, "the two ranks are not both in the communicator's LSA team");
    return SetupResult::Skipped;
  }
  int supports_gin =
      !require_gin || comm_properties.ginType != NCCL_GIN_TYPE_NONE;
  int all_support_gin;
  MPI_Allreduce(&supports_gin, &all_support_gin, 1, MPI_INT, MPI_LAND,
                MPI_COMM_WORLD);
  if (!all_support_gin) {
    skip(*state, "GIN is unavailable for this communicator");
    return SetupResult::Skipped;
  }

  DEVICE_API_CUDA_CHECK(
      cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
  DEVICE_API_NCCL_CHECK(ncclMemAlloc(&state->payload, payload_bytes));
  DEVICE_API_CUDA_CHECK(
      cudaMemsetAsync(state->payload, 0, payload_bytes, state->stream));
  DEVICE_API_NCCL_CHECK(
      ncclCommWindowRegister(state->comm, state->payload, payload_bytes,
                             &state->payload_window, NCCL_WIN_COLL_SYMMETRIC));

  if (require_lsa) {
    DEVICE_API_NCCL_CHECK(ncclMemAlloc(&state->result, payload_bytes));
    DEVICE_API_CUDA_CHECK(
        cudaMemsetAsync(state->result, 0, payload_bytes, state->stream));
    DEVICE_API_NCCL_CHECK(
        ncclCommWindowRegister(state->comm, state->result, payload_bytes,
                               &state->result_window, NCCL_WIN_COLL_SYMMETRIC));
  }

  if (require_gin) {
    DEVICE_API_NCCL_CHECK(
        ncclMemAlloc(&state->signal, sizeof(unsigned long long)));
    DEVICE_API_CUDA_CHECK(cudaMemsetAsync(
        state->signal, 0, sizeof(unsigned long long), state->stream));
    DEVICE_API_NCCL_CHECK(ncclCommWindowRegister(
        state->comm, state->signal, sizeof(unsigned long long),
        &state->signal_window, NCCL_WIN_COLL_SYMMETRIC));
  }
  DEVICE_API_CUDA_CHECK(cudaStreamSynchronize(state->stream));
  MPI_Barrier(MPI_COMM_WORLD);

  ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.lsaBarrierCount = require_lsa ? 1 : 0;
  requirements.ginForceEnable = require_gin;
  DEVICE_API_NCCL_CHECK(
      ncclDevCommCreate(state->comm, &requirements, &state->dev_comm));
  state->dev_comm_created = true;
  return SetupResult::Ready;
}

} // namespace device_api
