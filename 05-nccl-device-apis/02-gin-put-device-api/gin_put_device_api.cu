/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdint>
#include <cstdio>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("SKIP: this exercise needs NCCL headers 2.29 or newer for the "
           "device API.\n");
  }
  MPI_Finalize();
  return 0;
}

#else

#include "device_api_common.hpp"

namespace {
constexpr uint64_t kPutValue = 28;

__global__ void gin_put(ncclDevComm dev_comm, ncclWindow_t payload_window,
                        ncclWindow_t signal_window) {
#if __CUDA_ARCH__ >= 700
  const ncclTeam world = ncclTeamWorld(dev_comm);
  ncclGin gin(dev_comm, 0);

  if (world.rank == 0) {
    uint64_t *source =
        static_cast<uint64_t *>(ncclGetLocalPointer(payload_window, 0));
    *source = kPutValue;
    /*
     * Put source into rank 1's payload window and increment its signal
     * window with ncclGin_VASignalInc.
     */
  }

  if (world.rank == 1) {
    /* Wait for the signal before the target buffer is inspected by the host. */
  }
#endif
}
} // namespace

int main(int argc, char **argv) {
  device_api::State state;
  if (device_api::prepare(&state, &argc, &argv, sizeof(uint64_t), false,
                          true) != device_api::SetupResult::Ready) {
    device_api::finish(&state);
    return 0;
  }

  gin_put<<<1, 1, 0, state.stream>>>(state.dev_comm, state.payload_window,
                                     state.signal_window);
  DEVICE_API_CUDA_CHECK(cudaGetLastError());
  DEVICE_API_CUDA_CHECK(cudaStreamSynchronize(state.stream));

  uint64_t received = 0;
  if (state.rank == 1) {
    DEVICE_API_CUDA_CHECK(cudaMemcpy(&received, state.payload, sizeof(received),
                                     cudaMemcpyDeviceToHost));
  }
  int errors = state.rank == 1 && received != kPutValue;
  if (state.rank == 1 && errors == 0) {
    printf("Rank 1 received the GIN put value %llu.\n",
           static_cast<unsigned long long>(received));
  } else if (state.rank == 1) {
    fprintf(stderr, "Rank 1 received %llu instead of %llu.\n",
            static_cast<unsigned long long>(received),
            static_cast<unsigned long long>(kPutValue));
  }

  int total_errors;
  MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  device_api::finish(&state);
  return total_errors == 0 ? 0 : 1;
}

#endif
