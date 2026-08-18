/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

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
constexpr int kElements = 16;
constexpr int kThreads = 128;

__global__ void lsa_sum(ncclDevComm dev_comm, ncclWindow_t source_window,
                        ncclWindow_t result_window, int count) {
#if __CUDA_ARCH__ >= 700
  /*
   * Create an LSA CTA barrier, read each rank's source window with
   * ncclGetLsaPointer, and write the sum into result_window.
   */
  (void)dev_comm;
  (void)source_window;
  (void)result_window;
  (void)count;
#endif
}
} // namespace

int main(int argc, char **argv) {
  device_api::State state;
  if (device_api::prepare(&state, &argc, &argv, kElements * sizeof(float), true,
                          false) != device_api::SetupResult::Ready) {
    device_api::finish(&state);
    return 0;
  }

  float input[kElements];
  for (int i = 0; i < kElements; ++i)
    input[i] = static_cast<float>(state.rank + 1);
  DEVICE_API_CUDA_CHECK(cudaMemcpyAsync(state.payload, input, sizeof(input),
                                        cudaMemcpyHostToDevice, state.stream));
  DEVICE_API_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  MPI_Barrier(MPI_COMM_WORLD);

  lsa_sum<<<1, kThreads, 0, state.stream>>>(
      state.dev_comm, state.payload_window, state.result_window, kElements);
  DEVICE_API_CUDA_CHECK(cudaGetLastError());
  DEVICE_API_CUDA_CHECK(cudaStreamSynchronize(state.stream));

  float output[kElements];
  DEVICE_API_CUDA_CHECK(
      cudaMemcpy(output, state.result, sizeof(output), cudaMemcpyDeviceToHost));
  const float expected = static_cast<float>(state.size * (state.size + 1) / 2);
  int errors = 0;
  for (int i = 0; i < kElements; ++i) {
    if (output[i] != expected)
      ++errors;
  }
  if (errors == 0) {
    printf("Rank %d: LSA sum is %.1f for all %d elements.\n", state.rank,
           output[0], kElements);
  } else {
    fprintf(stderr, "Rank %d: LSA result did not match %.1f.\n", state.rank,
            expected);
  }

  int total_errors;
  MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  device_api::finish(&state);
  return total_errors == 0 ? 0 : 1;
}

#endif
