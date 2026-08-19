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

#include "../../05-nccl-device-apis/device_api_common.hpp"

#include <array>
#include <cmath>

namespace {
constexpr int kTile = 16;
constexpr int kElements = kTile * kTile;
constexpr int kThreads = kElements;

__global__ void fused_gemm_allreduce_nccl(const float *a, const float *b,
                                          ncclDevComm dev_comm,
                                          ncclWindow_t partial_window,
                                          ncclWindow_t result_window) {
#if __CUDA_ARCH__ >= 700
  int index = threadIdx.x;
  int row = index / kTile;
  int col = index % kTile;
  float value = 0.0f;
  for (int k = 0; k < kTile; ++k) {
    value += a[row * kTile + k] * b[k * kTile + col];
  }

  float *partial = static_cast<float *>(ncclGetLocalPointer(partial_window, 0));
  float *result = static_cast<float *>(ncclGetLocalPointer(result_window, 0));
  partial[index] = value;
  result[index] = value;
  __syncthreads();

  // TODO: Synchronize the LSA team after every rank has produced its tile.
  // TODO: Sum this element from every rank's partial window into result, then
  // synchronize the LSA team before the kernel returns.
  (void)dev_comm;
#endif
}

void fill_inputs(std::array<float, kElements> *a,
                 std::array<float, kElements> *b, int rank) {
  for (int row = 0; row < kTile; ++row) {
    for (int col = 0; col < kTile; ++col) {
      (*a)[row * kTile + col] = 0.25f * (rank + 1) + 0.01f * (row + col);
      (*b)[row * kTile + col] = 0.5f * (rank + 1) + 0.02f * (row - col);
    }
  }
}

void gemm_reference(const std::array<float, kElements> &a,
                    const std::array<float, kElements> &b,
                    std::array<float, kElements> *output) {
  for (int row = 0; row < kTile; ++row) {
    for (int col = 0; col < kTile; ++col) {
      float value = 0.0f;
      for (int k = 0; k < kTile; ++k) {
        value += a[row * kTile + k] * b[k * kTile + col];
      }
      (*output)[row * kTile + col] = value;
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  device_api::State state;
  if (device_api::prepare(&state, &argc, &argv, kElements * sizeof(float), true,
                          false) != device_api::SetupResult::Ready) {
    device_api::finish(&state);
    return 0;
  }

  float *a = nullptr;
  float *b = nullptr;
  DEVICE_API_CUDA_CHECK(cudaMalloc(&a, kElements * sizeof(*a)));
  DEVICE_API_CUDA_CHECK(cudaMalloc(&b, kElements * sizeof(*b)));

  std::array<float, kElements> host_a;
  std::array<float, kElements> host_b;
  std::array<float, kElements> local_reference;
  std::array<float, kElements> expected;
  std::array<float, kElements> observed;
  fill_inputs(&host_a, &host_b, state.rank);
  gemm_reference(host_a, host_b, &local_reference);
  MPI_Allreduce(local_reference.data(), expected.data(), kElements, MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);

  DEVICE_API_CUDA_CHECK(cudaMemcpyAsync(a, host_a.data(), sizeof(host_a),
                                        cudaMemcpyHostToDevice, state.stream));
  DEVICE_API_CUDA_CHECK(cudaMemcpyAsync(b, host_b.data(), sizeof(host_b),
                                        cudaMemcpyHostToDevice, state.stream));
  DEVICE_API_CUDA_CHECK(cudaStreamSynchronize(state.stream));
  MPI_Barrier(MPI_COMM_WORLD);

  fused_gemm_allreduce_nccl<<<1, kThreads, 0, state.stream>>>(
      a, b, state.dev_comm, state.payload_window, state.result_window);
  DEVICE_API_CUDA_CHECK(cudaGetLastError());
  DEVICE_API_CUDA_CHECK(cudaMemcpyAsync(observed.data(), state.result,
                                        sizeof(observed),
                                        cudaMemcpyDeviceToHost, state.stream));
  DEVICE_API_CUDA_CHECK(cudaStreamSynchronize(state.stream));

  int local_error = 0;
  for (int index = 0; index < kElements; ++index) {
    if (std::fabs(observed[index] - expected[index]) > 1.0e-4f) {
      fprintf(stderr, "Rank %d output[%d] = %.6f, expected %.6f\n", state.rank,
              index, observed[index], expected[index]);
      local_error = 1;
      break;
    }
  }
  if (!local_error) {
    printf("Rank %d: NCCL LSA fused all-reduce verified\n", state.rank);
  }

  int errors;
  MPI_Allreduce(&local_error, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  DEVICE_API_CUDA_CHECK(cudaFree(a));
  DEVICE_API_CUDA_CHECK(cudaFree(b));
  device_api::finish(&state);
  return errors == 0 ? 0 : 1;
}

#endif
