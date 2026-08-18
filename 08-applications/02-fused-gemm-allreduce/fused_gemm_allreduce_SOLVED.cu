/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/nvshmem_exercise.h"

#include <array>
#include <cmath>

constexpr int kTile = 16;
constexpr int kElements = kTile * kTile;
constexpr int kThreads = kElements;
constexpr uint64_t kReady = 1;

__global__ void fused_gemm_allreduce(const float *a, const float *b,
                                     float *output, float *inbox,
                                     uint64_t *ready, int peer) {
  int index = threadIdx.x;
  int row = index / kTile;
  int col = index % kTile;
  float value = 0.0f;
  for (int k = 0; k < kTile; ++k) {
    value += a[row * kTile + k] * b[k * kTile + col];
  }
  output[index] = value;
  __threadfence_system();
  __syncthreads();

  if (threadIdx.x == 0) {
    nvshmem_float_put_signal(inbox, output, kElements, ready, kReady,
                             NVSHMEM_SIGNAL_SET, peer);
    nvshmem_signal_wait_until(ready, NVSHMEM_CMP_EQ, kReady);
  }
  __syncthreads();
  output[index] += inbox[index];
}

static void fill_inputs(std::array<float, kElements> *a,
                        std::array<float, kElements> *b, int rank) {
  for (int row = 0; row < kTile; ++row) {
    for (int col = 0; col < kTile; ++col) {
      (*a)[row * kTile + col] = 0.25f * (rank + 1) + 0.01f * (row + col);
      (*b)[row * kTile + col] = 0.5f * (rank + 1) + 0.02f * (row - col);
    }
  }
}

static void gemm_reference(const std::array<float, kElements> &a,
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

static void launch_collectively(const void *kernel, void **args,
                                cudaStream_t stream) {
  int status = nvshmemx_collective_launch(kernel, dim3(1), dim3(kThreads), args,
                                          0, stream);
  if (status != NVSHMEMX_SUCCESS) {
    fprintf(stderr, "nvshmemx_collective_launch failed: %d\n", status);
    MPI_Abort(MPI_COMM_WORLD, status);
  }
}

int main(int argc, char **argv) {
  exercise_context_t context;
  exercise_init(&argc, &argv, &context);
  if (context.size != 2) {
    if (context.rank == 0)
      fprintf(stderr, "This exercise requires exactly two PEs\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  float *a = static_cast<float *>(nvshmem_malloc(kElements * sizeof(*a)));
  float *b = static_cast<float *>(nvshmem_malloc(kElements * sizeof(*b)));
  float *output =
      static_cast<float *>(nvshmem_malloc(kElements * sizeof(*output)));
  float *inbox =
      static_cast<float *>(nvshmem_malloc(kElements * sizeof(*inbox)));
  uint64_t *ready = static_cast<uint64_t *>(nvshmem_malloc(sizeof(*ready)));
  if (a == NULL || b == NULL || output == NULL || inbox == NULL ||
      ready == NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::array<float, kElements> host_a;
  std::array<float, kElements> host_b;
  std::array<float, kElements> local_reference;
  std::array<float, kElements> expected;
  std::array<float, kElements> observed;
  fill_inputs(&host_a, &host_b, context.rank);
  gemm_reference(host_a, host_b, &local_reference);
  MPI_Allreduce(local_reference.data(), expected.data(), kElements, MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);

  CUDA_CHECK(cudaMemcpyAsync(a, host_a.data(), kElements * sizeof(*a),
                             cudaMemcpyHostToDevice, context.stream));
  CUDA_CHECK(cudaMemcpyAsync(b, host_b.data(), kElements * sizeof(*b),
                             cudaMemcpyHostToDevice, context.stream));
  CUDA_CHECK(
      cudaMemsetAsync(output, 0, kElements * sizeof(*output), context.stream));
  CUDA_CHECK(
      cudaMemsetAsync(inbox, 0, kElements * sizeof(*inbox), context.stream));
  CUDA_CHECK(cudaMemsetAsync(ready, 0, sizeof(*ready), context.stream));
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  MPI_Barrier(MPI_COMM_WORLD);

  int peer = 1 - context.rank;
  void *args[] = {&a, &b, &output, &inbox, &ready, &peer};
  launch_collectively(reinterpret_cast<const void *>(fused_gemm_allreduce),
                      args, context.stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(observed.data(), output,
                             kElements * sizeof(*output),
                             cudaMemcpyDeviceToHost, context.stream));
  CUDA_CHECK(cudaStreamSynchronize(context.stream));

  int local_error = 0;
  for (int index = 0; index < kElements; ++index) {
    if (std::fabs(observed[index] - expected[index]) > 1.0e-4f) {
      fprintf(stderr, "PE %d output[%d] = %.6f, expected %.6f\n", context.rank,
              index, observed[index], expected[index]);
      local_error = 1;
      break;
    }
  }
  if (!local_error) {
    printf("PE %d fused all-reduce verified\n", context.rank);
  }

  int errors = exercise_sum_errors(local_error);
  nvshmem_free(a);
  nvshmem_free(b);
  nvshmem_free(output);
  nvshmem_free(inbox);
  nvshmem_free(ready);
  exercise_finalize(&context);
  return errors == 0 ? 0 : 1;
}
