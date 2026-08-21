/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/nvshmem_exercise.h"

constexpr int kPayload = 42;
constexpr uint64_t kReady = 1;

__global__ void put_signal_wait(const int *source, int *payload,
                                uint64_t *ready, int *result, int rank) {
  if (rank == 0) {
    /* TODO: put source and set ready on PE 1 with the device put-signal API. */
  } else {
    /* TODO: wait for ready == kReady with the device signal-wait API. */
  }
  *result = *payload;
}

static void launch_collectively(const void *kernel, void **args,
                                cudaStream_t stream) {
  int status =
      nvshmemx_collective_launch(kernel, dim3(1), dim3(1), args, 0, stream);
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

  int *source = static_cast<int *>(nvshmem_malloc(sizeof(*source)));
  int *payload = static_cast<int *>(nvshmem_malloc(sizeof(*payload)));
  uint64_t *ready = static_cast<uint64_t *>(nvshmem_malloc(sizeof(*ready)));
  int *result = static_cast<int *>(nvshmem_malloc(sizeof(*result)));
  if (source == NULL || payload == NULL || ready == NULL || result == NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  CUDA_CHECK(cudaMemsetAsync(source, 0, sizeof(*source), context.stream));
  CUDA_CHECK(cudaMemsetAsync(payload, 0, sizeof(*payload), context.stream));
  CUDA_CHECK(cudaMemsetAsync(ready, 0, sizeof(*ready), context.stream));
  CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(*result), context.stream));
  if (context.rank == 0) {
    CUDA_CHECK(cudaMemcpyAsync(source, &kPayload, sizeof(kPayload),
                               cudaMemcpyHostToDevice, context.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  MPI_Barrier(MPI_COMM_WORLD);

  void *args[] = {&source, &payload, &ready, &result, &context.rank};
  launch_collectively(reinterpret_cast<const void *>(put_signal_wait), args,
                      context.stream);
  CUDA_CHECK(cudaGetLastError());

  int observed = -1;
  CUDA_CHECK(cudaMemcpyAsync(&observed, result, sizeof(observed),
                             cudaMemcpyDeviceToHost, context.stream));
  CUDA_CHECK(cudaStreamSynchronize(context.stream));

  int expected = context.rank == 1 ? kPayload : 0;
  int local_error = observed != expected;
  if (local_error) {
    fprintf(stderr, "PE %d observed %d, expected %d\n", context.rank, observed,
            expected);
  } else {
    printf("PE %d observed %d\n", context.rank, observed);
  }

  int errors = exercise_sum_errors(local_error);
  nvshmem_free(source);
  nvshmem_free(payload);
  nvshmem_free(ready);
  nvshmem_free(result);
  exercise_finalize(&context);
  return errors == 0 ? 0 : 1;
}
