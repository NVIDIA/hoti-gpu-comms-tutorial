/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../09-alltoallv/common/nvshmem_exercise.h"

__global__ void put_to_next_pe(int *target, const int *source, int next_pe) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    nvshmem_int_put(target, source, 1, next_pe);
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

  int *source = (int *)nvshmem_malloc(sizeof(*source));
  int *target = (int *)nvshmem_malloc(sizeof(*target));
  if (source == NULL || target == NULL)
    MPI_Abort(MPI_COMM_WORLD, 1);
  int source_value = context.rank + 1;
  CUDA_CHECK(cudaMemcpyAsync(source, &source_value, sizeof(*source),
                             cudaMemcpyHostToDevice, context.stream));
  CUDA_CHECK(cudaMemsetAsync(target, 0, sizeof(*target), context.stream));

  int next_pe = (context.rank + 1) % context.size;
  put_to_next_pe<<<1, 1, 0, context.stream>>>(target, source, next_pe);
  CUDA_CHECK(cudaGetLastError());
  nvshmemx_barrier_all_on_stream(context.stream);

  int observed = -1;
  CUDA_CHECK(cudaMemcpyAsync(&observed, target, sizeof(observed),
                             cudaMemcpyDeviceToHost, context.stream));
  CUDA_CHECK(cudaStreamSynchronize(context.stream));

  int previous_pe = (context.rank + context.size - 1) % context.size;
  int expected = previous_pe + 1;
  int local_error = observed != expected;
  if (local_error) {
    fprintf(stderr, "PE %d observed %d, expected %d\n", context.rank, observed,
            expected);
  } else {
    printf("PE %d received %d from PE %d\n", context.rank, observed,
           previous_pe);
  }

  int errors = exercise_sum_errors(local_error);
  nvshmem_free(source);
  nvshmem_free(target);
  exercise_finalize(&context);
  return errors == 0 ? 0 : 1;
}
