/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../common/nvshmem_exercise.h"

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

  CUDA_CHECK(cudaMemsetAsync(source, 0, sizeof(*source), context.stream));
  CUDA_CHECK(cudaMemsetAsync(target, 0, sizeof(*target), context.stream));
  if (context.rank == 0) {
    int value = 42;
    CUDA_CHECK(cudaMemcpyAsync(source, &value, sizeof(value),
                               cudaMemcpyHostToDevice, context.stream));
  }

  /*
   * TODO: PE 0 should call nvshmemx_putmem_on_stream to put source into target
   * on PE 1. The put must be ordered on context.stream.
   */

  nvshmemx_barrier_all_on_stream(context.stream);

  int observed = -1;
  CUDA_CHECK(cudaMemcpyAsync(&observed, target, sizeof(observed),
                             cudaMemcpyDeviceToHost, context.stream));
  CUDA_CHECK(cudaStreamSynchronize(context.stream));

  int expected = context.rank == 1 ? 42 : 0;
  int local_error = observed != expected;
  if (local_error) {
    fprintf(stderr, "PE %d observed %d, expected %d\n", context.rank, observed,
            expected);
  } else {
    printf("PE %d observed %d\n", context.rank, observed);
  }

  int errors = exercise_sum_errors(local_error);
  nvshmem_free(source);
  nvshmem_free(target);
  exercise_finalize(&context);
  return errors == 0 ? 0 : 1;
}
