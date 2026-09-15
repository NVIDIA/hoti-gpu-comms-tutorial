/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../09-alltoallv/common/nvshmem_exercise.h"

__global__ void store_through_peer_pointer(int *symmetric_value, int value,
                                           int peer) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    /*
     * TODO: obtain the peer pointer with nvshmem_ptr. If it is non-null,
     * store value through it and call nvshmem_quiet().
     */
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

  int *symmetric_value = (int *)nvshmem_malloc(sizeof(*symmetric_value));
  if (symmetric_value == NULL)
    MPI_Abort(MPI_COMM_WORLD, 1);
  CUDA_CHECK(cudaMemsetAsync(symmetric_value, 0, sizeof(*symmetric_value),
                             context.stream));

  int peer = (context.rank + 1) % context.size;
  int local_supported = nvshmem_ptr(symmetric_value, peer) != NULL;
  int supported;
  MPI_Allreduce(&local_supported, &supported, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if (!supported) {
    if (context.rank == 0) {
      printf("nvshmem_ptr is unavailable for this allocation or topology; "
             "skipping\n");
    }
    nvshmem_free(symmetric_value);
    exercise_finalize(&context);
    return 0;
  }

  store_through_peer_pointer<<<1, 1, 0, context.stream>>>(
      symmetric_value, context.rank + 1, peer);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(context.stream));
  MPI_Barrier(MPI_COMM_WORLD);

  int observed = -1;
  CUDA_CHECK(cudaMemcpy(&observed, symmetric_value, sizeof(observed),
                        cudaMemcpyDeviceToHost));
  int previous_pe = (context.rank + context.size - 1) % context.size;
  int expected = previous_pe + 1;
  int local_error = observed != expected;
  if (local_error) {
    fprintf(stderr, "PE %d observed %d, expected %d\n", context.rank, observed,
            expected);
  } else {
    printf("PE %d read %d through a peer mapping\n", context.rank, observed);
  }

  int errors = exercise_sum_errors(local_error);
  nvshmem_free(symmetric_value);
  exercise_finalize(&context);
  return errors == 0 ? 0 : 1;
}
