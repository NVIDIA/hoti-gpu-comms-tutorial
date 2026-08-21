/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>

static void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, (int)status);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local_comm);

  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  int device_count;
  cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0 || (device_count != 1 && local_rank >= device_count)) {
    fprintf(stderr, "rank %d has no visible GPU for local rank %d\n", rank,
            local_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  cuda_check(cudaSetDevice(device_count == 1 ? 0 : local_rank),
             "cudaSetDevice");
  cuda_check(cudaFree(0), "cudaFree(0)");

  printf("hello from rank %d\n", rank);
  fflush(stdout);

  MPI_Finalize();
  return 0;
}
