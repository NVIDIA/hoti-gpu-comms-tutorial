/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef NVSHMEM_EXERCISE_H
#define NVSHMEM_EXERCISE_H

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(status));                                     \
      MPI_Abort(MPI_COMM_WORLD, status);                                       \
    }                                                                          \
  } while (0)

typedef struct {
  int rank;
  int size;
  cudaStream_t stream;
} exercise_context_t;

static inline void exercise_init(int *argc, char ***argv,
                                 exercise_context_t *context) {
  MPI_Init(argc, argv);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                      MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    if (mpi_rank == 0)
      fprintf(stderr, "No CUDA device is visible\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (device_count != 1 && local_rank >= device_count) {
    if (mpi_rank == 0)
      fprintf(stderr, "Not enough visible GPUs for local ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaSetDevice(device_count == 1 ? 0 : local_rank));
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  context->rank = nvshmem_my_pe();
  context->size = nvshmem_n_pes();
}

static inline int exercise_sum_errors(int local_error) {
  int errors;
  MPI_Allreduce(&local_error, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return errors;
}

static inline void exercise_finalize(exercise_context_t *context) {
  CUDA_CHECK(cudaStreamDestroy(context->stream));
  nvshmem_finalize();
  MPI_Finalize();
}

#endif
