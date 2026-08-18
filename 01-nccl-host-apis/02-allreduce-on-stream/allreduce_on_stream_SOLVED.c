/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
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

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t status = (call);                                              \
    if (status != ncclSuccess) {                                               \
      fprintf(stderr, "%s:%d: NCCL error: %s\n", __FILE__, __LINE__,           \
              ncclGetErrorString(status));                                     \
      MPI_Abort(MPI_COMM_WORLD, status);                                       \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0 || (device_count != 1 && local_rank >= device_count)) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CUDA_CHECK(cudaSetDevice(device_count == 1 ? 0 : local_rank));

  ncclUniqueId id;
  if (rank == 0)
    NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  float send = (float)(rank + 1);
  float recv = 0.0f;
  float *device_send, *device_recv;
  CUDA_CHECK(cudaMalloc((void **)&device_send, sizeof(*device_send)));
  CUDA_CHECK(cudaMalloc((void **)&device_recv, sizeof(*device_recv)));
  CUDA_CHECK(cudaMemcpyAsync(device_send, &send, sizeof(send),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(device_recv, 0, sizeof(*device_recv), stream));

  NCCL_CHECK(ncclAllReduce(device_send, device_recv, 1, ncclFloat, ncclSum,
                           comm, stream));

  CUDA_CHECK(cudaMemcpyAsync(&recv, device_recv, sizeof(recv),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float expected = (float)(size * (size + 1) / 2);
  int local_error = recv != expected;
  if (local_error) {
    fprintf(stderr, "Rank %d got %.1f, expected %.1f\n", rank, recv, expected);
  } else {
    printf("Rank %d: all-reduce result %.1f\n", rank, recv);
  }

  int errors;
  MPI_Allreduce(&local_error, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CUDA_CHECK(cudaFree(device_send));
  CUDA_CHECK(cudaFree(device_recv));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return errors == 0 ? 0 : 1;
}
