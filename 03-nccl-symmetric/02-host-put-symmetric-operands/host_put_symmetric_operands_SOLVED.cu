/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>

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

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("SKIP: this exercise needs NCCL headers 2.29 or newer for "
           "ncclPutSignal.\n");
  }
  MPI_Finalize();
  return 0;
}

#else

namespace {
constexpr int kElements = 16;
constexpr int kSignalIndex = 0;
constexpr int kContext = 0;
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    if (rank == 0)
      fprintf(stderr, "This exercise requires exactly two MPI ranks.\n");
    MPI_Finalize();
    return 1;
  }

  int runtime_version;
  NCCL_CHECK(ncclGetVersion(&runtime_version));
  int version_supported = runtime_version >= NCCL_VERSION(2, 29, 0);
  int all_versions_supported;
  MPI_Allreduce(&version_supported, &all_versions_supported, 1, MPI_INT,
                MPI_LAND, MPI_COMM_WORLD);
  if (!all_versions_supported) {
    if (rank == 0) {
      printf("SKIP: linked NCCL %d is older than 2.29.0, which added one-sided "
             "RMA.\n",
             runtime_version);
    }
    MPI_Finalize();
    return 0;
  }

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_free(&local_comm);

  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  int has_local_device =
      device_count > 0 && (device_count == 1 || local_rank < device_count);
  int all_have_local_devices;
  MPI_Allreduce(&has_local_device, &all_have_local_devices, 1, MPI_INT,
                MPI_LAND, MPI_COMM_WORLD);
  if (!all_have_local_devices) {
    if (rank == 0)
      printf("SKIP: not enough visible GPUs for one rank per GPU.\n");
    MPI_Finalize();
    return 0;
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

  int host_send[kElements];
  int host_recv[kElements] = {};
  for (int i = 0; i < kElements; ++i)
    host_send[i] = rank * 100 + i;

  void *device_send = nullptr;
  void *device_recv = nullptr;
  NCCL_CHECK(ncclMemAlloc(&device_send, sizeof(host_send)));
  NCCL_CHECK(ncclMemAlloc(&device_recv, sizeof(host_recv)));
  CUDA_CHECK(cudaMemcpyAsync(device_send, host_send, sizeof(host_send),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(device_recv, 0, sizeof(host_recv), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  ncclWindow_t send_window = nullptr;
  ncclWindow_t recv_window = nullptr;
  NCCL_CHECK(ncclCommWindowRegister(comm, device_send, sizeof(host_send),
                                    &send_window, NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(comm, device_recv, sizeof(host_recv),
                                    &recv_window, NCCL_WIN_COLL_SYMMETRIC));

  MPI_Barrier(MPI_COMM_WORLD);
  const int destination = (rank + 1) % size;
  const int source = (rank - 1 + size) % size;

  NCCL_CHECK(ncclPutSignal(device_send, kElements, ncclInt, destination,
                           recv_window, 0, kSignalIndex, kContext, 0, comm,
                           stream));
  ncclWaitSignalDesc_t wait_desc = {1, source, kSignalIndex, kContext};
  NCCL_CHECK(ncclWaitSignal(1, &wait_desc, comm, stream));

  CUDA_CHECK(cudaMemcpyAsync(host_recv, device_recv, sizeof(host_recv),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int errors = 0;
  for (int i = 0; i < kElements; ++i) {
    const int expected = source * 100 + i;
    if (host_recv[i] != expected)
      ++errors;
  }
  if (errors == 0) {
    printf("Rank %d received %d..%d from rank %d.\n", rank, host_recv[0],
           host_recv[kElements - 1], source);
  } else {
    fprintf(stderr, "Rank %d received an unexpected payload from rank %d.\n",
            rank, source);
  }

  int total_errors;
  MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  NCCL_CHECK(ncclCommWindowDeregister(comm, send_window));
  NCCL_CHECK(ncclCommWindowDeregister(comm, recv_window));
  NCCL_CHECK(ncclMemFree(device_send));
  NCCL_CHECK(ncclMemFree(device_recv));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return total_errors == 0 ? 0 : 1;
}

#endif
