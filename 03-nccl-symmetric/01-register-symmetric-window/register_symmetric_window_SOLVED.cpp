/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <mpi.h>

#define MPI_CALL(call)                                                         \
  {                                                                            \
    int mpi_status = call;                                                     \
    if (MPI_SUCCESS != mpi_status) {                                           \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                             \
      int mpi_error_string_length = 0;                                         \
      MPI_Error_string(mpi_status, mpi_error_string,                           \
                       &mpi_error_string_length);                              \
      if (NULL != mpi_error_string)                                            \
        fprintf(stderr,                                                        \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "         \
                "with %s "                                                     \
                "(%d).\n",                                                     \
                #call, __LINE__, __FILE__, mpi_error_string, mpi_status);      \
      else                                                                     \
        fprintf(stderr,                                                        \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "         \
                "with %d.\n",                                                  \
                #call, __LINE__, __FILE__, mpi_status);                        \
      exit(mpi_status);                                                        \
    }                                                                          \
  }

#include <cuda_runtime.h>
#include <unistd.h>

#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaSuccess != cudaStatus) {                                           \
      fprintf(stderr,                                                          \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "       \
              "with "                                                          \
              "%s (%d).\n",                                                    \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),       \
              cudaStatus);                                                     \
      exit(cudaStatus);                                                        \
    }                                                                          \
  }

#include <nccl.h>

#define NCCL_CALL(call)                                                        \
  {                                                                            \
    ncclResult_t ncclStatus = call;                                            \
    if (ncclSuccess != ncclStatus) {                                           \
      fprintf(stderr,                                                          \
              "ERROR: NCCL call \"%s\" in line %d of file %s failed "          \
              "with "                                                          \
              "%s (%d).\n",                                                    \
              #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus),       \
              ncclStatus);                                                     \
      exit(ncclStatus);                                                        \
    }                                                                          \
  }

#define NCCL_VERSION_SYMMETRIC NCCL_VERSION(2, 27, 6)
#define NCCL_SYMMETRIC_SUPPORT NCCL_VERSION_CODE >= NCCL_VERSION_SYMMETRIC

static size_t parse_buffer_size(int argc, char **argv, int rank) {
  size_t src_size = 1048576; // default 1 MiB
  int opt;
  const long long MIN_BYTES = 16LL;
  const long long MAX_BYTES = 1LL << 30; // 1 GiB
  optind = 1; // reset in case getopt was used elsewhere
  while ((opt = getopt(argc, argv, "b:")) != -1) {
    switch (opt) {
    case 'b': {
      long long v = atoll(optarg);
      if (v < MIN_BYTES)
        v = MIN_BYTES;
      if (v > MAX_BYTES)
        v = MAX_BYTES;
      src_size = (size_t)v;
      break;
    }
    default: {
      if (rank == 0)
        fprintf(stderr, "Usage: %s -b <bytes>\n", argv[0]);
      return 0; // signal failure
    }
    }
  }
  return src_size;
}

static int compare_src_dst(void *src, void *dst, int bytes, int world_size,
                           int rank) {
  int mismatches = 0;
  unsigned char *src_host = (unsigned char *)malloc(bytes);
  if (src_host == NULL) {
    fprintf(stderr, "Rank %d: host malloc failed during verification\n", rank);
    return 1;
  }

  unsigned char *dst_host = (unsigned char *)malloc(bytes * world_size);
  if (dst_host == NULL) {
    fprintf(stderr, "Rank %d: host malloc failed during verification\n", rank);
    return 1;
  }

  CUDA_RT_CALL(cudaMemcpy(src_host, (unsigned char *)src, bytes,
                          cudaMemcpyDeviceToHost));
  CUDA_RT_CALL(cudaMemcpy(dst_host, (unsigned char *)dst, world_size * bytes,
                          cudaMemcpyDeviceToHost));

  for (int r = 0; r < (world_size * bytes); ++r) {
    int exp_val = (r / bytes) + 1;
    if (dst_host[r] != exp_val) {
      fprintf(stderr, "Rank %d: dst[%d]=0x%02x, expected 0x%02x\n", rank, r,
              dst_host[r], exp_val);
      mismatches++;
    }
  }

  free(dst_host);
  free(src_host);

  if (mismatches == 0) {
    printf("Rank %d: AllGather verification passed.\n", rank);
  }

  return mismatches;
}

int main(int argc, char *argv[]) {

  MPI_CALL(MPI_Init(&argc, &argv));
  int rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int size;
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
  int num_devices = 0;
  CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

  ncclUniqueId nccl_uid;
  if (rank == 0)
    NCCL_CALL(ncclGetUniqueId(&nccl_uid));
  MPI_CALL(
      MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
  // MPI_Barrier ensures that all processs have completed the MPI_Bcast.
  // This can be required when combining MPI with other communication libraries
  // like NCCL.
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

  size_t src_size = parse_buffer_size(argc, argv, rank);
  if (src_size == 0) {
    MPI_CALL(MPI_Finalize());
    return 1;
  }

  int local_rank = -1;
  int local_size = 1;
  {
    MPI_Comm local_comm;
    MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &local_comm));

    MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CALL(MPI_Comm_size(local_comm, &local_size));

    MPI_CALL(MPI_Comm_free(&local_comm));
  }

  if (num_devices < 1) {
    fprintf(stderr, "ERROR No CUDA devices are visible to this rank.\n");
    MPI_CALL(MPI_Finalize());
    return 1;
  }
  if (num_devices != 1 && num_devices < local_size) {
    fprintf(stderr,
            "ERROR Number of visible devices (%d) is less than number of ranks "
            "on the node (%d)!\n",
            num_devices, local_size);
    MPI_CALL(MPI_Finalize());
    return 1;
  }
  if (num_devices == 1) {
    CUDA_RT_CALL(cudaSetDevice(0));
  } else {
    CUDA_RT_CALL(cudaSetDevice(local_rank));
  }
  CUDA_RT_CALL(cudaFree(0));

  ncclComm_t nccl_comm;
  NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));
  int nccl_version = 0;
  NCCL_CALL(ncclGetVersion(&nccl_version));
  if (nccl_version < 2276) {
    fprintf(stderr, "ERROR NCCL 2.27.6 or newer is required.\n");
    NCCL_CALL(ncclCommDestroy(nccl_comm));
    MPI_CALL(MPI_Finalize());
    return 1;
  }

  /* src_size set via -b option above */
  size_t dst_size =
      src_size *
      size; // result of allgather needs receive buffer to be sized accordingly
  void *src;
  void *dst;
  ncclWindow_t src_win;
  ncclWindow_t dst_win;

  NCCL_CALL(ncclMemAlloc(&src, src_size));
  NCCL_CALL(ncclMemAlloc(&dst, dst_size));
  CUDA_RT_CALL(
      cudaMemset(src, (1 + rank), src_size)); // initialize input data to rank
  CUDA_RT_CALL(cudaMemset(dst, 0xdb, dst_size)); // poision the output data
                                                 //
  CUDA_RT_CALL(cudaDeviceSynchronize());
  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // Passing NCCL_WIN_COLL_SYMMETRIC requires users to provide the symmetric
  // buffers among all ranks in collectives. Every rank needs to call
  // ncclCommWindowRegister to register its buffers.
  NCCL_CALL(ncclCommWindowRegister(nccl_comm, src, src_size, &src_win,
                                   NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CALL(ncclCommWindowRegister(nccl_comm, dst, dst_size, &dst_win,
                                   NCCL_WIN_COLL_SYMMETRIC));
  // Use the registered buffers for communication to enable symmetric
  // communication benefits.
  NCCL_CALL(ncclAllGather((char *)src, (char *)dst, src_size, ncclInt8,
                          nccl_comm, stream));
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  // Verify gathered data at expected offsets
  int verify_errors = compare_src_dst(src, dst, src_size, size, rank);
  if (verify_errors != 0) {
    fprintf(stderr, "Rank %d: verification failed with %d mismatches.\n", rank,
            verify_errors);
    return 1;
  }

  NCCL_CALL(ncclCommWindowDeregister(nccl_comm, src_win));
  NCCL_CALL(ncclCommWindowDeregister(nccl_comm, dst_win));

  NCCL_CALL(ncclMemFree(src));
  NCCL_CALL(ncclMemFree(dst));

  CUDA_RT_CALL(cudaStreamDestroy(stream));
  NCCL_CALL(ncclCommDestroy(nccl_comm));

  MPI_CALL(MPI_Finalize());
  return (0);
}
