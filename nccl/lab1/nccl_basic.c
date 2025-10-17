/* Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

#define NCCL_CALL(call)                                                                     \
    {                                                                                       \
        ncclResult_t  ncclStatus = call;                                                    \
        if (ncclSuccess != ncclStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: NCCL call \"%s\" in line %d of file %s failed "                 \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), ncclStatus); \
            exit( ncclStatus );                                                             \
        }                                                                                   \
    }
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size, rank;
    int h_send[N] = {0};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size != 2) {
        if (rank == 0) printf("This example requires 2 ranks\n");
        MPI_Finalize();
        return -1;
    }

    ncclUniqueId nccl_uid;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    int deviceC=0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceC));
    CUDA_CHECK(cudaSetDevice(rank));  // Assume one GPU per rank
    printf("We see %d devices on rank %d\n", deviceC, rank);

    ncclComm_t nccl_comm;
    NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));

    int *send_buf, *recv_buf;
    CUDA_CHECK(cudaMalloc((void **)&send_buf, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&recv_buf, N*sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (rank == 0) {
        for (int i = 0; i < N; ++i) h_send[i] = i;
        CUDA_CHECK(cudaMemcpyAsync(send_buf, h_send, N*sizeof(int), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
    }


    // NCCL Point-to-point communication (Send/Recv)
    if (rank == 0) {
        NCCL_CALL(ncclSend(send_buf, N, ncclInt, 1, nccl_comm, stream));
    } else if (rank == 1) {
        NCCL_CALL(ncclRecv(recv_buf, N, ncclInt, 0, nccl_comm, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (rank == 1) {
        int h_recv[N];
        CUDA_CHECK(cudaMemcpyAsync(h_recv, recv_buf, N*sizeof(int), cudaMemcpyDeviceToHost, stream));
	    CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("Rank 1 received data:");
        for (int i = 0; i < N; ++i) printf(" %d", h_recv[i]);
        printf("\n");
    }

    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));
    NCCL_CALL(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}