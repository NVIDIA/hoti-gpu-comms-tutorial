/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int errors = 0;
    cudaStream_t stream;
    int device_count;
    // Initialize NVSHMEM with MPI_COMM_WORLD
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int rank, nranks;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &nranks);

    if (nranks != 4) {
        if (rank == 0) printf("This example requires 4 ranks\n");
        MPI_Finalize();
        return -1;
    }

    attr.mpi_comm = &mpi_comm;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Device count: %d, rank: %d\n", device_count, rank);
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Set CUDA device and create CUDA stream
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    

    // Get PE number and number of PEs
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // Allocate symmetric memory using nvshmem_malloc
    size_t size = 1024; // bytes
    void *ptr = nvshmem_malloc(size);

    if (ptr == NULL) {
        printf("PE %d: nvshmem_malloc failed\n", mype);
	errors++;
    } else {
        printf("PE %d: Successfully allocated %zu bytes at %p\n", mype, size, ptr);
    }

    // Synchronize for all PEs 
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free the memory
    nvshmem_free(ptr);

    // Destroy the CUDA stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Finalize NVSHMEM and MPI
    nvshmem_finalize();
    MPI_Finalize();

    return (errors);
}
