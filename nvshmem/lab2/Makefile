# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NP = 4
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
NVSHMEM_HOME ?= /p/software/default/stages/2025/software/NVHPC/25.5-CUDA-12/Linux_x86_64/25.5/comm_libs/nvshmem/
MPICC ?= mpicc
GENCODE_FLAGS ?= -gencode arch=compute_80,code=sm_80

MPICC_FLAGS = -I$(CUDA_HOME)/include -I$(NVSHMEM_HOME)/include
LDFLAGS = -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -L$(CUDA_HOME)/lib64 -lcuda -lcudart 

nvshmem_basic: Makefile nvshmem_basic.cu
	nvcc -ccbin mpic++ $(MPICC_FLAGS) $(GENCODE_FLAGS) nvshmem_basic.cu $(LDFLAGS) -o nvshmem_basic

.PHONY.: clean
clean:
	rm -f nvshmem_basic

run: nvshmem_basic
	CUDA_VISIBLE_DEVICES=0,1,2,3 $(JSC_SUBMIT_CMD) -n $(NP) ./$<