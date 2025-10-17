# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NP ?= 4
NVCC=nvcc
MPICXX=mpicxx

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS	:= $(GENCODE_SM80)
NVCC_FLAGS = -Xptxas --optimize-float-atomics
NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++14
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++14
LD_FLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lcudart -ldl -lnccl

jacobi_solved: Makefile jacobi.cpp jacobi_kernel.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernel.o $(LD_FLAGS) -o jacobi_solved

jacobi: Makefile jacobi_unsolved.cpp jacobi_kernel.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi_unsolved.cpp jacobi_kernel.o $(LD_FLAGS) -o jacobi

jacobi_kernel.o: Makefile jacobi_kernel.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernel.cu -c

.PHONY.: clean
clean:
	rm -f jacobi jacobi_solved jacobi_kernel.o

run_solved: EXE=jacobi_solved
run_solved: jacobi_solved
run: EXE=jacobi
run: jacobi
run run_solved:
	CUDA_VISIBLE_DEVICES=0,1,2,3 $(JSC_SUBMIT_CMD) -n $(NP) ./${EXE}