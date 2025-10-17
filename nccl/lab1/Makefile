# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
NP = 2
MPICC ?= mpicc
SRUN ?= srun --mpi=pspmix --account exalab -N1 -n2 --partition dc-gpu-devel
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
MPICC_FLAGS = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LD_FLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lcudart -ldl -lnccl

nccl_basic: Makefile nccl_basic.c
	$(MPICC) $(MPICC_FLAGS) nccl_basic.c $(LD_FLAGS) -o nccl_basic

.PHONY.: clean
clean:
	rm -f nccl_basic

run: nccl_basic
	CUDA_VISIBLE_DEVICES=0,1 $(JSC_SUBMIT_CMD) -n $(NP) ./$<