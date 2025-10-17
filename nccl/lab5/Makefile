# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NP ?= 4
NVCC=nvcc
MPICXX=mpicxx
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
MPICXX_FLAGS = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++14
LD_FLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lcudart -ldl -lnccl

nccl_symmetric: Makefile nccl_symmetric_UNSOLVED.cpp
	$(MPICXX) $(MPICXX_FLAGS) nccl_symmetric_UNSOLVED.cpp -c nccl_symmetric_UNSOLVED.o
	$(MPICXX) $(MPICXX_FLAGS) nccl_symmetric_UNSOLVED.o -o nccl_symmetric $(LD_FLAGS)
nccl_symmetric_solved: Makefile nccl_symmetric.cpp
	$(MPICXX) $(MPICXX_FLAGS) nccl_symmetric.cpp -c nccl_symmetric.o
	$(MPICXX) $(MPICXX_FLAGS) nccl_symmetric.o -o nccl_symmetric_solved $(LD_FLAGS)

.PHONY.: clean
clean:
	rm -f nccl_symmetric_UNSOLVED.o nccl_symmetric.o nccl_symmetric

run_solved: EXE=nccl_symmetric_solved
run_solved: nccl_symmetric_solved
run: EXE=nccl_symmetric
run: nccl_symmetric
run run_solved:
	CUDA_VISIBLE_DEVICES=0,1,2,3 $(JSC_SUBMIT_CMD) -n $(NP) ./$<