# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
# This needs to be done to make sure nvJitLink 12.8 dependency is respected from CUDA 12.8 shipped with NVHPCSDK
CUDA_HOME = $(NVHPC)/Linux_x86_64/$(EBVERSIONNVHPC)/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
MPIRUN ?= mpirun
#CUDA_HOME ?= /p/software/default/stages/2025/software/NVHPC/25.5-CUDA-12/Linux_x86_64/25.5/cuda/
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set)
endif

GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_FLAGS = -Xptxas --optimize-float-atomics

NVCC_FLAGS += -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl $(GENCODE_FLAGS) -std=c++14 -I$(NVSHMEM_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(CUDA_HOME)/lib64 -lcuda -lcudart -ldl -lnvidia-ml
ifdef USE_LTO
	NVCC_FLAGS += -maxrregcount=32
	NVCC_LDFLAGS += -maxrregcount=32 -dlto
endif
jacobi: Makefile jacobi_unsolved.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_unsolved.cu -c -o jacobi.o
	$(NVCC) $(GENCODE_FLAGS) jacobi.o -o jacobi $(NVCC_LDFLAGS)

jacobi_solved: Makefile jacobi.cu
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -c -o jacobi.o
	$(NVCC) $(GENCODE_FLAGS) jacobi.o -o jacobi_solved $(NVCC_LDFLAGS)


.PHONY.: clean
clean:
	rm -f jacobi jacobi.o jacobi_solved 

run_solved: EXE=jacobi_solved
run_solved: jacobi_solved
run: EXE=jacobi
run: jacobi
run run_solved:
	CUDA_VISIBLE_DEVICES=0,1,2,3 $(JSC_SUBMIT_CMD) -n $(NP) ./${EXE}