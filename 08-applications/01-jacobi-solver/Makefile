# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.

NP ?= 1
NVCC ?= nvcc
MPICXX ?= mpicxx
CUDA_ARCH ?= 90
MPIRUN ?= mpirun
LAUNCHER ?= $(MPIRUN) -np $(NP)
RUN_ARGS ?=

ifneq ($(MAKECMDGOALS),clean)
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set; it must name the NVSHMEM install prefix)
endif
endif

NVSHMEM_LIBDIR ?= $(NVSHMEM_HOME)/lib
GENCODE_FLAGS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCC_FLAGS := -O3 -std=c++17 -lineinfo -rdc=true -ccbin=$(MPICXX) $(GENCODE_FLAGS) -I$(NVSHMEM_HOME)/include
NVCC_LDFLAGS := -rdc=true -ccbin=$(MPICXX) $(GENCODE_FLAGS) -L$(NVSHMEM_LIBDIR) -lnvshmem_host -lnvshmem_device -lcudart -lcuda -ldl

ifeq ($(USE_NVTX),1)
NVCC_FLAGS += -DUSE_NVTX
NVCC_LDFLAGS += -lnvToolsExt
endif

all: jacobi jacobi_SOLVED

jacobi: Makefile jacobi.cu
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -o $@ $(NVCC_LDFLAGS)

jacobi_SOLVED: Makefile jacobi_SOLVED.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_SOLVED.cu -o $@ $(NVCC_LDFLAGS)

run: jacobi
	$(LAUNCHER) ./jacobi $(RUN_ARGS)

run_SOLVED: jacobi_SOLVED
	$(LAUNCHER) ./jacobi_SOLVED $(RUN_ARGS)

.PHONY: all clean run run_SOLVED

clean:
	rm -f jacobi jacobi_SOLVED
