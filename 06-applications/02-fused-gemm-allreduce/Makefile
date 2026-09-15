NP ?= 2
NVCC ?= nvcc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
JSC_SUBMIT_CMD ?= $(MPIRUN)
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
NVSHMEM_HOME ?= /usr/local/nvshmem
CUDA_ARCH ?= 90
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

CUDA_LIBDIR ?= $(CUDA_HOME)/lib64
NCCL_LIBDIR ?= $(NCCL_HOME)/lib
NVSHMEM_LIBDIR ?= $(NVSHMEM_HOME)/lib

NVCCFLAGS += -O2 -std=c++17 -rdc=true -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
CUDA_CPPFLAGS = -I$(CUDA_HOME)/include
CUDA_LDFLAGS = -L$(CUDA_LIBDIR)
NVSHMEM_CPPFLAGS = -I$(NVSHMEM_HOME)/include
NVSHMEM_LDFLAGS = -L$(NVSHMEM_LIBDIR)
NVSHMEM_LDLIBS = -lnvshmem_host -lnvshmem_device -lcudart -lcuda -ldl
NCCL_CPPFLAGS = -I$(NCCL_HOME)/include
NCCL_LDFLAGS = -L$(NCCL_LIBDIR)
NCCL_LDLIBS = -lnccl -lcudart -lcuda

NVSHMEM_BINS := fused_gemm_allreduce fused_gemm_allreduce_SOLVED
NCCL_BINS := fused_gemm_allreduce_nccl fused_gemm_allreduce_nccl_SOLVED
BINS := $(NVSHMEM_BINS) $(NCCL_BINS)

.PHONY: all nvshmem nccl clean run run_SOLVED run_nvshmem \
        run_nvshmem_SOLVED run_nccl run_nccl_SOLVED

all: nvshmem nccl

nvshmem: $(NVSHMEM_BINS)

nccl: $(NCCL_BINS)

fused_gemm_allreduce: fused_gemm_allreduce.cu ../../09-alltoallv/common/nvshmem_exercise.h
	$(NVCC) -ccbin $(MPICXX) $(CUDA_CPPFLAGS) $(NVSHMEM_CPPFLAGS) $(NVCCFLAGS) $< $(CUDA_LDFLAGS) $(NVSHMEM_LDFLAGS) $(NVSHMEM_LDLIBS) -o $@

fused_gemm_allreduce_SOLVED: fused_gemm_allreduce_SOLVED.cu ../../09-alltoallv/common/nvshmem_exercise.h
	$(NVCC) -ccbin $(MPICXX) $(CUDA_CPPFLAGS) $(NVSHMEM_CPPFLAGS) $(NVCCFLAGS) $< $(CUDA_LDFLAGS) $(NVSHMEM_LDFLAGS) $(NVSHMEM_LDLIBS) -o $@

fused_gemm_allreduce_nccl: fused_gemm_allreduce_nccl.cu ../../05-nccl-device-apis/device_api_common.hpp
	$(NVCC) -ccbin $(MPICXX) $(CUDA_CPPFLAGS) $(NCCL_CPPFLAGS) $(NVCCFLAGS) $< $(CUDA_LDFLAGS) $(NCCL_LDFLAGS) $(NCCL_LDLIBS) -o $@

fused_gemm_allreduce_nccl_SOLVED: fused_gemm_allreduce_nccl_SOLVED.cu ../../05-nccl-device-apis/device_api_common.hpp
	$(NVCC) -ccbin $(MPICXX) $(CUDA_CPPFLAGS) $(NCCL_CPPFLAGS) $(NVCCFLAGS) $< $(CUDA_LDFLAGS) $(NCCL_LDFLAGS) $(NCCL_LDLIBS) -o $@

run: run_nvshmem

run_SOLVED: run_nvshmem_SOLVED

run_nvshmem: fused_gemm_allreduce
	$(RUN_ENV) $(LAUNCHER) ./$<

run_nvshmem_SOLVED: fused_gemm_allreduce_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$<

run_nccl: fused_gemm_allreduce_nccl
	$(RUN_ENV) $(LAUNCHER) ./$<

run_nccl_SOLVED: fused_gemm_allreduce_nccl_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$<

clean:
	rm -f $(BINS)
