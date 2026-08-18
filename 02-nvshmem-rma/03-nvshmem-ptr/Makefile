NP ?= 2
NVCC ?= nvcc
MPICXX ?= mpicxx
CUDA_HOME ?= /usr/local/cuda
NVSHMEM_HOME ?= /usr/local/nvshmem
JSC_SUBMIT_CMD ?= mpirun
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_ARCH ?= 90
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

CPPFLAGS := -I$(CUDA_HOME)/include -I$(NVSHMEM_HOME)/include
GENCODE_FLAGS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCCFLAGS := -std=c++17 -rdc=true $(GENCODE_FLAGS)
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(NVSHMEM_HOME)/lib
LDLIBS := -lcudart -lnvshmem_host -lnvshmem_device

.PHONY: all clean run run_SOLVED

all: nvshmem_ptr nvshmem_ptr_SOLVED

nvshmem_ptr: nvshmem_ptr.cu
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

nvshmem_ptr_SOLVED: nvshmem_ptr_SOLVED.cu
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: nvshmem_ptr
	$(RUN_ENV) $(LAUNCHER) ./nvshmem_ptr

run_SOLVED: nvshmem_ptr_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./nvshmem_ptr_SOLVED

clean:
	rm -f nvshmem_ptr nvshmem_ptr_SOLVED
