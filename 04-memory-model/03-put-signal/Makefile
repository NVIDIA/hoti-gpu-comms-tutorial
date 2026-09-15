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

all: put_signal put_signal_SOLVED

put_signal: put_signal.cu ../../09-alltoallv/common/nvshmem_exercise.h
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

put_signal_SOLVED: put_signal_SOLVED.cu ../../09-alltoallv/common/nvshmem_exercise.h
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: put_signal
	$(RUN_ENV) $(LAUNCHER) ./put_signal

run_SOLVED: put_signal_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./put_signal_SOLVED

clean:
	rm -f put_signal put_signal_SOLVED
