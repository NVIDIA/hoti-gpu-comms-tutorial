NP ?= 2
NVCC ?= nvcc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
LAUNCHER ?= $(MPIRUN) -np $(NP)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
CUDA_ARCH ?= 90
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

CPPFLAGS += -I.. -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
NVCCFLAGS += -O2 -std=c++17 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDLIBS += -lcudart -lnccl

BINS := lsa_device_api lsa_device_api_SOLVED

.PHONY: all clean run run_SOLVED

all: $(BINS)

lsa_device_api: lsa_device_api.cu ../device_api_common.hpp
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) lsa_device_api.cu $(LDFLAGS) $(LDLIBS) -o $@

lsa_device_api_SOLVED: lsa_device_api_SOLVED.cu ../device_api_common.hpp
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) lsa_device_api_SOLVED.cu $(LDFLAGS) $(LDLIBS) -o $@

run: lsa_device_api
	$(RUN_ENV) $(LAUNCHER) ./$<

run_SOLVED: lsa_device_api_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$<

clean:
	rm -f $(BINS)
