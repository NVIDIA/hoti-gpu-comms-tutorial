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

CPPFLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
NVCCFLAGS += -O2 -std=c++17 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDLIBS += -lcudart -lnccl

BINS := host_put_symmetric_operands host_put_symmetric_operands_SOLVED

.PHONY: all clean run run_SOLVED

all: $(BINS)

host_put_symmetric_operands: host_put_symmetric_operands.cu
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

host_put_symmetric_operands_SOLVED: host_put_symmetric_operands_SOLVED.cu
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: host_put_symmetric_operands
	$(RUN_ENV) $(LAUNCHER) ./$<

run_SOLVED: host_put_symmetric_operands_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$<

clean:
	rm -f $(BINS)
