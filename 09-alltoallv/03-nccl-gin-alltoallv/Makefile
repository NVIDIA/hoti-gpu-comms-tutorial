NP ?= 2
NVCC ?= nvcc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
JSC_SUBMIT_CMD ?= $(MPIRUN)
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
include ../alltoallv_build.mk
CUDA_VISIBLE_DEVICES ?=
RUN_ARGS ?=
NCCL_IB_MERGE_NICS ?= 0
NCCL_CROSS_NIC ?= 1

RUN_ENV = LD_LIBRARY_PATH=$(NCCL_HOME)/lib:$(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH) \
	NCCL_IB_MERGE_NICS=$(NCCL_IB_MERGE_NICS) \
	NCCL_CROSS_NIC=$(NCCL_CROSS_NIC) \
	$(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))
CPPFLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
NVCCFLAGS += -O3 -std=c++17 -lineinfo \
	$(GENCODE_FLAGS)
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDLIBS += -lcudart -lnccl

BINS := nccl_gin_alltoallv nccl_gin_alltoallv_SOLVED

.PHONY: all clean run run_SOLVED

all: $(BINS)

nccl_gin_alltoallv: nccl_gin_alltoallv.cu ../alltoallv_common.hpp ../nccl_alltoallv_setup.hpp
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

nccl_gin_alltoallv_SOLVED: nccl_gin_alltoallv_SOLVED.cu ../alltoallv_common.hpp ../nccl_alltoallv_setup.hpp
	$(NVCC) -ccbin $(MPICXX) $(CPPFLAGS) $(NVCCFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: nccl_gin_alltoallv
	$(RUN_ENV) $(LAUNCHER) ./$< $(RUN_ARGS)

run_SOLVED: nccl_gin_alltoallv_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$< $(RUN_ARGS)

clean:
	rm -f $(BINS)
