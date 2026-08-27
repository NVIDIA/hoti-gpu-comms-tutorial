NP ?= 4
NVCC ?= nvcc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
JSC_SUBMIT_CMD ?= $(MPIRUN)
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
NVSHMEM_HOME ?=
CUDA_ARCH ?= 90
CUDA_VISIBLE_DEVICES ?=
RUN_ARGS ?=
CHUNK_BYTES ?= 256K
NETWORK_CHUNK_BYTES ?= 4M
NETWORK_QPS ?=

ifneq ($(MAKECMDGOALS),clean)
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set; it must name the NVSHMEM install prefix)
endif
endif

NVSHMEM_LIBDIR ?= $(NVSHMEM_HOME)/lib
GENCODE_FLAGS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCC_FLAGS := -O3 -std=c++17 -lineinfo -rdc=true -ccbin=$(MPICXX) \
	$(GENCODE_FLAGS) -I$(NVSHMEM_HOME)/include
NVCC_LDFLAGS := -rdc=true -ccbin=$(MPICXX) $(GENCODE_FLAGS) \
	-L$(NVSHMEM_LIBDIR) -lnvshmem_host -lnvshmem_device -lcudart -lcuda -ldl
RUN_ENV = LD_LIBRARY_PATH=$(NVSHMEM_LIBDIR):$(LD_LIBRARY_PATH) \
	HOTI_ALLTOALLV_CHUNK_BYTES=$(CHUNK_BYTES) \
	HOTI_ALLTOALLV_NETWORK_CHUNK_BYTES=$(NETWORK_CHUNK_BYTES) \
	$(if $(strip $(NETWORK_QPS)),HOTI_ALLTOALLV_NETWORK_QPS=$(NETWORK_QPS)) \
	$(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

BINS := nvshmem_alltoallv nvshmem_alltoallv_SOLVED

.PHONY: all clean run run_SOLVED

all: $(BINS)

nvshmem_alltoallv: nvshmem_alltoallv.cu ../alltoallv_common.hpp \
	../../common/nvshmem_exercise.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(NVCC_LDFLAGS)

nvshmem_alltoallv_SOLVED: nvshmem_alltoallv_SOLVED.cu \
	../alltoallv_common.hpp ../../common/nvshmem_exercise.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(NVCC_LDFLAGS)

run: nvshmem_alltoallv
	$(RUN_ENV) $(LAUNCHER) ./$< $(RUN_ARGS)

run_SOLVED: nvshmem_alltoallv_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$< $(RUN_ARGS)

clean:
	rm -f $(BINS)
