# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NP ?= 2
NVCC ?= nvcc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
JSC_SUBMIT_CMD ?= $(MPIRUN)
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_HOME ?= /usr/local/cuda
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

CPPFLAGS += -I$(CUDA_HOME)/include
LDFLAGS += -L$(CUDA_HOME)/lib64
LDLIBS += -lcudart

BIN := hello_world

.PHONY: all clean run

all: $(BIN)

$(BIN): hello_world.cu
	$(NVCC) -ccbin $(MPICXX) -std=c++14 $(CPPFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: $(BIN)
	$(RUN_ENV) $(LAUNCHER) ./$(BIN)

clean:
	rm -f $(BIN)
