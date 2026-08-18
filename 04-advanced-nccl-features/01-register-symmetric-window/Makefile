# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NP ?= 4
MPICXX ?= mpicxx
MPIRUN ?= mpirun
LAUNCHER ?= $(MPIRUN) -np $(NP)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))
ARGS ?=

CPPFLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++14
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDLIBS += -lcudart -ldl -lnccl

BINS := register_symmetric_window register_symmetric_window_SOLVED

.PHONY: all clean run run_SOLVED

all: $(BINS)

register_symmetric_window: register_symmetric_window.cpp
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

register_symmetric_window_SOLVED: register_symmetric_window_SOLVED.cpp
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: register_symmetric_window
	$(RUN_ENV) $(LAUNCHER) ./$< $(ARGS)

run_SOLVED: register_symmetric_window_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$< $(ARGS)

clean:
	rm -f $(BINS)
