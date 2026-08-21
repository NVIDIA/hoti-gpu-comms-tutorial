# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NP ?= 2
MPICC ?= mpicc
MPIRUN ?= mpirun
JSC_SUBMIT_CMD ?= $(MPIRUN)
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

CPPFLAGS += -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
CFLAGS ?= -O2
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDLIBS += -lcudart -ldl -lnccl

BINS := send_recv_on_stream send_recv_on_stream_SOLVED

.PHONY: all clean run run_SOLVED

all: $(BINS)

send_recv_on_stream: send_recv_on_stream.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

send_recv_on_stream_SOLVED: send_recv_on_stream_SOLVED.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: send_recv_on_stream
	$(RUN_ENV) $(LAUNCHER) ./$<

run_SOLVED: send_recv_on_stream_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./$<

clean:
	rm -f $(BINS)
