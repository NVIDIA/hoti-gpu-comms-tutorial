NP ?= 2
MPICC ?= mpicc
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
JSC_SUBMIT_CMD ?= mpirun
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

CPPFLAGS := -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LDLIBS := -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lcudart -lnccl

.PHONY: all clean run run_SOLVED

all: allreduce_on_stream allreduce_on_stream_SOLVED

allreduce_on_stream: allreduce_on_stream.c
	$(MPICC) $(CPPFLAGS) $< $(LDLIBS) -o $@

allreduce_on_stream_SOLVED: allreduce_on_stream_SOLVED.c
	$(MPICC) $(CPPFLAGS) $< $(LDLIBS) -o $@

run: allreduce_on_stream
	$(RUN_ENV) $(LAUNCHER) ./allreduce_on_stream

run_SOLVED: allreduce_on_stream_SOLVED
	$(RUN_ENV) $(LAUNCHER) ./allreduce_on_stream_SOLVED

clean:
	rm -f allreduce_on_stream allreduce_on_stream_SOLVED
