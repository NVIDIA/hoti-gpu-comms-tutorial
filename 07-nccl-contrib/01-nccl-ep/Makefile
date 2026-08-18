PYTHON ?= python3
MPIEXEC ?= mpirun
LAUNCHER ?= $(MPIEXEC) -np $(NP)
MAKE_CMD ?= make
NP ?= 4
JOBS ?= 8
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

NCCL_EXTENSIONS_SRC ?=
NCCL_HOME ?= $(NCCL_EXTENSIONS_SRC)/third_party/nccl/build
CUDA_HOME ?= /usr/local/cuda
MPI_HOME ?=
COMPUTE_CAP ?= 90
NVCC_GENCODE ?= -gencode=arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP)
EP_TEST_ARGS ?= -a ll -t 32 -d 2048

NCCL_EP_SRC = $(NCCL_EXTENSIONS_SRC)/nccl_ep
NCCL_EXTENSIONS_PYTHON = $(NCCL_EXTENSIONS_SRC)/python

.PHONY: all help check check-src nccl ep install-python check-env run run_SOLVED

all: check nccl ep install-python

help:

	@echo "Set NCCL_EXTENSIONS_SRC, CUDA_HOME, MPI_HOME, and optionally NCCL_HOME."
	@echo "Run make nccl, make ep, make install-python, then make run_SOLVED or make run."

check:

	$(PYTHON) -m py_compile ep_test.py ep_test_SOLVED.py

check-src:

	@test -n "$(NCCL_EXTENSIONS_SRC)" || { echo "Set NCCL_EXTENSIONS_SRC to the nccl-extensions checkout."; exit 2; }
	@test -f "$(NCCL_EP_SRC)/Makefile" || { echo "NCCL EP source not found under $(NCCL_EXTENSIONS_SRC)."; exit 2; }
	@test -f "$(NCCL_EXTENSIONS_SRC)/third_party/nccl/makefiles/common.mk" || { echo "Initialize the nccl-extensions submodules with git submodule update --init --recursive."; exit 2; }

nccl: check-src

	$(MAKE_CMD) -C "$(NCCL_EP_SRC)" -j"$(JOBS)" nccl-submodule \
		CUDA_HOME="$(CUDA_HOME)" NVCC_GENCODE="$(NVCC_GENCODE)"

ep: check-src

	@test -f "$(NCCL_HOME)/include/nccl.h" || { echo "NCCL is not built under $(NCCL_HOME); run make nccl or set NCCL_HOME."; exit 2; }
	$(MAKE_CMD) -C "$(NCCL_EP_SRC)" -j"$(JOBS)" \
		MPI=1 MPI_HOME="$(MPI_HOME)" CUDA_HOME="$(CUDA_HOME)" \
		NCCL_HOME="$(NCCL_HOME)" NCCL_EP_BUILDDIR="$(NCCL_HOME)" \
		NVCC_GENCODE="$(NVCC_GENCODE)"

install-python: ep

	@mkdir -p "$(NCCL_EXTENSIONS_PYTHON)/nccl/ep/lib"
	cp -L "$(NCCL_HOME)/lib/libnccl_ep.so" "$(NCCL_EXTENSIONS_PYTHON)/nccl/ep/lib/libnccl_ep.so"
	$(PYTHON) -m pip install -r requirements.txt
	CUDA_HOME="$(CUDA_HOME)" $(PYTHON) -m pip install -e "$(NCCL_EXTENSIONS_PYTHON)[cu13]"

check-env: check

	$(PYTHON) -c 'import mpi4py, nccl.core, nccl.ep, numpy'

run: check-env

	LD_LIBRARY_PATH="$(NCCL_HOME)/lib:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH" \
	PATH="$(CUDA_HOME)/bin:$(MPI_HOME)/bin:$$PATH" \
	$(RUN_ENV) $(LAUNCHER) "$(PYTHON)" ep_test.py $(EP_TEST_ARGS)

run_SOLVED: check-env

	LD_LIBRARY_PATH="$(NCCL_HOME)/lib:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH" \
	PATH="$(CUDA_HOME)/bin:$(MPI_HOME)/bin:$$PATH" \
	$(RUN_ENV) $(LAUNCHER) "$(PYTHON)" ep_test_SOLVED.py $(EP_TEST_ARGS)
