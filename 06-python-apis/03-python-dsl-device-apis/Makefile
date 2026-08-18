# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

PYTHON ?= python3
MPIRUN ?= mpirun
NP ?= 2
LAUNCHER ?= $(MPIRUN) -np $(NP)
NCCL4PY_SOURCE ?=
NVIDIA_PYPI ?= https://pypi.nvidia.com
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

.PHONY: install install-nvshmem check-env check-nvshmem-env run run_SOLVED run-nvshmem run-nvshmem_SOLVED

install:
	test -d "$(NCCL4PY_SOURCE)"
	$(PYTHON) -m pip install -e "$(NCCL4PY_SOURCE)[cu13]"
	$(PYTHON) -m pip install cupy-cuda13x mpi4py

install-nvshmem:
	$(PYTHON) -m pip install --extra-index-url $(NVIDIA_PYPI) nvshmem4py-cu13==0.3.1 cupy-cuda13x mpi4py torch

check-env:
	$(PYTHON) check_env.py

check-nvshmem-env:
	$(PYTHON) check_nvshmem_env.py

run:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) ring_put.py

run_SOLVED:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) ring_put_SOLVED.py

run-nvshmem:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) ring_put_nvshmem_cute.py

run-nvshmem_SOLVED:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) ring_put_nvshmem_cute_SOLVED.py
