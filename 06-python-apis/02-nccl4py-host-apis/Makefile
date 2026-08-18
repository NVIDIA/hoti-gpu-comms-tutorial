# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

PYTHON ?= python3
MPIRUN ?= mpirun
NP ?= 2
LAUNCHER ?= $(MPIRUN) -np $(NP)
NCCL4PY_SOURCE ?=
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))

.PHONY: install check run-allreduce run-allreduce_SOLVED run-send-recv run-send-recv_SOLVED

install:
	test -d "$(NCCL4PY_SOURCE)"
	$(PYTHON) -m pip install -e "$(NCCL4PY_SOURCE)[cu13]"
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) -c 'from mpi4py import MPI; import torch; import nccl.core; print("NCCL4Py environment is importable")'

run-allreduce:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) allreduce.py

run-allreduce_SOLVED:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) allreduce_SOLVED.py

run-send-recv:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) send_recv.py

run-send-recv_SOLVED:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) send_recv_SOLVED.py
