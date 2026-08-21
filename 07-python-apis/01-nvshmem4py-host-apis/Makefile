# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

PYTHON ?= python3
MPIRUN ?= mpirun
NP ?= 2
JSC_SUBMIT_CMD ?= $(MPIRUN)
LAUNCHER ?= $(JSC_SUBMIT_CMD) -n $(NP)
CUDA_VISIBLE_DEVICES ?=
RUN_ENV = $(if $(strip $(CUDA_VISIBLE_DEVICES)),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES))
NVIDIA_PYPI ?= https://pypi.nvidia.com

.PHONY: install run run-put run-put_SOLVED run-put-signal run-put-signal_SOLVED

install:
	$(PYTHON) -m pip install --extra-index-url $(NVIDIA_PYPI) -r requirements.txt

run: run-put

run-put:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) put.py

run-put_SOLVED:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) put_SOLVED.py

run-put-signal:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) put_signal.py

run-put-signal_SOLVED:
	$(RUN_ENV) $(LAUNCHER) $(PYTHON) put_signal_SOLVED.py
