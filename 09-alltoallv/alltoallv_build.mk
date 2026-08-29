# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# The AlltoAllV exercises target GB200 (SM100) and GB300 (SM103) NVL72
# systems. Keep CUDA_ARCH as a compatibility alias for the other tutorial
# Makefiles, while CUDA_ARCHS permits a teaching-machine fat binary.
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
CUDA_ARCHS ?= $(CUDA_ARCH)
else
CUDA_ARCHS ?= 100 103
endif

# Native SASS covers both stated targets. PTX is opt-in because CUDA device
# linking may discard an embedded PTX image from an RDC executable; use
# CUDA_PTX_ARCH=103 only when a forward-JIT fallback is needed, and inspect the
# final executable with cuobjdump.
CUDA_PTX_ARCH ?=

GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
ifneq ($(strip $(CUDA_PTX_ARCH)),)
GENCODE_FLAGS += -gencode arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)
endif
