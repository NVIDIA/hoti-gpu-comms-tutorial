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

# Embed PTX for the highest requested architecture.  This is not a substitute
# for native SASS on the stated target, but it preserves a forward-compatible
# fallback for a newer compatible GPU.
CUDA_PTX_ARCH ?= $(lastword $(CUDA_ARCHS))

GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
ifneq ($(strip $(CUDA_PTX_ARCH)),)
GENCODE_FLAGS += -gencode arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)
endif
