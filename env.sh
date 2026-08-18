# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Example environment setup for CUDA, NCCL, and NVSHMEM
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/path/to/nccl-install
export NVSHMEM_HOME=/path/to/nvshmem-install

export PATH="$CUDA_HOME/bin:$NVSHMEM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NCCL_HOME/lib:$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
