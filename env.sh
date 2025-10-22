# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Example environment setup for NCCL and NVSHMEM
export NVSHMEM_HOME=/path/to/nvshmem/build/lib
export NCCL_HOME=/path/to/nccl-src/build/
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NVSHMEM_HOME/bin:$PATH
export CPATH=$NCCL_HOME/build:$NVSHMEM_HOME/include:$CPATH
