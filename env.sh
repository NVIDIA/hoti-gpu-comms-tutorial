# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

if command -v module >/dev/null 2>&1; then
  module load Stages/2026 GCC/14.3.0 CMake/3.31.8 CUDA/13 OpenMPI/5.0.8
fi

export HOTI_ROOT="${HOTI_ROOT:-/e/project1/training2633}"
export CUDA_HOME="${HOTI_CUDA_HOME:-/e/software/default/stages/2026/software/CUDA/13}"
export NCCL_HOME="${HOTI_NCCL_HOME:-$HOTI_ROOT/nvidia/install/nccl}"
export NVSHMEM_HOME="${HOTI_NVSHMEM_HOME:-$HOTI_ROOT/nvidia/install/nvshmem}"

export PATH="$CUDA_HOME/bin:$NVSHMEM_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$NVSHMEM_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for hoti_preload_library in "$NCCL_HOME/lib/libnccl.so.2" "$NVSHMEM_HOME/lib/libnvshmem_host.so.3"; do
  if [ -f "$hoti_preload_library" ]; then
    case ":${LD_PRELOAD:-}:" in
      *":$hoti_preload_library:"*) ;;
      *) export LD_PRELOAD="$hoti_preload_library${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
  fi
done
unset hoti_preload_library
export NCCL_NET="${NCCL_NET:-IB}"
export NVSHMEM_BOOTSTRAP="${NVSHMEM_BOOTSTRAP:-MPI}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-ibrc}"
