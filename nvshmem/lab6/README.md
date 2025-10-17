# NVSHMEM Python Tutorials

This folder contains Python tutorials and examples for using **NVSHMEM** (NVIDIA SHMEM) for distributed GPU programming. NVSHMEM provides a high-performance, scalable programming model for GPU-to-GPU communication across multiple nodes.

## Overview

NVSHMEM enables efficient one-sided communication between GPUs using Remote Memory Access (RMA) operations. These tutorials demonstrate how to use NVSHMEM4Py, the Python bindings for NVSHMEM, to build distributed GPU applications.

## Prerequisites

- NVIDIA GPU(s) with CUDA support
- CUDA 12.x toolkit
- Python 3.8+
- MPI implementation (e.g., OpenMPI, MPICH)

## Installation

### 0. Setup a virtual environment

```
module load virtualenv
python3 -m virtualenv myenv
source $(PWD)/myenv/bin/activate
```

### 1. Install NVSHMEM Runtime

NVSHMEM is shipped with the Python wheel `libnvidia-nvshmem-cu12`:

```bash
pip install libnvidia-nvshmem-cu12
```

### 2. Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The requirements include:
- `cupy-cuda12x` - GPU-accelerated NumPy-compatible array library
- `numpy` - Numerical computing library
- `mpi4py` - Python bindings for MPI
- `cuda-python==12.9` - Python bindings for CUDA

## Tutorial Files

### 1. `put.py` - One-Sided Communication
- Demonstrates one-sided Remote Memory Access (RMA) operations
- Shows how to use `put` operations to transfer data between GPUs
- Introduces CUDA streams and synchronization primitives

### 2. `put_signal.py` - Advanced Communication
- Advanced tutorial for asynchronous communication using signals
- Demonstrates how to avoid barriers using `put_signal` and `signal_wait`
- Includes a TODO section for hands-on learning

## Running the Tutorials

### Single Node, Multiple GPUs

```bash
make run
```

## Key Concepts

### Processing Elements (PEs)
- Each GPU process is a Processing Element
- PEs are numbered from 0 to n-1
- Use `nvshmem.core.my_pe()` to get current PE ID
- Use `nvshmem.core.n_pes()` to get total number of PEs

### Symmetric Memory
- Memory allocated with `nvshmem.core.array()` is accessible from all PEs
- Each PE sees the same virtual address for the same array
- Enables direct GPU-to-GPU communication without host involvement

### One-Sided Communication
- `put()`: Transfer data from local GPU to remote GPU
- `get()`: Retrieve data from remote GPU to local GPU
- `put_signal()`: Transfer data and signal completion
- No need for explicit receive operations on the target PE

### Teams
- Use `nvshmem.core.Teams.TEAM_WORLD` for global synchronization
- Create custom teams for subset communication patterns

## Best Practices

1. **Always use CUDA streams** for asynchronous operations
2. **Synchronize when needed** using barriers or signals
3. **Free allocated arrays** to prevent memory leaks
4. **Use appropriate data types** for your computation
5. **Profile communication patterns** to optimize performance

## Troubleshooting

### Common Issues

- **Library not found**: Ensure `LD_LIBRARY_PATH` includes NVSHMEM libraries
- **GPU out of memory**: Check symmetric heap size and array allocations
- **Communication errors**: Verify MPI setup and network configuration

### Debugging Tips

- Use `nvshmem.core.barrier()` to synchronize PEs
- Print PE information to verify process mapping
- Check CUDA error codes for GPU-related issues

## Additional Resources

- [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/)
- [NVSHMEM4Py API Reference](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/)
- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)
- [CuPy Documentation](https://docs.cupy.dev/)
