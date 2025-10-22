# NCCL Lab 3: Jacobi Iteration with Multi-GPU Communication

This lab implements a parallel Jacobi iteration solver using NCCL for distributed computing and CUDA for GPU acceleration. The solver uses NCCL (NVIDIA Collective Communications Library) for efficient multi-GPU communication.

## Project Overview

The Jacobi iteration method is an iterative algorithm for solving systems of linear equations. This implementation:
- Distributes the computational grid across multiple MPI processes
- Uses CUDA kernels for GPU-accelerated computation
- Implements boundary exchange between neighboring processes using NCCL
- Supports both single and double precision floating-point arithmetic

## Prerequisites

Before running this lab, ensure you have:

- **CUDA Toolkit** (version 11.0 or later)
- **MPI implementation** (OpenMPI or MPICH)
- **NCCL library** (NVIDIA Collective Communications Library)
- **C++ compiler** with C++14 support
- **NVTX** (NVIDIA Tools Extension) for profiling (optional)

## Environment Setup

### 1. Source the Environment
```bash
# If you haven't, load the environment
source </path/to/this/repo>/env.sh # Make sure this example env is populated correctly.
```
This script sets up the necessary environment variables for:
- CUDA paths
- NCCL library paths
- MPI configuration
- Compiler flags

### 2. Verify Dependencies
Check that the following are available in your environment:
```bash
nvcc --version          # CUDA compiler
mpicxx --version        # MPI C++ compiler
echo $CUDA_HOME         # Should point to CUDA installation
echo $NCCL_HOME         # Should point to NCCL installation
```

## Project Structure

```
lab3/
├── README.md              # This file
├── Makefile               # Build configuration
├── jacobi_unsolved.cpp    # Main application (student implementation)
├── jacobi.cpp             # Reference solution
└── jacobi_kernel.cu      # CUDA kernels for GPU computation
```

## Compilation

### Build Options

The Makefile provides several build targets:

- **`jacobi`** - Builds the student implementation from `jacobi_unsolved.cpp`
- **`jacobi_solved`** - Builds the reference solution from `jacobi.cpp`

### Compilation Commands

```bash
# Build the student implementation
make jacobi

# Build the reference solution
make jacobi_solved

# Clean build artifacts
make clean
```

### Compiler Flags

The build system uses the following flags:
- **CUDA**: `-gencode arch=compute_80,code=sm_80` (for Ampere GPUs)
- **C++**: `-std=c++14` with optimization flags
- **MPI**: Includes CUDA and NCCL headers
- **Linking**: Links against CUDA runtime, NCCL, and MPI libraries

## Running the Application

### Basic Execution

```bash
# Run the student implementation
make run

# Run the reference solution
make run_solved
```

### Configuration Options

You can customize the execution by modifying these variables:

- **`NP`** - Number of MPI processes (default: 4)
- **`CUDA_VISIBLE_DEVICES`** - GPU devices to use (default: 0,1,2,3)

Example with custom settings:
```bash
NP=8 make run                    # Use 8 MPI processes
CUDA_VISIBLE_DEVICES=0,1 make run  # Use only GPUs 0 and 1
```

### Expected Output

The application will:
1. Initialize the computational grid
2. Perform Jacobi iterations with boundary exchanges
3. Display convergence information
4. Report final L2 norm and execution time

## Implementation Details

### Algorithm

1. **Grid Distribution**: The 2D grid is distributed across MPI processes
2. **Boundary Initialization**: Sine wave boundary conditions are set
3. **Iterative Solver**: 
   - Compute new values using 5-point stencil
   - Exchange boundary data between processes
   - Calculate convergence criteria
4. **Convergence Check**: Continue until L2 norm is below threshold

### Key Components

- **`jacobi_kernel.cu`**: CUDA kernels for GPU computation
- **MPI Communication**: Boundary exchange between neighboring processes
- **NCCL Integration**: Efficient GPU-to-GPU communication
- **NVTX Profiling**: Performance analysis and visualization

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   ```bash
   # Ensure environment is sourced
   source $PROJECT_training2537/env.sh
   ```

2. **CUDA Compilation Errors**
   - Verify CUDA installation: `nvcc --version`
   - Check `$CUDA_HOME` environment variable

3. **MPI Runtime Errors**
   - Ensure MPI is properly installed: `mpicxx --version`
   - Check that the number of processes matches available resources

4. **NCCL Library Issues**
   - Verify NCCL installation: `ls $NCCL_HOME/lib`
   - Check that NCCL version is compatible with CUDA version

### Performance Tuning

- **Block Dimensions**: Adjust CUDA kernel block sizes in `jacobi_kernel.cu`
- **Process Count**: Match `NP` to available GPU count
- **Memory Allocation**: Monitor GPU memory usage during execution

## Development Workflow

1. **Implement** your solution in `jacobi_unsolved.cpp`
2. **Compile** with `make jacobi`
3. **Test** with `make run`
4. **Compare** results with reference solution: `make run_solved`
5. **Profile** using NVTX markers and NVIDIA tools

## Additional Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [MPI Standard](https://www.mpi-forum.org/docs/)
- [NVTX User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## Support

For technical issues or questions:
- Check the troubleshooting section above
- Review the reference implementation in `jacobi.cpp`
- Consult the CUDA and NCCL documentation
- Contact your lab instructor or TA
