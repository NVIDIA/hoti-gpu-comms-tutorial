# NVSHMEM Lab 4: Distributed Jacobi Solver

This lab demonstrates how to use **NVSHMEM** for distributed-memory parallelization of the Jacobi iterative solver on multiple GPUs. You will learn how to build, run, and experiment with a multi-GPU Jacobi solver using NVSHMEM and MPI.

## Overview

- **jacobi_unsolved.cu**: Starter code for the Jacobi solver (to be completed as an exercise).
- **jacobi.cu**: Complete solution for the Jacobi solver using NVSHMEM.
- **Makefile**: Build and run instructions for both the unsolved and solved versions.

## Environment Setup

Before compiling and running, make sure to load the required modules and set up your environment. For example:

```bash
# Setting up your environment (if not already done before)
source $PROJECT_training2537/env.sh
```

## Compile and Run

```bash
# To compile your application
make jacobi 
```

```bash
# To compile the reference solution
make jacobi_solved
```

```bash
# To run your application
make run
```

```bash
# To run the reference solution
make run_solved
```

