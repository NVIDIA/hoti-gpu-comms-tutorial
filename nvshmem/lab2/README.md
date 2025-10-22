# NVSHMEM Lab 2: Introduction to NVSHMEM

This lab demonstrates the basics of using **NVSHMEM** for distributed GPU programming in C++/CUDA. You will learn how to build and run a simple NVSHMEM application across multiple GPUs.

## Overview

- **nvshmem_basic.cu**: Example code showing basic NVSHMEM initialization, memory allocation, and simple communication between processes.
- **Makefile**: Build and run instructions for the example.

## Prerequisites

- NVIDIA GPUs with CUDA support (compute capability 8.0 or higher recommended)
- CUDA Toolkit (version 12.x)
- NVSHMEM library (pre-installed on most HPC clusters)
- MPI implementation (e.g., OpenMPI, MPICH)
- C++ compiler with MPI support (e.g., `mpic++`)

## Environment Setup

Before compiling and running, make sure to load the required modules and set up your environment. For example:

```bash
# If you haven't, load the environment
source </path/to/this/repo>/env.sh # Make sure this example env is populated correctly.
```

## Building and Running

```bash
# To compile your application
make
```

```bash
# To run your application
make run
```
