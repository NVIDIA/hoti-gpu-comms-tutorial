# NCCL Lab 5: NCCL Symmetric APIs

This lab demonstrates the NCCL Symmetric API by performing an `ncclAllGather` using
buffers registered as symmetric windows. You will build and run two programs:

- `nccl_symmetric` — assignment version (`nccl_symmetric_UNSOLVED.cpp`)
- `nccl_symmetric_solved` — reference solution (`nccl_symmetric.cpp`)


## Directory

```
lab5/
├── README.md
├── Makefile
├── nccl_symmetric_UNSOLVED.cpp   # Fill in the symmetric registration + AllGather
└── nccl_symmetric.cpp            # Completed reference
```

## Requirements

- NVIDIA GPUs (1 CUDA device per MPI rank)
- CUDA Toolkit 11.x or newer
- NCCL ≥ 2.27.6 (required for Symmetric API)
- MPI (Open MPI, MPICH, or equivalent)
- C++14-capable compiler and CUDA runtime libraries on the host


```bash
# If you haven't, load the environment
source </path/to/this/repo>/env.sh # Make sure this example env is populated correctly.
```

## Build

```bash
# Assignment build
make nccl_symmetric

# Reference solution build
make nccl_symmetric_solved

# Cleanup
make clean
```

## What the program does

Across `size` MPI ranks (one GPU per rank), each rank:

- Allocates a send buffer (`src`) and a receive buffer (`dst`) with `ncclMemAlloc`
- Registers both buffers as symmetric windows with `ncclCommWindowRegister`
- Runs `ncclAllGather(src -> dst)` so that `dst` contains every rank's `src`
- Verifies the gathered bytes and reports success per rank
- Deregisters the windows and frees memory

The UNSOLVED file contains TODOs to add the symmetric window registration, the
`ncclAllGather` using those windows, and window deregistration.

## Run

You can run directly with `mpirun` or use the Makefile convenience targets. Ensure the
number of MPI ranks matches the number of visible GPUs on each node.

```bash
# Example: 4 ranks on a single node with 4 GPUs
mpirun -np 4 ./nccl_symmetric

# Solution binary
mpirun -np 4 ./nccl_symmetric_solved

# Convenience target
make run_solved
make run
```

The Makefile also provides `run` and `run_solved` targets. `NP` controls the number of
MPI processes; `CUDA_VISIBLE_DEVICES` defaults to `0,1,2,3` in the recipe.

```bash
# Assignment run via Makefile
make run NP=4

# Solution run via Makefile
make run_solved NP=4
```

