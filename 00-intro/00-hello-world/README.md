# Hello world

Run this first. It does not use NCCL or NVSHMEM: it checks that MPI can start
the requested ranks, each local rank can select a visible GPU, and the CUDA
runtime can create a context.

The program is already complete. Each rank prints:

```text
hello from rank X
```

The order of the lines is not significant. With two ranks, expect one line
for rank 0 and one for rank 1.

## Build and run

This lab needs CUDA, MPI, and at least one visible GPU for each local MPI
rank. It does not need `NCCL_HOME` or `NVSHMEM_HOME`.

```bash
make
make run
```

The default run uses two ranks and leaves GPU visibility to the launcher. On
Jupiter, use:

```bash
make run LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
```

For a local workstation, set `NP` or `CUDA_VISIBLE_DEVICES` if needed.

If this fails, fix the CUDA or MPI environment before moving on to the NCCL
and NVSHMEM labs.
