# Jacobi solver

This is the end-to-end NVSHMEM device-API example. Jacobi relaxation repeatedly
updates each point of a 2D mesh from its neighbors. The mesh is split by rows
across GPUs, so every GPU owns an interior slab plus a top and bottom ghost
row. After each iteration it sends its newly computed edge rows to the two
neighboring slabs before the next iteration reads those ghosts.

```text
this PE's first computed row -> top neighbor's bottom ghost row
this PE's last computed row  -> bottom neighbor's top ghost row
```

The exercise shows three ways to make that handoff correct: per-thread device
puts with a global stream barrier, block-collective puts, and a neighborhood
signal/wait path. The [NVSHMEM API overview](https://docs.nvidia.com/nvshmem/api/api/overview.html)
and [device-API guidance](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html)
cover the ordering and completion rules behind them.

`jacobi.cu` is the exercise. `jacobi_SOLVED.cu` is the completed version. Both
also run a single-GPU calculation and compare the distributed result against it;
a completed run must report the multi-GPU result as correct rather than only
printing a timing.

## Build

Load CUDA, an MPI implementation, and NVSHMEM first. `NVSHMEM_HOME` must be the NVSHMEM install prefix containing `include/` and `lib/`; `mpicxx` and `nvcc` must be on `PATH`.

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"

# The Makefile defaults to 90 for GH200; override CUDA_ARCH for another GPU.
make CUDA_ARCH=90
make jacobi_SOLVED CUDA_ARCH=90
```

The first command builds the starter as `./jacobi`. The second builds the reference as `./jacobi_SOLVED`. Use `make clean` to remove both binaries. Set `USE_NVTX=1` while building if NVTX ranges are wanted.

## Run

The default launcher is `mpirun -np $(NP)`. A small two-GPU run is a good first check:

```bash
make NP=2 RUN_ARGS="-nx 1024 -ny 1024 -niter 100" run
make NP=2 RUN_ARGS="-nx 1024 -ny 1024 -niter 100" run_SOLVED
```

Use the scheduler's launcher when running in an allocation, for example:

```bash
make LAUNCHER="srun --ntasks=2 --gpus-per-task=1" RUN_ARGS="-nx 4096 -ny 4096 -niter 1000" run
```

Each rank selects its local GPU. Make sure the launcher assigns one visible GPU per rank. The program sizes `NVSHMEM_SYMMETRIC_SIZE` for its two mesh allocations when it is not already set; an explicitly set value must be large enough for the selected mesh.

## Exercise

Start with the default path, which uses per-element device puts in
`jacobi_kernel`. Keep the flags separate at first: each one changes a different
part of the communication/completion path.

1. After launching the Jacobi kernel, add `nvshmemx_barrier_all_on_stream(compute_stream)` so every GPU sees the boundary writes before swapping the two grids.
2. The L2 norm is copied asynchronously. Before the MPI all-reduce reads the previous host buffer, wait for `l2_norm_bufs[prev].copy_done`. The reference uses `cudaEventSynchronize` on that event, rather than synchronizing the whole compute stream.
3. Enable `-use_block_comm` to exercise `jacobi_block_comm_kernel`. Fill in the two `nvshmemx_float_put_nbi_block` calls: send this GPU's first computed row to the top neighbor's bottom ghost row and its last computed row to the bottom neighbor's top ghost row. Preserve the partial-block count at the right edge.
4. `-neighborhood_sync` is the optional version of the synchronization step. Complete `syncneighborhood_kernel` with two signal operations and a wait on the two local signal slots, then use the kernel instead of the global barrier. This is useful because Jacobi only communicates with its two neighbors.

`-norm_overlap` uses two L2-norm buffers and a separate reset stream. Its
existing event dependency keeps the reset copy ahead of the next use without a
host-side stream synchronize. Compare `jacobi.cu` with `jacobi_SOLVED.cu` after
each step, then try `-use_block_comm`, `-neighborhood_sync`, and `-norm_overlap`
individually and together. The final check remains the serial-versus-distributed
comparison printed at the end of the run.
