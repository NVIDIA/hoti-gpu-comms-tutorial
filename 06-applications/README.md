# Applications

These are longer examples where communication is part of the algorithm rather
than a standalone call. The Jacobi solver uses NVSHMEM device RMA. The fused
GEMM lab implements the same all-reduce with both NVSHMEM and the NCCL device
API:

| Lab | Dataflow | Communication boundary | Main constraint |
| --- | --- | --- | --- |
| Jacobi solver | Each GPU updates a row partition of a 2D mesh | Exchange top and bottom halo rows with the two neighboring GPUs each iteration | The next iteration must not read ghost rows before the writes arrive |
| Fused GEMM + all-reduce | Each GPU computes one 16x16 output tile, then combines both tiles inside the kernel | NVSHMEM put-with-signal or NCCL LSA barrier and peer loads | Publish the complete local tile before either GPU reads its peer |

The setup is explicit. Host code creates CUDA streams and prepares memory that
the kernels can address remotely. The NVSHMEM examples allocate from the
symmetric heap. The NCCL fused-GEMM version registers symmetric NCCL windows
and creates a device communicator with an LSA barrier. The application still
chooses the barrier, signal, event, or stream dependency required by its
dataflow.

## 1. Jacobi solver

The Jacobi solver splits a 2D mesh by rows. On every iteration, a GPU computes
its interior cells and writes its first and last computed rows into the ghost
rows of the neighboring partitions. The starter begins with per-element device
puts. The follow-on steps add a stream-ordered global barrier, an event-based
L2-norm handoff to MPI, block puts for the halo rows, and an optional
neighbor-only signal protocol. The comparison against a single-GPU calculation
is part of the program, so correctness stays visible as the synchronization
scheme changes.

This lab is intended for one GPU per PE with a CUDA/NVSHMEM installation that
can communicate across the selected GPUs. The Makefile defaults to
`CUDA_ARCH=90` for the GH200 systems used in the tutorial.

## 2. Fused GEMM + all-reduce

The fused GEMM example is a deliberately small version of the pattern. One
CTA computes a 16x16 output tile. The NVSHMEM version exchanges tiles with a
put-with-signal. The NCCL version publishes each tile in a symmetric window,
uses an LSA barrier, and loads both tiles directly through
`ncclGetPeerPointer`.

It is not intended as a tuned collective. It keeps the matrix small enough
that the relationship between computation, communication, and completion is
visible in one source file. Both implementations use paths available on the
GH200 lab systems.

1. [Jacobi solver](01-jacobi-solver)
2. [Fused GEMM and all-reduce](02-fused-gemm-allreduce)

Each directory has a starter and a `_SOLVED` reference. Run the reference on
the intended system first.
