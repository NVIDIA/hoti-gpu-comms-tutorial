# Applications

These are longer examples where communication is part of the algorithm rather
than a standalone call. Both use NVSHMEM from device code, but they make very
different tradeoffs:

| Lab | Dataflow | Communication boundary | Main constraint |
| --- | --- | --- | --- |
| Jacobi solver | Each GPU updates a row partition of a 2D mesh | Exchange top and bottom halo rows with the two neighboring GPUs each iteration | The next iteration must not read ghost rows before the writes arrive |
| Fused GEMM + all-reduce | Each GPU computes one 16x16 output tile and exchanges it with its peer | Device put-with-signal, then a local sum of the two tiles | The signal is the receiver-visible handoff for the tile |

The common setup is explicit. Host code initializes NVSHMEM, creates CUDA
streams, and allocates data that will be addressed remotely from the symmetric
heap. The kernels issue the data movement. Neither a device put nor a signal
makes unrelated stream work or host code safe to consume; the application
chooses the barrier, signal, event, or stream dependency appropriate to its
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
CTA computes a 16x16 output tile. Before that same kernel returns, thread zero
sends the tile to its peer with a device put-with-signal and waits for the
peer's matching handoff. The CTA then adds the received tile to its local tile.

It is not intended as a tuned collective. It keeps the matrix small enough
that the relationship between the computation, the RMA call, and the
completion protocol is visible in one source file. The code targets the normal
device RMA path used on the GH200 lab systems.

1. [Jacobi solver](01-jacobi-solver)
2. [Fused GEMM and all-reduce](02-fused-gemm-allreduce)

Each directory has a starter and a `_SOLVED` reference. Run the reference on
the intended system first.
