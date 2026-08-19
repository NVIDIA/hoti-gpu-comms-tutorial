# Introduction

Start with the completed [hello-world toolchain check](00-hello-world/). It
verifies MPI and CUDA before the NCCL and NVSHMEM labs add more dependencies.

## The process and GPU model

The C and CUDA examples launch one MPI process per GPU. In these exercises,
the MPI rank, NCCL rank, and NVSHMEM processing element (PE) normally identify
the same process. Each process selects a GPU and queues work on one or more
CUDA streams.

A CUDA stream orders GPU work. If a host-to-device copy, an NCCL call, and a
kernel are queued on the same stream, they execute in that order. The host
does not wait for the NCCL call to finish when it returns; the program
synchronizes only before it needs a result on the CPU. The first two NCCL labs
make that ordering explicit.

## NCCL and NVSHMEM in this tutorial

NCCL starts with host APIs. `ncclSend` and `ncclRecv` are a matched
point-to-point operation: the sender and receiver agree on the transfer.
`ncclAllReduce` is a collective: every rank in the communicator participates.
Both calls enqueue GPU work on a CUDA stream.

NVSHMEM starts from symmetric memory. Each PE allocates a corresponding object
on its symmetric heap. A PE can issue a `put` to another PE's object without
that target posting a receive. The target still needs an ordering or
notification step before it reads the new data. The RMA labs use the same
small two-rank setup to show a host-side put, a put issued by a kernel, and the
`nvshmem_ptr` path when a direct peer mapping is available.

Memory semantics follows immediately. It uses a device put as the baseline,
then compares a barrier, `quiet` with signal/wait, and `put_signal` so the
completion and notification rules stay explicit.

The advanced NCCL labs use symmetric windows. The host allocates and registers
matching operands on every rank, then uses those windows for collectives and
one-sided operations. The NCCL device APIs chapter then creates the device
communicator and window state on the host, passes it into a kernel, and covers
the LSA pointer path separately from the GIN put/signal/wait path.

## Tutorial sequence

0. **Hello world** — a completed CUDA/MPI toolchain check.
1. **NCCL host APIs** — send/receive and all-reduce on a stream.
2. **NVSHMEM RMA** — host puts, device puts, and `nvshmem_ptr`.
3. **Memory semantics** — barrier, quiet, signal/wait, and put-with-signal.
4. **Advanced NCCL features** — symmetric windows and host puts.
5. **NCCL device APIs** — host setup, LSA, and GIN device communication.
6. **Python APIs** — NVSHMEM4Py, NCCL4Py, and Python DSL device APIs.
7. **NCCL contrib and Extensions** — the surrounding ecosystem and an NCCL EP
   exercise.
8. **Applications** — a Jacobi solver and NVSHMEM/NCCL versions of a fused
   GEMM plus all-reduce kernel.

Most exercise directories contain a starter with the plain filename and a
checked reference with `_SOLVED` in its name. Hello world is deliberately
complete, so it is the exception. Use the leaf README for the required
environment, build command, launch command, and expected output. The
[repository README](../README.md) lists the full directory layout. Start with
[Hello world](00-hello-world/), then continue to [NCCL host APIs](../01-nccl-host-apis/).
