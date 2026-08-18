# Advanced NCCL features

The host-API chapter uses a communicator plus ordinary device pointers. That is
the right model for most `ncclAllReduce`, `ncclSend`, and `ncclRecv` calls. In
this chapter, NCCL is also given a description of the memory that a program
will communicate through. That description is a **window**.

A window is not a pointer. It is an NCCL handle for a registered allocation.
When every rank makes a compatible registration with
`NCCL_WIN_COLL_SYMMETRIC`, NCCL can use the same window-and-offset description
on every rank even though the CUDA virtual addresses differ. This is the basis
for the window collective in the first lab and host-side RMA in the second.

Start with the [NCCL window-registration documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-registration).
It covers the allocation requirements and the meaning of symmetric windows.

## Window lifetime

The same lifetime rule applies in both labs: a window stays registered
until no stream operation or CUDA kernel can access it. The sequence is:

```text
create communicator on every rank
allocate VMM-compatible GPU memory with ncclMemAlloc
register each allocation with ncclCommWindowRegister
enqueue NCCL work on a stream
synchronize the work
deregister the window
free the allocation and destroy the communicator
```

`ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC` is a collective setup
step: the communicator's ranks all participate with compatible allocations and
offsets. The registrations in a given process are represented by local
`ncclWindow_t` handles. Do not pass a handle from one process to another, and
do not free its allocation before `ncclCommWindowDeregister`.

## What changes in each lab

| Lab | Host work | GPU communication | What to look for |
| --- | --- | --- | --- |
| [1. Register a symmetric window](01-register-symmetric-window) | Register source and destination buffers around a normal collective. | NCCL executes `ncclAllGather` on the stream. | The collective still has its usual all-gather semantics; registration changes how NCCL can access its operands. |
| [2. Host put with symmetric operands](02-host-put-symmetric-operands) | Queue `ncclPutSignal` and `ncclWaitSignal` on a CUDA stream. | NCCL writes a peer's registered window, then updates its signal. | A put needs no `ncclRecv`, but the receiver must wait for the signal before consuming the payload. |

## Host-side RMA in one picture

The host RMA lab uses `ncclPutSignal`, not a paired send and receive:

```text
rank 0 stream:  ncclPutSignal(source, rank 1, rank-1 receive window) ───►
rank 1 stream:  ncclWaitSignal(from rank 0) ───► dependent copy or kernel
```

The CPU call only enqueues work. When the put has completed on the initiating
stream, its local source can be reused. When the matching wait has completed
on the target stream, the received data is visible to later work on that
stream. The [one-sided RMA API reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#one-sided-point-to-point-operations-rma)
defines those guarantees, including the current `sigIdx`, `ctx`, and `flags`
restrictions used by the exercise.

## Related chapters

The [Memory semantics](../03-memory-semantics/) chapter introduced the
ordering, completion, and notification questions through NVSHMEM. NCCL
windows and `ncclPutSignal` express the same kind of handoff through a
different API surface.

The [NCCL device APIs](../05-nccl-device-apis/) chapter then uses registered
symmetric memory in a CUDA kernel. It covers the host work needed to create an
`ncclDevComm`, then separates direct LSA access from GIN put and signal
operations.

## Before running

The window lab checks for NCCL 2.27.6 or newer. The RMA lab checks for NCCL
2.29 or newer, uses one visible GPU per MPI rank, and prints `SKIP` for an
unsupported configuration. The device APIs chapter documents its additional
CUDA, GPU, NIC, and topology requirements separately. Follow the printed
reason first, then consult the linked documentation rather than attempting to
force an unsupported configuration.
