# Host put with symmetric operands

This two-rank lab uses NCCL's host-side one-sided RMA API. Each rank has a
16-element send buffer and receive buffer. Rank 0 puts its payload into rank
1's receive window, while rank 1 puts its payload into rank 0's receive
window.

Unlike `ncclSend` and `ncclRecv`, there is no receive operation to post at the
target. The destination allocation must be registered before the put is
queued, and the target has to wait for its remote completion signal before it
uses the bytes.

```text
rank 0: Put rank-0 values into rank-1 receive window ──► signal rank 1
rank 1: Put rank-1 values into rank-0 receive window ──► signal rank 0
rank 0: Wait for rank-1 signal, then copy/verify rank-1 values
rank 1: Wait for rank-0 signal, then copy/verify rank-0 values
```

The [NCCL one-sided point-to-point reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#one-sided-point-to-point-operations-rma)
defines the important ordering rule: when `ncclWaitSignal` has completed on a
CUDA stream, the corresponding remote data is visible to work that follows it
on that stream.

## What the code sets up

Both send and receive allocations come from `ncclMemAlloc` and are registered
with `NCCL_WIN_COLL_SYMMETRIC`. The lab registers the source as well as the
target, so the local buffer supplied to `ncclPutSignal` is part of the same
symmetric-operand setup. The peer's `recv_window` tells NCCL which registered
target allocation to write, and byte offset `0` selects the beginning of that
window.

The solved code queues these operations on one nonblocking CUDA stream:

```cpp
ncclPutSignal(device_send, kElements, ncclInt, destination, recv_window, 0,
              kSignalIndex, kContext, 0, comm, stream);
ncclWaitSignal(1, &wait_desc, comm, stream);
cudaMemcpyAsync(host_recv, device_recv, ..., cudaMemcpyDeviceToHost, stream);
```

The host calls return after the work has been enqueued. Do not overwrite
`device_send` until the stream has completed the put, and do not inspect
`device_recv` until the same stream has completed the wait and the following
copy.

## Files and exercise

- `host_put_symmetric_operands.cu` is the starter.
- `host_put_symmetric_operands_SOLVED.cu` is the checked reference.

At the marked location, enqueue:

1. `ncclPutSignal` to the other rank, with the local send pointer, the peer's
   receive window, signal index `0`, context `0`, and flags `0`.
2. A single `ncclWaitSignalDesc_t` for the other rank with one expected
   signal, index `0`, and context `0`.

Leave the device-to-host copy after the wait on the same stream. Moving it
before the wait turns the result into a race with the remote write.

## Build and run

```bash
make
make run_SOLVED
```

The default Makefile launch uses exactly two MPI ranks and leaves GPU
visibility to the launcher. Set `LAUNCHER`, `CUDA_HOME`, `NCCL_HOME`, or
`CUDA_ARCH` for the local system. On Jupiter, use
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. A successful reference run
reports that each rank received its predecessor's range: rank 0 receives
`100..115`, and rank 1 receives `0..15`.

The program prints `SKIP` when the headers or runtime are older than NCCL 2.29
or when it cannot assign one GPU to each rank. Host RMA also has CUDA and
platform requirements documented in the [device-initiated communication guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#requirements).
