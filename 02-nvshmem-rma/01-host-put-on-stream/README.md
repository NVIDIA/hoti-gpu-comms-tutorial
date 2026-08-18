# Host put on a stream

This is the smallest host-launched NVSHMEM RMA transfer in the tutorial. It
runs with two MPI processes and two GPUs, which NVSHMEM calls PEs. Each process
selects its local GPU, initializes NVSHMEM with `MPI_COMM_WORLD`, and creates a
nonblocking CUDA stream.

Both PEs collectively allocate two one-element symmetric objects:

- `source`, initialized to `42` on PE 0 and `0` on PE 1;
- `target`, initialized to `0` on both PEs.

The address of `target` is a local symmetric address. Passing that address with
PE 1 tells NVSHMEM to write the corresponding `target` object on PE 1; it does
not mean that PE 0 has a raw pointer to PE 1's allocation.

## Data flow

PE 0 enqueues a four-byte transfer on `context.stream`:

```text
PE 0: source = 42  -- nvshmemx_putmem_on_stream -->  PE 1: target
```

PE 1 does not post a receive. That is the one-sided part of RMA: the initiating
PE supplies the source, destination, byte count, target PE, and stream. The
target takes part only in the completion protocol.

The program queues `nvshmemx_barrier_all_on_stream` after the put, then queues a
device-to-host copy of each PE's local `target` on the same stream. Synchronizing
the stream makes that sequence complete before the host checks the value. The
barrier is deliberately heavier than an application normally needs; later labs
can replace it with a signal/wait protocol.

## Your task

Complete the TODO guarded by `context.rank == 0` with
`nvshmemx_putmem_on_stream`:

- destination: `target`;
- source: `source`;
- byte count: `sizeof(*target)`;
- destination PE: `1`;
- stream: `context.stream`.

Do not replace the stream operation with a synchronous host put or move the
barrier. The point of this lab is that the host queues the transfer in CUDA
stream order with the initialization and later readback.

## Build and run

```bash
make
make run
```

The Makefile defaults to `CUDA_ARCH=90` and leaves GPU visibility to the
launcher. Set `CUDA_ARCH` for another GPU architecture. On a system that
needs a scheduler launcher, override `LAUNCHER`, for example
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. Use `make run_SOLVED` only
after attempting `host_put_on_stream.cu`.

The two output lines may arrive in either order. A passing run reports:

```text
PE 0 observed 0
PE 1 observed 42
```

## API reference

- [NVSHMEM Remote Memory Access](https://docs.nvidia.com/nvshmem/api/gen/api/rma.html)
  documents `nvshmemx_putmem_on_stream`, symmetric destinations, and put
  completion semantics.
- [NVSHMEM memory model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)
  explains symmetric objects and symmetric addresses.
