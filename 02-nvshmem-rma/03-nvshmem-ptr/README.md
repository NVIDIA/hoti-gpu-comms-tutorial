# `nvshmem_ptr`

`nvshmem_ptr` asks NVSHMEM whether a symmetric object on another PE can be
accessed through a directly usable pointer. This is not an RMA operation: when
the query succeeds, the kernel uses ordinary CUDA loads and stores through the
returned local pointer.

The exercise runs with exactly two MPI processes and two GPUs. Each PE
collectively allocates one symmetric `int`, initializes its local copy to zero,
and chooses the other PE as its peer.

## Availability and data flow

Before launching the kernel, every PE calls `nvshmem_ptr(symmetric_value, peer)`
on the host. An MPI all-reduce requires the pointer to be available on both
PEs. If either query returns `NULL`, rank 0 prints a skip message and the
program exits successfully. That is an expected outcome for an allocation or
GPU topology that does not provide a direct mapping.

When the mapping is available, each GPU launches one thread:

```text
PE 0 kernel: store 1 through a pointer to PE 1's symmetric_value
PE 1 kernel: store 2 through a pointer to PE 0's symmetric_value
```

The kernel gets the peer pointer itself, stores through it, and calls device
`nvshmem_quiet()` from the same thread. Quiet completes and orders the direct
store to symmetric memory, but it is local and non-collective: the call does
not notify the peer. The host therefore synchronizes each CUDA stream to make
sure its kernel and quiet have finished, then all PEs enter `MPI_Barrier`
before copying their local values back. The pointer query alone provides
neither completion nor synchronization.

The returned pointer is a local direct-access address, not a symmetric address.
Do not pass it to an NVSHMEM RMA routine.

## Your task

Complete `store_through_peer_pointer` in `nvshmem_ptr.cu`:

1. Query `nvshmem_ptr(symmetric_value, peer)` inside the kernel.
2. Keep a null check around the returned pointer.
3. Store `value` through the returned pointer and follow the store with
   `nvshmem_quiet()` in the same thread.

The host-side availability check remains in the starter so that the kernel is
not launched on a topology without a peer mapping. The checked reference is
`nvshmem_ptr_SOLVED.cu`.

## Build and run

```bash
make
make run
```

The Makefile defaults to `CUDA_ARCH=90` and leaves GPU visibility to the
launcher. Set `CUDA_ARCH` for another GPU architecture. Set
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"` on Jupiter. Use
`make run_SOLVED` to build and run the reference.

On a supported direct-access path, the output order is arbitrary and a passing
run includes:

```text
PE 0 read 2 through a peer mapping
PE 1 read 1 through a peer mapping
```

On an unsupported path, this is also a successful result:

```text
nvshmem_ptr is unavailable for this allocation or topology; skipping
```

## API reference

- [`nvshmem_ptr`](https://docs.nvidia.com/nvshmem/api/gen/api/setup.html#nvshmem-ptr)
  documents the host/device query, the direct-access pointer, and its
  null-return behavior.
- [NVSHMEM memory model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)
  explains the distinction between a symmetric address and a direct-access
  pointer returned by `nvshmem_ptr`.
- [`nvshmem_quiet`](https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html#nvshmem-quiet)
  documents completion and ordering for direct stores and explains why quiet
  is not peer notification.
