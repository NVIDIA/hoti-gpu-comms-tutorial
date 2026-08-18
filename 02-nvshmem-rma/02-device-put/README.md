# Device put

This exercise moves the RMA call from host code into a CUDA kernel. Run it with
exactly two MPI processes and two GPUs. As in the previous lab, each process is
an NVSHMEM PE, selects a local GPU, and creates a CUDA stream. Each PE then
collectively allocates two symmetric `int` objects: `source` and `target`.
`source` holds the value that this PE will send; `target` is the matching remote
destination. The code initializes `source` to `context.rank + 1` and its local
`target` to zero on that stream.

## Data flow

Every PE launches one CUDA thread. That thread copies its local `source` value
to the same symmetric `target` address on the next PE:

```text
PE 0 kernel: copy source = 1 to PE 1 target
PE 1 kernel: copy source = 2 to PE 0 target
```

The kernel receives the local symmetric addresses `source` and `target`. In the
device call, `next_pe` selects the matching `target` allocation on the remote
PE. Both the initiator and target therefore change on each rank: every GPU
issues one put and receives one value from its neighbor.

## Device API scopes

NVSHMEM provides three device-side forms of the typed put used here:

```c++
nvshmem_int_put(dest, source, nelems, pe);
nvshmemx_int_put_warp(dest, source, nelems, pe);
nvshmemx_int_put_block(dest, source, nelems, pe);
```

The first form is issued by one CUDA thread. The `_warp` and `_block` forms
are collective over their CUDA thread group: every thread in the warp or
block calls the routine with identical arguments. NVSHMEM can use those
threads to move a larger payload in parallel.

This exercise transfers one integer from a `<<<1, 1>>>` kernel, so it uses
the single-thread `nvshmem_int_put` form. The warp and block forms are
introduced here, but are not needed to complete the lab.

After the kernels are launched on `context.stream`, the code queues
`nvshmemx_barrier_all_on_stream` on that same stream. Only after the stream
synchronizes does the host copy its local `target` back and validate it. The
put is device-side; the barrier is host-enqueued and stream-ordered. The
barrier is the completion and handoff mechanism for this small example; a
real kernel sequence may use signals, waits, or another narrower protocol.

## Your task

Complete the TODO in `put_to_next_pe` with the typed device-side put for an
`int`. It must copy one element from `source` into the matching `target`
allocation on `next_pe` from the single thread already selected by the `if`
statement. Use the single-thread form shown above; do not substitute the warp
or block variants. Keep the kernel launch, stream, and barrier unchanged.

The starter is `device_put.cu`. The checked reference is
`device_put_SOLVED.cu`.

## Build and run

```bash
make
make run
```

The Makefile defaults to `CUDA_ARCH=90` and leaves GPU visibility to the
launcher. Set `CUDA_ARCH` for another GPU architecture. Set
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"` on Jupiter. `make run_SOLVED`
builds and runs the reference after you have tried the starter.

The output order is not significant. A passing run includes:

```text
PE 0 received 2 from PE 1
PE 1 received 1 from PE 0
```

## API reference

- [NVSHMEM Remote Memory Access](https://docs.nvidia.com/nvshmem/api/gen/api/rma.html)
  lists the typed `nvshmem_TYPENAME_put` host and device APIs and their remote
  completion behavior. In this lab, both operands are symmetric allocations.
- [NVSHMEM memory model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)
  describes why `target` must be a symmetric address.
