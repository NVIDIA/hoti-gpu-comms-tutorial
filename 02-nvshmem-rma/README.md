# NVSHMEM RMA

NVSHMEM implements a partitioned global address space across GPUs. Each
process is a processing element (PE), and each PE has private memory plus
objects that other PEs can access. The objects used for communication are
**symmetric**: every PE collectively creates a corresponding allocation with
the same type and size. In these labs, `nvshmem_malloc` creates the symmetric
source and destination objects on the GPU.

An RMA operation names its target as a local symmetric address plus a PE:

```c++
nvshmemx_putmem_on_stream(target, source, bytes, target_pe, stream);
```

The initiating PE supplies the source, destination, length, and target PE.
The target PE does not post a matching receive. This is the main difference
from the NCCL send/receive lab: the transfer and the protocol used to tell the
target that data is ready are separate decisions.

The remote destination of an NVSHMEM RMA call must be an address within a
symmetric allocation. It is expressed as a local pointer on the calling PE;
NVSHMEM uses the target PE argument to reach the corresponding object
remotely. In these labs both source and destination are symmetric. An ordinary
CUDA allocation cannot be the remote destination.

## Completion, ordering, and visibility

A blocking `put` has **local completion** when the operation completes:
NVSHMEM has consumed the source data on the initiating PE. For a
`*_on_stream` call, returning to the host only means the work was enqueued;
the stream must reach the operation first. Local completion still does not
mean the target PE can safely read the new value. Use the appropriate ordering
and completion operation before consuming remote data:

- `nvshmem_fence` orders blocking RMA operations from one PE.
- `nvshmem_quiet` completes prior symmetric-memory operations from the calling
  PE, including direct stores through a pointer returned by `nvshmem_ptr`, but
  does not notify a peer.
- A barrier, or `quiet` followed by a remote signal and a target-side wait,
  makes the handoff explicit between PEs.

These labs use `nvshmemx_barrier_all_on_stream` after the put. The barrier is
queued after the RMA work on the same stream, so PE 1 can inspect its local
target after its stream synchronizes. Real applications often use a smaller
protocol, such as a signal and wait, rather than a global barrier.

The [Memory model](../04-memory-model/) chapter uses the same
one-integer put with three handoff protocols: a barrier, `quiet` plus an
explicit signal/wait, and `put_signal`. Read it when you want to separate
initiator completion from target notification.

## Exercises

All three exercises run with exactly two PEs. Start from the plain source file
and use the matching `_SOLVED` file only to check your work.

1. [Host put on a stream](01-host-put-on-stream) — PE 0 enqueues
   `nvshmemx_putmem_on_stream` to copy `42` into PE 1's symmetric target.
   This is the host-launched, stream-ordered form of RMA.
2. [Device put](02-device-put) — the accompanying material introduces the
   thread, warp, and block forms of a typed NVSHMEM `put`. The lab deliberately
   uses the thread form to send one value to the next PE from inside a kernel.
3. [nvshmem_ptr](03-nvshmem-ptr) — query whether the peer's symmetric object
   has a direct mapping. When `nvshmem_ptr` returns non-null, the kernel can
   use ordinary loads and stores through that pointer. It follows its direct
   store with device `nvshmem_quiet`, then uses an MPI barrier for the peer
   handoff. Unsupported allocations or topologies skip cleanly.

`nvshmem_ptr` is not a replacement for RMA. It is available only when the
runtime can expose a directly accessible mapping, and the pointer it returns
is not a symmetric address to pass back to an NVSHMEM RMA routine.

## Build settings

The chapter Makefiles default to `CUDA_ARCH=90` and leave GPU visibility to
the launcher.
The device-put and `nvshmem_ptr` labs use relocatable device code because they
call NVSHMEM device APIs from their kernels.

```bash
make CUDA_ARCH=90
make run LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
```

## Further reading

- [NVSHMEM introduction](https://docs.nvidia.com/nvshmem/api/introduction.html)
  describes the GPU-focused OpenSHMEM programming model.
- [Memory model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)
  covers symmetric objects, ordering, and visibility.
- [Remote memory access API](https://docs.nvidia.com/nvshmem/api/gen/api/rma.html)
  documents `put`, `get`, stream, device, and nonblocking variants.
- [`nvshmem_ptr` API](https://docs.nvidia.com/nvshmem/api/gen/api/setup.html#nvshmem-ptr)
  describes its direct-access and null-return semantics.
