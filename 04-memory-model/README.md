# Memory model

One-sided communication moves data without a matching receive, but the target
still needs a defined point at which device code may consume that data. This
chapter keeps the transfer fixed—PE 0 sends one integer to PE 1—and changes
only the protocol around it.

Each lab allocates symmetric `source`, `payload`, `result`, and, where needed,
`ready` objects. The host initializes them, completes a setup-only MPI barrier,
then every PE enters the same one-thread CUDA kernel through
`nvshmemx_collective_launch`. The kernel reads its local `payload` into
`result` only after the protocol permits that read. The host copies `result`
back after the kernel finishes.

All communication and synchronization below occurs in device code. The CUDA
stream orders setup, collective launch, and readback, but the ordering,
completion, notification, and consume operation happen in the kernel. A
one-thread, one-block collective launch is intentional: it gives one
thread-scoped device collective instance per PE.

A blocking device put copies data out of the source on the initiating PE, but
it does not create a receiver-visible handoff. It neither notifies the target
PE nor defines the delivery order of a separate transfer.

## The three handoff protocols

| Pattern | What it establishes | What it does not establish |
| --- | --- | --- |
| `put` + `barrier_all` | Prior RMA work completes and every participating PE reaches the handoff. | A narrow producer/consumer path; it is collective. |
| `put` + `quiet` | Prior symmetric-memory operations issued by the calling PE complete. | A notification or synchronization event on another PE. |
| `put` + `quiet` + signal/wait | The producer completes the payload, then notifies the receiver; the receiver waits before reading. | Ordering for unrelated transfers. |
| `put_signal` + wait | Delivery of one payload is coupled to delivery of its corresponding signal. | Ordering or completion for any other put. |

The distinction between the middle two rows is the point of the chapter:
`quiet` is local. Calling it on PE 0 does not make PE 1 aware that the payload
is ready. A separate remote signal and an NVSHMEM wait are required to create
a point-to-point handoff after a plain put. `put_signal` fuses that payload and
notification relationship for one operation, so this specific transfer needs
neither a separate quiet nor a global barrier.

Do not replace the device waits in these labs with ordinary CUDA polling.
NVSHMEM wait/test operations perform the consistency work needed for the
receiver to consume data after a flag is observed.

## Exercises

Start with the plain-named source file in each directory. The `_SOLVED` file is
the completed reference.

1. [Put + barrier](01-put-barrier) — the collective baseline. PE 0's device
   thread puts `42` into PE 1's payload; both device threads enter a barrier
   before loading their local payload into `result`.
2. [Put + quiet + signal/wait](02-put-quiet-signal-wait) — device-side local
   completion at the sender followed by explicit remote notification and a
   device-side target wait.
3. [Put-signal + wait](03-put-signal) — one device put-with-signal couples the
   payload with its flag and removes the separate quiet/signal sequence for
   that transfer.

## Build settings

These kernels call NVSHMEM device APIs, so their Makefiles enable relocatable
device code. They default to `CUDA_ARCH=90` and leave GPU visibility to the
launcher.

```bash
make CUDA_ARCH=90
make run LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
```

## Further reading

- [NVSHMEM memory ordering](https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html)
  explains `fence`, `quiet`, and device-side completion.
- [NVSHMEM signaling operations](https://docs.nvidia.com/nvshmem/api/gen/api/signal.html)
  documents the relation between a put-with-signal payload and its signal.
- [NVSHMEM point-to-point synchronization](https://docs.nvidia.com/nvshmem/api/gen/api/sync.html)
  documents the device wait/test APIs.
- [NVSHMEM collective communication](https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html)
  covers device collective participation and collective launch.

NVSHMEM makes ordering, completion, and notification particularly easy to
reason about because the API surface exposes them separately. The preceding
[NCCL symmetric memory](../03-nccl-symmetric/) chapter uses registered
windows, put-with-signal, and stream ordering to expose the same concerns
through NCCL. The following [NCCL device APIs](../05-nccl-device-apis/) chapter
applies them through device communicators, LSA, and GIN.
