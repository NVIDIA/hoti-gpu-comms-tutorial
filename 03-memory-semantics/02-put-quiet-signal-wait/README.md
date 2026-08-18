# Put + quiet + signal/wait

This is the same one-integer transfer as the barrier lab, but it does not use
a collective handoff. PE 0's device thread sends `42` to PE 1, completes its
own prior RMA work with `quiet`, then writes a separate symmetric `ready` flag
on PE 1. PE 1's device thread waits on that flag before it loads `payload` into
`result`.

```text
PE 0 kernel:  put(payload on PE 1) -> quiet -> signal(ready = 1 on PE 1)
PE 1 kernel:                                        wait(ready == 1) -> result = payload
```

The host initializes the symmetric objects and completes an MPI barrier before
the kernels are launched. It then collectively launches one thread on each PE
on each PE's `context.stream`. That setup barrier is not part of the data
handoff.

## Why `quiet` is not enough

Device `nvshmem_quiet` completes symmetric-memory operations issued by the
calling PE before it. Here, it means PE 0 has finished the put before it sends
the flag. It is local and non-collective: PE 1 does not learn that PE 0 called
`quiet`, and a `quiet` on PE 0 does not block PE 1.

The device signal and device wait provide the missing receiver-visible event.
PE 1 executes `nvshmem_signal_wait_until` before its load from `payload`; the
wait supplies the consistency needed for that load. Do not replace it with
ordinary CUDA loads in a polling loop.

## Your task

On PE 0, write these three device operations in order:

```c++
nvshmem_int_put(payload, source, 1, 1);
nvshmem_quiet();
nvshmemx_signal_op(ready, kReady, NVSHMEM_SIGNAL_SET, 1);
```

On PE 1, write:

```c++
nvshmem_signal_wait_until(ready, NVSHMEM_CMP_EQ, kReady);
```

The supplied `*result = *payload` follows those operations in the same
one-thread kernel. A correct run reports `0` on PE 0 and `42` on PE 1.

## Build and run

```bash
make
make run
```

The Makefile defaults to `CUDA_ARCH=90` and leaves GPU visibility to the
launcher. Set `CUDA_ARCH` for another GPU architecture. On Jupiter, use
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. The reference program is
available through `make run_SOLVED` after you have attempted the starter.

## API reference

- [NVSHMEM memory ordering](https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html)
  documents `quiet` as local completion rather than peer notification.
- [NVSHMEM signaling operations](https://docs.nvidia.com/nvshmem/api/gen/api/signal.html)
  shows the same put, quiet, signal, and wait pattern.
- [Point-to-point synchronization](https://docs.nvidia.com/nvshmem/api/gen/api/sync.html)
  documents the device wait operation.
