# Put + barrier

This is the baseline handoff. PE 0's device thread puts one integer into PE
1's symmetric `payload` object. Then the one device thread on every PE joins a
world barrier before loading its own `payload` into local symmetric `result`.
The host validates `result`, so the consuming load is part of the device-side
protocol rather than a host read after the fact.

```text
PE 0 kernel: source = 42 -> put(payload on PE 1) -> barrier_all -> result = payload
PE 1 kernel:                   local payload = 0 -> barrier_all -> result = payload
```

The host does only setup and verification: it initializes the symmetric
objects, completes an MPI barrier that occurs before any RMA work, then calls
`nvshmemx_collective_launch` with one block and one thread on each PE. The
single thread on each GPU calls `nvshmem_barrier_all` once. That is the device
collective this lab is about.

`nvshmem_barrier_all` is collective. Both PEs must call it in the same program
order. It waits for the group and completes prior remote memory updates, so PE
1's later device load from `payload` is valid. It is deliberately broad for
one producer and one consumer, but it is a clear first answer to “when is the
remote write ready?”

## Your task

Inside the `rank == 0` block of `put_then_barrier`, use the typed device put:

```c++
nvshmem_int_put(payload, source, 1, 1);
```

Do not move the barrier or the `*result = *payload` load. PE 0 should retain
`0`; PE 1 should observe `42`.

## Build and run

```bash
make
make run
```

The Makefile defaults to `CUDA_ARCH=90` and leaves GPU visibility to the
launcher. Set `CUDA_ARCH` for another GPU architecture. On Jupiter, use
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. Try the starter first;
`make run_SOLVED` runs `put_barrier_SOLVED.cu`.

## What this establishes

The barrier gives both completion and group coordination. That is more than a
plain put supplies, and more coordination than a point-to-point transfer
usually needs. The next lab has the sender call `quiet` to complete its put,
then uses an explicit signal and wait to notify the receiver.

## API reference

- [NVSHMEM collective communication](https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html)
  documents device barrier participation and completion.
- [NVSHMEM remote memory access](https://docs.nvidia.com/nvshmem/api/gen/api/rma.html)
  documents typed device put operations.
