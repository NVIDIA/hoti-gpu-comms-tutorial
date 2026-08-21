# Put-signal + wait

This is the compact point-to-point form of the preceding lab. PE 0's device
thread performs one put-with-signal operation; PE 1's device thread waits for
the corresponding signal before it loads the payload into `result`. The
payload, target PE, signal object, signal value, and signal operator are
supplied in one device API call.

```text
PE 0 kernel:  put_signal(payload, ready = 1 on PE 1)
PE 1 kernel:  wait(ready == 1) -> result = payload
```

The host completes the same setup-only MPI barrier as the other labs, then
uses `nvshmemx_collective_launch` to launch one device thread per PE. No host
NVSHMEM RMA operation implements the handoff.

## What the fused operation guarantees

Use the blocking `nvshmem_int_put_signal` form in this lab, not the `_nbi_`
form. When PE 1 observes the signal update, NVSHMEM guarantees that the
destination words from this same put-with-signal operation have been delivered.
That replaces the `quiet` plus separate signal in the preceding lab, and it
avoids the global barrier used in the baseline.

That guarantee covers only this payload/signal pair. Seeing the flag says
nothing about a different put issued before or after this one. A
multi-transfer protocol still needs its own fence, quiet, signaling, or
collective ordering as appropriate.

## Your task

On PE 0, write:

```c++
nvshmem_int_put_signal(payload, source, 1, ready, kReady, NVSHMEM_SIGNAL_SET, 1);
```

On PE 1, write:

```c++
nvshmem_signal_wait_until(ready, NVSHMEM_CMP_EQ, kReady);
```

The supplied device load into `result` follows the wait. A correct run prints
`PE 0 observed 0` and `PE 1 observed 42`.

## Build and run

```bash
make
make run
```

The Makefile defaults to `CUDA_ARCH=90` and leaves GPU visibility to the
launcher. Set `CUDA_ARCH` for another GPU architecture. On Jupiter, use
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. Use `make run_SOLVED` to run
the completed reference after trying the starter.

## API reference

- [NVSHMEM signaling operations](https://docs.nvidia.com/nvshmem/api/gen/api/signal.html)
  documents the delivery-to-signal relation and its single-operation scope.
- [Point-to-point synchronization](https://docs.nvidia.com/nvshmem/api/gen/api/sync.html)
  documents the device wait that establishes the target-side dependency.
