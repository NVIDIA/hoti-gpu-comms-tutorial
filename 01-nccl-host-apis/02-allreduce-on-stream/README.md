# All-reduce on a stream

This lab replaces a matched point-to-point transfer with a collective. Every
rank in the communicator contributes one device value and calls
`ncclAllReduce` on its CUDA stream. With `ncclSum`, every rank receives the
same sum of all rank inputs.

The reference gives rank `r` the value `r + 1`. For `K` ranks, the expected
result on every GPU is therefore:

```text
1 + 2 + ... + K = K * (K + 1) / 2
```

Unlike the send/receive exercise, no rank is merely a sender or merely a
receiver. Every rank makes the collective call and every rank owns a result.
All participants must use the same count and data type for this call. See
[NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
for the full all-reduce definition and the other collective semantics.

## What the stream does here

The reference queues a host-to-device copy of the local value, initializes the
receive buffer, calls `ncclAllReduce`, then queues a device-to-host copy of the
result. `cudaStreamSynchronize` makes the host result safe to inspect. The
collective payload stays in device memory; the host copy is only for the small
correctness check.

## Files

- `allreduce_on_stream.c` is the starter. The communicator, stream, buffers,
  input value, and verification are already present.
- `allreduce_on_stream_SOLVED.c` supplies the `ncclAllReduce` call.
- `Makefile` builds both variants and provides launch targets.

## Build and run

The lab needs MPI, CUDA, NCCL, and one visible GPU per MPI rank. The default
launch uses two ranks.

    make
    make run_SOLVED

Override the complete launcher or number of ranks for the local environment:

    make run_SOLVED NP=4
    make run_SOLVED LAUNCHER="srun --ntasks=2 --gpus-per-task=1"

With two ranks, the reference prints `Rank 0: all-reduce result 3.0` and
`Rank 1: all-reduce result 3.0`. Use `make run` after filling in the starter.

## Exercise

At the marked TODO, enqueue `ncclAllReduce` with:

1. `device_send` as the input and `device_recv` as the output.
2. One `ncclFloat` element per rank.
3. `ncclSum` as the reduction operator.
4. The existing communicator and CUDA stream.

Do not add a host-side reduction. The purpose of the lab is to see the
collective operate on the GPU buffers and to verify its result after the stream
has completed.
