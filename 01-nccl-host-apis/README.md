# NCCL host APIs

NCCL is the GPU communication library used by these first two labs. MPI starts
one process per GPU, and every process joins the same `ncclComm_t`
communicator. The host selects a CUDA device, allocates device buffers, and
enqueues NCCL work on a CUDA stream.

NCCL calls in this chapter submit work to the stream; they do not make the CPU
wait for the transfer or collective to finish. The device buffers and host
memory used by asynchronous copies must remain valid until the stream reaches
a completion point. Both labs call `cudaStreamSynchronize` before inspecting a
result on the CPU.

## Two-sided point-to-point communication

`ncclSend` and `ncclRecv` form a matched transfer between two ranks. The
sender supplies a device buffer, element count, data type, and peer rank. The
receiver supplies a matching receive with the same count and data type. This
is the right model when one GPU has data for one other GPU, or when an
application builds a larger exchange from several such pairs.

The first lab uses the simplest case: rank 0 copies 16 integers to its GPU and
sends them to rank 1. Rank 1 receives the data into its GPU buffer, copies it
back to host memory, and checks the values. The stream order is:

```text
rank 0: host-to-device copy -> ncclSend
rank 1:                         ncclRecv -> device-to-host copy
```

The two ranks are a deliberate small example, not a limitation of NCCL
point-to-point. For several sends and receives that must make progress
together, use `ncclGroupStart` and `ncclGroupEnd` as described in the
[NCCL point-to-point documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html).

## Collective communication

A collective is one operation carried out by every rank in a communicator. For
one collective call, all ranks must call the same operation with compatible
arguments. In particular, NCCL requires the same count and data type across
the ranks. Skipping a rank or passing inconsistent arguments is undefined and
can hang, crash, or corrupt data.

`ncclAllReduce` combines the input from every rank with a reduction operator
such as sum, min, or max, then writes the result on every rank. For a sum over
`K` ranks, each element has the defined result:

```text
out[i] = in[0][i] + in[1][i] + ... + in[K - 1][i]
```

![NCCL all-reduce: each rank receives the reduction of all ranks](figures/nccl-allreduce.png)

All-gather has a different, equally specific result: it places each rank's
input in rank order and gives the assembled buffer to every rank. The second
diagram is included so the distinction is visible before the later examples
use more than one collective.

![NCCL all-gather: each rank receives all rank inputs in rank order](figures/nccl-allgather.png)

The diagrams above are reproduced from the [NCCL Collective Operations
documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).
That page defines the other standard collectives and their input/output
semantics.

## Exercises

1. [Send and receive on a stream](01-send-recv-on-stream) builds the two-GPU
   copy above. It shows the ordering between CUDA copies and a matched
   `ncclSend`/`ncclRecv` pair.
2. [All-reduce on a stream](02-allreduce-on-stream) gives each rank a local
   value, enqueues `ncclAllReduce(..., ncclSum, ...)`, and verifies that every
   rank receives the same sum.

Each exercise README states its CUDA, MPI, NCCL, GPU-count, build, and launch
requirements. Start with the reference executable to confirm the environment,
then fill in the plain starter file and compare it with the `_SOLVED` version.
