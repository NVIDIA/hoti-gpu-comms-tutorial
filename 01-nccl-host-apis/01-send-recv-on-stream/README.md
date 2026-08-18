# Send and receive on a stream

This is a two-rank NCCL point-to-point exercise. Rank 0 fills 16 integers,
copies them to GPU memory, and sends that device buffer to rank 1. Rank 1
receives into device memory, copies the values back to the host, and checks
that it received `0` through `15`.

`ncclSend` and `ncclRecv` are a two-sided pair: one rank sends and its peer
posts the matching receive with the same count and data type. This lab has one
sender and one receiver, so it models a direct copy from rank 0's GPU to rank
1's GPU. See
the [NCCL point-to-point documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)
for the general rule and for grouped multi-peer exchanges.

## What the stream does here

The program uses one CUDA stream per rank. The reference queues the work in
this order:

```text
rank 0: cudaMemcpyAsync(host -> device), ncclSend
rank 1:                                ncclRecv, cudaMemcpyAsync(device -> host)
```

The stream guarantees that the send reads data after rank 0's copy and that
rank 1 copies data back only after its receive. `cudaStreamSynchronize` waits
for the queued work before rank 1 reads `h_recv`. The host does not need a
separate event for this two-rank, one-stream example.

The program assumes one node with two MPI ranks and uses each rank's assigned
GPU.

## Files

- `send_recv_on_stream.c` is the starter.
- `send_recv_on_stream_SOLVED.c` queues both copies and the NCCL operation on
  one stream, then validates the values on rank 1.
- `Makefile` builds both versions and provides launch targets.

## Build and run

The lab needs MPI, CUDA, NCCL, and two visible GPUs. Set `CUDA_HOME` and
`NCCL_HOME` if they are not installed in `/usr/local/cuda` and `/usr`.

    make
    make run_SOLVED

`make run_SOLVED` uses `mpirun -np 2` and leaves GPU visibility to the
launcher. Override the complete launcher for Jupiter:

    make run_SOLVED LAUNCHER="srun --ntasks=2 --gpus-per-task=1"

The reference prints `Rank 1 received data: 0 1 ... 15` and exits
successfully. Use `make run` after filling in the starter.

## Exercise

1. On rank 0, queue the host-to-device copy on `stream`.
2. Queue `ncclSend` on rank 0 and `ncclRecv` on rank 1 using that same stream.
3. On rank 1, queue the device-to-host copy after `ncclRecv`.
4. Synchronize once after the queued work, then print and check the host
   result.
