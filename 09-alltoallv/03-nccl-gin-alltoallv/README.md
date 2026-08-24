# NCCL GIN AlltoAllV

GIN lets a GPU kernel initiate network operations directly. This exercise uses
full GIN connectivity: every rank can name every other world rank as the peer
of a `gin.put`. It is the network baseline for AlltoAllV and is easiest to
measure with one GPU on each of two or more nodes, so every non-self transfer
crosses InfiniBand.

The send and receive layout is the same as in the LSA exercise. A sender uses
`DevicePlanEntry::send_offset` for its local source and
`DevicePlanEntry::remote_recv_offset` for the byte offset in the destination's
registered receive window.

## CTA sharding and signals

One large transfer per rank pair would leave most CTAs idle. Instead, CTA `b`
owns shard `b` of every source-to-destination segment:

```text
source r -> destination p

send segment:  [ CTA 0 ][ CTA 1 ][ CTA 2 ] ... [ CTA B-1 ]
                    |       |       |                 |
signal index:       0       1       2                B-1
```

Each non-empty shard becomes one GIN put. Every CTA uses GIN context 0 and has
its own signal index, `blockIdx.x`. On a receiver, the incoming counts tell CTA
`b` exactly how many sources have a non-empty shard `b`, so it knows the signal
value to wait for even for the `sparse` pattern.

The synchronization sequence in each CTA is:

```text
read signal baseline
        |
world GIN barrier (acquire, no put/get fence)
        |
thread 0 issues puts with weak signal increments
        |
wait for baseline + expected incoming puts
        |
flush local GIN source use
        |
world GIN barrier (release, no put/get fence)
```

A weak signal covers exactly the put to which it is attached. When the
receiver observes all expected increments, every corresponding payload shard
is visible. `gin.flush` provides a different guarantee: it makes the issuing
CTA's source buffers safe to reuse, but does not announce remote completion.
The closing barrier is the collective rendezvous after those two local facts
have been established.

The relevant NCCL 2.31.2 device APIs are:

```cpp
ncclGin(ncclDevComm const &comm, int context_index);

void ncclGin::put(
    ncclTeam team, int peer,
    ncclWindow_t destination_window, size_t destination_byte_offset,
    ncclWindow_t source_window, size_t source_byte_offset, size_t bytes,
    ncclGin_WeakSignalInc remote_action);

uint64_t ncclGin::readSignal(ncclGinSignal_t signal);

void ncclGin::waitSignal(
    Coop coop, ncclGinSignal_t signal, uint64_t least);

void ncclGin::flush(Coop coop);

ncclGinBarrierSession(
    Coop coop, ncclGin gin, ncclTeamTagWorld team, uint32_t index);

void ncclGinBarrierSession::sync(
    Coop coop, cuda::memory_order order, ncclGinFenceLevel fence);
```

The put offsets and size are bytes. Only thread 0 in each CTA issues puts;
`waitSignal` and `flush` are called cooperatively by the full CTA.

## Exercise

Open `nccl_gin_alltoallv.cu` and complete the TODOs in
`nccl_gin_alltoallv_kernel`:

1. synchronize the world GIN barrier after reading the signal baseline;
2. issue each non-empty shard with `gin.put` and
   `ncclGin_WeakSignalInc`;
3. wait for the calculated number of incoming shard signals;
4. flush the CTA's outgoing operations and close the world barrier.

The starter already supplies the shard calculation, expected-arrival count,
host setup, registered windows, timing loop, and validation. The completed
reference is `nccl_gin_alltoallv_SOLVED.cu`.

## Build

GIN kernels must be compiled with headers matching the NCCL runtime library.
This exercise uses the NCCL 2.31.2 device API:

```bash
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda CUDA_ARCH=90
```

The Makefile puts `NCCL_HOME/lib` first in `LD_LIBRARY_PATH` for its run
targets. You can confirm the selected runtime before launching with
`ldd ./nccl_gin_alltoallv_SOLVED | grep nccl`.

## Run over InfiniBand

The clearest GIN-only placement uses one GPU per node:

```bash
make run_SOLVED NP=2 \
  LAUNCHER="srun --nodes=2 --ntasks-per-node=1 --gpus-per-task=1"
```

The run target defaults to `NCCL_IB_MERGE_NICS=0` and `NCCL_CROSS_NIC=1`.
Full GIN needs arbitrary peer connectivity and is unavailable when
`NCCL_CROSS_NIC=0`. Keeping the physical NICs separate also lets NCCL build
the GIN connections instead of presenting merged NICs as one device. Both
settings are Make variables and can be overridden for a system with a
different validated network configuration.

For a larger or sparse run:

```bash
make run_SOLVED NP=4 \
  LAUNCHER="srun --nodes=4 --ntasks-per-node=1 --gpus-per-task=1" \
  RUN_ARGS="--pattern sparse --bytes-per-rank 16M --warmup 10 --iters 50"
```

The program queries GIN support after NCCL constructs the communicator. If
full GIN connectivity is unavailable for the selected placement or transport,
it prints `SKIP`. A successful run ends with output like:

```text
NCCL topology: world=2, LSA=1, rail=2
NCCL GIN AlltoAllV correctness: PASS
NCCL GIN AlltoAllV performance: ... ms/iteration, ... GB/s aggregate
```

The reported bandwidth counts payload sent to other ranks and uses the
slowest rank's elapsed time.

Further reading: [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) and [GIN operations, signals, and barriers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html).
