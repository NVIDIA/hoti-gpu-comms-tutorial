# NCCL GIN AlltoAllV

GIN lets a GPU kernel initiate network operations directly. This exercise uses
full GIN connectivity: every rank can name every other world rank as the peer
of a `gin.put`. It is the network baseline for AlltoAllV and is easiest to
measure with one GPU in each NVLink domain, so every non-self transfer crosses
InfiniBand.

The send and receive layout is the same as in the LSA exercise. A sender uses
`DevicePlanEntry::send_offset` for its local source and
`DevicePlanEntry::remote_recv_offset` for the byte offset in the destination's
registered receive window.

## CTA sharding and signals

One large transfer per rank pair would leave most GIN contexts idle. Each CTA
therefore owns one shard of one route:

```text
CTA:             0       1       2       3       4       5   ...
peer, round:   +1,0    +2,0    +3,0    +2,1    +3,1    +1,1  ...
GIN context:     0       1       2       3       4       5   ...
```

The example above has four ranks, so there are three non-self routes. The first
round assigns one route to each CTA. The next round shifts the route assignment
by one slot. The shifts spread each route over different GIN contexts instead
of pinning one peer to one QP. When a communicator has multiple GIN
connections, NCCL also stripes those context IDs over the connections. Each
non-empty non-self shard is one GIN put. All CTA threads copy a shard of the
self segment directly between the local send and receive buffers; self traffic
does not consume a GIN route.

The host requests one GIN context per CTA. NCCL may create a different number,
for example by rounding the request up across GIN connections. The kernel
assigns CTAs to the created contexts round-robin. Each CTA uses `blockIdx.x` as
its signal index. Its route identifies the one source from which it may receive
a shard, so the incoming count tells it whether to wait for zero or one signal.
`--blocks` must be at least the number of non-self routes, or `ranks - 1`.
More blocks create more route shards and request more GIN contexts; they are
not simply an occupancy knob.

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

The put offsets and size are bytes. `ncclDevComm::ginContextCount` is the
number of contexts NCCL actually made; it can differ from the host request in
either direction. Only thread 0 in each CTA issues puts; `waitSignal` and
`flush` are called cooperatively by the full CTA.

## Exercise

Open `nccl_gin_alltoallv.cu` and complete the TODOs in
`nccl_gin_alltoallv_kernel`:

1. synchronize the world GIN barrier after reading the signal baseline;
2. issue each non-empty non-self shard with `gin.put` and
   `ncclGin_WeakSignalInc`;
3. wait for the calculated number of incoming shard signals;
4. flush the CTA's outgoing operations and close the world barrier.

The starter already supplies the route assignment, local self copy, shard
calculation, expected-arrival count, host setup, registered windows, timing
loop, and validation. The completed reference is
`nccl_gin_alltoallv_SOLVED.cu`.

## Build

GIN kernels must be compiled with headers matching the NCCL runtime library.
This exercise uses the NCCL 2.31.2 device API:

```bash
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda
```

The Makefile puts `NCCL_HOME/lib` first in `LD_LIBRARY_PATH` for its run
targets. It defaults to native `sm_100` code plus `compute_100` PTX for
GB200 and GB300. Use `CUDA_ARCHS='90 100'` for a compatible fat binary, or
`CUDA_ARCH=90` for a GH200-only build. You can confirm the selected runtime
before launching with `ldd ./nccl_gin_alltoallv_SOLVED | grep nccl`.

## Run over InfiniBand

The clearest GIN-only placement uses one GPU per NVLink domain. On Lyris,
`--segment=1 --spread-segments` places each selected compute tray in a
different NVL72 base block:

```bash
make run_SOLVED NP=4 \
  LAUNCHER="srun --mpi=pmix_v5 --nodes=4 --ntasks=4 --ntasks-per-node=1 --segment=1 --spread-segments --cpu-bind=none" \
  RUN_ARGS="--pattern offdiagonal --bytes-per-rank 256M --blocks 48 --threads 256 --warmup 20 --iters 100"
```

A 48-CTA launch was the best large-message starting point in the four-rank
Lyris sweep used for this lab. It is a measured tuning point, not a portable
default: sweep `--blocks` again when the rank count, message distribution, or
GPU and NIC topology changes.

The run target defaults to `NCCL_IB_MERGE_NICS=0` and `NCCL_CROSS_NIC=1`.
Full GIN needs arbitrary peer connectivity and is unavailable when
`NCCL_CROSS_NIC=0`. Keeping the physical NICs separate also lets NCCL build
the GIN connections instead of presenting merged NICs as one device. Both
settings are Make variables and can be overridden for a system with a
different validated network configuration.

For a larger or sparse run:

```bash
make run_SOLVED NP=4 \
  LAUNCHER="srun --mpi=pmix_v5 --nodes=4 --ntasks=4 --ntasks-per-node=1 --segment=1 --spread-segments --cpu-bind=none" \
  RUN_ARGS="--pattern sparse --bytes-per-rank 16M --warmup 10 --iters 50"
```

The program queries GIN support after NCCL constructs the communicator. If
full GIN connectivity is unavailable for the selected placement or transport,
it prints `SKIP`. A successful run ends with output like:

```text
NCCL topology: world=4, LSA=1, rail=4
NCCL GIN AlltoAllV correctness: PASS
NCCL GIN AlltoAllV performance: ... ms/iteration, ... GB/s logical non-self
NCCL GIN AlltoAllV placement payload rates (NCCL LSA): 0.000 GB/s same-domain non-self, ... GB/s cross-domain
```

The reported bandwidth counts payload sent to other ranks and uses the
slowest rank's elapsed time.

Further reading: [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) and [GIN operations, signals, and barriers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html).
