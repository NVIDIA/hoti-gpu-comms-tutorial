# NCCL AlltoAllV with LSA and railed GIN

This version is for several LSA domains with more than one GPU in each domain.
An LSA domain may span several Linux hosts. The algorithm uses two
communication scopes:

- LSA pointers for GPUs in the same local LSA domain;
- railed GIN between GPUs with the same LSA rank in different domains.

Railed GIN cannot address an arbitrary world rank. In a four-GPU LSA domain,
LSA rank 2's rail team contains LSA rank 2 in every other domain, but not LSA
ranks 0, 1, or 3. A cross-domain message therefore needs an ingress GPU and
one LSA scatter:

```text
source LSA domain                  destination LSA domain

GPU 0 ==== rail 0 =================> GPU 0 -- LSA --> final local GPU
GPU 1 ==== rail 1 =================> GPU 1 -- LSA --> final local GPU
GPU 2 ==== rail 2 =================> GPU 2 -- LSA --> final local GPU
GPU 3 ==== rail 3 =================> GPU 3 -- LSA --> final local GPU
```

Each source GPU sends one variable-sized message for each remote destination
GPU. The GIN put reads directly from the registered send window and writes a
fixed slot in the matching ingress GPU's inbox. The ingress GPU then copies
that slot through an LSA pointer to the final destination. There is no pack
buffer or packet header in the data path.

## Inbox layout

Every ingress GPU has the same symmetric inbox layout:

```text
                 destination LSA rank
                 0          1          2          3
source domain  +----------+----------+----------+----------+
       0       | message  | message  | message  | message  |
               +----------+----------+----------+----------+
       1       | message  | message  | message  | message  |
               +----------+----------+----------+----------+
```

A slot is large enough for the largest source/destination pair in the current
plan. The source-domain coordinate is `rail.rank`; the destination coordinate
is the LSA rank in the remote domain. The source GPU's LSA rank is implicit:
LSA rank 2 sends only to LSA rank 2 through railed GIN.

Large messages are divided into shards inside that slot. Shard boundaries are
16-byte aligned, and the last shard owns any scalar tail. The host aims for
about 1 MiB per shard. For balanced traffic, it reserves a share of the CTA
grid for every remote route:

```text
route_shards = min(
    max(1, blocks / remote_routes),
    max(1, ceil(largest_remote_pair_bytes / 1 MiB)))
```

That cap leaves room for all routes to make progress. It is too conservative
when the largest remote pair is at least half of the largest per-rank remote
send volume. For that case, the cap becomes `blocks`, allowing the hot route to
use the whole grid. `route_shards` is one global value, so this also raises the
shard count for the other remote routes; zero-length shards do not issue puts.
The calculation happens once on the host, outside the timed region. Small
messages still use one put per route, and sharding does not change the inbox
allocation or add headers.

Same-domain messages are sharded across the whole CTA grid as well. Each CTA
owns one contiguous range and starts its peer loop at a different LSA rank.
The rotation keeps the CTAs from all writing through the same NVLink route at
once.

The ingress GPU needs the receive offset chosen by each final local GPU. It
reads that GPU's registered plan through `ncclGetLsaPointer(plan_window, ...)`,
then indexes the plan by the source world rank.

## CTAs, contexts, and signals

GIN signals belong to a GIN context. A sender and receiver must therefore use
the same context for a shard. The helper `route_shard_block` first computes
the forward domain distance:

```text
domain_delta = (destination_domain - source_domain + domain_count) % domain_count
```

It then maps `(domain_delta - 1, destination LSA rank, shard)` to a CTA and
GIN context. Using the relative domain distance instead of an absolute
source/destination pair gives every source domain the same spread across the
CTA grid. The sender and receiver calculate the same CTA. The signal index is
`blockIdx.x` within that matched context.

By default the host requests one GIN context per CTA. `--gin-contexts N` can
request `1..--blocks` contexts instead, allowing context sharing to be swept
without changing CTA count. The kernel uses `dev_comm.ginContextCount` because
the actual count may differ from the host request and uses NCCL's default
GPU-wide resource sharing:

```cpp
ncclGin gin(dev_comm, context);
```

CTA `b` uses context `b % dev_comm.ginContextCount`. The modulo form keeps
adjacent route shards on different contexts when NCCL creates fewer contexts
than CTAs and remains valid when several CTAs share a context.

The device communicator is created with matching resources:

```cpp
requirements.ginContextCount = blocks;
requirements.barrierCount = blocks;
requirements.ginSignalCount = blocks;
requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;

ncclDevCommCreate(comm, &requirements, &dev_comm);
```

`--gin-queue-depth N` sets `requirements.ginQueueDepth` (zero leaves NCCL's
default). The setup prints the requested and created GIN resources and skips a
run whose resources differ across ranks or cannot satisfy the CTA signal IDs.

The launch is split into two kernels:

1. `send_and_deliver_local` enters a per-CTA world barrier, copies same-domain
   messages with LSA pointers, and issues the cross-domain shard puts.
2. `wait_and_scatter` counts the non-empty incoming shards assigned to each
   CTA, waits once for all of them, copies them to the final local GPUs,
   flushes the same GIN context's outgoing puts, and enters the final world
   barrier.

Keeping the sends in a kernel with no remote waits avoids filling the GPU with
waiting CTAs before all producer CTAs have run.

Each non-empty shard attaches one weak signal increment to its put. The plan
does not change during the program, so a CTA expects the same number of
increments on every launch. At epoch `e`, it waits for
`e * expected_nonempty_shards`. An empty shard neither signals nor contributes
to that threshold.

The weak signal makes its own inbox shard visible before the receiver observes
the increment. It does not make the sender's source range safe to reuse;
`gin.flush` provides that local completion guarantee. The flush is delayed
until the second kernel so outgoing puts can remain in flight while the CTA
waits for and scatters incoming data. The final world barrier runs after every
CTA has flushed its outgoing context and completed its LSA scatter. The next
launch can then reuse the send buffer and inbox slots.

The main device APIs are:

```cpp
ncclGin(ncclDevComm const &comm, int context_index);

void *ncclGetLocalPointer(ncclWindow_t window, size_t byte_offset);

void *ncclGetLsaPointer(
    ncclWindow_t window, size_t byte_offset, int lsa_rank);

int ncclTeamRankToWorld(
    ncclDevComm const &comm, ncclTeam team, int team_rank);

void ncclGin::put(
    ncclTeam team, int peer,
    ncclWindow_t destination_window, size_t destination_byte_offset,
    ncclWindow_t source_window, size_t source_byte_offset, size_t bytes,
    ncclGin_WeakSignalInc remote_action);

void ncclGin::waitSignal(
    Coop coop, ncclGinSignal_t signal, uint64_t least);

void ncclGin::flush(Coop coop);

ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGin gin, uint32_t index);
```

`ncclGetLocalPointer` names memory on the calling GPU.
`ncclGetLsaPointer` names the same registered window on an LSA peer. The peer
argument to `gin.put(rail, ...)` is a rail-team rank, while the AlltoAllV plan
is indexed by world rank.

## Exercise

Open `nccl_lsa_gin_alltoallv.cu` and complete its three TODOs:

1. copy this CTA's contiguous same-domain shard to its LSA target;
2. put each non-empty remote shard directly into its fixed inbox slot and
   attach a weak increment of this CTA's signal;
3. wait for this CTA's cumulative signal threshold, then call the supplied
   helper that scatters its assigned shards through LSA pointers.

The starter supplies the topology mapping, shard calculation, symmetric
allocation, registered windows, GIN-context mapping, barriers, source-reuse
flush, scatter traversal, launch loop, timing, and validation. The reference
is `nccl_lsa_gin_alltoallv_SOLVED.cu`.

## Build and run

This lab needs NCCL 2.31.2 or newer, at least two LSA domains, uniform LSA team
sizes, contiguous world ranks within each LSA team, and railed GIN support. The
setup checks that layout before launching the kernel.

```bash
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda

make run_SOLVED NP=8 \
  LAUNCHER='srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1'
```

That common two-host command is valid only when each host is a separate LSA
domain. On Lyris, use two two-tray segments placed in separate NVL72 base
blocks:

```bash
make run_SOLVED NP=16 \
  LAUNCHER='srun --mpi=pmix_v5 --nodes=4 --ntasks=16 --ntasks-per-node=4 --segment=2 --spread-segments' \
  RUN_ARGS='--pattern offdiagonal --bytes-per-rank 256M --blocks 40 --threads 512 --warmup 10 --iters 50'
```

The build defaults to native `sm_100` (GB200) and `sm_103` (GB300) code. Use
`CUDA_ARCHS='90 100 103'` for a compatible fat binary, or `CUDA_ARCH=90` for a
GH200-only build. The run target defaults to
`NCCL_IB_MERGE_NICS=0` and `NCCL_CROSS_NIC=0` so
NCCL builds corresponding GPU/NIC rails. Override those variables only when a
system has a different validated mapping.

Use the same workload controls as the other labs:

```bash
make run_SOLVED NP=8 \
  LAUNCHER='srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1' \
  RUN_ARGS='--pattern sparse --bytes-per-rank 64M --blocks 40 --threads 512 --iters 50'
```

When the mixed path is below its target, measure the two sequential kernels
before changing its algorithm:

```bash
make run_SOLVED NP=16 \
  LAUNCHER='srun --mpi=pmix_v5 --nodes=4 --ntasks=16 --ntasks-per-node=4 --segment=2 --spread-segments --cpu-bind=none' \
  RUN_ARGS='--pattern offdiagonal --bytes-per-rank 256M --blocks 40 --threads 512 --warmup 10 --iters 50 --profile-phases'
```

The additional line reports `send + local delivery` separately from `wait +
scatter + flush`. Use it to choose the next algorithmic experiment, then turn
the flag off for the throughput number because the optional per-iteration CUDA
events add measurement overhead.

CTA count affects both the LSA copy and the requested GIN-context count. Start
with the default for small messages. On the Lyris placement above, start
large-message tuning with `--blocks 40 --threads 512`, then sweep the CTA
count. The best value depends on the LSA-team size, remote-domain count, GPU,
and message distribution; measure again on the tutorial system.

The program prints `SKIP` if the placement does not form uniform LSA and rail
teams or if railed GIN is unavailable. For performance comparisons, use the
cross-domain placement rate for the network part. Each cross-domain byte is
read by GIN and written to the inbox. A byte whose final destination has a
different LSA rank is then read and written once more by the LSA scatter.

This implementation uses Hopper-compatible loads, stores, LSA pointers, and
GIN operations. It does not require NVLS, multimem instructions, or a
Blackwell-only feature.
