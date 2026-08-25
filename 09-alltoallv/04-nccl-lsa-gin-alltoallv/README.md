# NCCL AlltoAllV with LSA and railed GIN

This version is for multiple nodes with several GPUs per node. It uses two
different communication scopes:

- LSA pointers for GPUs in the same local LSA domain;
- railed GIN between GPUs in the same local position on different nodes.

Railed GIN cannot address an arbitrary world rank. On a four-GPU node, GPU 2's
rail team contains GPU 2 on every other node, but not GPUs 0, 1, or 3 on those
nodes. A remote message therefore needs an ingress GPU and one local scatter:

```text
source node                         destination node

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
source node  +----------+----------+----------+----------+
     0       | message  | message  | message  | message  |
             +----------+----------+----------+----------+
     1       | message  | message  | message  | message  |
             +----------+----------+----------+----------+
```

A slot is large enough for the largest source/destination pair in the current
plan. The source-node coordinate is `rail.rank`; the destination coordinate is
the LSA rank on the remote node. The source GPU's local position is implicit:
rail GPU 2 sends only to rail GPU 2.

Large messages are divided into shards inside that slot. Shard boundaries are
16-byte aligned, and the last shard owns any scalar tail. The host aims for
about 4 MiB per shard but never creates more shards than the CTA grid can
spread across the remote routes:

```text
route_shards = min(
    max(1, blocks / remote_routes),
    max(1, ceil(largest_pair_bytes / 4 MiB)))
```

Small messages therefore stay as one put per route. Larger messages can use
several GIN contexts without changing the inbox allocation or adding headers.

The ingress GPU needs the receive offset chosen by each final local GPU. It
reads that GPU's registered plan through `ncclGetLsaPointer(plan_window, ...)`,
then indexes the plan by the source world rank.

## CTAs, contexts, and signals

GIN signals belong to a GIN context. A sender and receiver must therefore use
the same context for a shard. The helper `route_shard_block` maps

```text
(source node, destination node, destination LSA rank, shard)
    -> CTA -> GIN context
```

the same way on both sides. The signal index is `blockIdx.x` within that
matched context.

The host requests one GIN context per CTA. The kernel uses
`dev_comm.ginContextCount` because NCCL may create fewer contexts than were
requested. The device communicator is created with matching resources:

```cpp
requirements.ginContextCount = blocks;
requirements.barrierCount = blocks;
requirements.ginSignalCount = blocks;
requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;

ncclDevCommCreate(comm, &requirements, &dev_comm);
```

The launch is split into two kernels:

1. `send_and_deliver_local` copies same-node messages with LSA pointers,
   issues the remote shard puts, and flushes every issuing context.
2. `wait_and_scatter` counts the non-empty incoming shards assigned to each
   CTA, waits once for all of them, and copies them to the final local GPUs.

Keeping the sends in a kernel with no remote waits avoids filling the GPU with
waiting CTAs before all producer CTAs have run.

Each non-empty shard attaches one weak signal increment to its put. The plan
does not change during the program, so a CTA expects the same number of
increments on every launch. At epoch `e`, it waits for
`e * expected_nonempty_shards`. An empty shard neither signals nor contributes
to that threshold.

The weak signal makes its own inbox shard visible to the receiver.
`gin.flush` is separate: it makes the sender's source range safe to reuse. The
closing world barrier runs after every LSA scatter and completes the collective
before the next launch can reuse an inbox slot.

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

1. copy each same-node message to its LSA target;
2. put each non-empty remote shard directly into its fixed inbox slot and
   attach a weak increment of this CTA's signal;
3. wait for this CTA's cumulative signal threshold, then call the supplied
   helper that scatters its assigned shards through LSA pointers.

The starter supplies the topology mapping, shard calculation, symmetric
allocation, registered windows, GIN-context mapping, barriers, flushes,
scatter traversal, launch loop, timing, and validation. The reference is
`nccl_lsa_gin_alltoallv_SOLVED.cu`.

## Build and run

This lab needs NCCL 2.31.2 or newer, at least two nodes, uniform LSA team sizes,
contiguous world ranks within each node, and railed GIN support. The setup
checks that layout before launching the kernel.

```bash
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda CUDA_ARCH=90

make run_SOLVED NP=8 \
  LAUNCHER='srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1'
```

The run target defaults to `NCCL_IB_MERGE_NICS=0` and `NCCL_CROSS_NIC=0` so
NCCL builds corresponding GPU/NIC rails. Override those variables only when a
system has a different validated mapping.

Use the same workload controls as the other labs:

```bash
make run_SOLVED NP=8 \
  LAUNCHER='srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1' \
  RUN_ARGS='--pattern sparse --bytes-per-rank 64M --blocks 16 --iters 50'
```

CTA count affects both the LSA copy and the requested GIN-context count. Start
with the default for small messages and measure `--blocks 16` and `--blocks
64` for larger messages.

The program prints `SKIP` if the placement does not form uniform LSA and rail
teams or if railed GIN is unavailable. For performance comparisons, use the
inter-host placement rate for the network part. Each remote byte is read by
GIN, written to the inbox, read for the scatter, and written to the final
receive buffer.

This implementation uses Hopper-compatible loads, stores, LSA pointers, and
GIN operations. It does not require NVLS, multimem instructions, or a
Blackwell-only feature.
