# NCCL LSA AlltoAllV

This version of AlltoAllV is for a group of GPUs that can directly load and
store one another's memory. NCCL calls such a group an LSA team. On the GH200
tutorial system, a run confined to one NVLink-connected domain should form one
LSA team containing every rank in the communicator.

Rank `r` has one send segment for each destination `p`:

```text
local send window on r                   receive window on p

send_offsets[p]                          recv_offsets[r]
       |                                       |
       v                                       v
       +----------------------+                +----------------------+
       | send_counts[p] items | -- NVLink -->  | recv_counts[r] items |
       +----------------------+                +----------------------+
```

The sender already knows the destination's chosen receive offset. Setup
exchanges those offsets with MPI and stores them in
`DevicePlanEntry::remote_recv_offset`. The timed kernel performs no host-side
communication.

## What the kernel does

Every CTA works on a different shard of every source-to-destination segment.
For each LSA peer, the kernel:

1. translates the LSA-team rank to its world rank so it can select the correct
   plan entry;
2. gets the local source address with `ncclGetLocalPointer`;
3. gets the destination address with `ncclGetLsaPointer`;
4. copies aligned 16-byte vectors, followed by any remaining elements.

The peer order is offset by both the source rank and the CTA index. At a given
step, different sources write different destinations, and different CTAs do
not all work on the same peer at once. Every CTA owns a different slice of
every message, so changing the visit order does not change which bytes it
copies.

The kernel uses one LSA barrier per CTA. The acquire barrier at entry ensures
that every rank has entered the operation before stores begin. The
acquire-release barrier at exit publishes this rank's peer stores and ensures
that stores from the other ranks are visible before the receiving kernel
returns to its CUDA stream. This lab requests ordinary LSA barriers and passes
`multimem=false`; it does not require NVLS or multimem instructions.

The relevant NCCL 2.31.2 device APIs are:

```cpp
ncclLsaBarrierSession(
    Coop coop, ncclDevComm const &comm, ncclTeamTagLsa team,
    uint32_t index, bool multimem = false);

void ncclLsaBarrierSession::sync(Coop coop, cuda::memory_order order);

void *ncclGetLocalPointer(ncclWindow_t window, size_t byte_offset);

void *ncclGetLsaPointer(
    ncclWindow_t window, size_t byte_offset, int lsa_peer);

int ncclTeamRankToWorld(
    ncclDevComm const &comm, ncclTeam team, int team_rank);
```

`ncclGetLsaPointer` takes an LSA-team rank, while the plan is indexed by world
rank. Convert the LSA peer to a world rank before indexing the plan; do not
assume the two rank spaces have the same numbering.

## Exercise

Open `nccl_lsa_alltoallv.cu` and complete the TODOs in
`nccl_lsa_alltoallv_kernel`:

1. enter the per-CTA LSA barrier with acquire ordering;
2. obtain the local source pointer and the peer receive pointer for each plan
   entry, then call the supplied vector-copy helper;
3. leave the barrier with acquire-release ordering.

All allocation, collective window registration, plan construction, warmup,
timing, and full-buffer validation are already present. Compare with
`nccl_lsa_alltoallv_SOLVED.cu` after working through the kernel.

## Build

Use NCCL 2.31.2 headers and the matching NCCL runtime library:

```bash
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda
```

Keep the headers and runtime from the same NCCL build. NCCL device kernels are
compiled against implementation details in those headers. The default emits
native `sm_100` (GB200) and `sm_103` (GB300) code. Use
`CUDA_ARCHS='90 100 103'` for a compatible fat binary, or `CUDA_ARCH=90` for a
GH200-only build.

## Run on NVLink

Run one MPI rank per GPU in a single LSA domain. For example, inside a Slurm
allocation:

```bash
make run_SOLVED NP=4 \
  LAUNCHER="srun --nodes=1 --ntasks=4 --gpus-per-task=1" \
  RUN_ARGS="--blocks 128"
```

On Lyris, eight GPUs across two trays in one NVL72 form one LSA domain. Lyris
does not expose GPU GRES, so omit `--gpus-per-task`:

```bash
make run_SOLVED NP=8 \
  LAUNCHER="srun --mpi=pmix_v5 --nodes=2 --ntasks=8 --ntasks-per-node=4 --segment=2" \
  RUN_ARGS="--pattern offdiagonal --bytes-per-rank 256M --blocks 1024 --threads 256 --warmup 20 --iters 100"
```

The `1024 x 256` launch was the best 256 MiB/rank starting point in the Lyris
sweep used for this lab. Sweep the CTA count again on another GPU, message
size, or LSA topology.

Change the traffic pattern and payload with `RUN_ARGS`:

```bash
make run_SOLVED NP=4 \
  LAUNCHER="srun --nodes=1 --ntasks=4 --gpus-per-task=1" \
  RUN_ARGS="--pattern sparse --bytes-per-rank 16M --blocks 128 --warmup 10 --iters 50"
```

For large messages, CTA count controls how finely each peer segment is split.
Start with `--blocks 128` and measure; a small grid can leave much of the
NVLink copy bandwidth unused, while the best value depends on the GPU and
message size.

If the communicator is not one LSA domain, the program prints `SKIP` rather
than attempting invalid peer accesses. A successful run ends with output like:

```text
NCCL topology: world=4, LSA=4, rail=1
NCCL LSA AlltoAllV correctness: PASS
NCCL LSA AlltoAllV performance: ... ms/iteration, ... GB/s logical non-self
NCCL LSA AlltoAllV placement payload rates (NCCL LSA): ... GB/s same-domain non-self, 0.000 GB/s cross-domain
```

The reported bandwidth counts payload sent to other ranks and uses the
slowest rank's elapsed time.

Further reading: [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) and [Device memory and LSA pointers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html).
