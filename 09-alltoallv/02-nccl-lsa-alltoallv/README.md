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

The algorithm is the simplest one in this chapter: every rank stores every
outgoing message straight into its peer's receive window, all at the same
time. There is no handshake per message and no staging buffer. Only two things
have to be decided - when it is safe to start, and which bytes each CTA owns.

For each LSA peer, the kernel:

1. translates the LSA-team rank to its world rank so it can select the correct
   plan entry;
2. takes one contiguous, 16-byte-aligned slice of that message with
   `shard_slice(count, blockIdx.x, gridDim.x, ...)`;
3. gets the local source address with `ncclGetLocalPointer`;
4. gets the destination address with `ncclGetLsaPointer`;
5. copies aligned 16-byte vectors, followed by any remaining elements.

The peer order is offset by both the source rank and the CTA index. At a given
step, different sources write different destinations, and different CTAs do
not all work on the same peer at once. Every CTA owns a different slice of
every message, so changing the visit order does not change which bytes it
copies.

`copy_values` keeps four 16-byte stores in flight per thread rather than one.
This exposes more independent stores to the NVLink write path without changing
which bytes each thread owns.

[04-nccl-lsa-gin-alltoallv](../04-nccl-lsa-gin-alltoallv/) shares `shard_slice`
and the same peer-rotation idea, but keeps its `copy_values` at one store in
flight on purpose. The extra registers cost more occupancy than they buy in a
kernel that is network-bound. The helper is a good place to look at the two
labs side by side.

The implementation stays with 16-byte vectors. Wider source-level stores are
not useful when the compiler lowers them into multiple sparse 16-byte stores.

The kernel uses one LSA barrier per CTA. The acquire barrier at entry ensures
that every rank has entered the operation before stores begin. The
acquire-release barrier at exit publishes this rank's peer stores and ensures
that stores from the other ranks are visible before the receiving kernel
returns to its CUDA stream. This lab requests ordinary LSA barriers and passes
`multimem=false`.

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
