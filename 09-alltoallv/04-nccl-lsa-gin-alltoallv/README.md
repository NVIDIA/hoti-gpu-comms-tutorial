# NCCL AlltoAllV with LSA and railed GIN

This version is for multiple nodes with several GPUs per node. It assigns a
different path to each part of the route:

- LSA stores move data between GPUs in the same local LSA domain.
- Railed GIN moves data between matching GPU positions on different nodes.

Railed GIN is not a cheaper form of full GIN. A member of `ncclTeamRail`
cannot address an arbitrary world rank. On a four-GPU node, GPU 2's rail team
contains GPU 2 on every other node; it does not contain GPU 0, 1, or 3 on
those nodes. Calling `gin.put` on that team with an arbitrary destination is
invalid.

The collective therefore takes two hops for a remote destination:

```text
source node                         destination node

GPU 0 ==== rail 0 =================> GPU 0 -- LSA --> final local GPU
GPU 1 ==== rail 1 =================> GPU 1 -- LSA --> final local GPU
GPU 2 ==== rail 2 =================> GPU 2 -- LSA --> final local GPU
GPU 3 ==== rail 3 =================> GPU 3 -- LSA --> final local GPU
```

Every source GPU packs all of its messages for one destination node into one
packet. It sends that packet to the GPU at the same local position on the
destination node. The receiving GPU reads the packet header and scatters each
submessage to its final local GPU through an LSA pointer. This keeps every
rail active while reducing network operations from one per destination GPU to
one per destination node.

## Packet format

The host allocates symmetric outbox and inbox windows. Each remote-node slot
starts with one `HybridPacketItem` for each destination LSA rank:

```text
+----------------------+-------------------------------+
| item 0 ... item L-1  | aligned payloads for L GPUs  |
+----------------------+-------------------------------+
```

An item records the payload size, its offset in the packet, and its final
offset in the destination GPU's receive window. The packet is compact: zero
length messages occupy an item but no payload space.

## Ordering and completion

The implementation uses three handoffs.

1. An LSA barrier brackets local stores and packet construction.
2. Each remote `gin.put` carries a weak signal indexed by the source node.
   That signal covers its own packet, which is exactly what the receiver needs
   before reading the header and payload.
3. After the signal waits, `gin.flush` makes the local outbox safe to reuse.
   A closing hybrid barrier makes all LSA scatters visible and completes the
   collective before the next iteration.

Signal values are cumulative. The host passes an `epoch` that increases on
every launch, so a signal slot never needs to be reset while another rank may
still update it.

The two kernels use these NCCL device APIs:

```cpp
void *ncclGetLocalPointer(ncclWindow_t window, size_t byte_offset);

void *ncclGetLsaPointer(
    ncclWindow_t window, size_t byte_offset, int lsa_rank);

int ncclTeamRankToWorld(
    ncclDevComm const &comm, ncclTeam team, int team_rank);

ncclBarrierSession(
    Coop coop, ncclTeamTagLsa, ncclDevComm const &comm, uint32_t index);

ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGin gin, uint32_t index);

void ncclGin::put(
    ncclTeam team, int peer,
    ncclWindow_t destination_window, size_t destination_byte_offset,
    ncclWindow_t source_window, size_t source_byte_offset, size_t bytes,
    ncclGin_WeakSignalInc remote_action);

void ncclGin::waitSignal(
    Coop coop, ncclGinSignal_t signal, uint64_t least);

void ncclGin::flush(Coop coop);
```

`ncclGetLocalPointer` names this rank's storage. `ncclGetLsaPointer` names the
same offset on an LSA peer. `ncclTeamRankToWorld` converts a rail or LSA team
rank before it is used to index the world-sized AlltoAllV plan. The weak
signal belongs to one packet put: observing it makes that packet readable,
while `flush` separately makes the sender's outbox safe to reuse.

## Exercise

Complete the two kernels in `nccl_lsa_gin_alltoallv.cu`.

In `pack_and_deliver_local`:

1. Use `ncclGetLsaPointer` to deliver messages whose destination is in the
   local LSA team.
2. Use `ncclTeamRankToWorld` to find the first world rank on each destination
   node.
3. Fill that node's packet header and copy the variable-sized payloads into
   its outbox slot.

In `exchange_rails_and_scatter`:

1. Construct `ncclGin` on context 0 and a world
   `ncclBarrierSession<ncclCoopCta>`.
2. Send one packet to every remote member of `ncclTeamRail` with
   `gin.put(..., ncclGin_WeakSignalInc{source_node})`.
3. Wait for the current epoch from every remote source node and call
   `gin.flush` on the issuing CTA.
4. Read each inbox header and copy its payloads through LSA pointers to their
   final receive offsets.

The starter already contains the plan exchange, symmetric allocation and
window registration, device-communicator requirements, launch loop,
validation, and timing. The reference is
`nccl_lsa_gin_alltoallv_SOLVED.cu`.

## Build and run

This lab needs at least two nodes, a uniform LSA team size on every node, and
railed GIN support in NCCL 2.31.2 or newer.

```bash
make
make run_SOLVED NP=4 \
  LAUNCHER='srun --nodes=2 --ntasks=4 --ntasks-per-node=2 --gpus-per-task=1'
```

The run target defaults to `NCCL_IB_MERGE_NICS=0` and `NCCL_CROSS_NIC=0`.
That asks NCCL for corresponding GPU/NIC rails rather than arbitrary
cross-NIC connections. The variables can be overridden on the Make command
line if a system has a different validated mapping. The Makefile also puts
`NCCL_HOME/lib` first in `LD_LIBRARY_PATH` so the device headers and runtime
library come from the same installation.

The four-rank command is the smallest mixed placement. To exercise every GPU
and rail on two eight-GPU nodes, use 16 tasks with 8 tasks per node.

Pass the same workload controls used by the other AlltoAllV labs through
`RUN_ARGS`:

```bash
make run_SOLVED NP=4 \
  LAUNCHER='srun --nodes=2 --ntasks=4 --ntasks-per-node=2 --gpus-per-task=1' \
  RUN_ARGS='--pattern sparse --bytes-per-rank 16M --iters 50'
```

The program prints the discovered world, LSA, and rail sizes. It prints
`SKIP` instead of guessing when the placement does not form uniform LSA and
rail teams or when railed GIN is unavailable. A successful run reports both
correctness and the slowest-rank iteration time.

This implementation uses ordinary Hopper-compatible loads, stores, and GIN
operations. It does not require NVLS, multimem instructions, or a
Blackwell-only device feature.
