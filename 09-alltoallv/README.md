# A topology-aware AlltoAllV

All-to-all communication appears in mixture-of-experts routing, distributed
hash tables, graph workloads, transposes, and repartitioning. `AlltoAllV` is
the irregular form: every source can send a different amount of data to every
destination.

For rank `r`, the collective consumes:

- `send_counts[d]`: elements sent from `r` to destination `d`;
- `send_offsets[d]`: the start of that message in `r`'s send buffer;
- `recv_counts[s]`: elements received by `r` from source `s`;
- `recv_offsets[s]`: where that message belongs in `r`'s receive buffer.

The counts have to agree across each source/destination pair, but neither the
counts nor the offsets need to be equal. That makes the operation more than a
fixed-size all-to-all with a different name: work assignment, metadata
exchange, notification, and buffer reuse all have to handle empty and
unbalanced messages correctly.

```text
source rank 0                         destination rank 2

send buffer                           receive buffer
+------+----------+---+              +---+------+----------+
| -> 1 |   -> 2   |gap|   ------->   |gap| <- 0 |   <- 3   |
+------+----------+---+              +---+------+----------+
         count=variable                    offset=variable
```

This chapter builds the same contract with NVSHMEM and with NCCL device APIs.
The common test harness deliberately generates aligned but non-contiguous
offsets and checks untouched gaps as well as payload values. A solution that
accidentally implements fixed-size all-to-all will fail validation.

## Why topology changes the NCCL algorithm

The NCCL labs use three communication scopes:

| Scope | NCCL path | What the GPU can address |
| --- | --- | --- |
| One LSA domain | LSA | Symmetric windows of load/store-accessible peers |
| Several LSA domains | Full GIN | A registered window on any world peer |
| Several LSA domains | LSA + railed GIN | LSA peers and the matching LSA rank in other domains |

LSA and full GIN use a direct algorithm: each rank copies or puts one
variable-sized message to every destination's advertised receive offset.
Railed GIN is different. It connects one LSA rank to the same LSA rank in
another domain. The hybrid implementation puts each message into a fixed
inbox slot on that ingress GPU, then scatters it with LSA inside the receiving
domain. Large messages are sharded across GIN contexts while keeping the same
slot layout.

The NVSHMEM lab uses one source implementation across these topologies. It
uses `nvshmem_ptr` to detect direct peer mappings, chooses a block-scoped put
for that path, and uses explicit QP handles for network peers. The correctness
protocol remains the same.

## Exercises

1. [NVSHMEM AlltoAllV](01-nvshmem-alltoallv/) exchanges the metadata on the
   device and uses one put-and-signal protocol across NVLink and IB.
2. [NCCL LSA AlltoAllV](02-nccl-lsa-alltoallv/) writes directly through LSA
   pointers. Run it inside one LSA domain.
3. [NCCL GIN AlltoAllV](03-nccl-gin-alltoallv/) sends variable-sized puts to
   arbitrary world peers. Run it where full GIN is available.
4. [NCCL LSA + railed GIN AlltoAllV](04-nccl-lsa-gin-alltoallv/) uses all
   LSA peers and all network rails across multiple LSA domains.

Each directory contains a starter, a matching `_SOLVED` reference, a
Makefile, and the exact API and topology requirements for that implementation.

## What changes between NVSHMEM and NCCL

The data-movement contract is identical in all four labs. The route selection
and completion mechanism are not:

| Question | NVSHMEM | NCCL device API |
| --- | --- | --- |
| Which source file handles the placement? | One implementation chooses each route with `nvshmem_ptr`. | Separate LSA, full-GIN, and LSA + railed-GIN implementations. |
| How does a direct peer receive data? | A block-scoped put-with-signal. | Threads store through an LSA pointer. |
| How does a network peer receive data? | A QP-specific put-with-signal. | A full-GIN put, or a railed-GIN put followed by an LSA scatter. |
| What tells the receiver the payload is ready? | The signal attached to each put. | An LSA barrier for direct stores or the weak signal attached to a GIN put. |
| What protects source-buffer reuse? | A sender-local quiet over the QPs. | An LSA barrier or a sender-local GIN flush. |

Use the same offdiagonal payload and placement when comparing the APIs. Record
the route counts, chunk or shard size, CTA count, QP or GIN-context count,
logical rate, matching-primitive rate, and hardware SoL. A faster run with a
different route count is a topology change, not an implementation comparison.

## Shared workload and output

Every executable accepts the same workload options:

```text
--pattern uniform|offdiagonal|skewed|sparse
--bytes-per-rank N[K|M|G]
--warmup N
--iters N
--blocks N
--threads N
```

`--bytes-per-rank` is each source rank's total logical payload before aligned
gaps. `uniform` divides it across every rank, including the source itself.
`offdiagonal` divides it only among other ranks. Use it to isolate the network
with one GPU per direct-access domain. `skewed` is the default and gives one
destination a much larger share. `sparse` includes zero-count pairs.

The programs first validate one iteration, then time warm and measured
iterations. The output separates self, same-domain, and cross-domain bytes.
NCCL uses its LSA team as the domain. NVSHMEM uses the set of PEs for which
`nvshmem_ptr` returns a direct pointer. On an NVL72, two GPUs can be on
different Linux hosts and still be NVLink peers.

The main logical bandwidth is `(same-domain + cross-domain bytes) /
slowest-rank time`; self copies are timed but are not in that numerator. The
two placement rates use the same elapsed time and show the NVLink and IB parts
separately. The combined logical rate from a mixed run is not an IB bandwidth
number. The hybrid algorithm writes each cross-domain byte to an ingress
inbox, then uses one local scatter when the destination has a different LSA
rank. That extra local hop is the cost of rail-only connectivity.

## The three tuning steps

The solved versions use the same three changes in forms appropriate to each
API:

1. **Shard a large peer message.** One transfer does not create enough
   parallel work. NCCL stripes shards over GIN contexts; NVSHMEM uses smaller
   direct chunks and larger network chunks.
2. **Map the shards to the hardware.** Work is spread over CTAs, contexts, QP
   handles, and rails. The hybrid NCCL path sends only to the matching rail
   peer, then scatters with LSA.
3. **Separate issue from handoff.** Nonblocking puts run before the receiver
   waits. Put-with-signal or a weak GIN signal makes the payload visible; one
   delayed quiet or flush protects buffer reuse. There is no quiet or flush
   after every shard.

Use `--blocks` to expose the first two effects. The NVSHMEM lab also exposes
`HOTI_ALLTOALLV_CHUNK_BYTES`, `HOTI_ALLTOALLV_NETWORK_CHUNK_BYTES`, and
`HOTI_ALLTOALLV_NETWORK_QPS`. Change one setting at a time and keep the
offdiagonal payload fixed.

## Performance target on GB300 NVL72

Use 75% of the hardware path SoL as the stretch target for large, balanced,
offdiagonal traffic. Also measure the matching primitive on the same
allocation: direct peer copy for NVLink and a large put for GIN or NVSHMEM
IBGDA. If the primitive itself is below the hardware line rate, report both
percentages and require the AlltoAllV to reach at least 75% of that measured
primitive reference. The primitive is a data-path baseline, not necessarily a
strict upper bound: the collective may expose more parallel work. This keeps
an API or transport limit visible instead of crediting it to the collective.
Do not apply a fixed percentage to small, sparse, or strongly skewed messages;
launch and load-imbalance costs dominate those cases.

For the Lyris placements below, the one-way hardware rates are 900
GB/s of NVLink per GPU (half of the
[1.8 TB/s bidirectional rate](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/network-logical-architecture.html))
and 100 GB/s for each
[800 Gb/s ConnectX-8 rail](https://www.nvidia.com/en-us/data-center/gb300-nvl72/).
This gives:

| Placement | Large-message logical SoL | 75% target |
| --- | ---: | ---: |
| 8 GPUs in one NVL72 | 7.2 TB/s | 5.4 TB/s |
| 4 GPUs in four NVL72s, one rail each | 0.4 TB/s | 0.3 TB/s |
| 2 NVL72s x 8 GPUs, LSA + railed GIN | 3.0 TB/s | 2.25 TB/s |

These are aggregate logical send rates across all ranks, matching the metric
printed by the programs; they are not per-GPU bandwidths.

The IB-only row deliberately selects one HCA per rank so the NCCL and
NVSHMEM runs have the same denominator. An NVSHMEM run that enables four
800 Gb/s HCAs for each single-PE tray has a 1.6 TB/s aggregate raw ceiling,
not 0.4 TB/s. Label that as a separate multi-port result.

The hybrid number is traffic weighted. With two eight-GPU domains,
offdiagonal AlltoAllV sends `7/15` of its bytes inside an LSA domain and
`8/15` across the network. The network is the limiting path, so the combined
logical SoL is `1.6 TB/s / (8/15) = 3.0 TB/s`.

## Lyris reference comparison

These are the best validated 256 MiB/rank offdiagonal results from the tuning
runs. The NVSHMEM column uses the same source in all three rows. The NCCL
column switches between the topology-specific implementations.

| Placement | NVSHMEM | NCCL device API | NCCL implementation |
| --- | ---: | ---: | --- |
| 8 GPUs in one NVL72 | 3.76 TB/s (52% raw SoL) | 4.67 TB/s (65%) | LSA |
| 4 GPUs in four NVL72s, one rail each | 205.7 GB/s (51%) | 225.2 GB/s (56%) | Full GIN |
| 2 NVL72s x 8 GPUs | 1.52 TB/s (51%) | 1.45 TB/s (48%) | LSA + railed GIN |

Every run passed the initial full-buffer check and the receive-buffer reuse
check. The single-rail NVSHMEM and NCCL figures were measured back-to-back on
the same allocation. The other rows report each implementation's best
validated run.

None reaches 75% of raw hardware SoL. The data-path references were below raw
SoL as well. In their matching tuning runs, the solved NVSHMEM implementation
reached 98% of the direct reference, 101% of the single-rail reference, and
about 100% of the mixed reference. The NCCL implementations reached 87% of
the LSA reference, 88% of the full-GIN reference, and 89% of the strict hybrid
component roof. All six therefore clear the practical target of 75% of their
measured references. Those percentages come from paired reference/solution
runs, not from dividing the best-of-sweep table above. The full-GIN pair used
64 MiB/rank; the other pairs used 256 MiB/rank.

The comparison is topology dependent. NCCL led on the direct LSA and
single-rail full-GIN placements. NVSHMEM's single mixed kernel slightly led
the staged NCCL algorithm on the two-NVL72 placement.

The harness uses MPI only for bootstrap, metadata needed to construct the
reference answer, error reduction, and benchmark alignment. MPI is not the
data path being measured.

## Topology test matrix

Use the solved binaries to establish a baseline before changing the starters.
On Jupiter, `CUDA_ARCH=90` is already the default.

For NVLink within one LSA domain:

```bash
make -C 01-nvshmem-alltoallv
make -C 01-nvshmem-alltoallv run_SOLVED NP=4 \
  LAUNCHER='srun --nodes=1 --ntasks=4 --gpus-per-task=1'

make -C 02-nccl-lsa-alltoallv
make -C 02-nccl-lsa-alltoallv run_SOLVED NP=4 \
  LAUNCHER='srun --nodes=1 --ntasks=4 --gpus-per-task=1' \
  RUN_ARGS='--blocks 128'
```

On Lyris, `--segment` is the number of compute trays placed in one NVL72 base
block. These allocations produce the three benchmark placements:

```text
NVLink:   --nodes=2 --segment=2
IB:       --nodes=4 --segment=1 --spread-segments
Hybrid:   --nodes=4 --segment=2 --spread-segments
```

Use four tasks and GPUs per tray for the NVLink and hybrid rates in the table.
Use one task and GPU per tray for the IB-only rate. Lyris does not expose its
GPUs as Slurm GRES, so omit `--gpus-per-task`; each program selects a GPU from
the MPI local rank. The leaf commands select PMIx explicitly. Add `--overlap`
when launching them from inside an existing `srun` step. The leaf READMEs give
complete commands and explain a legitimate `SKIP` result.

The NCCL labs require NCCL 2.31.2 or newer. They use Hopper-compatible LSA and
GIN APIs only; none uses NVLS, multimem instructions, or a Blackwell-only
feature.
