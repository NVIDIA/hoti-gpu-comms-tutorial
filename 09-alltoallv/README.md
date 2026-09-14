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
Each AlltoAllV invocation uses one payload kernel; one-time metadata setup and
optional validation helpers are outside the collective itself.

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
--gin-contexts N
--gin-queue-depth N
--profile-phases
```

`--bytes-per-rank` is each source rank's total logical payload before aligned
gaps. `uniform` divides it across every rank, including the source itself.
`offdiagonal` divides it only among other ranks. Use it to isolate the network
with one GPU per direct-access domain. `skewed` is the default and gives one
destination a much larger share. `sparse` includes zero-count pairs.

`--gin-contexts` and `--gin-queue-depth` apply to the NCCL GIN and LSA +
railed-GIN exercises. A context value of `0` (the default) requests one GIN
context per CTA, so it preserves the original `--blocks` behavior. Set it to
`1..--blocks` to sweep context sharing independently of CTA count. On the
railed-GIN exercise, sharing contexts measurably hurts: 8 or 16 contexts across
40 CTAs cost about 13% against one context per CTA.

`--profile-phases` applies to the solved LSA + railed-GIN implementation. It
keeps the one-kernel collective intact and stamps five per-CTA timestamps, then
splits the measured time across issue plus local delivery, the wait for inbound
shards, the LSA scatter, the GIN flush, and the completion barrier. It is a
diagnostic run, not a headline-performance run. It is also the fastest way to
see why the hybrid sits well under the rail line rate; see
[04-nccl-lsa-gin-alltoallv](04-nccl-lsa-gin-alltoallv/).

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
