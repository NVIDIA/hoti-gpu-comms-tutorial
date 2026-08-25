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

There are three useful communication scopes in these labs:

| Scope | NCCL path | What the GPU can address |
| --- | --- | --- |
| One LSA domain | LSA | Symmetric windows of load/store-accessible peers |
| Arbitrary GIN peers | Full GIN | A registered window on any world peer |
| Several multi-GPU nodes | LSA + railed GIN | Local LSA peers and the matching GPU position on other nodes |

The first two map naturally to a direct algorithm: each rank copies or puts
one variable-sized message to every destination's advertised receive offset.
Railed GIN is different. It connects GPU 0 to GPU 0 across nodes, GPU 1 to GPU
1, and so on. The hybrid implementation puts each message into a fixed inbox
slot on the matching remote GPU, then scatters it with LSA on the receiving
node. Large messages are sharded across GIN contexts while keeping the same
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
   local GPUs and all network rails across multiple nodes.

Each directory contains a starter, a matching `_SOLVED` reference, a
Makefile, and the exact API and topology requirements for that implementation.

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
`offdiagonal` divides it only among other ranks and is the useful pattern for
isolating the network with one GPU per node. `skewed` is the default and gives
one destination a much larger share. `sparse` includes zero-count pairs.

The programs first validate one iteration, then time warm and measured
iterations. The output separates self, same-host, and inter-host bytes. The
main logical bandwidth is `(same-host + inter-host bytes) / slowest-rank
time`; self copies are timed but are not in that numerator. The two placement
rates use the same elapsed time and show the same-host and inter-host parts
separately.

Use the inter-host rate for an IB comparison only after confirming that the
chosen ranks are in different NVLink domains. On a multi-node NVLink system,
two hosts can still have a direct GPU mapping. The combined logical rate from
a mixed NVLink-plus-IB run is not an IB bandwidth number. The hybrid algorithm
reads each remote byte for the GIN transfer, writes it to the ingress inbox,
then reads and writes it once more for the LSA scatter. That extra local hop is
the cost of rail-only connectivity.

The harness uses MPI only for bootstrap, metadata needed to construct the
reference answer, error reduction, and benchmark alignment. MPI is not the
data path being measured.

## Topology test matrix

Use the solved binaries to establish a baseline before changing the starters.
On Jupiter, `CUDA_ARCH=90` is already the default.

For NVLink or another one-node LSA domain:

```bash
make -C 01-nvshmem-alltoallv
make -C 01-nvshmem-alltoallv run_SOLVED NP=4 \
  LAUNCHER='srun --nodes=1 --ntasks=4 --gpus-per-task=1'

make -C 02-nccl-lsa-alltoallv
make -C 02-nccl-lsa-alltoallv run_SOLVED NP=4 \
  LAUNCHER='srun --nodes=1 --ntasks=4 --gpus-per-task=1' \
  RUN_ARGS='--blocks 128'
```

For an IB-only placement, select one GPU per node and run the NVSHMEM and full
GIN versions. For NVLink plus IB, select the same number of GPUs per node and
run the NVSHMEM and hybrid NCCL versions. The leaf READMEs give complete
commands and explain a legitimate `SKIP` result.

The NCCL labs require NCCL 2.31.2 or newer. They use Hopper-compatible LSA and
GIN APIs only; none uses NVLS, multimem instructions, or a Blackwell-only
feature.
