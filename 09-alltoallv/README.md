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
1, and so on. The hybrid implementation first packs messages by destination
node, sends each packet on the source GPU's rail, and then scatters it with LSA
on the receiving node.

NVSHMEM exposes a single put API across these topologies. Its implementation
uses `nvshmem_ptr` to detect direct peer mappings and chooses a block-scoped
put for that path while retaining an ordinary device put for network peers.
The correctness protocol remains the same.

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
--pattern uniform|skewed|sparse
--bytes-per-rank N[K|M|G]
--warmup N
--iters N
--blocks N
--threads N
```

`--bytes-per-rank` is each source rank's total logical payload before aligned
gaps. `skewed` is the default and gives one destination a much larger share.
`sparse` includes zero-count pairs. The programs first run one iteration and
validate every rank, then time warm and measured iterations. Reported
aggregate GB/s counts non-self payload bytes once; it is useful for comparing
these implementations under the same placement, not as a claim about a
particular link's line rate.

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
  LAUNCHER='srun --nodes=1 --ntasks=4 --gpus-per-task=1'
```

For an IB-only placement, select one GPU per node and run the NVSHMEM and full
GIN versions. For NVLink plus IB, select the same number of GPUs per node and
run the NVSHMEM and hybrid NCCL versions. The leaf READMEs give complete
commands and explain a legitimate `SKIP` result.

The NCCL labs require NCCL 2.31.2 or newer. They use Hopper-compatible LSA and
GIN APIs only; none uses NVLS, multimem instructions, or a Blackwell-only
feature.
