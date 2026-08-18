# LSA device API

LSA means *load/store accessible*. NCCL has mapped each member's symmetric
window so that a CUDA kernel can obtain a pointer to an LSA peer and access it
with normal load and store instructions. This lab uses that property to write
a deliberately small all-reduce in one CUDA kernel.

Rank `r` starts with 16 floats whose value is `r + 1`. With the required two
ranks, the kernel reads both source windows and writes `1 + 2 = 3` into every
element of each rank's separate result window.

```text
rank 0 source: 1 1 1 ...        rank 1 source: 2 2 2 ...
          \                         /
           \-- each CTA reads both -/
                    result: 3 3 3 ... on both ranks
```

This loop is intentionally easy to read, not an optimized collective. The
[NCCL device-API guide's LSA example](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel)
uses the same teaching pattern and explains why production kernels use more
specialized building blocks.

## Host-side setup

The setup is in [`../device_api_common.hpp`](../device_api_common.hpp). It
creates an `ncclDevComm`, a symmetric source window, a symmetric result window,
and one LSA barrier. It also verifies that both ranks belong to the local LSA
team before launching the kernel. If that is not true for the selected GPUs or
topology, the program prints `SKIP`.

The source and result windows are separate on purpose. A rank must not
overwrite its input while another rank may still be reading it.

## Exercise

Complete `lsa_sum` in `lsa_device_api.cu`:

1. Construct `ncclLsaBarrierSession<ncclCoopCta>` for `blockIdx.x` and enter
   it with acquire ordering. Every thread in the CTA participates.
2. For every assigned element, iterate over `ncclTeamLsa(dev_comm)` and use
   `ncclGetLsaPointer(source_window, 0, peer)` to load each peer's value.
3. Store the sum through `ncclGetLocalPointer(result_window, 0)`.
4. Leave through the same barrier with release ordering before the kernel
   returns.

The first barrier ensures that all ranks have initialized their source windows
before any rank reads them. The second prevents one rank from tearing down or
reusing its state while a peer can still be in the shared phase.

## Build and expected output

```bash
make
make run_SOLVED
```

Set `CUDA_ARCH`, `CUDA_HOME`, `NCCL_HOME`, or `LAUNCHER` as needed. On
Jupiter, use `LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. With two ranks,
the reference prints:

```text
Rank 0: LSA sum is 3.0 for all 16 elements.
Rank 1: LSA sum is 3.0 for all 16 elements.
```

An unsupported LSA team, device API, NCCL runtime, or GPU capability produces
`SKIP`. Consult the [NCCL device-API requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#requirements)
before changing the program.
