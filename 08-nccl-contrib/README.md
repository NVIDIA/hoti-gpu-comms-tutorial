# NCCL contrib and Extensions

NCCL is also a foundation for communication libraries and language bindings.
Its public host and device APIs let those projects add workload-specific
operations without changing NCCL core.

There are two places to look for this work:

- [`contrib/` on the NCCL staging branch](https://github.com/NVIDIA/nccl/tree/staging/contrib)
  is the incubator for community and partner projects. The projects use public
  NCCL APIs, have their own maintainers, and are outside NCCL core's release
  quality standards. Current examples include custom device collectives,
  `nccl4rust`, NCCL Checkpoint, UB-X, and NIIN.
- [NCCL Extensions](https://github.com/NVIDIA/nccl-extensions) packages
  reusable communication patterns for AI workloads. These projects have moved
  beyond the staging incubator, but the repository is still evolving and its
  APIs remain subject to change. It currently contains NCCL EP for
  Mixture-of-Experts token routing and NCCL M2N for tensor transfers between
  disjoint process groups.

NVSHMEM has a similar
[`contrib/` area on its development branch](https://github.com/NVIDIA/nvshmem/tree/devel/contrib).
It currently includes `nvshmem4rust`; like NCCL contrib, those projects are
maintained separately from the core library.

## NCCL EP

This chapter uses NCCL EP rather than implementing a new communication
primitive. An MoE layer routes each input token to one or more experts, which
may live on another GPU:

```text
token activations + top-k expert indices
                 |
              dispatch
                 v
       local expert computation
                 |
               combine
                 v
      output in source-token order
```

NCCL EP performs the dispatch and combine phases with NCCL's LSA and GIN
device APIs. The application still owns expert selection, expert computation,
GPU buffers, and their lifetimes.

The supplied Python harness creates deterministic token data and all required
tensors. The exercise is to call the NCCL EP facade to create a group and
handle, dispatch the tokens, and combine the expert output. Completion,
synchronization, verification, and teardown are already present.

1. [Use NCCL EP](01-nccl-ep/)
