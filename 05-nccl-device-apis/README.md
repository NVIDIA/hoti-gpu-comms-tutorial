# NCCL device APIs

The [Advanced NCCL features](../04-advanced-nccl-features/) chapter registers
symmetric windows for host-side operations. This chapter uses that same
registered-memory model from a CUDA kernel. The kernel does not create a
communicator and it does not call host APIs. The host prepares a device
communicator and symmetric windows, passes those values as kernel arguments,
and keeps them alive until the kernel is done.

Both exercises use [`device_api_common.hpp`](device_api_common.hpp). Read it
before starting either kernel: it is the lifecycle code that makes the device
calls valid.

## What the common setup does

`device_api::prepare` follows the same sequence an application would use:

1. Require exactly two MPI ranks, select one CUDA device per rank, and create
   `ncclComm_t`.
2. Call `ncclCommQueryProperties` and skip if this communicator does not
   support the device API. The GIN lab also skips when GIN is unavailable.
3. Allocate the payload, result, and signal memory with `ncclMemAlloc`, then
   register the required allocations as symmetric windows.
4. Initialize `ncclDevCommRequirements` and request the resources used by the
   kernel: an LSA barrier for the LSA lab or GIN connectivity for the GIN lab.
5. Call `ncclDevCommCreate` collectively and pass the returned `ncclDevComm`
   and windows to a regular CUDA kernel launch.

Cleanup runs in the reverse direction: synchronize the stream, destroy the
device communicator, deregister the windows, free their allocations, then
destroy `ncclComm_t`. A kernel must not still be running when
`ncclDevCommDestroy` is called.

The [NCCL device API host-setup reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html)
describes `ncclCommQueryProperties`, the requirements object, and the
collective nature of `ncclDevCommCreate`.

## What is passed to the kernel

`ncclDevComm` is a device-side description of the communicator. Pass it by
value as the examples do, but do not build application logic around its
internal fields; NCCL documents only a limited set as stable. `ncclWindow_t`
identifies a registered allocation and is also passed as a kernel argument.

Inside the kernel, `ncclCoopCta()` identifies the participating threads of one
CTA. It is used for the LSA barrier and the GIN signal wait. It does **not**
require or imply a CUDA cooperative-grid launch.

## The two labs

1. [LSA device API](01-lsa-device-api) uses direct load/store access to
   symmetric peer windows. It needs the two ranks to be in one LSA team.
2. [GIN put device API](02-gin-put-device-api) issues a remote GPU-initiated
   put and waits for its remote completion signal. It is availability-dependent
   on the queried communicator and on the system's network configuration.

Build both with `make`, then run either completed reference:

```bash
make
make run-lsa_SOLVED
make run-gin_SOLVED
```

## Capability checks are expected

These focused labs are intentionally small, not portable fallbacks. They print
`SKIP` for an older NCCL runtime, fewer than two ranks/GPUs, GPU compute
capability below 7.0, missing device-API support, an incomplete LSA team, or
unavailable GIN. A skip tells you which prerequisite to inspect; it is not a
signal to force environment variables or bypass the check.

The [device-initiated communication guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#requirements)
lists the CUDA, GPU, NIC, driver, and topology requirements. It also documents
the cross-version limitation that GIN kernels need recompilation when NCCL is
upgraded.

The [Memory semantics](../03-memory-semantics/) chapter introduced the same
ordering, completion, and notification questions through NVSHMEM.
