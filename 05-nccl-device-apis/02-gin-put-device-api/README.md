# GIN put device API

GIN is NCCL's GPU-initiated networking path. This lab keeps the operation to
one value so that the remote-completion protocol is impossible to miss: rank 0
writes `28` into its local payload window, then its CUDA kernel puts that
64-bit value into rank 1's matching window. The put also increments a signal
in rank 1's signal window. Rank 1 waits for that signal in its kernel before
the host reads and checks the payload.

```text
rank 0 kernel: payload = 28; gin.put(payload window, rank 1)
                              └── remote action: increment rank-1 signal
rank 1 kernel: waitSignal(signal == 1); now payload is safe to consume
```

The signal is required because starting a target-side kernel says nothing
about when rank 0's remote write has arrived. The remote action gives the
destination a stream- and kernel-visible completion condition.

## Host-side setup and availability

[`../device_api_common.hpp`](../device_api_common.hpp) creates the ordinary
communicator, queries its properties, allocates and symmetrically registers the
payload and signal windows, and creates `ncclDevComm`. It requests GIN in
`ncclDevCommRequirements` only after the communicator reports GIN support.

GIN is not a generic replacement for every local device store. It needs the
documented GPU, NIC, driver, GPUDirect RDMA, and topology support. The program
therefore prints `SKIP` when GIN cannot be created on the selected system.
Use the [NCCL device-initiated communication requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#requirements)
to assess a platform rather than assuming that a GPU with device-API support
also has GIN support.

The starter's common setup currently uses `ginForceEnable`. NCCL documents
that field as deprecated in favor of `ginConnectionType`; it remains here to
match the lab's reference setup, not as guidance for new production code. The
[host-side setup reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#nccldevcommrequirements)
has the current requirements fields.

## Exercise

Complete `gin_put` in `gin_put_device_api.cu`:

1. On world rank 0, obtain the local payload pointer and store `kPutValue`.
2. Call `gin.put` on `ncclTeamWorld(dev_comm)` for peer 1, using the payload
   window as both source and destination window at byte offset `0`.
3. Attach `ncclGin_VASignalInc{signal_window, 0}` as the remote action.
4. On world rank 1, call
   `gin.waitSignal(ncclCoopCta(), signal_window, 0, 1)` before returning.

The kernel intentionally launches with one thread, so `ncclCoopCta()` contains
one participant. Larger kernels must make the cooperative scope and all
participants' ordering explicit; this lab is only about the put and its remote
completion signal.

## Build and expected output

```bash
make
make run_SOLVED
```

The Makefile launches two MPI ranks by default. Override `CUDA_ARCH`,
`CUDA_HOME`, `NCCL_HOME`, or `LAUNCHER` as needed. On Jupiter, use
`LAUNCHER="srun --ntasks=2 --gpus-per-task=1"`. A successful run prints:

```text
Rank 1 received the GIN put value 28.
```

The starter is not expected to reach that message until its TODOs are complete.
If the reference prints `SKIP`, retain the capability check and investigate the
reported prerequisite instead of removing it.
