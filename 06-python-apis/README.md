# Python APIs

These labs use Python to drive the same GPU communication mechanisms used in
the C++ chapters. Python does not make the communication synchronous: the host
sets up storage and communicators, queues work to a GPU stream, and must still
establish when a remote write is ready to read.

MPI supplies one process per GPU and is used here for bootstrap and final
checks. It is not on the data path of the NCCL or NVSHMEM operations. The
payloads live in GPU memory: CuPy-backed symmetric arrays in the NVSHMEM4Py
exercises, and PyTorch tensors or raw CUDA allocations in the NCCL4Py and DSL
exercises.

| Lab | Communication model | What the Python code controls |
| --- | --- | --- |
| NVSHMEM4Py host APIs | One-sided RMA to a named PE | Symmetric allocation, `put`/`put_signal`, and stream-ordered completion |
| NCCL4Py host APIs | Matched point-to-point or a collective communicator operation | Tensor storage, NCCL communicator setup, and calls issued from the host |
| Python DSL for device APIs | NCCL or NVSHMEM call made by a JIT-compiled GPU kernel | CuTe compilation, device communicator/window setup, and device-side RMA |

The first two labs are host APIs: Python makes the call, and the GPU carries
out the queued work. The final lab is different: Python builds and launches a
kernel whose code calls NCCL or NVSHMEM directly. Its NCCL exercise registers
windows and creates a device communicator on the host, then uses GIN from the
CuTe kernel. The companion NVSHMEM version uses NVSHMEM's CuTe bindings.

For API details beyond this tutorial, see the [NVSHMEM API
reference](https://docs.nvidia.com/nvshmem/api/index.html) and the [NCCL User
Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html).

## How to work through the chapter

Start with the two NVSHMEM4Py examples. `put.py` uses a world barrier after a
remote put, while `put_signal.py` uses a signal and a remote wait so the
receiving PE can proceed without a collective barrier. Then move to NCCL4Py:
the all-reduce shows one operation invoked by every rank in a communicator;
the send/receive example shows the matching operations required for a
two-rank exchange. Finish with the NCCL CuTe ring put after the RMA model is
clear, then compare it with the NVSHMEM version in the same directory.

1. [NVSHMEM4Py host APIs](01-nvshmem4py-host-apis)
2. [NCCL4Py host APIs](02-nccl4py-host-apis)
3. [Python DSL for device APIs](03-python-dsl-device-apis)

Every exercise keeps a starter named for the task beside a completed
`_SOLVED.py` reference. Run the solution first if you need to establish that
the local Python, CUDA, MPI, and communication-library environment is sound.
