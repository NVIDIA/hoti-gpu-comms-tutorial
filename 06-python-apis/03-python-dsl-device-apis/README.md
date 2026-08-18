# Python DSL device APIs

This lab puts one integer around a two-rank ring from a CuTe kernel. Rank 0
sends `1` to rank 1, and rank 1 sends `2` to rank 0. The main exercise uses
the NCCL device API and GIN; the directory also keeps the NVSHMEM CuTe version
for comparison.

```text
rank 0: send 1  ----->  rank 1: receive 1
rank 0: receive 2  <-----  rank 1: send 2
```

The NCCL version uses two registered windows. The send window must remain
unchanged until GIN has consumed it, while the receive window can be written by
the preceding rank. Using one window for both would allow the incoming put to
overwrite a rank's source before its outgoing put has read it.

## Files

- `ring_put.py` is the NCCL starter. The communicator, registered windows,
  device communicator, CuTe kernel, and launch are already set up. The two
  TODOs are the device-side put-with-signal and the matching wait.
- `ring_put_SOLVED.py` is the completed NCCL reference.
- `ring_put_nvshmem_cute.py` and `ring_put_nvshmem_cute_SOLVED.py` are the
  equivalent NVSHMEM CuTe starter and solution.
- `check_env.py` checks the NCCL4Py and CuTe pieces used by `ring_put.py`.
- `check_nvshmem_env.py` checks the NVSHMEM companion environment.

## NCCL setup

`ring_put.py` expects NCCL4Py with the CuTe device bindings, NCCL 2.30.7 or
newer, a matching `libnccl_device.bc`, and `nvidia-cutlass-dsl` 4.5.2 or
newer. The NCCL and bitcode builds must match. Importing
`nccl.core.device.cute` locates and links the bitcode automatically; this lab
does not use NVSHMEM's explicit JIT-link and kernel-registration sequence.

Build NCCL4Py from the [public NCCL source tree](https://github.com/NVIDIA/nccl/tree/master/bindings/nccl4py). With CUDA 13 and `CUDA_HOME` configured, clone NCCL outside this tutorial checkout, install the NCCL path, and run the environment check:

```bash
git clone https://github.com/NVIDIA/nccl.git /path/to/nccl
make install NCCL4PY_SOURCE=/path/to/nccl/bindings/nccl4py
make check-env
```

The NCCL4Py `cu13` extra installs the matching NCCL, CUDA Python, and CuTe DSL
dependencies. The install target adds CuPy and MPI for this exercise.

The host setup registers separate source and destination buffers, then requests
one full-connectivity GIN context and one signal slot:

```python
send_win = comm.register_window(send_buf)
recv_win = comm.register_window(recv_buf)
requirements = nccl.NCCLDevCommRequirements(
    gin_connection_type=nccl.NcclGinConnectionType.FULL,
    gin_signal_count=1,
)
dev_comm = comm.create_dev_comm(requirements=requirements)
```

## Exercise

Inside `ring_put_kernel`, `world` names the communicator team, `gin` selects
GIN context 0, and `coop` names the participating CTA. Complete the two TODOs:

1. Call `gin.put` to copy the local `send` tensor into `peer`'s `recv` window.
   Attach signal 0 and increment it by one when the payload arrives.
2. Call `gin.wait_signal` so this rank does not read its receive buffer until
   the preceding rank's signal reaches one.

Both operations are called by the CTA; do not put them inside a single-thread
branch. The signal makes the incoming payload visible to the wait. It does not
complete unrelated GIN operations or provide a barrier across the communicator.

Run the starter and solution with two ranks:

```bash
make run
make run_SOLVED
```

The solution prints `rank 0: received 2` and `rank 1: received 1`. A scheduler
launcher can be supplied in the usual way:

```bash
make run_SOLVED LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
```

The companion NVSHMEM version is available through `make run-nvshmem` and
`make run-nvshmem_SOLVED` after `make check-nvshmem-env`.

The NVSHMEM companion needs NVSHMEM4Py with its CuTe bindings and device
bitcode matching the GPU architecture. Install its public CUDA 13 package set
before running the companion check:

```bash
make install-nvshmem
make check-nvshmem-env
```

The check prints the selected bitcode library before the exercise runs.

The NCCL calls here follow the current
[public NCCL4Py CuTe examples](https://github.com/NVIDIA/nccl/tree/master/bindings/nccl4py/examples/cute).
