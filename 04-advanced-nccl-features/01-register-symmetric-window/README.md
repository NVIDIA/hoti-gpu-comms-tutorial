# Register symmetric memory into a window

This is the first step toward the RMA and device-API labs: take two ordinary
GPU allocations and register them as NCCL symmetric windows. The communication
operation is still a familiar `ncclAllGather`; the point of the exercise is to
make the buffer setup and teardown explicit.

Each MPI rank allocates one source buffer and one destination buffer with
`ncclMemAlloc`. It fills its source with a rank-specific byte pattern. All
ranks then register both buffers with `NCCL_WIN_COLL_SYMMETRIC` and call:

```cpp
ncclAllGather(src, dst, src_size, ncclInt8, comm, stream);
```

After the stream has completed, every destination buffer contains each rank's
source segment in rank order. With four ranks, the destination holds four
segments; every rank verifies the same result.

## Why register both buffers?

Window registration describes the operands that NCCL will use. For a
collective, all ranks in the communicator must follow the same registration
model: do not mix registered and unregistered operands in the same operation.
For a symmetric window, every rank participates in compatible registration and
uses the same offset within the registered allocation. The pointers do not
have to have the same CUDA virtual address on each GPU.

This lab uses `ncclMemAlloc` because window registration requires
VMM-compatible memory. See [Window Registration in the NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-registration)
for the allocation and symmetry requirements, plus the equivalent NCCL example.

## Files

- `register_symmetric_window.cpp` is the starter. Communicator creation,
  allocation, initialization, and verification are present; registration,
  `ncclAllGather`, and deregistration are TODOs.
- `register_symmetric_window_SOLVED.cpp` is the checked reference.
- `Makefile` builds both versions and provides run targets.

## Exercise

Complete the missing calls in this order:

1. Create `src_win` and `dst_win` with `ncclCommWindowRegister`, passing
   `NCCL_WIN_COLL_SYMMETRIC` for each allocation.
2. Enqueue the all-gather on the existing nonblocking CUDA stream.
3. Synchronize that stream before copying the destination to the host and
   checking it.
4. Deregister both windows before freeing either allocation.

The handles are local bookkeeping objects. Keep them alive with their
allocations; neither one is a pointer you can dereference or exchange with a
peer rank.

## Build and run

Use one MPI rank per visible GPU:

```bash
make
make run_SOLVED NP=4
```

The Makefile defaults to `mpirun` and leaves GPU visibility to the launcher.
Common overrides are:

```bash
make run_SOLVED NP=2
make run_SOLVED LAUNCHER="srun --ntasks=4 --gpus-per-task=1"
make run_SOLVED NP=4 ARGS="-b 65536"
```

`ARGS=-b ...` selects the per-rank source size in bytes; the default is 1 MiB.
A successful run
prints `Rank <n>: AllGather verification passed.` on every rank. The starter
will not reach that result until its TODOs are implemented.

## Requirements and failure modes

The program checks for NCCL 2.27.6 or newer and requires one visible GPU per
local MPI rank. If registration fails, first confirm that both the headers and
linked NCCL runtime provide `ncclMemAlloc` and `ncclCommWindowRegister`, then
check the VMM allocation requirements in the NCCL documentation. Do not
substitute `cudaMalloc` merely to make the example compile: that changes the
property the lab is teaching.
