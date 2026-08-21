# Fused GEMM + all-reduce

Each GPU computes one 16x16 `float` GEMM tile. The kernel then communicates
the local tiles and leaves the elementwise sum on both GPUs before it returns.
Keeping the matrix small makes the compute-to-communication handoff visible in
one kernel; this is a teaching example, not a tuned tensor-core GEMM.

The lab has NVSHMEM and NCCL implementations of the same operation:

| Implementation | Publish the local tile | Wait for the peer | Read the peer tile |
| --- | --- | --- | --- |
| NVSHMEM | Device put-with-signal into the peer's inbox | Wait on the local signal | Read the local inbox |
| NCCL | Store into a symmetric NCCL window | LSA barrier | Load the peer's window with an LSA pointer |

Both versions run on two GPUs. The NCCL version uses the ordinary LSA path for
a single NVLink domain. It does not use NVLS, `multimem`, or any Blackwell-only
instructions.

## Source files

- `fused_gemm_allreduce.cu`: NVSHMEM starter
- `fused_gemm_allreduce_SOLVED.cu`: NVSHMEM reference
- `fused_gemm_allreduce_nccl.cu`: NCCL starter
- `fused_gemm_allreduce_nccl_SOLVED.cu`: NCCL reference

Both references compute the expected local GEMMs on the CPU and use
`MPI_Allreduce` to form the expected two-GPU result. Every output element is
checked after the fused kernel completes.

## Build

Load MPI so `mpicxx` and `mpirun` are on `PATH` (or pass `MPICXX` and
`MPIRUN` to `make`). Set the CUDA, NCCL, and NVSHMEM installation prefixes,
then build all four binaries:

~~~bash
export CUDA_HOME=/path/to/cuda
export NCCL_HOME=/path/to/nccl
export NVSHMEM_HOME=/path/to/nvshmem
export PATH="$CUDA_HOME/bin:$NVSHMEM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$NVSHMEM_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

make
~~~

`make nvshmem` builds only the NVSHMEM pair. `make nccl` builds only the NCCL
pair. `make run` and `make run_SOLVED` remain aliases for the NVSHMEM version.
`CUDA_ARCH` defaults to `90` for the GH200 lab systems.

## Run the NVSHMEM version

Run one PE per GPU. Pass the complete launcher through `LAUNCHER`:

~~~bash
make run_nvshmem_SOLVED \
  LAUNCHER="$NVSHMEM_HOME/bin/nvshmrun -n 2 -ppn 2"
~~~

## Run the NCCL version

The NCCL exercise needs NCCL 2.29 or newer and a communicator whose two ranks
are in the same LSA team. It follows the same window-and-barrier model as
NCCL's [device API LSA all-reduce example](https://github.com/NVIDIA/nccl/tree/master/docs/examples/06_device_api/01_allreduce_lsa).

~~~bash
make run_nccl_SOLVED LAUNCHER="mpirun -np 2"
~~~

On Jupiter, either binary can be launched with two Slurm tasks and one GPU per
task. The CUDA device-selection helper maps rank-local visibility correctly:

~~~bash
make run_nccl_SOLVED \
  LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
~~~

## NVSHMEM exercise

Start with `fused_gemm_allreduce.cu`.

1. After the CTA produces its local tile, have thread zero put the complete
   tile into the peer's inbox and set the peer's `ready` signal to `kReady`.
2. Have thread zero wait for the local `ready` signal before releasing the CTA
   to add the received tile.

~~~cpp
nvshmem_float_put_signal(
    float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

uint64_t nvshmem_signal_wait_until(
    uint64_t *sig_addr, int cmp, uint64_t cmp_value);
~~~

The fused put-with-signal orders its payload before its signal. Seeing the
signal is therefore the receiver-visible handoff for that tile; the kernel
does not need a separate `quiet` or global barrier. This is a one-shot signal
value. A loop would need a new value or an explicit reset.

## NCCL exercise

Start with `fused_gemm_allreduce_nccl.cu`. The host setup has already created
a device communicator, registered symmetric partial and result windows, and
reserved one LSA barrier for the single CTA.

1. Construct an `ncclLsaBarrierSession<ncclCoopCta>` for barrier index
   `blockIdx.x`.
2. After every thread writes its local GEMM element to `partial_window`, call
   the barrier with `cuda::memory_order_acq_rel`. The release side publishes
   the local stores; the acquire side makes the peers' stores visible after
   all ranks arrive.
3. Iterate `ncclTeamWorld(dev_comm)` and use `ncclGetPeerPointer` to load this
   element from every rank's partial window. The setup has already verified
   that both world ranks are in the LSA team. Store the sum in the local result
   window.
4. Call the same barrier with `cuda::memory_order_release` before returning.

The functions used in the kernel are:

~~~cpp
ncclTeam ncclTeamWorld(ncclDevComm const &comm);
// Return the communicator's world team.

void *ncclGetLocalPointer(ncclWindow_t window, size_t offset);
// Return this rank's local address for a registered window.

void *ncclGetPeerPointer(ncclWindow_t window, size_t offset, int peer);
// Return an LSA address that directly accesses a world peer's registered window.

barrier.sync(ncclCoopCta(), cuda::memory_order order);
// Make every thread in the CTA participate in the cross-rank LSA barrier.
~~~

The NCCL implementation uses separate partial and result windows. Writing the
sum back into the partial window could overwrite a value before the other GPU
loads it.

A successful reference run prints
`NCCL LSA fused all-reduce verified` from both ranks.
