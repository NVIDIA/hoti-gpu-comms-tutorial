# Fused GEMM + all-reduce

This lab puts the communication stage inside the same CUDA kernel that
computes the GEMM tile. A single CTA produces one 16x16 `float` output tile.
After the CTA has finished the local GEMM, one thread sends that tile to the
peer's symmetric inbox with a device put-with-signal. It waits for the peer's
matching signal, then the CTA adds the received tile to its local tile.

~~~text
local GEMM tile -> put-with-signal -> wait for peer -> local sum
~~~

This is a two-PE all-reduce expressed with the device RMA tools introduced
earlier in the tutorial. It is deliberately small rather than tuned: the
point is to make the handoff protocol and the boundary between compute and
communication easy to inspect. The system fence before the CTA rendezvous
makes the per-thread GEMM stores available to the sending thread. The fused
put-with-signal orders its payload before the peer's signal, so seeing the
signal is enough to read the remote contribution; there is no separate
`quiet` or global barrier in the kernel.

This is a one-shot handoff. A loop around the kernel would need a new signal
value or an explicit reset before the next iteration.

`fused_gemm_allreduce.cu` is the starter and
`fused_gemm_allreduce_SOLVED.cu` is the completed reference. The reference
computes a CPU GEMM on each PE and uses MPI to form the expected two-PE
all-reduce result before comparing every output element.

## Build

Load CUDA, MPI, and NVSHMEM, then set the NVSHMEM installation prefix:

~~~bash
export NVSHMEM_HOME=/path/to/nvshmem

cmake -S . -B build-SOLVED \
  -DNVSHMEM_HOME="$NVSHMEM_HOME" \
  -DHOTI_GEMM_VARIANT=SOLVED
cmake --build build-SOLVED -j
~~~

The default CUDA target is `90`, which matches the GH200 lab systems. Set
`-DHOTI_CUDA_ARCH=<architecture>` only when building for a different system.

## Run

Run one PE per GPU. The CUDA device-selection helper handles both a process
that can see all local GPUs and a scheduler launch where each process sees one
GPU:

~~~bash
"$NVSHMEM_HOME/bin/nvshmrun" -n 2 -ppn 2 \
  ./build-SOLVED/fused_gemm_allreduce
~~~

On Jupiter, the corresponding Slurm form is:

~~~bash
srun --ntasks=2 --gpus-per-task=1 ./build-SOLVED/fused_gemm_allreduce
~~~

A successful reference run prints `fused all-reduce verified` from every PE.

## Exercise

Start with `fused_gemm_allreduce.cu`.

1. After the CTA has produced its local tile, have thread zero use
   `nvshmem_float_put_signal` to put the whole tile into the peer's inbox and
   set the peer's symmetric `ready` value to `kReady`.
2. Before the CTA reads its inbox, have thread zero wait for the local
   `ready` signal with `nvshmem_signal_wait_until`, then release the CTA with
   `__syncthreads()`.

The put-with-signal API used in the first step is:

~~~cpp
nvshmem_float_put_signal(
    float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
~~~

The signal address is remote and symmetric just like the destination pointer.
The reference is the same program with those two handoff steps filled in.
