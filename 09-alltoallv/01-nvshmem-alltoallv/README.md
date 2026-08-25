# NVSHMEM AlltoAllV

AlltoAllV sends a different amount of data to every PE. On PE `r`, the
segment beginning at `send_offsets[p]` and containing `send_counts[p]`
elements belongs to destination `p`. After the collective, PE `p` finds the
segment from source `r` at `recv_offsets[r]`.

The counts are deliberately uneven in this exercise. The default `skewed`
pattern includes a large destination, while `sparse` also includes zero-byte
pairs. `offdiagonal` excludes self traffic for a clean network measurement.
Every element encodes its source PE, destination PE, and index so the
program can check the complete receive buffer before reporting performance.

`nvshmem_alltoallv.cu` is the exercise and
`nvshmem_alltoallv_SOLVED.cu` is the completed reference. Both use the same
host setup, device-generated route plan, correctness check, warmup, and timing
loop.

## One implementation for three placements

The code does not have separate NVLink and InfiniBand executables. It decides
how to send each source-to-destination segment on the GPU:

```text
nvshmem_ptr(receive address, destination)
        |
        +-- non-null: chunked block-scoped puts-with-signal (direct path)
        |
        `-- null:     chunked thread-scoped puts-with-signal (network path)
```

On a single node, directly mapped peers normally take the first path. PEs on
different nodes take the network path. A multi-node run with several PEs per
node uses both paths from the same kernel.

Direct and network transfers use different chunk sizes. The smaller direct
chunks give many CTAs work on the NVLink path. Network chunks are larger so
the kernel can issue concurrent operations across available NICs without
turning a large message into thousands of tiny RMAs. Both paths remain in one
kernel and use the same completion protocol.

Each call begins with a block-scoped world barrier on the same CUDA stream.
That handshake says every PE has finished consuming the previous receive
buffer before any PE can overwrite it. Every direct or network chunk has a
separate signal slot. The put-with-signal orders its payload before its signal,
and a second kernel waits for the signal counts the senders supplied during
setup before the CUDA stream can consume the buffer. Signal values increase on
every iteration, so the benchmark reuses the signal table without clearing it.
The [NVSHMEM signaling reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
defines the payload-before-signal guarantee used here.

## Device route setup

The CPU constructs the input counts and a reference result, but it does not
provide remote receive offsets to the communication kernel. A collectively
launched, one-CTA setup kernel builds that information:

1. `nvshmemx_uint64_alltoall_block` exchanges the byte count for every
   source/destination pair.
2. Thread 0 computes aligned receive offsets from the incoming counts.
3. A second block all-to-all returns each receiver's chosen offset to the
   corresponding sender.
4. Each sender uses `nvshmem_ptr` to choose its route, calculates how many
   signals it will produce for each destination, and exchanges those counts
   with a third block all-to-all.

The last exchange matters because direct accessibility is directional. A
receiver cannot use its own `nvshmem_ptr` result for a source PE to infer the
route that source chose in the opposite direction.

This setup is outside the timed loop. The send and receive allocations use the
maximum capacity over all PEs because every NVSHMEM symmetric allocation must
be made collectively with a compatible size and ordering.

## Build

Load CUDA, MPI, and NVSHMEM, then set `NVSHMEM_HOME` to the installation
prefix containing `include/` and `lib/`:

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"

make CUDA_ARCH=90
```

The Makefile builds the starter and `_SOLVED` reference. It defaults to
`CUDA_ARCH=90` for the GH200 tutorial systems and uses GNU Make like the other
C/CUDA exercises.

## Run

Run the solved version first. Four PEs and a 4 MiB payload per PE are the
defaults:

```bash
make run_SOLVED NP=4
```

Pass command-line options through `RUN_ARGS`:

```bash
make run_SOLVED NP=4 \
  RUN_ARGS="--pattern sparse --bytes-per-rank 16M --warmup 10 --iters 50"
```

`CHUNK_BYTES` controls direct-path and self-copy chunks;
`NETWORK_CHUNK_BYTES` controls the larger network chunks. They accept the same
`K`, `M`, and `G` suffixes as `--bytes-per-rank` and must be multiples of 16
bytes:

```bash
make run_SOLVED NP=4 CHUNK_BYTES=256K NETWORK_CHUNK_BYTES=4M
```

Placement determines which transport paths the kernel exercises. With a
Slurm allocation, representative launch shapes are:

```bash
# NVLink/direct peer paths within one node
make run_SOLVED NP=4 \
  LAUNCHER="srun --nodes=1 --ntasks=4 --gpus-per-task=1"

# InfiniBand: one PE and GPU on each of two nodes
make run_SOLVED NP=2 \
  LAUNCHER="srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=1" \
  RUN_ARGS="--pattern offdiagonal"

# Mixed: two direct peers per node and InfiniBand between nodes
make run_SOLVED NP=4 \
  LAUNCHER="srun --nodes=2 --ntasks=4 --ntasks-per-node=2 --gpus-per-task=1" \
  RUN_ARGS="--pattern uniform"
```

Use the node and task counts allowed by the current allocation. NVSHMEM must
be built with an InfiniBand transport for the inter-node forms. The tutorial
`env.sh` selects MPI bootstrap and IBRC on JUPITER. IBRC is the compatible
proxy-backed baseline; on an installation built for IBGDA, set
`NVSHMEM_IB_ENABLE_IBGDA=1` to run the same kernel with GPU-initiated network
progress. No source change is required.

The two-node offdiagonal run should report `0 direct, 2 network`. The two-node,
two-PE-per-node run should report `4 direct, 8 network`. Those counts verify
the intended placement before interpreting the timing number. In a mixed run,
the logical non-self rate includes both direct and network payload bytes. The
placement breakdown separates same-host and inter-host bytes; confirm the
actual route with the `NVSHMEM routes` line before treating inter-host bytes as
InfiniBand traffic. Multi-node NVLink systems can map an inter-host peer
directly.

## Exercise

Complete the three communication functions in `nvshmem_alltoallv.cu`.

1. In `exchange_plan`, use block-scoped all-to-all collectives to exchange
   counts, receive offsets, and the sender-selected signal counts.
2. In `send_chunks`, assign destination/chunk pairs in chunk-major order so
   adjacent CTAs begin on different destinations. Copy self and direct-peer
   segments in the smaller direct chunks. For a network PE, issue each larger
   network chunk with one thread-scoped put-and-signal.
3. In `wait_for_chunks`, wait for the signal count exchanged by each source.
   Use the source PE and chunk index to address the correct signal slot.

The APIs used in those functions are:

```cpp
int nvshmemx_uint64_alltoall_block(
    nvshmem_team_t team, uint64_t *dest, const uint64_t *source,
    size_t nelems);

void *nvshmem_ptr(const void *symmetric_address, int pe);

void nvshmemx_putmem_signal_block(
    void *dest, const void *source, size_t bytes,
    uint64_t *signal_address, uint64_t signal, int signal_op, int pe);

void nvshmem_putmem_signal(
    void *dest, const void *source, size_t bytes,
    uint64_t *signal_address, uint64_t signal, int signal_op, int pe);

uint64_t nvshmem_signal_wait_until(
    uint64_t *signal_address, int comparison, uint64_t value);

void nvshmemx_barrier_all_block();
```

The setup, entry-barrier, and wait kernels are launched with
`nvshmemx_collective_launch`. The wait kernel uses NVSHMEM synchronization
APIs, and the other two contain block collectives. The multi-CTA send phase
uses an ordinary CUDA launch and completes before the one-CTA wait phase is
launched on the same stream. Waiting CTAs therefore cannot block unscheduled
send CTAs. See the
[collective-launch contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)
for the residency requirement behind this split.

A successful reference run includes:

```text
NVSHMEM device plan: PASS
NVSHMEM routes: ... direct, ... network
NVSHMEM AlltoAllV correctness: PASS
NVSHMEM AlltoAllV performance: ... ms/iteration, ... GB/s logical non-self
NVSHMEM AlltoAllV placement payload rates: ... GB/s same-host non-self, ... GB/s inter-host
```

This lab implements an out-of-place collective for 32-bit values. The route
plan is rebuilt when the counts change, and the fixed chunk size is a tuning
parameter rather than an automatic policy. The traffic summary and split
payload rates describe MPI rank placement. The route summary reports what
`nvshmem_ptr` actually mapped directly and what used the network path.
