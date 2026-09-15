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
        `-- null:     chunked QP-specific puts-with-signal (network path)
```

The direct domain is defined by `nvshmem_ptr`, not by Linux host boundaries.
It often follows NVLink reachability, and on an NVL72 it can span several
hosts. Peers that are not directly mapped take the network path. A run that
contains direct peers and peers outside that domain uses both paths from the
same kernel.

Direct and network transfers use different chunk sizes. The smaller direct
chunks give many CTAs work on the NVLink path. Network chunks are larger so
an IBGDA transport has enough independent operations to use its available
QPs and NICs without turning a large message into thousands of tiny RMAs.
Both paths remain in one kernel and use the same completion protocol.

The setup collectively requests NVSHMEM QP handles and copies them to the GPU.
Network chunks choose a handle from both their destination and chunk index. On
an IBGDA system with several selected HCAs, that lets different chunks use
different rails. A transport that does not provide a custom QP returns
`NVSHMEMX_QP_DEFAULT`; the same device call then falls back to the default
NVSHMEM path. Unless `NETWORK_QPS` is set, the program requests 16 QPs for a
network-only run, eight for a mixed run, and one unused handle for a
direct-only run.

Each collective call is one cooperatively launched kernel. Its controller CTA
performs the block-scoped world barrier, then a CUDA grid barrier releases the
resident producer CTAs together. That cross-PE handshake says every PE has
finished consuming the previous receive buffer before any PE can overwrite it.
Every direct or network chunk has a separate signal slot, and the nonblocking
put-with-signal orders its payload before its signal without quieting after
every chunk. After all producer CTAs finish, a grid barrier lets the controller
CTA quiet all default and custom QPs and wait for the route-plan signal counts
before the kernel returns. The controller's `nvshmemx_barrier_all_block` is
the cross-PE operation; the CUDA grid barriers only synchronize CTAs on one
GPU. The network put itself is always issued by one thread; only the
cooperation used by `quiet` changes. A direct-only run uses thread-scoped
quiet, a network-only run uses warp-scoped quiet, and a mixed run uses
block-scoped quiet. The choice is collective and is made once from the routes
reported by `nvshmem_ptr`.

Because the kernel uses cooperative launch, all requested CTAs must be
resident concurrently. Before the timing loop, the program verifies
cooperative-launch support and queries the fused kernel's cooperative grid
limit across all PEs. It rejects an oversized `--blocks` value rather than
risking a producer/receiver deadlock.

Signals and quiet answer different questions. The receiver waits for signals
before reading its local receive buffer. Quiet is local to the sender and
makes its source buffer reusable; it does not notify a receiver. Signal values
increase on every iteration, so the benchmark reuses the signal table without
clearing it. The entry barrier in the next call prevents any PE from
overwriting a receive buffer while another PE is still consuming the previous
result. Issuing an NBI operation alone is not a completion guarantee.
The [NVSHMEM signaling reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)
defines the payload-before-signal guarantee used here.

## The three tuning steps

Start with one correct put-with-signal per message, then make three changes:

1. Split long messages into chunks so several CTAs can work at once. Direct
   peers use 256 KiB chunks; network peers use 4 MiB chunks.
2. Route each chunk with `nvshmem_ptr`. Directly mapped peers use the ordinary
   block put-with-signal, while unmapped peers use an explicit QP so IBGDA can
   initiate the transfer from the GPU.
3. Spread network chunks over several QPs and cooperate on the final quiet.
   The QP index includes both the destination and chunk number, and the quiet
   scope changes with the route mix.

Each step changes one visible part of the code. Chunk sizes, QP count, CTA
count, and quiet scope are printed before the timing result, so a performance
comparison can be tied back to the mechanism that changed.

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
prefix containing `include/` and `lib/`. This exercise needs a current NVSHMEM
installation with the explicit-QP APIs and `NVSHMEMX_QP_ALL`:

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"

make
```

The Makefile builds native `sm_100` (GB200) and `sm_103` (GB300) code. To
build a portable teaching-machine fat binary with a CUDA toolkit that supports
all three architectures, use
`make CUDA_ARCHS='90 100 103'`. For a GH200-only system, use
`make CUDA_ARCH=90`. The Makefile builds the starter and `_SOLVED` reference
and otherwise uses GNU Make like the other C/CUDA exercises.

## Exercise

Complete the three marked communication regions in `nvshmem_alltoallv.cu`.

1. In `exchange_plan`, use block-scoped all-to-all collectives to exchange
   counts, receive offsets, and the sender-selected signal counts.
2. In the producer section of `nvshmem_alltoallv_kernel`, assign
   destination/chunk pairs in chunk-major order so adjacent CTAs begin on
   different destinations, and rotate the first destination by the source rank
   to avoid synchronized incast. Copy self and direct-peer segments in the
   smaller direct chunks. For a network PE, issue each larger network chunk
   with a QP-specific thread-scoped NBI put-and-signal. Select the handle from
   the destination and chunk index.
3. In the completion section of `nvshmem_alltoallv_kernel`, wait for the
   signal count exchanged by each source. Use the source PE and chunk index to
   address the correct signal slot, and quiet all QPs so the sender can safely
   reuse its input. Use the supplied scope: one thread for an all-direct
   placement, one warp for network-only, or the whole block when direct and
   network routes are mixed. The controller-CTA and grid-synchronization
   scaffold is already supplied.

The APIs used in those functions are:

```cpp
int nvshmemx_uint64_alltoall_block(
    nvshmem_team_t team, uint64_t *dest, const uint64_t *source,
    size_t nelems);

void *nvshmem_ptr(const void *symmetric_address, int pe);

void nvshmemx_putmem_signal_nbi_block(
    void *dest, const void *source, size_t bytes,
    uint64_t *signal_address, uint64_t signal, int signal_op, int pe);

int nvshmemx_qp_create(
    int num_qps, nvshmemx_qp_handle_t **out_qp_array);

void nvshmemx_qp_uint_put_signal_nbi(
    uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *signal_address, uint64_t signal, int signal_op, int pe,
    nvshmemx_qp_handle_t qp);

uint64_t nvshmem_signal_wait_until(
    uint64_t *signal_address, int comparison, uint64_t value);

void nvshmemx_qp_quiet(
    int pe, nvshmemx_qp_handle_t *qps, int num_qps);

void nvshmemx_qp_quiet_warp(
    int pe, nvshmemx_qp_handle_t *qps, int num_qps);

void nvshmemx_qp_quiet_block(
    int pe, nvshmemx_qp_handle_t *qps, int num_qps);

void nvshmemx_barrier_all_block();
```

`nvshmemx_qp_create` is collective over `NVSHMEM_TEAM_WORLD`, so every PE must
request the same number of handles. NVSHMEM allocates the returned host array,
and the API requires it to remain allocated through finalization; this program
frees it only after `nvshmem_finalize`. In the fused collective kernel's
controller CTA, a handle value of `NVSHMEMX_QP_ALL` with `NVSHMEMX_PE_ALL`
makes the quiet cover default and custom QPs. The
[NVSHMEM QP reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/qp.html)
defines the handle fallback, lifetime, and synchronization rules.

The one-CTA route-plan setup is launched with `nvshmemx_collective_launch` once
before validation and timing. Each subsequent AlltoAllV call is one
multi-CTA `nvshmemx_collective_launch`: its controller CTA uses the
block-scoped NVSHMEM barrier and completion APIs, while CUDA grid barriers
separate the producer and completion phases locally. The collective-launch
contract requires the requested grid to be concurrently resident, which the
program preflights before it starts the timed loop. See the
[collective-launch contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)
for those residency requirements.

This lab implements an out-of-place collective for 32-bit values. The route
plan is rebuilt when the counts change, and the fixed chunk size is a tuning
parameter rather than an automatic policy.
