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

Each call begins with a block-scoped world barrier on the same CUDA stream.
That handshake says every PE has finished consuming the previous receive
buffer before any PE can overwrite it. Every direct or network chunk has a
separate signal slot. The nonblocking put-with-signal orders its payload before
its signal without quieting after every chunk. A second kernel waits for the
signal counts the senders supplied during setup before the CUDA stream can
consume the buffer. That kernel also quiets all default and custom QPs so the
local send buffer is safe to reuse. The network put itself is always
issued by one thread; only the cooperation used by `quiet` changes. A
direct-only run uses thread-scoped quiet, a network-only run uses warp-scoped
quiet, and a mixed run uses block-scoped quiet. The choice is collective and
is made once from the routes reported by `nvshmem_ptr`.

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

The Makefile builds native `sm_100` code plus `compute_100` PTX, matching the
GB200 and GB300 NVL72 target. To build a portable teaching-machine fat binary
with a CUDA toolkit that supports both architectures, use
`make CUDA_ARCHS='90 100'`. For a GH200-only system, use
`make CUDA_ARCH=90`. The Makefile builds the starter and `_SOLVED` reference
and otherwise uses GNU Make like the other C/CUDA exercises.

## Run

Run the solved version first. Four PEs and a 4 MiB payload per PE are the
defaults:

```bash
make run_SOLVED NP=4
```

That default is a quick correctness run. Use a larger payload, more warmup,
and more CTAs when measuring bandwidth.

Pass command-line options through `RUN_ARGS`:

```bash
make run_SOLVED NP=4 \
  RUN_ARGS="--pattern sparse --bytes-per-rank 16M --warmup 10 --iters 50"
```

`CHUNK_BYTES` controls direct-path and self-copy chunks;
`NETWORK_CHUNK_BYTES` controls the larger network chunks. `NETWORK_QPS` can
override the route-based QP default. Chunk sizes accept the same `K`, `M`, and
`G` suffixes as `--bytes-per-rank` and must be multiples of 16 bytes. An
explicit `NETWORK_QPS` value must be positive and identical on every PE:

```bash
make run_SOLVED NP=4 \
  CHUNK_BYTES=256K NETWORK_CHUNK_BYTES=4M NETWORK_QPS=8
```

Placement determines which transport paths the kernel exercises. On Lyris,
the following allocation shapes select the three cases used in the
performance comparison:

```bash
# Direct only: eight PEs in one NVL72, spread over two compute trays
make run_SOLVED NP=8 \
  LAUNCHER="srun --mpi=pmix_v5 --nodes=2 --ntasks=8 --ntasks-per-node=4 --segment=2 --cpu-bind=none" \
  RUN_ARGS="--pattern offdiagonal --bytes-per-rank 256M --blocks 128 --threads 256 --warmup 20 --iters 100"

# Network only: one PE in each of four NVL72 systems
make run_SOLVED NP=4 \
  LAUNCHER="srun --mpi=pmix_v5 --nodes=4 --ntasks=4 --ntasks-per-node=1 --segment=1 --spread-segments --cpu-bind=none" \
  RUN_ARGS="--pattern offdiagonal --bytes-per-rank 256M --blocks 128 --threads 256 --warmup 20 --iters 100"

# Mixed: eight direct peers per NVL72 and network traffic between two NVL72s
make run_SOLVED NP=16 \
  LAUNCHER="srun --mpi=pmix_v5 --nodes=4 --ntasks=16 --ntasks-per-node=4 --segment=2 --spread-segments --cpu-bind=none" \
  RUN_ARGS="--pattern offdiagonal --bytes-per-rank 256M --blocks 128 --threads 256 --warmup 20 --iters 100"
```

The `--segment` and `--spread-segments` options are Lyris allocation controls.
Check the site documentation before copying them to another cluster. If the
command is launched from inside an existing `srun` step, add `--overlap`.

NVSHMEM must be built with an InfiniBand transport for the network forms. The
tutorial `env.sh` selects MPI bootstrap and IBRC on JUPITER. IBRC is the
compatible proxy-backed baseline; on an installation built for IBGDA, set
`NVSHMEM_IB_ENABLE_IBGDA=1` to run the same kernel with GPU-initiated network
progress. No source change is required.

The network-only run should report `0 direct, 12 network`. The 16-PE mixed run
should report `112 direct, 128 network`. Those counts verify the intended
placement before interpreting the timing number. In a mixed run,
the logical non-self rate includes both direct and network payload bytes. The
placement breakdown separates directly mapped peers from cross-domain peers;
confirm the actual route with the `NVSHMEM routes` line before treating
cross-domain bytes as InfiniBand traffic. Multi-node NVLink systems can map an
inter-host peer directly. Direct and network payload totals use the exact
outgoing `nvshmem_ptr` decisions. The smaller "different domain rank"
subcategory assumes the direct peers form symmetric domains, as they do on
the NVL72 allocation used here.

## Reference measurement on GB300 Lyris

The table below uses 256 MiB per PE, 128 CTAs, 256 threads per CTA, a 256 KiB
direct chunk, and a 4 MiB network chunk. The reported bandwidth counts each
non-self payload byte once at the sender.

| Placement | PEs | Measured | Raw send ceiling | Raw ceiling reached |
| --- | ---: | ---: | ---: | ---: |
| One NVL72, direct only | 8 | 3.76 TB/s | 7.2 TB/s | 52% |
| Four GB300 NVL72s, one 800 Gb/s rail per PE | 4 | 205.7 GB/s | 400 GB/s | 51% |
| Two NVL72s, direct plus one network rail per PE | 16 | 1.52 TB/s | 3.0 TB/s | 51% |

The raw NVLink number uses half of the documented 1.8 TB/s bidirectional
bandwidth per GPU because this benchmark counts sent bytes, not both link
directions. A selected 800 Gb/s ConnectX-8 rail contributes 100 GB/s of send
bandwidth. For the mixed case, 7/15 of the payload is direct and 8/15 is
network traffic; the 3.0 TB/s ceiling assumes those paths overlap and the
network portion is the longer one. On GB200, a selected 400 Gb/s ConnectX-7
rail is 50 GB/s, so the corresponding four-PE network and 16-PE hybrid
ceilings are 0.2 TB/s and 1.5 TB/s. Always calculate the ceiling from the
active rails and their negotiated speed before comparing a run. See the
[NVL72 reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html)
and the [GB300 NVL72 system specifications](https://www.nvidia.com/en-us/data-center/gb300-nvl72/).

These are application-level ceilings, not promises for every message size.
Put-with-signal processing, the entry barrier, completion, and routing balance
all reduce the application rate. Measure a matching direct copy or put on the
same allocation and report both the primitive and raw-hardware percentages.

## Exercise

Complete the three communication functions in `nvshmem_alltoallv.cu`.

1. In `exchange_plan`, use block-scoped all-to-all collectives to exchange
   counts, receive offsets, and the sender-selected signal counts.
2. In `send_chunks`, assign destination/chunk pairs in chunk-major order so
   adjacent CTAs begin on different destinations, and rotate the first
   destination by the source rank to avoid synchronized incast. Copy self and
   direct-peer segments in the smaller direct chunks. For a network PE, issue
   each larger network chunk with a QP-specific thread-scoped NBI
   put-and-signal. Select the handle from the destination and chunk index.
3. In `wait_for_chunks`, wait for the signal count exchanged by each source.
   Use the source PE and chunk index to address the correct signal slot, and
   quiet all QPs so the sender can safely reuse its input. Use the supplied
   scope: one thread for an all-direct placement, one warp for network-only,
   or the whole block when direct and network routes are mixed.

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
frees it only after `nvshmem_finalize`. In the wait kernel, a handle value of
`NVSHMEMX_QP_ALL` with `NVSHMEMX_PE_ALL` makes the quiet cover default and
custom QPs. The
[NVSHMEM QP reference](https://docs.nvidia.com/nvshmem/api/latest/gen/api/qp.html)
defines the handle fallback, lifetime, and synchronization rules.

The setup, entry-barrier, and wait kernels are launched with
`nvshmemx_collective_launch`. The wait kernel uses NVSHMEM synchronization
APIs, and the other two contain block collectives. The multi-CTA send phase
uses an ordinary CUDA launch. The one-CTA wait phase is queued after it on the
same stream and cannot begin execution until the send phase completes. Waiting
CTAs therefore cannot block unscheduled send CTAs. See the
[collective-launch contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)
for the residency requirement behind this split.

A successful reference run includes:

```text
NVSHMEM device plan: PASS
NVSHMEM routes: ... direct, ... network
NVSHMEM AlltoAllV correctness: PASS
NVSHMEM AlltoAllV performance: ... ms/iteration, ... GB/s logical non-self
NVSHMEM AlltoAllV placement payload rates (NVSHMEM direct peer): ... GB/s same-domain non-self, ... GB/s cross-domain
```

This lab implements an out-of-place collective for 32-bit values. The route
plan is rebuilt when the counts change, and the fixed chunk size is a tuning
parameter rather than an automatic policy. The traffic summary and split
payload rates use the direct-peer domains reported by `nvshmem_ptr`. The route
summary reports what was mapped directly and what used the network path.
