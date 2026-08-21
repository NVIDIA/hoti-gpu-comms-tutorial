# NVSHMEM4Py host APIs

This lab is the Python version of host-side NVSHMEM RMA. There are two MPI
ranks, one GPU per rank, and an identically shaped allocation on both ranks.
NVSHMEM calls that allocation *symmetric*: a pointer obtained on one PE names
the corresponding allocation on another PE.

The host initiates the transfer, but the call is queued on a CUDA stream. Work
queued before the put completes first; work queued after it observes the normal
stream order. This is how a Python GPU program can add RMA without a host-side
device synchronize between every operation. See the [NVSHMEM on-stream API
overview](https://docs.nvidia.com/nvshmem/api/api/overview.html) and the
[host-API guidance](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/apis.html).

```text
PE 0 array = 1.0  -- put on CUDA stream -->  PE 1 symmetric array
                                            |
                                      barrier or signal wait
                                            |
                                      inspect the received data
```

## Files

- `put.py` / `put_SOLVED.py`: host put followed by a stream-ordered world barrier.
- `put_signal.py` / `put_signal_SOLVED.py`: host put-with-signal and a signal wait, with no collective barrier in the data handoff.
- `requirements.txt`: Python dependencies.
- `Makefile`: installation and starter/solution launch targets.

Each unsolved file is named for its exercise. Its `_SOLVED.py` counterpart is
the completed reference; use it to compare after attempting the TODOs, not as a
second program to edit.

## Setup

Use a CUDA 13 Python environment with an MPI implementation available to `mpi4py`:

```bash
python3 -m venv .venv
. .venv/bin/activate
make install
```

The install target uses NVIDIA's Python package index to install
`nvshmem4py-cu13`, its matching NVSHMEM runtime, and the CUDA Python packages.
If the site provides NVSHMEM through modules, load the matching environment and
install only the packages required there.

## Run

Each program needs two PEs and two visible GPUs:

```bash
make run-put
make run-put_SOLVED
make run-put-signal
make run-put-signal_SOLVED
```

The default launcher is `mpirun -np 2` and leaves GPU visibility to the
launcher. On Jupiter, override the complete launcher:

```bash
make run-put-signal_SOLVED LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
```

## Exercises

### `put.py`: put followed by a world barrier

PE 0 fills its 10-by-10 float array with `1.0`; PE 1 starts at zero. On PE 0,
enqueue `nvshmem.core.put` to PE 1. Then have every PE enqueue a `TEAM_WORLD`
barrier on the same stream before `dev.sync()`. The barrier is the point where
PE 1 may consume the remote write. With the solution, PE 1 prints an array of
ones.

### `put_signal.py`: put followed by a one-sided notification

A global barrier is too broad when one producer has one consumer. Keep the
data path from PE 0 to PE 1, but use the supplied symmetric signal slot as the
ready flag. PE 0 calls `put_signal` with value `1` and `SignalOp.SIGNAL_SET`;
PE 1 queues `signal_wait` for `1` with `ComparisonType.CMP_EQ`, both on the
existing stream. After `stream.sync()`, PE 1 should print the copied array.
The signal establishes the handoff for this pair; it is not a replacement for
synchronization among a larger group of PEs.

Compare each result with the corresponding `_SOLVED.py` program. The useful
check is PE 1's array after synchronization, not whether the host returned
immediately from the enqueue call.
