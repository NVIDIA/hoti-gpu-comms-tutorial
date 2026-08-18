# Use NCCL EP

NCCL EP provides `dispatch` and `combine` operations for the communication
around a Mixture-of-Experts layer. `dispatch` sends token activations to the
ranks that own their selected experts. After local expert work, `combine`
returns the expert outputs to the ranks that originated the tokens.

```text
tokens + top-k expert indices
          | dispatch
          v
  expert-owner ranks
          | local expert work
          | combine
          v
source ranks, in source-token order
```

The lab calls the Python facade in `nccl.ep`. MPI assigns ranks and exchanges
the NCCL unique ID; token payloads move through NCCL EP, not MPI. The default
run uses the low-latency algorithm and deterministic one-hop routing. Rank
`r` sends to the next rank and receives from the preceding rank.

`ep_test.py` is the starter and `ep_test_SOLVED.py` is the completed reference.
The communicator, routing tensor, token buffers, expert-output stub,
verification, synchronization, and teardown are supplied in both files.

## Prerequisites

- Linux with one MPI process per GPU.
- CUDA 13 or newer.
- Hopper or newer GPUs (`sm_90` or later).
- An MPI implementation with headers and `mpirun` or a site launcher.
- A recursive clone of
  [NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions).

The NCCL Extensions repository vendors a compatible NCCL checkout under
`third_party/nccl`. NCCL EP currently requires NCCL 2.29 or newer with Device
API and GIN support. The test accepts 2 or 4 ranks, or a rank count divisible
by 8; the workshop command uses four local GPUs.

## Setup

Clone NCCL Extensions with its submodules and point this exercise at the
checkout:

```bash
git clone --recursive https://github.com/NVIDIA/nccl-extensions.git

export NCCL_EXTENSIONS_SRC=/path/to/nccl-extensions
export NCCL_HOME="$NCCL_EXTENSIONS_SRC/third_party/nccl/build"
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/path/to/mpi
export COMPUTE_CAP=90

export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Set `COMPUTE_CAP` to the target GPU architecture. For example, use `90` for
H100. A normal source setup builds the vendored NCCL first, then NCCL EP and
its Python package:

```bash
make nccl
make ep
make install-python
make check-env
```

`make ep` runs the `nccl_ep/Makefile` from the Extensions repository and puts
`libnccl_ep.so` in `NCCL_HOME/lib`. `make install-python` stages that library
in the Extensions Python package and installs the package in editable mode.
To use a separate NCCL build, set `NCCL_HOME` to that build before running
`make ep`; the recursive Extensions checkout is still needed for its shared
build files and submodules.

## Exercise

Complete four TODOs in `ep_test.py`:

1. Create an EP group from the NCCL communicator and `GroupConfig`.
2. Create a handle that binds the layout and top-k routing tensor.
3. Dispatch the supplied input and output descriptors on the CUDA stream.
4. Combine the supplied expert output back into source-token order.

The relevant facade methods are `Group.create`, `group.create_handle`,
`handle.dispatch`, and `handle.combine`. The supplied code calls `complete()`
only for the low-latency `send_only` path, where dispatch or combine is staged
across two calls. The default path performs the full operation in one call.
The stream synchronization remains after each phase so its output is ready for
verification and for the next phase.

Run the reference first to check the environment, then complete the starter:

```bash
make run_SOLVED NP=4 EP_TEST_ARGS='-a ll -t 32 -d 2048'
make run NP=4 EP_TEST_ARGS='-a ll -t 32 -d 2048'
```

On Jupiter, use Slurm to assign one GPU to each of the four ranks:

```bash
make run_SOLVED NP=4 LAUNCHER="srun --ntasks=4 --gpus-per-task=1" EP_TEST_ARGS='-a ll -t 32 -d 2048'
```

Each source rank fills its checked token elements with `0x1000 + rank`.
Dispatch verification checks that the data came from the preceding rank, and
combine verification checks that the expert-output stub returned every token
to its source rank.

The harness also retains high-throughput and cached modes for follow-up work.
Establish the low-latency run first because the other paths have additional
topology and GIN requirements.

The upstream
[NCCL EP README](https://github.com/NVIDIA/nccl-extensions/tree/main/nccl_ep)
documents the complete C and Python APIs, tensor layouts, and build outputs.
