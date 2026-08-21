# NCCL4Py host APIs

NCCL4Py exposes NCCL's host API to Python. MPI is only used here to give every
rank the same NCCL unique ID and to collect the pass/fail result; the payloads
move between GPU tensors through NCCL.

There are two communication patterns in this directory. `send_recv.py` pairs
two GPUs and copies a different buffer in each direction. `allreduce.py` is a
collective: every rank in one communicator calls the same operation, and NCCL
gives every rank the reduced result. Those are different contracts, not two
spellings of the same operation. Read the NCCL [usage guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html)
and [point-to-point API reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html)
before starting.

Each exercise has a task-named source file and a completed `_SOLVED.py`
reference.

## Files

- `allreduce.py` / `allreduce_SOLVED.py`: in-place sum across all ranks.
- `send_recv.py` / `send_recv_SOLVED.py`: a two-rank grouped point-to-point exchange.
- `requirements.txt`: supporting Python dependencies installed by `make install`.
- `Makefile`: dependency, environment, starter, and solution targets.

## Setup

Build NCCL4Py from the [public NCCL source tree](https://github.com/NVIDIA/nccl/tree/master/bindings/nccl4py). With CUDA 13 and `CUDA_HOME` configured, clone NCCL outside this tutorial checkout and pass its NCCL4Py directory to the install target:

```bash
git clone https://github.com/NVIDIA/nccl.git /path/to/nccl
make install NCCL4PY_SOURCE=/path/to/nccl/bindings/nccl4py
make check
```

The source install uses NCCL4Py's `cu13` extra, which supplies the matching
NCCL and CUDA Python dependencies. `requirements.txt` adds MPI and PyTorch for
these exercises.

Run one MPI process per CUDA device. These examples select `cuda:rank modulo
visible-device-count`; with Jupiter's one-GPU-per-task Slurm launch, each rank
therefore uses its single visible CUDA device, numbered 0.

## Run

```bash
make run-allreduce NP=4
make run-allreduce_SOLVED NP=4
make run-send-recv
make run-send-recv_SOLVED
```

On Jupiter, supply one GPU per rank through Slurm:

```bash
make run-allreduce_SOLVED NP=4 LAUNCHER="srun --ntasks=4 --gpus-per-task=1"
make run-send-recv_SOLVED LAUNCHER="srun --ntasks=2 --gpus-per-task=1"
```

The point-to-point exercise requires exactly two ranks.

## Exercises

### `allreduce.py`: one result from every rank

Rank *r* starts with a one-element tensor containing `r`. Add an in-place NCCL
sum over the communicator. For `NP=4`, every rank must finish with
`0 + 1 + 2 + 3 = 6`; the supplied MPI check fails if any rank has another
value. This is the ordinary data-parallel pattern: the whole group participates
and every member receives a result with the same defined meaning.

### `send_recv.py`: two independent copies

Rank 0 starts with `100` and rank 1 starts with `101`. Each rank sends its
value to the other and receives the peer's value into a separate tensor. Put
the matching `send` and `recv` calls in `nccl.group()`. The group lets NCCL
launch the mutually dependent pair together; without it, independently
submitted point-to-point operations can wait on an ordering the peer has not
yet supplied. On a successful two-rank run, rank 0 receives `101` and rank 1
receives `100`.

Run the starter first, then compare with the corresponding `_SOLVED.py` file.
The output makes the distinction visible: a send/receive has one specified
peer, while an all-reduce requires the communicator's full group.
