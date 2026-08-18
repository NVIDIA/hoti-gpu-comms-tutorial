#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NCCL4Py host API exercise: AllReduce.

Usage:
mpirun -np 4 python allreduce.py
"""

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py is required. Install it with: pip install mpi4py")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install it with: pip install torch")
    sys.exit(1)

import nccl.core as nccl


def main():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    nranks = mpi_comm.Get_size()

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        if rank == 0:
            print("ERROR: this example requires a visible CUDA device")
        return 1

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    unique_id = nccl.get_unique_id() if rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)
    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    data = torch.tensor([float(rank)], dtype=torch.float32, device=device)
    # TODO: Reduce data into itself with nccl.SUM on every rank.
    torch.cuda.synchronize()

    expected = float(nranks * (nranks - 1) // 2)
    actual = float(data[0].item())
    local_ok = actual == expected
    all_ok = mpi_comm.allreduce(int(local_ok), op=MPI.MIN)

    print(f"Rank {rank}: all-reduce result = {actual:.0f} (expected {expected:.0f})")
    nccl_comm.destroy()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
