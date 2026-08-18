#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NCCL4Py host API exercise: grouped Send/Recv.

Usage:
mpirun -np 2 python send_recv.py
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

    if nranks != 2:
        if rank == 0:
            print("ERROR: this exercise requires exactly two MPI ranks")
        return 1
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        if rank == 0:
            print("ERROR: this exercise requires a visible CUDA device")
        return 1

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    unique_id = nccl.get_unique_id() if rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)
    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    send_data = torch.tensor([float(100 + rank)], dtype=torch.float32, device=device)
    recv_data = torch.zeros(1, dtype=torch.float32, device=device)
    peer = 1 - rank

    # TODO: Wrap the matching send and receive in nccl.group().
    # TODO: Queue nccl_comm.send(send_data, peer=peer) and nccl_comm.recv(recv_data, peer=peer).

    torch.cuda.synchronize()

    expected = float(100 + peer)
    received = float(recv_data[0].item())
    local_ok = received == expected
    all_ok = mpi_comm.allreduce(int(local_ok), op=MPI.MIN)

    print(f"Rank {rank}: sent {send_data[0].item():.0f}, received {received:.0f} (expected {expected:.0f})")
    nccl_comm.destroy()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
