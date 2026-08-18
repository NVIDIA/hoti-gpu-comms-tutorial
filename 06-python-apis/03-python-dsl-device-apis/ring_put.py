#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from cuda.core import Device, system
from mpi4py import MPI

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


SIGNAL_ID = 0


@cute.kernel
def ring_put_kernel(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
):
    world = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()
    send = send_win.tensor(cutlass.Int32, cute.make_layout(1))
    recv = recv_win.tensor(cutlass.Int32, cute.make_layout(1))
    peer = (world.rank + 1) % world.nRanks

    # TODO: put send into peer's recv window and increment SIGNAL_ID.
    # TODO: wait until this rank's SIGNAL_ID reaches 1.


@cute.jit
def ring_put(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
):
    ring_put_kernel(dev_comm, send_win, recv_win).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


def main():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    nranks = mpi_comm.Get_size()

    if nranks != 2:
        if rank == 0:
            print("ERROR: this exercise requires exactly two MPI ranks")
        return 1
    if system.get_num_devices() == 0:
        if rank == 0:
            print("ERROR: this exercise requires a visible CUDA device")
        return 1

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)
    comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    if not comm.device_api_support or comm.gin_type == nccl.NcclGinType.NONE:
        if rank == 0:
            print(
                "ERROR: the NCCL device API and a GIN transport are required "
                f"(device_api_support={comm.device_api_support}, gin_type={comm.gin_type.name})"
            )
        comm.destroy()
        return 1

    send_buf = nccl.cupy.empty(1, dtype="int32")
    recv_buf = nccl.cupy.empty(1, dtype="int32")
    send_buf[0] = rank + 1
    recv_buf[0] = 0
    device.sync()

    send_win = comm.register_window(send_buf)
    recv_win = comm.register_window(recv_buf)
    assert send_win is not None and send_win.is_valid
    assert recv_win is not None and recv_win.is_valid

    requirements = nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        gin_signal_count=SIGNAL_ID + 1,
    )
    dev_comm = comm.create_dev_comm(requirements=requirements)

    ring_put(
        nccl_cute.DevComm(dev_comm),
        nccl_cute.Window(send_win),
        nccl_cute.Window(recv_win),
    )
    device.sync()

    expected = ((rank - 1) % nranks) + 1
    actual = int(recv_buf[0].item())
    local_ok = int(actual == expected)
    all_ok = mpi_comm.allreduce(local_ok, op=MPI.MIN)
    print(f"rank {rank}: received {actual} (expected {expected})")

    dev_comm.close()
    send_win.close()
    recv_win.close()
    comm.destroy()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
