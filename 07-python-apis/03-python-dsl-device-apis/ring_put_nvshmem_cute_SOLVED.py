#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from mpi4py import MPI
import torch

import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
from cutlass.cute.runtime import from_dlpack
from cuda.core import Device, system

import nvshmem.core
from nvshmem.bindings.device.cute import int_p as cute_int_p
from nvshmem.bindings.device.cute import my_pe as cute_my_pe
from nvshmem.bindings.device.cute import n_pes as cute_n_pes


@cute.kernel
def ring_put_kernel(destination: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    if thread_idx == 0:
        my_pe = cute_my_pe()
        peer = (my_pe + 1) % cute_n_pes()
        cute_int_p(destination.iterator, my_pe + 1, peer)


@cute.jit
def ring_put(destination: cute.Tensor):
    ring_put_kernel(destination).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
    )


def main():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    if mpi_comm.Get_size() != 2:
        if rank == 0:
            print("ERROR: this example requires exactly two MPI ranks")
        return 1
    if system.get_num_devices() == 0:
        if rank == 0:
            print("ERROR: this example requires a visible CUDA device")
        return 1

    local_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    device_id = 0 if system.get_num_devices() == 1 else local_rank % system.get_num_devices()
    device = Device(device_id)
    device.set_current()
    nvshmem.core.init(mpi_comm=mpi_comm, initializer_method="mpi")

    my_pe = nvshmem.core.my_pe()
    npes = nvshmem.core.n_pes()
    tensor = nvshmem.core.tensor(1, dtype=torch.int32)
    tensor.fill_(0)
    device.sync()
    tensor_cute = from_dlpack(tensor).mark_layout_dynamic()

    bitcode = nvshmem.core.find_device_bitcode_library()
    compiled = cute.compile(ring_put, tensor_cute, options=f" --link-libraries={bitcode}")
    compiled = compiled.to(device.device_id)
    cuda_library = compiled.jit_module.cuda_library
    kernel = nvshmem.core.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.core.library_init(kernel)

    compiled(tensor_cute)
    device.sync()
    stream = device.create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    expected = (my_pe + 1) % npes + 1
    actual = int(tensor[0].item())
    local_ok = int(actual == expected)
    all_ok = mpi_comm.allreduce(local_ok, op=MPI.MIN)
    print(f"PE {my_pe}: value is {actual} (expected {expected})")

    nvshmem.core.library_finalize(kernel)
    nvshmem.core.free_tensor(tensor)
    nvshmem.core.finalize()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
