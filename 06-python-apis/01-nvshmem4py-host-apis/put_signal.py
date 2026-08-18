"""
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
"""

"""Host-initiated NVSHMEM put-with-signal starter."""

import sys

import numpy as np
from cuda.core import Device, system
from mpi4py import MPI

import nvshmem.core


def mpi_init():
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    device = Device(local_rank_per_node)
    device.set_current()
    nvshmem.core.init(device=device, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    return device


def main():
    device = mpi_init()
    my_pe = nvshmem.core.my_pe()
    if nvshmem.core.n_pes() != 2:
        if my_pe == 0:
            print("ERROR: this exercise requires exactly two PEs")
        nvshmem.core.finalize()
        return 1

    stream = device.create_stream()
    array = nvshmem.core.array(shape=(10, 10), dtype="float32")
    signal = nvshmem.core.array(shape=(1,), dtype="uint64")
    signal_buffer, _, _ = nvshmem.core.array_get_buffer(signal)

    array[:] = 0.0
    signal[:] = 0
    if my_pe == 0:
        array[:] = 1.0
    device.sync()
    MPI.COMM_WORLD.Barrier()

    # TODO: On PE 0, call put_signal with array, signal_buffer, value 1,
    # SignalOp.SIGNAL_SET, remote_pe=1, and stream=stream.
    # TODO: On PE 1, wait on signal_buffer for value 1 with
    # ComparisonType.CMP_EQ on the same stream.
    stream.sync()

    local_ok = int(np.allclose(array.get(), 1.0))
    all_ok = MPI.COMM_WORLD.allreduce(local_ok, op=MPI.MIN)
    print(f"PE {my_pe}: array is all ones = {bool(local_ok)}")

    nvshmem.core.free_array(array)
    nvshmem.core.free_array(signal)
    nvshmem.core.finalize()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
