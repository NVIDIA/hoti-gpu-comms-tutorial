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

"""
Using NVSHMEM to do one-sided RMA
"""
import numpy as np
import nvshmem.core
from cuda.core import Device, system

from mpi4py import MPI

def mpi_init():
    # This uses the MPI communicator to perform initialization of NVSHMEM
    global dev
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    device_id = 0 if system.get_num_devices() == 1 else local_rank % system.get_num_devices()
    dev = Device(device_id)
    dev.set_current()
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

if __name__ == "__main__":
    mpi_init()
    # This is a CuPy array that is allocated on the NVSHMEM Symmetric heap
    array = nvshmem.core.array(shape=(10, 10), dtype="float32")

    if nvshmem.core.my_pe() == 0:
        array[:] = 1.0
    dev.sync()
    stream = dev.create_stream()
    print(f"Array before operation on pe {nvshmem.core.my_pe()} of {nvshmem.core.n_pes()}: {array}")

    if nvshmem.core.my_pe() == 0:
        # Put from array on PE 0 to array on PE 1 using stream `stream`
        nvshmem.core.put(array, array, 1, stream=stream)
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=stream)
    dev.sync()

    print(f"Array after operation on pe {nvshmem.core.my_pe()} of {nvshmem.core.n_pes()}: {array}")
    local_ok = int(np.allclose(array.get(), 1.0))
    all_ok = MPI.COMM_WORLD.allreduce(local_ok, op=MPI.MIN)
    print(f"PE {nvshmem.core.my_pe()}: array is all ones = {bool(local_ok)}")
    nvshmem.core.free_array(array)
    nvshmem.core.finalize()
    if not all_ok:
        raise SystemExit(1)
