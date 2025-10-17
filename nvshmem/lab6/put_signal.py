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
from cuda.core.experimental import Device, system
import os

from mpi4py import MPI

def mpi_init():
    # This uses the MPI communicator to perform initialization of NVSHMEM
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    global dev
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

if __name__ == "__main__":
    mpi_init()
    stream = dev.create_stream()
    # This is a CuPy array that is allocated on the NVSHMEM Symmetric heap
    array = nvshmem.core.array(shape=(10, 10), dtype="float32")
    
    if nvshmem.core.my_pe() == 0:
        array[:] = 1.0
    print(f"Array on pe {nvshmem.core.my_pe()} of {nvshmem.core.n_pes()}: {array}")

    # TODO: Copy the array from PE 0 to PE 1 without a barrier using put_signal and signal_wait
    # See https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/rma.html for more details

    print(f"Array on pe {nvshmem.core.my_pe()} of {nvshmem.core.n_pes()}: {array}")
    nvshmem.core.finalize()

