#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys


def main():
    try:
        import cutlass.cute
        from cuda.core import system
        import nccl.core as nccl
        import nccl.core.device.cute as nccl_cute
    except ImportError as error:
        print(f"ERROR: missing Python dependency: {error}")
        return 1

    if system.get_num_devices() == 0:
        print("ERROR: no visible CUDA device")
        return 1

    required_symbols = (
        nccl.NCCLDevCommRequirements,
        nccl.NcclGinConnectionType,
        nccl_cute.DevComm,
        nccl_cute.Window,
        nccl_cute.GinBackendMask,
        nccl_cute.cta,
    )
    if any(symbol is None for symbol in required_symbols):
        print("ERROR: the installed NCCL4Py lacks the CuTe device API")
        return 1

    print(f"CUDA devices visible: {system.get_num_devices()}")
    print(f"NCCL4Py version: {nccl.__version__}")
    print("NCCL CuTe device bindings: available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
