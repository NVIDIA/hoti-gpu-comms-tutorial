#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys


def main():
    try:
        import cutlass.cute
        import nvshmem.core
        import nvshmem.bindings.device.cute
        import torch
    except ImportError as error:
        print(f"ERROR: missing Python dependency: {error}")
        return 1

    if not torch.cuda.is_available():
        print("ERROR: no visible CUDA device")
        return 1

    bitcode = Path(nvshmem.core.find_device_bitcode_library())
    if not bitcode.is_file():
        print(f"ERROR: NVSHMEM device bitcode was not found: {bitcode}")
        return 1

    print(f"CUDA devices visible: {torch.cuda.device_count()}")
    print(f"NVSHMEM device bitcode: {bitcode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
