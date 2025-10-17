# GPU Communication Libraries for Accelerating HPC and AI Applications

This repository accompanies the interactive HOTI 2025 tutorial on GPU communication libraries, covering NVIDIA Collective Communication Library (NCCL) and NVSHMEM (including Python bindings). It contains hands-on labs with ready-to-build examples and reference solutions.

Links:
- Tutorial homepage: [GPU Communication Libraries for Accelerating HPC and AI Applications @ HotI 2025](https://hoti.org/tutorials-nccl-nvshmem.html)
- Video recording: [YouTube](https://www.youtube.com/watch?v=rlA5QreHekk&list=PLBM5Lly_T4yRGBFgforeMTDpjasC_PV7r&index=31)

## Prerequisites

- NVIDIA GPUs with CUDA support (Ampere or newer recommended)
- CUDA Toolkit (12.x recommended)
- MPI implementation (e.g., OpenMPI or MPICH)
- NCCL installed and visible to your toolchain
- NVSHMEM installed (for C/C++) and NVSHMEM Python runtime (for Python labs)
- Python 3.9+ for NVSHMEM Python labs

## Environment Setup

Set the following environment variables so the build system and runtime can find CUDA, NCCL, and NVSHMEM. The paths below are examples; adjust to your system.

```bash
export NVSHMEM_HOME=/path/to/nvshmem/build/lib
export NCCL_HOME=/path/to/nccl-src/build/
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NVSHMEM_HOME/bin:$PATH
export CPATH=$NCCL_HOME/build:$NVSHMEM_HOME/include:$CPATH
```

You may also need `CUDA_HOME` if not set by your environment modules:

```bash
export CUDA_HOME=/usr/local/cuda
```

Verify your toolchain:

```bash
nvcc --version
mpicxx --version || mpicc --version
python3 -V
```

## Repository Structure

```
nccl/
  lab1/     # NCCL basics (unsolved + solved)
  lab3/     # Jacobi with NCCL (unsolved + solved)
  lab5/     # NCCL symmetric memory kernels (unsolved + solved)
nvshmem/
  lab2/     # NVSHMEM basics (C++/CUDA - unsolved + solved)
  lab4/     # Jacobi with NVSHMEM (unsolved + solved)
  lab6/     # NVSHMEM Python bindings (put, put_signal)
```

Each lab includes a `Makefile` with standard targets to build and run.

## Building and Running

Unless noted, the examples assume 2–4 GPUs on a single node. Control the number of MPI processes with `NP` and the visible GPUs with `CUDA_VISIBLE_DEVICES`.

### NCCL Labs

- `nccl/lab3` (Jacobi):
  ```bash
  cd nccl/lab3
  make jacobi              # build unsolved version
  make jacobi_solved       # build reference solution
  make run                 # run unsolved (default NP=4)
  make run_solved          # run solved (default NP=4)
  ```

- `nccl/lab5` (Symmetric kernels):
  ```bash
  cd nccl/lab5
  make nccl_symmetric            # build unsolved
  make nccl_symmetric_solved     # build reference solution
  make run                       # run unsolved (default NP=4)
  make run_solved                # run solved (default NP=4)
  ```

### NVSHMEM Labs (C++/CUDA)

- `nvshmem/lab2` (Basics):
  ```bash
  cd nvshmem/lab2
  make               # build
  make run           # run (default NP=4)
  ```

- `nvshmem/lab4` (Jacobi):
  ```bash
  cd nvshmem/lab4
  make jacobi            # build unsolved
  make jacobi_solved     # build reference solution
  make run               # run unsolved (default NP=1 unless NP is set)
  make run_solved        # run solved
  ```

### NVSHMEM Python Labs

- `nvshmem/lab6`:
  - Install Python dependencies:
    ```bash
    cd nvshmem/lab6
    pip install -r requirements.txt
    ```
  - Run the Python example with two processes on two GPUs:
    ```bash
    make run   # runs: CUDA_VISIBLE_DEVICES=0,1 $(JSC_SUBMIT_CMD) -n 2 python3 put_signal.py
    ```

Notes:
- Some Makefiles rely on `JSC_SUBMIT_CMD` (cluster launcher wrapper). This is because the tutorial was hosted using hardware from Forschungszentrum Jülich (JSC). On a workstation, you can set `JSC_SUBMIT_CMD` to `mpirun` or `srun` as appropriate, e.g.:
  ```bash
  export JSC_SUBMIT_CMD=mpirun
  ```
- You can override `NP` at invocation time: `NP=8 make run`.

## Troubleshooting

- Ensure `LD_LIBRARY_PATH` contains both CUDA, NCCL, and NVSHMEM `lib` directories.
- If `NVSHMEM_HOME` is required by a Makefile, confirm it is set and points to a valid install.
- Match `NP` to the number of GPUs specified by `CUDA_VISIBLE_DEVICES`.
- For NVSHMEM Python, verify that `libnvidia-nvshmem-cu12` and `cuda-python` versions are compatible with your CUDA driver/runtime.

## Credits and Attribution

This material was presented as an interactive tutorial at Hot Interconnects 2025 (HOTI 2025):
- Tutorial homepage: [hoti.org/tutorials-nccl-nvshmem.html](https://hoti.org/tutorials-nccl-nvshmem.html)
- Recording: [YouTube](https://www.youtube.com/watch?v=rlA5QreHekk&list=PLBM5Lly_T4yRGBFgforeMTDpjasC_PV7r&index=31)

This tutorial was co-hosted by NVIDIA and Forschungszentrum Jülich (JSC). JSC supported the workshop by supplying hardware access for participants. 
