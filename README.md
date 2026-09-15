# GPU Communication Libraries for Accelerating HPC and AI Applications

This repository contains the hands-on exercises for the HOTI 2026 GPU
communications tutorial. The labs cover NCCL and NVSHMEM, from host-launched
operations through device APIs, Python bindings, and fused applications.

Each exercise has a plain-named starter and a matching `_SOLVED` reference,
except the completed hello-world toolchain check. Start with the chapter README
for context, then use the leaf README for the build command, launcher,
expected result, and TODOs.

## Prerequisites

- Linux with NVIDIA GPUs, a CUDA Toolkit, and an MPI implementation with
  `mpicxx` or `mpicc`.
- NCCL and/or NVSHMEM installations matching the labs you plan to run.
- An MPI launcher configuration that assigns one GPU to each local rank.

The basic exercises run on a normal multi-GPU CUDA/MPI system. Exact library
versions, GPU count, topology, and Python environment vary by chapter. The
NCCL ecosystem and application chapters have additional requirements, so read
the leaf README before building any lab.

## Environment setup

For a standalone JUPITER checkout, source the included environment before
building or launching a lab:

```bash
source ./env.sh
```

It loads the JUPITER Booster module stack and selects the project-built NCCL
and NVSHMEM installations. The NVSHMEM setup uses MPI bootstrap and forces
IBRC for remote PEs. Override the project defaults by setting `HOTI_ROOT`,
`HOTI_CUDA_HOME`, `HOTI_NCCL_HOME`, or `HOTI_NVSHMEM_HOME` before sourcing the
script.

### HOTI course account on JUPITER

Use the course environment and its synced material tree when you have a JSC
HOTI account. Run these commands in an interactive SSH or Jupyter Terminal:

```bash
source "$PROJECT_training2633/env.sh"
jsc-material-sync
cd "$HOME/HotI26"
```

The course environment supplies `JSC_SUBMIT_CMD`; Makefile run targets honor
it automatically and submit through the Booster course allocation.

For another system, set `CUDA_HOME`, `NCCL_HOME`, and `NVSHMEM_HOME` to
installations containing the required `include/` and library directories.

Verify the tools first:

```bash
nvcc --version
mpicxx --version || mpicc --version
mpirun --version
python3 -V  # Chapter 7 only
```

## Jupiter launch

The tutorial systems are Jupiter Booster GH200 nodes, so the Makefile-based
labs before chapter 9 default to `CUDA_ARCH=90`. Chapter 9 targets GB200 and
GB300 NVL72 systems and instead defaults to native `sm_100` (GB200) and
`sm_103` (GB300) code. Request one GPU per Slurm task. Slurm then exposes each
task's assigned GPU as CUDA device 0; the labs handle that convention as well
as a local launch where all GPUs are visible.

~~~bash
salloc -p booster --nodes=1 --ntasks=2 --gpus-per-task=1

cd 01-nccl-host-apis/01-send-recv-on-stream
make
make run_SOLVED LAUNCHER='srun --ntasks=2 --gpus-per-task=1'
~~~

Every Makefile that launches multiple PEs accepts a complete `LAUNCHER`
override. Do not set a global `CUDA_VISIBLE_DEVICES=0,1` in this Slurm form:
Slurm supplies a separate one-GPU mask to each rank. For a local workstation,
leave GPU visibility unset or set `CUDA_VISIBLE_DEVICES` explicitly and use
the Makefile's default launcher.

## How the labs work

Start by verifying CUDA and MPI with the completed hello-world exercise:

```bash
cd 00-hello-world
make
make run
```

Then use the same build/run workflow in the coding labs. The first NCCL
exercise uses:

```bash
cd 01-nccl-host-apis/01-send-recv-on-stream
make
make run_SOLVED
make run  # after completing the starter
```

Run the solved version first to validate the environment, then fill in the
starter and use `make run`. C/CUDA Makefiles use a complete `LAUNCHER`
variable; some also expose `NP`, `CUDA_VISIBLE_DEVICES`, or architecture
variables. Use the leaf README and Makefile for the values supported by that
lab.

Chapter 7 installs the public Python dependencies through its leaf Makefiles.
The NCCL4Py and device-DSL labs build against the public
[NCCL4Py source](https://github.com/NVIDIA/nccl/tree/master/bindings/nccl4py)
specified by `NCCL4PY_SOURCE`; their leaf READMEs show the complete setup.
Chapters 6 and 8 have specialized build steps; follow their local READMEs.

## Tutorial layout

| Chapter | Topic | Exercises |
| --- | --- | --- |
| 0 | Hello world | CUDA/MPI toolchain check |
| 1 | NCCL host APIs | Send/receive on a stream; all-reduce on a stream |
| 2 | NVSHMEM RMA | Host put; device put; `nvshmem_ptr` |
| 3 | NCCL symmetric memory | Register symmetric memory; host PUT with symmetric operands |
| 4 | Memory model | Put + barrier; put + quiet + signal/wait; put-signal |
| 5 | NCCL device APIs | LSA device API; GIN put device API |
| 6 | Applications | Jacobi solver; fused GEMM + all-reduce with NVSHMEM and NCCL LSA |
| 7 | Python APIs | NVSHMEM4Py; NCCL4Py; Python device-API DSL |
| 8 | NCCL contrib and Extensions | Use NCCL EP |
| 9 | Topology-aware AlltoAllV | NVSHMEM; NCCL LSA; NCCL GIN; NCCL LSA + railed GIN |

The directory names follow the same order:

```text
00-hello-world/
01-nccl-host-apis/
02-nvshmem-rma/
03-nccl-symmetric/
04-memory-model/
05-nccl-device-apis/
06-applications/
07-python-apis/
08-nccl-contrib/
09-alltoallv/
```

`09-alltoallv/common/nvshmem_exercise.h` supplies shared CUDA, MPI, and
NVSHMEM setup reused by the NVSHMEM RMA, memory-model, fused-GEMM, and
AlltoAllV exercises.
`05-nccl-device-apis/device_api_common.hpp` supplies the NCCL communicator,
symmetric-window, and device-communicator setup reused by the NCCL fused-GEMM
implementation.
`09-alltoallv/alltoallv_common.hpp` generates one irregular communication
plan and validates it consistently across the NVSHMEM and NCCL versions.

## Troubleshooting

- If a compiler or linker cannot find NCCL or NVSHMEM, check that `NCCL_HOME`
  and `NVSHMEM_HOME` name installation prefixes, then check `LD_LIBRARY_PATH`.
- Match the MPI process count to the visible GPUs. The leaf README specifies
  the expected rank count and any required topology.
- Set the architecture variable documented by the leaf Makefile when its
  default does not match your GPU.
- A `SKIP` result in an `nvshmem_ptr`, NCCL window, or NCCL device-API lab can
  be a valid capability report. Read the printed reason and the leaf README.
- Keep the Python packages, CUDA, driver, and `nccl.core` installation
compatible for Chapter 7.

For system-specific commands, use the leaf README.
