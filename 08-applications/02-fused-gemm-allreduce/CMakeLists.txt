cmake_minimum_required(VERSION 3.19)

set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE PATH "NVSHMEM installation prefix")
set(HOTI_GEMM_VARIANT "starter" CACHE STRING "Build the starter or solved exercise")
set_property(CACHE HOTI_GEMM_VARIANT PROPERTY STRINGS starter SOLVED)
set(HOTI_CUDA_ARCH 90 CACHE STRING "CUDA architecture for the tutorial GPUs")

set(CMAKE_CUDA_ARCHITECTURES "${HOTI_CUDA_ARCH}")

project(FusedGemmAllReduce LANGUAGES CXX CUDA)

if(NOT NVSHMEM_HOME)
  message(FATAL_ERROR "Set NVSHMEM_HOME or pass -DNVSHMEM_HOME=<install-prefix>.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(NVSHMEM QUIET CONFIG HINTS "${NVSHMEM_HOME}/lib/cmake/nvshmem")

if(HOTI_GEMM_VARIANT STREQUAL "starter")
  set(gemm_source fused_gemm_allreduce.cu)
elseif(HOTI_GEMM_VARIANT STREQUAL "SOLVED")
  set(gemm_source fused_gemm_allreduce_SOLVED.cu)
else()
  message(FATAL_ERROR "HOTI_GEMM_VARIANT must be starter or SOLVED.")
endif()

add_executable(fused_gemm_allreduce "${gemm_source}")

target_compile_features(fused_gemm_allreduce PRIVATE cxx_std_17)
target_link_libraries(fused_gemm_allreduce PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  MPI::MPI_CXX)
if(NVSHMEM_FOUND)
  target_link_libraries(fused_gemm_allreduce PRIVATE
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device)
else()
  find_path(NVSHMEM_INCLUDE_DIR nvshmem.h HINTS "${NVSHMEM_HOME}/include" REQUIRED)
  find_library(NVSHMEM_HOST_LIBRARY nvshmem_host HINTS "${NVSHMEM_HOME}/lib" REQUIRED)
  find_library(NVSHMEM_DEVICE_LIBRARY nvshmem_device HINTS "${NVSHMEM_HOME}/lib" REQUIRED)
  target_include_directories(fused_gemm_allreduce PRIVATE "${NVSHMEM_INCLUDE_DIR}")
  target_link_libraries(fused_gemm_allreduce PRIVATE
    "${NVSHMEM_HOST_LIBRARY}"
    "${NVSHMEM_DEVICE_LIBRARY}")
endif()
set_target_properties(fused_gemm_allreduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
