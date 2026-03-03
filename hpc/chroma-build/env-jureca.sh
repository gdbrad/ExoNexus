#!/bin/bash
## MODULES
module --force purge
module load Stages/2025 GCC/13.3.0
module load OpenMPI
module load MPI-settings/CUDA
module load UCX-settings/RC-CUDA
module load CMake
module load HDF5
module load GMP
module load Python
module load OpenBLAS

moudle list 

QMP=qmp
QDP=qdpxx
QUDA=quda
CHROMA=chroma
LALIBE=lalibe
H5=ON

## COMPILE ENV
export GPU_ARCH=sm_80
export CC=mpicc
export CXX=mpicxx
export CMAKE=cmake
export CMAKE_EXTRA_FLAGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=1"
export CMAKE_MAKE_FLAGS="--parallel $(nproc)"

## RUNTIME ENV
export CODE_BASE_DIR="/p/scratch/exotichadrons/chroma-distillation-superb"
export LD_LIBRARY_PATH="$CODE_BASE_DIR/install/chroma/lib:$CODE_BASE_DIR/install/quda/lib:$CODE_BASE_DIR/install/qdpjit/lib:$CODE_BASE_DIR/install/qmp/lib:$CODE_BASE_DIR/install/llvm/lib:$LD_LIBRARY_PATH"
export QUDA_RESOURCE_PATH="$CODE_BASE_DIR/quda_resources"
export QUDA_TUNE_VERSION_CHECK=0

