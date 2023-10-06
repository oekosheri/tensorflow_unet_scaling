#!/usr/local_rwth/bin/zsh


# clean env first
module purge
# open MPI module and GCC
module load gompi/2020a
module load GCCcore/.11.3.0
# CUDA and cudnn
module load cuDNN/8.6.0.163-CUDA-11.8.0
# NCCL and CMake
module load NCCL/2.11.4-CUDA-11.8.0
module load CMake/3.21.1
# load Python
module load Python/3.9.6
export CUDA_DIR=/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/CUDA/11.8.0