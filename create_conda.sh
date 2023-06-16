#!/usr/local_rwth/bin/zsh

export ENV_PREFIX=$PWD/env
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_MPI=1
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITHOUT_PYTORCH=1
conda env create --prefix $ENV_PREFIX --file environment1.yml --force


