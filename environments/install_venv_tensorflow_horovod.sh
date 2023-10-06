#!/usr/local_rwth/bin/zsh

# create virtual env
python3 -m venv horovod-env
source horovod-env/bin/activate

pip install tensorflow==2.12.0 --no-cache-dir
HOROVOD_NCCL_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/NCCL/2.11.4-GCCcore-11.3.0-CUDA-11.8.0 \
HOROVOD_GPU_OPERATIONS=NCCL \
HOROVOD_WITH_MPI=1 \
HOROVOD_WITH_TENSORFLOW=1 \
pip install --no-cache-dir -v horovod

