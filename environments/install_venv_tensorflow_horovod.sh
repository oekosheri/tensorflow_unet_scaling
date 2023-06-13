#!/usr/local_rwth/bin/zsh

# create virtual environment based on loaded python version
python3 -m venv ${TMP_ENV_NAME}
# activate environment
source ${TMP_ENV_NAME}/bin/activate

# install pytorch and supporting libraries
pip install tensorflow==2.7
pip3 install scikit-learn
pip3 install pandas
pip3 install opencv-python

# build and install horovod (will be linked against loaded MPI version)
HOROVOD_GPU_OPERATIONS=NCCL     \
HOROVOD_WITH_MPI=1              \
HOROVOD_WITH_TENSORFLOW=1    \
HOROVOD_WITHOUT_PYTORCH=1          \
pip3 install --no-cache-dir horovod
