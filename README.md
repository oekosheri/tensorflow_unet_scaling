## GPU acceleration of Unet using Tensorflow native environment and Horovod

On the root directory you will find the scripts to run UNet implemented in Tensorflow using a GPU data parallel scheme in Horovod Tensorflow. In the native Tensorflow directory you will find the scripts to run the same training jobs using Tensorflow native environment without Horovod. The goal is to compare the paralleisation performance of Horovod Tensorflow vs native tensorflow for a UNet algorithm. The data used here is an open microscopy data for semantic segmentation: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7639190.svg)](https://doi.org/10.5281/zenodo.7639190). These calculations have all been done on the [RWTH high performance computing cluster](https://help.itc.rwth-aachen.de/), using Tesla V100 GPUs. 

### Virtual environment:

To install Horovod for Tensorflow a virtual environment was created. The same environment will be used for both Tensorflow native and Horovod trainings so that the results are comparable. Read this README to see which softwares and environments need to be loaded before creating the vitual envs and running the jobs. For both native and Horovod we use NCCL as the backend for collective communications. We also use open MPI for spawning the processes. 
