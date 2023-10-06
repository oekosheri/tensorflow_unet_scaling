## GPU acceleration of Unet using Tensorflow native environment and Horovod

On the root directory you will find the scripts to run UNet implemented in Tensorflow using a GPU data parallel scheme in Horovod Tensorflow. In the native Tensorflow directory you will find the scripts to run the same training jobs using Tensorflow native environment without Horovod. The goal is to compare the paralleisation performance of Horovod Tensorflow vs native tensorflow for a UNet algorithm. The data used here is an open microscopy data for semantic segmentation: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7639190.svg)](https://doi.org/10.5281/zenodo.7639190). These calculations have all been done on the [RWTH high performance computing cluster](https://help.itc.rwth-aachen.de/), using Tesla V100 GPUs. 

### Virtual environment

To install Horovod for Tensorflow a virtual environment was created. The same environment will be used for both Tensorflow native and Horovod trainings so that the results are comparable. Read this [README](./environments/README.md) to see which softwares and environments need to be loaded before creating the vitual envs and running the jobs. For both native and Horovod we use NCCL as the backend for collective communications. We also use open MPI for spawning the parallel processes. 

### Data parallel scheme

n a data parallel scheme, the mini-batch size is fixed per GPU (worker) and is usually the batch size that maxes out the GPU memory. In our case here it was 16. By having more GPUs the effective batch size increases and therefore run time decreases. This is a very common method of deep learning parallelisation. The drawback maybe that it can eventually lead to poor convergence and therefore model metrics (in our case intersection over union (IOU)) deteriorate. 

To implement the data parallel scheme the following necessary steps have been taken:

Submission:

- For native Tensorflow during the submission process, some SLURM environmental [variables](./tensorflow_native/setup_dist_env.sh) have been set up which will help us access the size and ranks of workers during training. Additionally to run Tensorflow on more than one node, a [TF_CONFIG](./tensorflow_native/tensorflow_create_tfconfig.py) env variable needs to be set that lists the IP addresses of all workers. The linked python file helps set it up for our computing cluster. For Horovod Tensorflow no env variables are required to be set up.

Training:  

- 
