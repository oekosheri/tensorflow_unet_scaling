#!/usr/local_rwth/bin/zsh
#SBATCH --time=2:30:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tag_task
#SBATCH --account=p0020572

localDir=`pwd`


source ../environments/load_env_rocky.sh
source ../environments/horovod-env/bin/activate

# mpirun the program
${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c 'bash script.sh'


# save the log file
cp log.csv  ../logs/log_hvd_${SLURM_NTASKS}.csv
rm core.*

