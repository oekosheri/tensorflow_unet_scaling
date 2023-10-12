#!/usr/local_rwth/bin/zsh
#SBATCH --time=2:30:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tag_task
#SBATCH --account=********

localDir=`pwd`



source ../../environments/load_env_rocky.sh
source ../../environments/horovod-env/bin/activate


comm_1="${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c '\
source setup.sh  && bash script.sh'"

comm_2="source setup.sh && bash script.sh"


command=tag_command


if [ $command = 1 ]

then

    eval  "${comm_1}"

else

    eval  "${comm_2}"

fi

# save the log file
cp log.csv  ../../logs/log_${SLURM_NTASKS}.csv
rm core.*

