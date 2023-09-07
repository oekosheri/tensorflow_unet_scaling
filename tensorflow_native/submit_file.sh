#!/usr/local_rwth/bin/zsh
#SBATCH --time=2:30:00
#SBATCH --partition=c18g
#SBATCH --nodes=tag_node
#SBATCH --ntasks-per-node=tag_task
#SBATCH --cpus-per-task=tag_cpu
#SBATCH --gres=gpu:tag_task
#SBATCH --account=p0020572

localDir=`pwd`



source ../../environments/load_env_rocky.sh
source ../../environments/horovod-env/bin/activate


# source ~/.zshrc
# conda activate tensor_new


# module list
# echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
# echo "R_WLM_ABAQUSHOSTLIST: ${R_WLM_ABAQUSHOSTLIST}"
# echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"


comm_1="${MPIEXEC} ${FLAGS_MPI_BATCH} zsh -c '\
source setup.sh  && bash script.sh' --export=R_WLM_ABAQUSHOSTLIST"

comm_2="source setup.sh && bash script.sh"


command=tag_command


if [ $command = 1 ]

then

    eval  "${comm_1}"

else

    eval  "${comm_2}"

fi

# save the log file
cp log.csv  ../Logs/log_${SLURM_NTASKS}.csv


