
tasks = [1,2,4,6,8,10,12,14,16]  #nr. nodes/gpus

def main():

    for i in range(len(tasks)):
        t = tasks[i]

        file = open("job_p"+str(t)+".sh","w")

        text = """#!/bin/bash


#SBATCH --ntasks-per-node=1
#SBATCH -n """+str(t)+"""
#SBATCH -N """+str(t)+"""
#SBATCH -o output.p"""+str(t)+""".r%a.out
#SBATCH -e error.p"""+str(t)+""".r%a.out
#SBATCH --time=03:00:00
#SBATCH --mem=46000
#SBATCH -J nhr4cs_"""+str(t)+"""
#SBATCH --exclusive
#SBATCH --gres=gpu:1

ml --force purge
ml restore dl3
ml -f unload nvidia-driver/.default

export TMPDIR=/scratch
export CUDA_VISIBLE_DEVICES=0

SECONDS=0;

PSP_OPENIB=1 PSP_UCP=0 srun python training.py --log_name log_g"""+str(t)+"""_r$SLURM_ARRAY_TASK_ID --mask_dir masks_collective --image_dir images_collective

echo $SECONDS
"""

        file.write(text)

        file.close()

if __name__ == "__main__":
    main()
