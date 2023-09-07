localDir=`pwd`
run_file=$localDir/run_file.sh
submit_file=$localDir/submit_file.sh
program=$localDir/training.py



epochs=200
bs=16
name="Indents_"


for gpu in   1 2 4
do

        for augment in 0
        do


            mkdir -p $localDir/$name$gpu_$augment
            cd $localDir/$name$gpu_$augment


            if [ $gpu = 1 ]
            then

                tasks=1
                node=1

            else

                tasks=2
                node=$((${gpu}/2))

            fi

            # effective batch size
            batch=$((${gpu}*bs))

            # adapting run file
            sed -e "s|tag_program|${program}|g" ${run_file}  |\
            sed -e "s/\<tag_epoch\>/${epochs}/g"| \
            sed -e "s/\<tag_batch\>/${batch}/g"| \
            sed -e "s/\<tag_aug\>/${augment}/g" > script.sh

            # adapting submit file
            sed -e "s/\<tag_task\>/${tasks}/g" ${submit_file}|\
            sed -e "s/\<tag_node\>/${node}/g" > sub_${node}.sh

            # submit job
            sbatch sub_${node}.sh


    done
done



