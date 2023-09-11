localDir=`pwd`
run_file=$localDir/run_file.sh
submit_file=$localDir/submit_file.sh
program1=$localDir/tensorflow_create_tfconfig.py
program2=$localDir/training.py
setup=$localDir/setup_dist_env.sh


epochs=200
bs=16
name="Indents_"


# rm -r $localDir/Logs
# mkdir $localDir/Logs

for gpu in 1 2 4
do

        for augment in 0
        do


            mkdir -p $localDir/$name$gpu$augment
            cd $localDir/$name$gpu$augment


            if [ $gpu = 1 ]
            then
                tasks=1
                command=2
                node=1
            elif [ $gpu = 2 ]
            then
                tasks=2
                command=2
                node=1
            else
                tasks=2
                command=1
                node=$((${gpu}/2))

            fi

            batch=$((${gpu}*bs))
            # adapting run file
            sed -e "s|tag_program1|${program1}|g" ${run_file}  |\
            sed -e "s|tag_program2|${program2}|g"|\
            sed -e "s/\<tag_epoch\>/${epochs}/g"| \
            sed -e "s/\<tag_batch\>/${batch}/g"| \
            sed -e "s/\<tag_aug\>/${augment}/g" > script.sh

            # adapting submit file
            sed -e "s/\<tag_task\>/${tasks}/g" ${submit_file}|\
            sed -e "s/\<tag_node\>/${node}/g" | \
            sed -e "s|setup.sh|${setup}|g" | \
            sed -e "s/\<tag_command\>/${command}/g" > sub_${node}.sh
            sbatch sub_${node}.sh
        done

done



