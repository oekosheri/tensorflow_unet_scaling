```
srun -A deepsea -N 1 -p dp-esb -t 01:00:00 --pty --interactive bash


ml restore dl2

ml -f unload nvidia-driver/.default

```
