#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-1:00 #--time=1-18:00
#SBATCH --array=1-2
#SBATCH --mem=8000M
time ./pipeline_3.exe $(($SLURM_ARRAY_TASK_ID*2)) #*10
