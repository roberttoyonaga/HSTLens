#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-24:00
#SBATCH --array=1-1472
#SBATCH --mem=8000M
time ./pipeline_3.exe $(($SLURM_ARRAY_TASK_ID*5)) 
