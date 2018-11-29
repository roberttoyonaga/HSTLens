#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=00:14:00
#SBATCH --array=1-736 
#SBATCH --mem=8000M
./pipeline_2.exe $(($SLURM_ARRAY_TASK_ID*10))
