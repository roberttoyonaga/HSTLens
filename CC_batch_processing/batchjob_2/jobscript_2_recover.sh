#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-16:00
#SBATCH --array=1-220 
#SBATCH --mem=8000M
./pipeline_2_recover.exe $(($SLURM_ARRAY_TASK_ID*20))

#8h array 1-176 25 jobs each