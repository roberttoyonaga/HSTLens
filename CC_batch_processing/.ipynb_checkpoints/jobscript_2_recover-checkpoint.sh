#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-08:00
#SBATCH --array=1-176 
#SBATCH --mem=8000M
./pipeline_2_recover.exe $(($SLURM_ARRAY_TASK_ID*25))

#8h array 1-176 25 jobs each