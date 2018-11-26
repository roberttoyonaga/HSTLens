#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-7:00
#SBATCH --array=1-368 
#SBATCH --mem=8000M
./pipeline_2.exe $(($SLURM_ARRAY_TASK_ID*20))
