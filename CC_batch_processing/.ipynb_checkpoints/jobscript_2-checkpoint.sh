#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-17:00
#SBATCH --array=1-180 
#SBATCH --mem=5000M
./pipeline.exe $(($SLURM_ARRAY_TASK_ID*50))
