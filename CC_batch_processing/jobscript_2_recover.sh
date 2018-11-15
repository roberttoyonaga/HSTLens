#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-15:00
#SBATCH --array=1-88 
#SBATCH --mem=8000M
./pipeline.exe $(($SLURM_ARRAY_TASK_ID*50))
