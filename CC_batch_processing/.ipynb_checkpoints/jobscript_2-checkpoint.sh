#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-0:20
#SBATCH --array=1-2 
#SBATCH --mem=5000M
./pipeline.exe $(($SLURM_ARRAY_TASK_ID*2))
