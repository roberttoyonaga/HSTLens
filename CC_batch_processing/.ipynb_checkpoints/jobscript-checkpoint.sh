#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --time=0-0:20
#SBATCH --array=1-9000
#SBATCH --mem=4000M
./pipeline.exe $SLURM_ARRAY_TASK_ID
