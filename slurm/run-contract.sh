#!/bin/bash
#SBATCH --job-name=contract        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=02:00:00          
#SBATCH --partition=dc-cpu
#SBATCH --account=exotichadrons

python3 contract.py