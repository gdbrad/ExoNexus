#!/bin/bash
#SBATCH --job-name=kaon_cpu_mpi 
#SBATCH --account=exncmf  
#SBATCH --nodes=2         
#SBATCH --time=1:00:00          
#SBATCH --output=contract-log/kaon-gevp%a.log        
#SBATCH --partition=dc-cpu-devel
#SBATCH --error=contract-log/kaon-gevp_%a.err
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1

module load Stages/2025
module load GCC
module load OpenMPI
module load Python
module load h5py
module load PyYAML
module load sympy
module load mpi4py


# Use the pre-defined YAML configuration file
INI=ini/kaon.ini.yml

# srun --exclusive --ntasks-per-node=100 python3 contract_gevp.py --ini $INI --strange
srun --ntasks-per-node=100 python3 contract_gevp.py --ini $INI --strange

