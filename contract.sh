#!/bin/bash
#SBATCH --job-name=kaon_cpu_mpi 
#SBATCH --account=exncmf  
#SBATCH --nodes=1        
#SBATCH --time=00:20:00          
#SBATCH --output=contract-log/charm.log        
#SBATCH --partition=dc-cpu-devel
#SBATCH --error=contract-log/charm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load Stages/2025
module load GCC
module load OpenMPI
module load Python
module load h5py
module load PyYAML
module load sympy
module load mpi4py

INI=ini/kaon.ini.yml

# srun --exclusive --ntasks-per-node=100 python3 contract_gevp.py --ini $INI --strange
#srun --ntasks-per-node=100 python3 contract_gevp.py --ini $INI --strange
#srun python3 contract_gevp.py --ini $INI --strange
srun python3 src/two_pt_corr.py --cfg_id 31 --flavor 'charm_strange' --task 1
