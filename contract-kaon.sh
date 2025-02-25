#!/bin/bash -x
#SBATCH --job-name=kaon_cpu_mpi 
#SBATCH --account=exncmf  
#SBATCH --nodes=1      
#SBATCH --time=10:00:00          
#SBATCH --output=h5-kaon/kaon-gevp_%j.log
#SBATCH --partition=dc-cpu
#SBATCH --error=h5-kaon/kaon-gevp_%j.err
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=1

module load Stages/2025
module load GCC
module load OpenMPI
module load Python
module load h5py
module load PyYAML
module load sympy
module load mpi4py

# INI=ini/kaon.ini.yml
INI=ini/kaon.ini.yml


# srun --exclusive --ntasks-per-node=100 python3 contract_gevp.py --ini $INI --strange
#srun --ntasks-per-node=90 python3 contract_gevp.py --ini $INI --strange
for i in {0..3}; do
    srun --ntasks=50 --exclusive python3 contract_gevp.py --ini $INI --strange &
done

wait  
# echo "Executing program..."

# srun python3 contract_gevp.py --ini $INI --strange

