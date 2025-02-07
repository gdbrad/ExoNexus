#!/bin/bash
#SBATCH --job-name=kaon_cpu_parallel 
#SBATCH --account=exncmf  
#SBATCH --nodes=2          
#SBATCH --time=1:00:00          
#SBATCH --output=contract-log/kaon-mpi_%a.log        
#SBATCH --partition=dc-cpu-devel
#SBATCH --error=contract-log/kaon-mpi_%a.err

module load Stages/2024  GCCcore/.12.3.0
module load GCC
module load Python/3.11.3
module load h5py/3.9.0
module load PyYAML/6.0
module load sympy/1.12
module load OpenMPI
module load mpi4py/3.1.4

export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16


# Use the pre-defined YAML configuration file
INI=ini/kaon.ini.yml
# INI=ini/a1mp.ini.yml

# srun --threads-per-core=1 --ntasks-per-node=100 python3 contract_gevp.py --ini $INI --strange 
srun --cpus-per-task=32 --ntasks-per-node=64 python3 contract_gevp.py --ini $INI --strange

# srun --threads-per-core=1 --ntasks-per-node=100 python3 contract_gevp.py --ini $INI 

