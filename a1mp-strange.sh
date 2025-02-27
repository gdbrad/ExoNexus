#!/bin/bash
#SBATCH --job-name=a1mp_strange_mom 
#SBATCH --account=exncmf  
#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=32          
#SBATCH --time=24:00:00             
#SBATCH --output=h5-a1_mp_strange/test-a1%a.log        
#SBATCH --partition=dc-cpu
#SBATCH --array=1-20

module load Stages/2025
module load Python
module load h5py
module load GCC
module load OpenMPI
module load PyYAML
module load sympy
module load mpi4py

export OMP_NUM_THREADS=32

source /p/scratch/exotichadrons/exotraction/exo/bin/activate

#SBATCH --ntasks-per-node=1 
NUM_CONFIGS=10   # Number of configs to process per job
NUM_VECS=96      # Number of eigenvectors
NUM_TSRC=24      # Number of source time slices
CFG_STEP=10     
START_CFG=$(( 11 + (SLURM_ARRAY_TASK_ID - 1) * NUM_CONFIGS * CFG_STEP ))

# List of invalid configurations
INVALID_CFGS="21 171 1001 1061 1271 1371 1451 1531 1591 1611 1641 1711 1781 1851 1871 1901 1941 1991"

# Generate the list of valid config IDs
CFG_IDS=""
for cfg in $(seq $START_CFG $CFG_STEP $(( START_CFG + (NUM_CONFIGS - 1) * CFG_STEP ))); do
    if [[ ! " $INVALID_CFGS " =~ " $cfg " ]]; then
        CFG_IDS+="$cfg "
    fi
done

CFG_IDS=$(echo $CFG_IDS | tr ' ' ',')
INI=ini/a1mp_strange.ini.yml

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start Config: ${START_CFG}"
echo "Valid Config IDs: ${CFG_IDS}"

if [[ -n "$CFG_IDS" ]]; then
    srun python3 src/contract_gevp_multi_mom.py --ini $INI --cfg_ids ${CFG_IDS} --task ${SLURM_ARRAY_TASK_ID}
else
    echo "No valid configurations for this job."
fi
