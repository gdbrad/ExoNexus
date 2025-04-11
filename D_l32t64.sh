#!/bin/bash
#SBATCH --job-name=pi_meson 
#SBATCH --account=exncmf  
#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=32          
#SBATCH --time=00:30:00             
#SBATCH --output=D_all_%a.log        
#SBATCH --partition=dc-cpu-devel
#SBATCH --array=1-2

module load Stages/2025 GCCcore/.13.3.0
module load Python/3.12.3
module load h5py
module load GCC
module load OpenMPI
module load PyYAML
module load sympy
module load mpi4py

export OMP_NUM_THREADS=32

source /p/scratch/exotichadrons/exotraction/exo/bin/activate

#SBATCH --ntasks-per-node=1 
NUM_CONFIGS=2  
NUM_VECS=64    
NUM_TSRC=24    
CFG_STEP=50     
START_CFG=$(( 400 + (SLURM_ARRAY_TASK_ID - 1) * NUM_CONFIGS * CFG_STEP ))

INVALID_CFGS="1991"

CFG_IDS=()
for cfg in $(seq $START_CFG $CFG_STEP $(( START_CFG + (NUM_CONFIGS - 1) * CFG_STEP ))); do
    if [[ ! " $INVALID_CFGS " =~ " $cfg " ]]; then
        CFG_IDS+=("$cfg")
    fi
done

CFG_IDS=$(echo $CFG_IDS | tr ' ' ',')

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start Config: ${START_CFG}"
echo "Valid Config IDs: ${CFG_IDS}"

CFG_ID=${CFG_IDS[$SLURM_ARRAY_TASK_ID - 1]}

if [[ -n "$CFG_ID" ]]; then
    echo "Running for cfg_id: ${CFG_ID}"
    srun python3 src/two_pt_corr.py --cfg_id ${CFG_ID} --flavor light_charm,light_light --task ${SLURM_ARRAY_TASK_ID}
else
    echo "No valid configuration for this job."
    exit 1
fi
