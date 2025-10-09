#!/bin/bash
#SBATCH --job-name=l40t64_pion_serial
#SBATCH --account=exncmf  
#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=32          
#SBATCH --time=00:30:00             
#SBATCH --output=l40t64-pion_serial_%a.log        
#SBATCH --partition=dc-cpu-devel
#SBATCH --array=0-1

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
NUM_CONFIGS=1 
NUM_VECS=32
LT=64
ENS='eric-l40t64-serial'
CFG_STEP=50     
START_CFG=950
END_CFG=1050

CFG_IDS=()
for cfg in $(seq $START_CFG $CFG_STEP $END_CFG); do
    if [[ ! " $INVALID_CFGS " =~ " $cfg " ]]; then
        CFG_IDS+=("$cfg")
    fi
done

# Get the configuration ID for this task
CFG_ID=${CFG_IDS[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Valid Config IDs: ${CFG_IDS[*]}"
echo "Selected Config ID: ${CFG_ID}"

if [[ -n "$CFG_ID" ]]; then
    echo "Running for cfg_id: ${CFG_ID}"
    srun python3 ../src/two_pt_corr.py --lt ${LT} --nvecs ${NUM_VECS} --ens ${ENS} --cfg_id ${CFG_ID} --flavor light_light --task $((SLURM_ARRAY_TASK_ID + 1)) --ntsrc 8
else
    echo "No valid configuration for this job."
    exit 1
fi
