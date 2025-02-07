#!/bin/bash
#SBATCH --job-name=pion_2pt_64 
#SBATCH --account=exncmf  
#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=16          
#SBATCH --time=03:00:00             
#SBATCH --output=test-pion_64_%j.log        
#SBATCH --partition=dc-cpu
#SBATCH --array=1-20

module load Python/3.11.3  
module load h5py/3.9.0
module load tqdm/4.66.1
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16


source /p/scratch/exotichadrons/exolaunch/contractions/contract/bin/activate
NUM_CONFIGS=10   # Number of configs to process per job
NUM_VECS=64      # Number of eigenvectors
NUM_TSRC=24      # Number of source time slices
CFG_STEP=10     
START_CFG=$(( 11 + (SLURM_ARRAY_TASK_ID - 1) * NUM_CONFIGS * CFG_STEP ))

CFG_IDS=$(seq $START_CFG $CFG_STEP $(( START_CFG + (NUM_CONFIGS - 1) * CFG_STEP )))

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start Config: ${START_CFG}"
echo "Generated Config IDs: $(echo ${CFG_IDS} | tr ' ' ',')"

# echo "srun python3 test-create-master.py --cfg_ids $(echo ${CFG_IDS} | tr ' ' ',') --nvec ${NUM_VECS} --ntsrc ${NUM_TSRC} --task ${SLURM_ARRAY_TASK_ID}"

srun python3 test-create-master.py --cfg_ids $(echo ${CFG_IDS} | tr ' ' ',') --nvec ${NUM_VECS} --ntsrc ${NUM_TSRC} --task ${SLURM_ARRAY_TASK_ID}
