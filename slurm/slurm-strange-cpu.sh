#!/bin/bash
#SBATCH --job-name=strange_gevp_cpu 
#SBATCH --account=exncmf  
#SBATCH --nodes=1                   
#SBATCH --time=06:00:00          
#SBATCH --output=contract-log/a1m_strange_gevp_%a.log        
#SBATCH --partition=dc-cpu
module load Stages/2024  GCCcore/.12.3.0
module load Python/3.11.3
module load h5py/3.9.0
module load tqdm/4.66.1
module load PyYAML/6.0
module load sympy/1.12
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16

#source /p/scratch/exotichadrons/exotraction/exotraction/bin/activate
NUM_CONFIGS=1   # Number of configs to process per job
CFG_STEP=10
START_CFG=$(( 11 + (SLURM_ARRAY_TASK_ID - 1) * NUM_CONFIGS * CFG_STEP ))

CFG_IDS=$(seq $START_CFG $CFG_STEP $(( START_CFG + (NUM_CONFIGS - 1) * CFG_STEP )))
CFG_ID=61
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start Config: ${START_CFG}"
echo "Generated Config IDs: $(echo ${CFG_IDS} | tr ' ' ',')"

# Use the pre-defined YAML configuration file
INI=ini/a1mp-strange.ini.yml

# Run the Python script with the YAML file
# srun --cpus-per-task=16 python3 contract_2pt_matrix_gpu.py --ini $INI --cfg_ids $(echo ${CFG_IDS} | tr ' ' ',') --task_id ${SLURM_ARRAY_TASK_ID} --strange
# srun --cpus-per-task=16 python3 contract_2pt_matrix_gpu.py --ini $INI --cfg_ids 31 --task_id ${SLURM_ARRAY_TASK_ID} --strange
srun --ntasks-per-node=128 --cpus-per-task=1 python3 contract_2pt_matrix_gpu.py --ini $INI --cfg_ids $(echo ${CFG_IDS} | tr ' ' ',') --task_id ${SLURM_ARRAY_TASK_ID} --strange


#srun --cpus-per-task=16 python3 contract_2pt_matrix.py --ini $INI --cfg_ids 11 --task_id ${SLURM_ARRAY_TASK_ID}
