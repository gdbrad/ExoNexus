#!/bin/bash
#SBATCH --job-name=light_gevp 
#SBATCH --account=exotichadrons 
#SBATCH --nodes=1       
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00             
#SBATCH --output=contract-log/a1m_light_gevp%a.log        
#SBATCH --partition=dc-gpu
#SBATCH --array=1-40
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=none

module load Stages/2024  GCCcore/.12.3.0
module load Python/3.11.3
module load h5py/3.9.0
module load tqdm/4.66.1
module load PyYAML/6.0
module load sympy/1.12
module load CuPy/12.2.0-CUDA-12

export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16

#source /p/scratch/exotichadrons/exotraction/exotraction/bin/activate
NUM_CONFIGS=5   # Number of configs to process per job
CFG_STEP=10
START_CFG=$(( 11 + (SLURM_ARRAY_TASK_ID - 1) * NUM_CONFIGS * CFG_STEP ))

CFG_IDS=$(seq $START_CFG $CFG_STEP $(( START_CFG + (NUM_CONFIGS - 1) * CFG_STEP )))

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start Config: ${START_CFG}"
echo "Generated Config IDs: $(echo ${CFG_IDS} | tr ' ' ',')"

INI=ini/a1mp.ini.yml

#srun --cpus-per-task=16 python3 contract_2pt_matrix.py --ini $INI --cfg_ids $(echo ${CFG_IDS} | tr ' ' ',') --task_id ${SLURM_ARRAY_TASK_ID}
#CUDA_VISIBLE_DEVICES=0 srun -n 1 --gres=gpu:1 -c 16 python3 contract_2pt_matrix_gpu.py --ini $INI --cfg_ids 11 --task_id ${SLURM_ARRAY_TASK_ID} --gpu

CUDA_VISIBLE_DEVICES=0 srun -n 1 --gres=gpu:1 -c 16 python3 contract_2pt_matrix_gpu.py --ini $INI --cfg_ids $(echo ${CFG_IDS} | tr ' ' ',') --task_id ${SLURM_ARRAY_TASK_ID} --gpu

