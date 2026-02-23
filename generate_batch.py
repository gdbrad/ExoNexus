import yaml
from datetime import datetime
import os 
import argparse

def generate_batch_contraction(ini_file:str):
    with open(ini_file, 'r') as f:
        config = yaml.safe_load(f)

    slurm = config['slurm']
    env = config['environment']
    params = config['parameters']

    now = datetime.now()
    run_dir = f'run_{params['ens']}'
    os.makedirs(run_dir,exist_ok=True)
    with open(f'{run_dir}/run_{now.day}_{params['flavor']}_{params['irrep']}.sh', 'w') as sh:
        sh.write(f'''\
    #!/bin/bash
    #SBATCH --job-name={slurm['job_name']}
    #SBATCH --account={slurm['account']}
    #SBATCH --nodes={slurm['nodes']}
    #SBATCH --cpus-per-task={slurm['cpus_per_task']}
    #SBATCH --time={slurm['time']}
    #SBATCH --output={slurm['output']}
    #SBATCH --partition={slurm['partition']}
    #SBATCH --array={slurm['array']}
    #SBATCH --ntasks-per-node={slurm['ntasks_per_node']}

    module load Stages/2025 GCCcore/.13.3.0
    module load Python/3.12.3
    module load h5py
    module load GCC
    module load OpenMPI
    module load PyYAML
    module load sympy

    # Set environment variables
    export OMP_NUM_THREADS={env['omp_num_threads']}

    # Activate virtual environment
    source {env['virtual_env']}

    # Define parameters
    NUM_CONFIGS={params['num_configs']}
    NUM_VECS={params['num_vecs']}
    LT={params['lt']}
    ENS='{params['ens']}'
    CFG_STEP={params['cfg_step']}
    START_CFG={params['start_cfg']}
    END_CFG={params['end_cfg']}

    CFG_IDS=()
    for cfg in $(seq $START_CFG $CFG_STEP $END_CFG); do
        if [[ ! " $INVALID_CFGS " =~ " $cfg " ]]; then
            CFG_IDS+=("$cfg")
        fi
    done

    # Get the configuration ID for this task
    CFG_ID=${{CFG_IDS[$SLURM_ARRAY_TASK_ID]}}

    echo "SLURM_ARRAY_TASK_ID: ${{SLURM_ARRAY_TASK_ID}}"
    echo "Valid Config IDs: ${{CFG_IDS[*]}}"
    echo "Selected Config ID: ${{CFG_ID}}"

    if [[ -n "$CFG_ID" ]]; then
        echo "Running for cfg_id: ${{CFG_ID}}"
        srun python3 /p/scratch/exotichadrons/exotraction/src/two_pt_corr.py --lt {params['lt']} --nvecs {params['num_vecs']} --ens {params['ens']} --cfg_id ${{CFG_ID}} --flavor {params['flavor']} --task $((SLURM_ARRAY_TASK_ID + 1)) --ntsrc {params['ntsrc']}
    else
        echo "No valid configuration for this job."
        exit 1
    fi
    ''')
    
def main(ini_file): 
    generate_batch_contraction(ini_file=ini_file)

if __name__== '__main__': 
    parser = argparse.ArgumentParser(description="generate slurm batch script for contractions on the cpu node.")
    parser.add_argument('--ini', type=str, required=True, help="exotraction input file")

    args = parser.parse_args()
    main(ini_file=args.ini)

