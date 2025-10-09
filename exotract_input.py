from dataclasses import dataclass
import yaml
from datetime import datetime
import os 
import argparse
import re 
from collections import List 

from src import * 
'''
to calculate a 2pt correlator we first 
- Read in source operators from an input list 
- Read in sink operators from an input list 
- parse the operator string and force them to be ``QuantumNum`` objects 
- Select the irreps commensurate with the given momentum, P^2

this generates parameters to run exotractions 
- generates directory of contraction batch scripts
- 
'''
@dataclass
class QuantumNum:
    name: str
    had: int
    F: str
    flavor: str
    twoI: int
    S: int
    P: int
    C: int
    gamma: str
    gamma_i: bool
    deriv: str
    mom: str 


class Exotraction:
    '''parameters for exotraction
    this interfaces with an input file 
    ''' 

    def __init__(self,ini):
        self.ini = ini
    
        with open(self.ini, 'r') as f:
            self.params = yaml.safe_load(f)


    def parse_op_string(self) -> List[QuantumNum]:
        """parse string and extract attributes so that two_pt_mhi can access them
        hadspec example 
        "XXpion_pionxD0_J0__J0_D2A2__-101xxpion_pionxD0_J0__J0_D2A2__110__F3,1_D2B1P,1__011XXpion_pionxD0_J0__J0_D2A2__0-1-1__F5,1_T1mM,1__000"
         """
        # first grab the insertion_operators (bare) 
         # Split the string into individual operator components
        op_string = self.params['op_string']
        op_list = op_string.split('XX')[1:]  # Skip initial 'XX' if present
        quantum_nums = []


        for op in op_list:
            # Split into main components
            parts = op.split('__')
            # if len(parts) != 3:
            #     raise ValueError(f"Invalid operator format: {op}")

            # Part 1: Flavor and displacement
            flavor_part, momentum_part, gamma_deriv_part = parts

            # Flavor and hadron type
            flavor_match = re.match(r'^(\d{3}[a-zA-Z0-9]x|D0_J0x|.+?)(\w+)$', flavor_part)
            if not flavor_match:
                raise ValueError(f"Invalid flavor format: {flavor_part}")
            flavor_prefix, disp = flavor_match.groups()
            flavor = 'light'  # Default, adjust based on prefix if needed
            had = 1  # Meson (no parity in name)
            if 'D0' in flavor_prefix:
                flavor = 'charm'

            # Momentum and quantum numbers
            mom_match = re.match(r'(\w+)_(\w+)__(\w+)__([-0-1]+)', momentum_part)
            if not mom_match:
                raise ValueError(f"Invalid momentum format: {momentum_part}")
            disp_type, spin, irrep, mom_str = mom_match.groups()
            # Convert momentum string (e.g., '-101') to 3-tuple (e.g., (-1, 0, 1))
            if len(mom_str) != 3:
                raise ValueError(f"Invalid momentum format, expected 3 digits: {mom_str}")
            try:
                mom = tuple(int(c) if c != '-' else -1 for c in mom_str)  # e.g., '-101' -> (-1, 0, 1)
            except ValueError:
                raise ValueError(f"Invalid momentum components: {mom_str}")
            twoI = 0  # Assume isospin 0 for pions
            S = 0     # Assume spin 0 for pions
            P = 1     # Assume positive parity for pions
            C = 1     # Assume positive charge conjugation for pions

            # Gamma and derivative
            gamma_deriv_match = re.match(r'(\w+,\d)_(\w+,\d)__([-0-1]+)', gamma_deriv_part)
            if not gamma_deriv_match:
                raise ValueError(f"Invalid gamma/deriv format: {gamma_deriv_part}")
            gamma, deriv, proj = gamma_deriv_match.groups()
            gamma_i = False  # Assume not summed over gamma_i unless specified
            deriv_type = deriv.split(',')[0]  # e.g., 'D2B1P' -> 'D2B1P'
            if deriv_type.startswith('D'):
                deriv_type = 'D'
            elif deriv_type.startswith('B'):
                deriv_type = 'B'
            elif deriv_type == 'nabla':
                deriv_type = 'nabla'
            else:
                deriv_type = None

            # Construct name
            name = f"{flavor_prefix}{disp}_{gamma}_{deriv}"

            # Create QuantumNum object
            qn = QuantumNum(
                name=name,
                had=had,
                F=irrep,  # Use irrep as F (e.g., 'D2A2')
                flavor=flavor,
                twoI=twoI,
                S=S,
                P=P,
                C=C,
                gamma=gamma,  # e.g., 'F3,1'
                gamma_i=gamma_i,
                deriv=deriv_type,
                mom=mom  # Store as 3-tuple, e.g., (-1, 0, 1)
            )
            quantum_nums.append(qn)

        return quantum_nums



    def generate_corr2pt(self): 
        print('reading src operators..')
        src_ops_list = [params.src_ops]

        print('reading snk operators')
        snk_ops_list = [params.snk_ops]

        print('reading operator map')
        
        operators = params.op_list

        # parse list 

        print('building two point correlator object')


    def generate_batch(self):
        with open(self.ini, 'r') as f:
            config = yaml.safe_load(f)

        # Extract parameters
        slurm = config['slurm']
        env = config['environment']
        params = config['parameters']

        # Generate the shell script
        now = datetime.now()
        run_dir = f'run_{params['ens']}_{params['channel']}'
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
        module load mpi4py


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
            srun python3 ../src/two_pt_corr.py --lt {params['lt']} --nvecs {params['num_vecs']} --ens {params['ens']} --cfg_id ${{CFG_ID}} --flavor {params['flavor']} --task $((SLURM_ARRAY_TASK_ID + 1)) --ntsrc {params['ntsrc']}
        else
            echo "No valid configuration for this job."
            exit 1
        fi
        ''')
        

def main(): 
    exotract = Exotraction(ini=args.ini)
    exotract.generate_corr2pt()
    exotract.generate_batch()

if __name__== '__main__': 
    parser = argparse.ArgumentParser(description="Process peram and meson files for one or two flavor systems.")
    parser.add_argument('--ini', type=str, required=True, help="exotraction input file")

    args = parser.parse_args()
    main()


    




