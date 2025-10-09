import h5py
import numpy as np
import yaml 
import os
import argparse
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
import gamma as gamma
from typing import List
import operator_factory
from operator_factory import QuantumNum
# Constants
BASE_PATH = os.path.abspath('/p/scratch/exotichadrons/exolaunch')

def get_file_path(directory, filename, cfg_id):
    """Construct file path and check if it exists."""
    full_path = os.path.join(directory, filename.format(cfg_id=cfg_id))
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}. Skipping.")
        return None
    return full_path

def load_op_map(channel: str):
    """Loads a QuantumNum object from operator_factory."""
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")
    

D_ins = QuantumNum(name='D',had=1, F="A1", flavor='charm',twoI=1, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_0')
D_star_ins = QuantumNum(name='D_star',had=1, F="T1", flavor='charm',twoI=1, S=0, P=-1, C=1, gamma=gamma.IDEN,gamma_i=True,deriv=None,mom='mom_0_0_0')

# 1/2 * 1/2 cg coeffs 
# isospin 1,0 
# use projection coeffs from su2 

# mixing 0,1 so opposite in terms of signal 
# 0 has disco diagrams
"""
into isospin 0, charge conjugation eigenstate 
1++, 2 particle in 1++ but this mixes with all other states with same quantum numbers 

correlator will be lightest state with those quant numbers 

for pipi, has same qauantum numbers as 1 particle scalar 

to resolve  a lattice scattering state, take account everything below. if you have pions or 1po below it, you need to add to the basis 

dd_star has two open charms 

when couple together in color space 3bar x 3 is 9 dim , 
color octet or color singlet 
adjoint meson is color octet 
2 meson operator multiply color singlet x color singlet = singlet 
3 bar x 3 x 3bar x 3 = 

d, dstar are doublets 

if want 0 isosopn witn I_z = 0 , 1/sqrt(2) 

treat flavors separately 


"""

def D_Dstar_twopt(operators,
                  cfg_id, 
                  dirs,
                    h5_group, 
                    nvecs: int,
                    ntsrcs:int,
                    LT: int,
                    flavor_contents: List[str], 
                    tsrc_avg=True):
    """Process one configuration for given flavor systems, compute di-meson correlator with direct and crossing terms."""
    # file naming conventions for a single configuration
    file_specs = {
        'light': (dirs['light'], f"peram_{nvecs}_cfg{{cfg_id}}.h5"),  # Light perambulator
        'charm': (dirs['charm'], f"peram_charm_{nvecs}_cfg{{cfg_id}}.h5"),  # charm perambulator

        'meson': (dirs['meson'], f"meson-{nvecs}_cfg{{cfg_id}}.h5"),
    }

    paths = {key: get_file_path(dir, template, cfg_id) for key, (dir, template) in file_specs.items()}
    if not paths['light'] or not paths['meson']:
        return False

    print(f"Reading light perambulator file: {paths['light']}")
    print(f"Reading charm perambulator file: {paths['charm']}")
    print(f"Reading meson elementals file: {paths['meson']}")

    # load common data
    meson_elemental = load_elemental(paths['meson'], LT, nvecs, mom='mom_0_0_0', disp='disp')
    peram_light = load_peram(paths['light'], LT, nvecs, ntsrcs)  # Explicitly light
    peram_charm = load_peram(paths['charm'], LT, nvecs, ntsrcs)  # Explicitly light


    # flavor combinations for perambulators
    flavor_map = {
        'light_light': (peram_light, peram_light, paths['light'], paths['light']),
        'light_charm': (peram_light, peram_charm, paths['light'], paths['charm']),
        'charm_charm': (peram_charm, peram_charm, paths['charm'], paths['charm']),


    }

    # store the correlators and perambulator data for each flavor system
    # write out the single particle correlators and two-particle correlators 
    """
    Direct:
    D A (creation operator): <phi_t_A * tau_A * phi_0_A * tau_A_back>
    D B (creation operator): <phi_t_B * tau_B * phi_0_B * tau_B_back>

    pi-pi creation operator = pion A x pion B 
    Di-meson direct: C_A(t) * C_B(t), where C_A and C_B are the individual meson correlators.
    """
    correlators = {}
    peram_data = {}
    for idx, flavor_content in enumerate(flavor_contents, 1):  # Start index at 1 for meson1, meson2
        if flavor_content not in flavor_map:
            print(f"Flavor '{flavor_content}' not recognized, defaulting to light_light.")
            flavor_content = 'light_light'

        peram, peram_back_data, peram_file, peram_back_file = flavor_map[flavor_content]
        if peram_back_data is None:
            print(f"Required {flavor_content} back perambulator file missing: {peram_back_file}. Skipping.")
            return False
        peram_back = reverse_perambulator_time(peram_back_data)
        print(f"Flavor '{flavor_content}' (meson {idx}):")
        print(f"  Perambulator loaded: {peram_file}")
        print(f"  Reverse perambulator loaded: {peram_back_file}")

        # store perambulator data for computation of crossing terms
        peram_data[flavor_content] = (peram, peram_back)

        # direct diagram
        meson_data = np.zeros((ntsrcs, LT), dtype=np.cdouble)  # Store tsrc x Lt data
        phi_A_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])
        phi_B_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])

        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                for tsrc in range(ntsrcs):
                    for t in range(LT):
                        phi_A_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                        phi_B_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                        tau_A = peram[tsrc, t, :, :, :, :]
                        tau_B = peram[tsrc, t, :, :, :, :]
                        tau_A_back = peram_back[tsrc, t, :, :, :, :]
                        tau_B_back = peram_back[tsrc, t, :, :, :, :]
                        dimeson_A = np.einsum("ijab,jkbc,klcd,lida", phi_A_t, tau_A, phi_A_0, tau_A_back, optimize='optimal')
                        dimeson_B = np.einsum("ijab,jkbc,klcd,lida", phi_B_t, tau_B, phi_B_0, tau_B_back, optimize='optimal')

                        meson_data[tsrc, t] = dimeson_A * dimeson_B 

        meson_data = meson_data.real

        # tsrc avg
        key_prefix = f'meson{idx}_{flavor_content}'  #  assign unique prefix for each meson
        if tsrc_avg:
            for tsrc in range(ntsrcs):
                meson_data[tsrc] = np.roll(meson_data[tsrc], -4 * tsrc)  # shift tsrc to origin to get symmetric corr 
            meson_avg = meson_data.mean(axis=0)  # do tsrc avg here 
            correlators[flavor_content] = meson_avg
            h5_group.create_dataset(f'{key_prefix}/cfg_{cfg_id}_tsrc_avg', data=meson_avg)
        else:
            correlators[flavor_content] = meson_data
            for tsrc in range(ntsrcs):
                h5_group.create_dataset(f'{key_prefix}/tsrc_{tsrc}/cfg_{cfg_id}', data=meson_data[tsrc])

        print(f"Correlator for {flavor_content} (meson {idx}) computed successfully{' with tsrc averaging' if tsrc_avg else ''}.")


    print(f"Cfg {cfg_id} processed successfully.")
    return True

def main(in_file,
         cfg_ids,
         task_id: int) -> None:
    """Process a single configuration for one or two flavor systems."""
    
    if os.path.exists(in_file):
        with open(in_file, 'r') as f:
            ini = yaml.safe_load(f)

    # Extract parameters
    slurm = ini['slurm']
    env = ini['environment']
    params = ini['parameters']
    operator_list = ini['operator_list']

    nvecs = params['nvecs']
    ntsrc = params['ntsrc']
    h5_path = os.path.abspath(env['h5_base_path'])
    nt = params['lt']

    # parse operator list strings 

    mom_list = ini.get('mom_list', ['1 0 0', '0 1 0', '0 0 1'])
    tsrc_avg = params['tsrc_avg']
    mom_avg = params['mom_avg']
    ens = params['ens']

    dirs = {
        'light': os.path.join(BASE_PATH, ens, 'perams_sdb', f'numvec{nvecs}'),
        'charm': os.path.join(BASE_PATH, ens, 'perams_charm_sdb', f'numvec{nvecs}'),
        'meson': os.path.join(BASE_PATH, ens, 'meson_sdb', f'numvec{nvecs}'),
    }




    flavor_contents = params['flavor']
    # output file name based on flavor combination
    group_name = '_'.join(flavor_contents) if len(flavor_contents) > 1 else flavor_contents[0]
    # FIX THIS 
    h5_output_file = f'{ens}{flavor_contents}_2pt_task{task_id}.h5'

    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f"{group_name}_000")
        for cfg_id in cfg_ids:
            try:
                if not D_Dstar_twopt(cfg_id, dirs, h5_group, flavor_contents):
                    print(f"Skipping configuration {cfg_id} due to missing files.")
            except FileNotFoundError as e:
                print(f"Error: {e}")

            print(f"Configuration {cfg_id} processed & saved to {h5_output_file} under group '{group_name}_000'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for one or two flavor systems.")
    parser.add_argument('--ini', type=str, required=True)
    parser.add_argument('--cfg_ids', type=str, required=True, help="Single configuration ID to process")
    parser.add_argument('--task', type=int, required=True, help="Task ID for this run")

    args = parser.parse_args()
    parser.add_argument('--task', type=int, required=True, help="SLURM array task ID")
    args = parser.parse_args()
    cfg_ids = [int(cfg) for cfg in args.cfg_ids.split(',')]
    main(args.ini, cfg_ids=cfg_ids, task_id=args.task)