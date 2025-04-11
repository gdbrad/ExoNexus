import numpy as np
import os
from opt_einsum import contract as oe_contract
import h5py
import argparse
import time
import yaml
import sys
from typing import Iterable, List, Dict
from gamma import gamma
import operator_factory as operator_factory
from operator_factory import QuantumNum
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from contract_routines import *

"""
The fundamental objects that will be coagulated into a two-point correlator are: 
- gamma matrix structure 
- covariant derivative operator eg. displacement operator. This is identity for local operators 
- projection operator for non-zero momentum, for zero total momentum this is the identity 

So for a specified operator at the sink we must set the 1. gamma index and 2. derivative index 
To spell out the logic, we 
1. start with a "bare" operator using the ``QuantumNum`` dataclass
2. apply some functor to the OperatorBare -> OperatorDisplacement, where the arrow is some displacement operator 

3. 


"""

def load_op_map(channel: str):
    """Loads a QuantumNum object from operator_factory."""
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")
    

def correlator_matrix(
    operators: Dict[str, 'QuantumNum'],
    mom_list: List[str],
    tsrc_avg: bool,
    mom_avg: bool,
    meson_dir: str,
    peram_dir: str,
    peram_strange_dir: str,
    peram_charm_dir: str,
    nt: int,
    channel: str,
    cfg_id: str,
    ntsrc: int,
    nvec: int,
    h5_group: 'h5py.Group',
    irrep: str = 'A1'  # Default to A1, change to A2 for pions at p=1
) -> bool:

    FLAVOR_ORDER = {'light': 0, 'strange': 1, 'charm': 2}
    nmom = len(mom_list)
    nop = len(operators)
    
    # A2 projection coefficients for C4v (simplified for mom_list averaging)
    A2_COEFFS = {
        '1 0 0': 1.0,  
        '0 1 0': -1.0, 
        '0 0 1': -1.0 
    } if irrep == 'A2' else {mom: 1.0 for mom in mom_list}  # A1 is symmetric
    
    # Check and load files
    peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
    peram_file = os.path.join(peram_dir, peram_filename)
    if not os.path.isfile(peram_file):
        print(f"Peram file {peram_file} not found. Skipping cfg {cfg_id}.")
        return False
    
    meson_filename = f"meson-{nvec}_cfg{cfg_id}.h5"
    meson_file = os.path.join(meson_dir, meson_filename)
    if not os.path.isfile(meson_file):
        print(f"Meson file {meson_file} not found. Skipping cfg {cfg_id}.")
        return False
    
    peram = load_peram(peram_file, nt, nvec, ntsrc)
    peram_flavors = {'light': peram}
    unique_flavors = set(op.flavor for op in operators.values())
    
    if 'strange' in unique_flavors:
        peram_strange_filename = f"peram_strange_nv{nvec}_cfg{cfg_id}.h5"
        peram_strange_file = os.path.join(peram_strange_dir, peram_strange_filename)
        if os.path.isfile(peram_strange_file):
            peram_flavors['strange'] = load_peram(peram_strange_file, nt, nvec, ntsrc)
    
    if 'charm' in unique_flavors:
        peram_charm_filename = f"peram_charm_{nvec}_cfg{cfg_id}.h5"
        peram_charm_file = os.path.join(peram_charm_dir, peram_charm_filename)
        if os.path.isfile(peram_charm_file):
            peram_flavors['charm'] = load_peram(peram_charm_file, nt, nvec, ntsrc)

    peram_back = {flavor: reverse_perambulator_time(peram) for flavor, peram in peram_flavors.items()}

    # Initialize output array based on averaging options
    if mom_avg and tsrc_avg:
        meson_matrix = np.zeros((nop, nop, nt), dtype=np.cdouble)
    elif mom_avg:
        meson_matrix = np.zeros((nop, nop, ntsrc, nt), dtype=np.cdouble)
    elif tsrc_avg:
        meson_matrix = np.zeros((nmom, nop, nop, nt), dtype=np.cdouble)
    else:
        meson_matrix = np.zeros((nmom, nop, nop, ntsrc, nt), dtype=np.cdouble)

    print(f"Processing cfg {cfg_id}: {peram_file}, {meson_file}, irrep {irrep}")
    
    # Temporary storage for contractions
    temp_matrix = np.zeros((ntsrc, nt), dtype=np.cdouble)

    for mom_idx, mom in enumerate(mom_list):
        coeff = A2_COEFFS.get(mom, 1.0)  # Default to 1.0 if not specified
        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                src_flavor_weight = FLAVOR_ORDER.get(src_op.flavor, 0)
                snk_flavor_weight = FLAVOR_ORDER.get(snk_op.flavor, 0)
                forward_flavor = src_op.flavor if src_flavor_weight <= snk_flavor_weight else snk_op.flavor
                backward_flavor = snk_op.flavor if src_flavor_weight <= snk_flavor_weight else src_op.flavor
                
                if forward_flavor not in peram_flavors or backward_flavor not in peram_flavors:
                    print(f"Skipping {src_name}-{snk_name}: Missing peram for {forward_flavor} or {backward_flavor}")
                    continue
                
                for tsrc in range(ntsrc):
                    for t in range(nt):
                        tau = peram_flavors[forward_flavor][tsrc, t, :, :, :, :]
                        tau_ = peram_back[backward_flavor][tsrc, t, :, :, :, :]

                        if src_op.deriv is None:
                            phi_0, _ = contract_local(meson_file, nt, nvec, src_op, t, mom)
                        elif src_op.deriv == "nabla":
                            phi_0, _ = contract_nabla(meson_file, nt, nvec, src_op, t, mom)
                        elif src_op.deriv in ["B", "D"]:
                            phi_0, _ = contract_B_D(meson_file, nt, nvec, src_op, t, mom, add=(src_op.deriv == "D"))
                        else:
                            continue

                        if snk_op.deriv is None:
                            _, phi_t = contract_local(meson_file, nt, nvec, snk_op, t, mom)
                        elif snk_op.deriv == "nabla":
                            _, phi_t = contract_nabla(meson_file, nt, nvec, snk_op, t, mom)
                        elif snk_op.deriv in ["B", "D"]:
                            _, phi_t = contract_B_D(meson_file, nt, nvec, snk_op, t, mom, add=(snk_op.deriv == "D"))
                        else:
                            continue

                        print(f'Contracting {src_name}-{snk_name}, mom {mom}, tsrc {tsrc}, t {t}, '
                              f'forward: {forward_flavor}, backward: {backward_flavor}')

                        correlation = oe_contract("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')
                        correlation *= coeff  # Apply A2 projection coefficient
                        temp_matrix[tsrc, t] = correlation.real

                    # Store results based on averaging options
                    if mom_avg and tsrc_avg:
                        meson_matrix[src_idx, snk_idx] += temp_matrix.mean(axis=0) / nmom
                    elif mom_avg:
                        meson_matrix[src_idx, snk_idx, tsrc] = temp_matrix.mean(axis=0)
                    elif tsrc_avg:
                        temp_matrix[tsrc] = np.roll(temp_matrix[tsrc], -4 * tsrc)
                        meson_matrix[mom_idx, src_idx, snk_idx] = temp_matrix.mean(axis=0)
                    else:
                        meson_matrix[mom_idx, src_idx, snk_idx, tsrc] = temp_matrix[tsrc]

    # Write data to HDF5
    for src_idx, (src_name, src_op) in enumerate(operators.items()):
        for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
            if mom_avg and tsrc_avg:
                dataset_name = f"{src_op.name}_{snk_op.name}/cfg_{cfg_id}"
                h5_group.create_dataset(dataset_name, data=meson_matrix[src_idx, snk_idx])
            elif mom_avg:
                for tsrc in range(ntsrc):
                    dataset_name = f"{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                    h5_group.create_dataset(dataset_name, data=meson_matrix[src_idx, snk_idx, tsrc])
            elif tsrc_avg:
                for mom_idx, mom in enumerate(mom_list):
                    dataset_name = f"mom_{mom}/{src_op.name}_{snk_op.name}/cfg_{cfg_id}"
                    h5_group.create_dataset(dataset_name, data=meson_matrix[mom_idx, src_idx, snk_idx])
            else:
                for mom_idx, mom in enumerate(mom_list):
                    for tsrc in range(ntsrc):
                        dataset_name = f"mom_{mom}/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        h5_group.create_dataset(dataset_name, data=meson_matrix[mom_idx, src_idx, snk_idx, tsrc])

    print(f"Cfg {cfg_id} processed successfully{' (mom averaged)' if mom_avg else ''}{' (tsrc averaged)' if tsrc_avg else ''}, irrep {irrep}.")
    return True

def main(in_file, cfg_ids, task_id: int) -> None:
    if os.path.exists(in_file):
        with open(in_file, 'r') as f:
            ini = yaml.safe_load(f)
    else:
        ini = {
            'channel': 'a1_mp_nomix',
            'nvec': 96,
            'ntsrc': 24,
            'h5_base_path': '/p/scratch/exotichadrons/exolaunch',
            'nt': 96,
            'mom_list': ['1 0 0', '0 1 0', '0 0 1'],
            'tsrc_avg': False,
            'mom_avg': False,
            'irrep': 'A2',
            'psq': '000',
            'ens': 'gio-L32T96'
        }

    channel = ini['channel']
    nvec = ini['nvec']
    ntsrc = ini['ntsrc']
    h5_path = os.path.abspath(ini['h5_base_path'])
    nt = ini['nt']
    mom_list = ini.get('mom_list', ['1 0 0', '0 1 0', '0 0 1'])
    tsrc_avg = ini.get('tsrc_avg', True)
    mom_avg = ini.get('mom_avg', True)
    irrep = ini.get('irrep', 'A2')
    psq = ini.get('psq', '000')
    ens = ini.get('ens', 'gio-L32T96')

    if ens == 'b3.6_ms0.25_mud-0.013_s32t64':
        peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}')
    else:
        peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}', f'tsrc-{ntsrc}')

    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{nvec}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')
    peram_charm_dir = os.path.join(h5_path, 'perams_charm_sdb', f'numvec{nvec}')

    operators = load_op_map(channel)

    h5_output_file = f'{ens}-{channel}_nvec_{nvec}_task{task_id}.h5'
    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f'{channel}_{psq}')
        for cfg_id in cfg_ids:
            try:
                processed = correlator_matrix(
                    operators=operators,
                    mom_list=mom_list,
                    tsrc_avg=tsrc_avg,
                    mom_avg=mom_avg,
                    channel=channel,
                    cfg_id=cfg_id,
                    nvec=nvec,
                    ntsrc=ntsrc,
                    peram_dir=peram_dir,
                    meson_dir=meson_dir,
                    peram_strange_dir=peram_strange_dir,
                    peram_charm_dir=peram_charm_dir,
                    nt=nt,
                    h5_group=h5_group,
                    irrep=irrep
                )
                if not processed:
                    print(f"Skipping cfg {cfg_id}: Missing files.")
            except Exception as e:
                print(f"Error processing cfg {cfg_id}: {e}")

    print(f"All cfgs processed & saved to {h5_output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for multiple configurations.")
    parser.add_argument('--ini', type=str, required=True)
    parser.add_argument('--cfg_ids', type=str, required=True, help="Comma-separated list of config IDs")
    parser.add_argument('--task', type=int, required=True, help="SLURM array task ID")
    args = parser.parse_args()
    cfg_ids = [int(cfg) for cfg in args.cfg_ids.split(',')]
    main(args.ini, cfg_ids=cfg_ids, task_id=args.task)