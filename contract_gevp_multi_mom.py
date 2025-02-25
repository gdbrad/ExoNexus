import h5py
import numpy as np
import os
import argparse
import time
import yaml
import sys
from typing import Iterable,List,Dict

from gamma import gamma
import operator_factory
from operator_factory import QuantumNum
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from contract_routines import *
from opt_einsum import contract

timestr = time.strftime("%Y-%m-%d")
gamma_i = [gamma[1], gamma[2], gamma[3], gamma[4]]
# flavor ordering for fwd and bkwd propagation direction (lightest to heaviest)
# TODO: this might matter for flavor mixing 
FLAVOR_ORDER = {'light': 0, 'strange': 1, 'charm': 2} 

def load_op_map(channel:str):
    '''loads a ``QuantumNum`` object from ``operator_factory`` eg. the insertion between perambulator(light,strange,or charm) and elemental
    '''
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")


def correlator_matrix(
    operators: Dict[str,QuantumNum], #passed by channel given in input file to ``load_op_map``
    mom_list: List[str],
    tsrc_avg: bool, #avg over all tsrc locations before writing out correlators
    meson_dir: str,
    peram_dir: str, 
    peram_strange_dir: str,
    peram_charm_dir: str,
    nt: int,
    channel: str,
    cfg_id: str,
    ntsrc: int,
    nvec: int,
    h5_group:h5py.Group ):
    '''handles list of momenta tuples, which are contained in the elemental files as strings'''

    nmom = len(mom_list)
    nop = len(operators)
    
    # load base (light) perambulator
    peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
    peram_file = os.path.join(peram_dir, peram_filename)
    peram = load_peram(peram_file, nt, nvec, ntsrc)

    peram_flavors = {'light': peram}
    unique_flavors = set(op.flavor for op in operators.values())

    if 'strange' in unique_flavors:
        peram_strange_filename = f"peram_strange_nv{nvec}_cfg{cfg_id}.h5"
        peram_strange_file = os.path.join(peram_strange_dir, peram_strange_filename)
        peram_flavors['strange'] = load_peram(peram_strange_file, nt, nvec, ntsrc)
    if 'charm' in unique_flavors:
        peram_charm_filename = f"peram_charm_nv{nvec}_cfg{cfg_id}.h5"
        peram_charm_file = os.path.join(peram_charm_dir, peram_charm_filename)
        peram_flavors['charm'] = load_peram(peram_charm_file, nt, nvec, ntsrc)


    meson_filename = f"meson-{nvec}_cfg{cfg_id}.h5"
    meson_file = os.path.join(meson_dir, meson_filename)

    # Precompute reversed perambulators
    peram_back = {flavor: reverse_perambulator_time(peram) for flavor, peram in peram_flavors.items()}

    # initialize correlator matrix to fill with raw correlator data 
    # this will be reshaped once all timeslices are accessed 
    # meson_matrix = np.zeros((nmom,nop,nop,nt), dtype=np.cdouble)
    if tsrc_avg:
        # Only store the averaged result: (nmom, nop, nop, nt)
        meson_matrix = np.zeros((nmom, nop, nop, nt), dtype=np.cdouble)
        temp_matrix = np.zeros((nmom, nop, nop, ntsrc, nt), dtype=np.cdouble)  # Temporary storage
    else:
        # Store all tsrc: (nmom, nop, nop, nt)
        meson_matrix = np.zeros((nmom, nop, nop, nt), dtype=np.cdouble)

    # Determine dominant flavor for file naming
    #dominant_flavor = next((f for f in ['charm', 'strange', 'light'] if f in unique_flavors), 'light')
    h5_file = f"h5-{channel}/test_gevp_{channel}_{cfg_id}.h5"

    with h5py.File(h5_file, "w") as h5_group:
        # momenta loop if momenta tuple is given is a list 
        for mom_idx,mom in enumerate(mom_list):
            for src_idx, (src_name, src_op) in enumerate(operators.items()):
                for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                    # Update operator momentum
                    if src_op.mom != mom:
                        src_op.mom = mom
                    if snk_op.mom != mom:
                        snk_op.mom = mom
                    
                    # Determine forward and backward propagation based on flavor order
                    src_flavor_weight = FLAVOR_ORDER[src_op.flavor]
                    snk_flavor_weight = FLAVOR_ORDER[snk_op.flavor]
                    
                    if src_flavor_weight <= snk_flavor_weight:
                        # Source is lighter or equal, propagates forward; sink backward
                        fwd_flavor = src_op.flavor
                        bkwd_flavor = snk_op.flavor
                    else:
                        # Sink is lighter, propagates forward; source backward
                        fwd_flavor = snk_op.flavor
                        bkwd_flavor = src_op.flavor
                    for tsrc in range(ntsrc):
                        for t in range(nt):
                            tau = peram_flavors[fwd_flavor][tsrc, t, :, :, :, :]
                            tau_ = peram_back[bkwd_flavor][tsrc, t, :, :, :, :]

                            if src_op.deriv is None:
                                phi_0, _ = contract_local(meson_file, nt, nvec, src_op, t,mom)
                            elif src_op.deriv == "nabla":
                                phi_0, _ = contract_nabla(meson_file, nt, nvec, src_op, t,mom)
                            elif src_op.deriv in ["B", "D"]:
                                phi_0, _ = contract_B_D(meson_file,nt,nvec,src_op, t,mom, add=(src_op.deriv == "D"))
                            else:
                                continue

                            if snk_op.deriv is None:
                                _, phi_t = contract_local(meson_file, nt, nvec, snk_op, t ,mom)
                            elif snk_op.deriv == "nabla":
                                _, phi_t = contract_nabla(meson_file, nt, nvec, snk_op, t ,mom)
                            elif snk_op.deriv in ["B", "D"]:
                                _, phi_t = contract_B_D(meson_file,nt,nvec,snk_op, t, mom, add=(snk_op.deriv == "D"))
                            else:
                                continue

                            print(f'Contracting {src_name}-{snk_name}, mom {mom}, tsrc {tsrc}, t {t}, '
                                  f'forward: {fwd_flavor}, backward: {bkwd_flavor}')

                            correlation = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize="optimal")
                            
                            if tsrc_avg:
                                temp_matrix[mom_idx, src_idx, snk_idx, tsrc, t] = correlation
                            else:
                                meson_matrix[mom_idx, src_idx, snk_idx, t] = correlation

                        if not tsrc_avg:
                            group_name = f"/mom_{mom}/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                            h5_group.create_dataset(group_name, data=meson_matrix[mom_idx, src_idx, snk_idx, :])

                    # Perform tsrc averaging if requested
                    if tsrc_avg:
                        # Apply time shifts and average over tsrc
                        for i in range(ntsrc):
                            temp_matrix[mom_idx, src_idx, snk_idx, i, :] = np.roll(
                                temp_matrix[mom_idx, src_idx, snk_idx, i, :], -4 * i
                            )
                        meson_matrix[mom_idx, src_idx, snk_idx, :] = temp_matrix[mom_idx, src_idx, snk_idx, :].mean(axis=0)
                        
                        # Write averaged data
                        group_name = f"/mom_{mom}/{src_op.name}_{snk_op.name}/cfg_{cfg_id}"
                        h5_group.create_dataset(group_name, data=meson_matrix[mom_idx, src_idx, snk_idx, :])

    print("HDF5 file successfully written with GEVP data.")

def main(in_file,cfg_ids,task_id):
    with open(in_file, 'r') as f:
        ini = yaml.safe_load(f)

    channel = ini['channel']
    nvec = ini['nvec']
    ntsrc = ini['ntsrc']
    h5_path = os.path.abspath(ini['h5_base_path'])
    nt = ini['nt']
    mom_list = ini.get('mom_list')
    tsrc_avg = ini.get('tsrc_avg',True)

    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}', f'tsrc-{ntsrc}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{nvec}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')
    peram_charm_dir = os.path.join(h5_path, 'perams_charm_sdb')

    operators = load_op_map(channel)
    h5_output_file = f'h5-{channel}_nvec_{nvec}_tsrc_{ntsrc}_task{task_id}.h5'
    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f'{channel}_000')
        for cfg_id in cfg_ids:
            try:
                print(f"Processing configuration {cfg_id}...")
                correlator_matrix(
                    operators=operators,
                    mom_list=mom_list,
                    tsrc_avg=tsrc_avg,
                    channel=channel,
                    cfg_id=cfg_id,
                    nvec=nvec,
                    ntsrc=ntsrc,
                    peram_dir=peram_dir,
                    meson_dir=meson_dir,
                    peram_strange_dir=peram_strange_dir,
                    peram_charm_dir=peram_charm_dir,
                    nt=nt,
                    h5_group=h5_group
                )
            except FileNotFoundError as e:
                print(e)

    print("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a YAML input file.")
    parser.add_argument('--ini', type=str, required=True)
    parser.add_argument('--cfg_ids', type=str, required=True, help="Comma-separated list of config IDs")
    parser.add_argument('--task', type=int, required=True, help="SLURM array task ID or unique identifier for this run")

    args = parser.parse_args()
    cfg_ids =  [int(cfg) for cfg in args.cfg_ids.split(',')]

    main(args.ini,cfg_ids=cfg_ids,task_id=args.task)