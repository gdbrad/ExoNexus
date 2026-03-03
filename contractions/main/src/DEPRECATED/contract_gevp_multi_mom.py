import h5py
import numpy as np
import os
import argparse
import time
import yaml
import sys
from typing import Iterable,List,Dict

import operator_factory
from operator_factory import QuantumNum
from insertion_factory import gamma
from contract_routines import *
from file_io import FileIO


timestr = time.strftime("%Y-%m-%d")
gamma_i = [gamma[1], gamma[2], gamma[3], gamma[4]]
# flavor ordering for fwd and bkwd propagation direction (lightest to heaviest)
# TODO: this might matter for flavor mixing 
FLAVOR_ORDER = {'light': 0, 'strange': 1, 'charm': 2} 

def load_op_map(channel:str):
    '''loads a ``QuantumNum`` object from ``operator_factory`` eg. the insertion between perambulator(light,strange,or charm) and elemental

    CG and subduction coeffs are applied when building the operator here, not in the correlator loop
    '''
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")


def two_pt_corr_matrix(
        operators: Dict[str,QuantumNum],
        nvecs:int,
        LT:int,
        cfg_id:int,
        file_io: FileIO,
        h5_group,
        flavor_contents: List[str],
        ntsrc: int,
        tsrc_step: int,
        mom_list: List[str],
        mom_avg: bool,
        tsrc_avg: bool = False,
        three_bar: bool = False
        ) -> bool:
    
    
    #passed by channel given in input file to ``load_op_map``
    

    '''handles list of momenta tuples, which are contained in the elemental files as strings
    Avoids redundant momenta averaging or post-projection if the operators from ``Operator_factory`` are already subduced 
    '''
    # ----------------- data loading --------------------------------------- #

    # Get required flavors and file specifications using FileIO
    required_flavors = file_io.get_required_flavors
    file_specs = file_io.file_specs()

    # Check paths for required flavors and meson elemental
    paths = {'meson': file_io.get_file_path('meson')}
    for flavor in required_flavors:
        if flavor in file_specs:
            paths[flavor] = file_io.get_file_path(flavor)

    if not paths['meson'] or not all(paths.get(flavor) for flavor in required_flavors):
        print(f"Missing required files for cfg {cfg_id}. Skipping.")
        return False

    print(f"Reading meson elementals file: {paths['meson']}")
    for flavor in required_flavors:
        print(f"Reading {flavor} perambulator file: {paths[flavor]}")

    # Use preloaded data from FileIO
    if file_io.meson_elemental is None:
        print(f"Meson elemental data not loaded for cfg {cfg_id}. Skipping.")
        return False
    meson_elemental = file_io.meson_elemental
    peram_light = file_io.peram_light
    peram_strange = file_io.peram_strange
    peram_charm = file_io.peram_charm

    # Store perambulator data for each flavor system
    peram_data = file_io.peram_data()
#--------------------------------------------------------------------------#


# initialize correlator matrix to fill with raw correlator data 
    nmom = len(mom_list)
    nop = len(operators)
    if mom_avg and tsrc_avg:
        meson_matrix = np.zeros((nop, nop, LT), dtype=np.cdouble)
        temp_matrix = np.zeros((nmom, nop, nop, ntsrc, LT), dtype=np.cdouble) #before averaging
        temp_matrix = np.zeros((nmom, nop, nop, ntsrc, LT), dtype=np.cdouble)
    elif mom_avg:
        meson_matrix = np.zeros((nop, nop, LT), dtype=np.cdouble)
    elif tsrc_avg:
        meson_matrix = np.zeros((nmom, nop, nop, LT), dtype=np.cdouble)
        # before averaging
        temp_matrix = np.zeros((nmom, nop, nop, ntsrc, LT), dtype=np.cdouble)
    else:
        meson_matrix = np.zeros((nmom, nop, nop, LT), dtype=np.cdouble)

    # Determine dominant flavor for file naming
    #dominant_flavor = next((f for f in ['charm', 'strange', 'light'] if f in unique_flavors), 'light')
    # h5_file = f"h5-{channel}/gevp_{channel}_{cfg_id}.h5"

    # with h5py.File(h5_file, "w") as h5_group:
        # momenta loop if momenta tuple is given is a list 
    irrep = next(iter(operators.values())).F
    for mom_idx,mom in enumerate(mom_list):
        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                # Update operator momentum
                # if src_op.mom != mom:
                #     src_op.mom = mom
                # if snk_op.mom != mom:
                #     snk_op.mom = mom
                
                # Determine forward and backward propagation based on flavor order
                src_flavor_weight = FLAVOR_ORDER[src_op.flavor]
                snk_flavor_weight = FLAVOR_ORDER[snk_op.flavor]
                forward_flavor = src_op.flavor if src_flavor_weight <= snk_flavor_weight else snk_op.flavor
                backward_flavor = snk_op.flavor if src_flavor_weight <= snk_flavor_weight else src_op.flavor
                
                if src_flavor_weight <= snk_flavor_weight:
                    # Source is lighter or equal, propagates forward; sink backward
                    fwd_flavor = src_op.flavor
                    bkwd_flavor = snk_op.flavor
                else:
                    # Sink is lighter, propagates forward; source backward
                    fwd_flavor = snk_op.flavor
                    bkwd_flavor = src_op.flavor
                for tsrc in range(ntsrc):
                    for t in range(LT):
                        tau = peram_flavors[fwd_flavor][tsrc, t, :, :, :, :]
                        tau_ = peram_back[bkwd_flavor][tsrc, t, :, :, :, :]

                        if src_op.deriv is None:
                            phi_0, _ = contract_local(meson_file, LT, nvec, src_op, t,mom)
                        elif src_op.deriv == "nabla":
                            phi_0, _ = contract_nabla(meson_file, LT, nvec, src_op, t,mom)
                        elif src_op.deriv in ["B", "D"]:
                            phi_0, _ = contract_B_D(meson_file,LT,nvec,src_op, t,mom, add=(src_op.deriv == "D"))
                        else:
                            continue

                        if snk_op.deriv is None:
                            _, phi_t = contract_local(meson_file, LT, nvec, snk_op, t ,mom)
                        elif snk_op.deriv == "nabla":
                            _, phi_t = contract_nabla(meson_file, LT, nvec, snk_op, t ,mom)
                        elif snk_op.deriv in ["B", "D"]:
                            _, phi_t = contract_B_D(meson_file,LT,nvec,snk_op, t, mom, add=(snk_op.deriv == "D"))
                        else:
                            continue

                        print(f'Contracting {src_name}-{snk_name}, mom {mom}, tsrc {tsrc}, t {t}, '
                                f'forward: {fwd_flavor}, backward: {bkwd_flavor}')

                        correlation = contract("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize="optimal")
                        
                        if mom_avg and tsrc_avg:
                            temp_matrix[mom_idx, src_idx, snk_idx, tsrc, t] = correlation
                        elif mom_avg:
                            meson_matrix[src_idx, snk_idx, t] += correlation.real / nmom
                        elif tsrc_avg:
                            temp_matrix[mom_idx, src_idx, snk_idx, tsrc, t] = correlation
                        else:
                            meson_matrix[mom_idx, src_idx, snk_idx, t] = correlation.real

                    if not (mom_avg or tsrc_avg):
                        group_name = f"/mom_{mom}/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        h5_group.create_dataset(group_name, data=meson_matrix[mom_idx, src_idx, snk_idx, :])
                    elif mom_avg and not tsrc_avg:
                        group_name = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        h5_group.create_dataset(group_name, data=meson_matrix[src_idx, snk_idx, :])
                # Perform tsrc averaging if requested
                if tsrc_avg:
                    # Apply time shifts and average over tsrc
                    for i in range(ntsrc):
                        temp_matrix[mom_idx, src_idx, snk_idx, i, :] = np.roll(
                            temp_matrix[mom_idx, src_idx, snk_idx, i, :], -tsrc_step * i
                        )
                    if mom_avg:
                        meson_matrix[src_idx, snk_idx, :] += temp_matrix[mom_idx, src_idx, snk_idx, :].mean(axis=0).real / nmom
                    else:
                        meson_matrix[mom_idx, src_idx, snk_idx, :] = temp_matrix[mom_idx, src_idx, snk_idx, :].mean(axis=0).real
                        group_name = f"/mom_{mom}/{src_op.name}_{snk_op.name}/cfg_{cfg_id}"
                        h5_group.create_dataset(group_name, data=meson_matrix[mom_idx, src_idx, snk_idx, :])
    if mom_avg and tsrc_avg:
        group_name = f"/{src_op.name}_{snk_op.name}/cfg_{cfg_id}"
        h5_group.create_dataset(group_name, data=meson_matrix[src_idx, snk_idx, :])

    print(f"Cfg {cfg_id} processed successfully{' (mom averaged)' if mom_avg else ''}{' (tsrc averaged)' if tsrc_avg else ''}, irrep {irrep}.")
    return True

def main(in_file,cfg_ids,task_id):
    with open(in_file, 'r') as f:
        ini = yaml.safe_load(f)

    channel = ini['channel']
    nvec = ini['nvec']
    ntsrc = ini['ntsrc']
    h5_path = os.path.abspath(ini['h5_base_path'])
    nt = ini['nt']
    mom_list = ini.get('mom_list')
    psq = ini['psq']
    tsrc_avg = ini.get('tsrc_avg',True)

    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}', f'tsrc-{ntsrc}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{nvec}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')
    peram_charm_dir = os.path.join(h5_path, 'perams_charm_sdb')

    operators = load_op_map(channel)
    # h5_output_file = f'h5-{channel}_nvec_{nvec}_tsrc_{ntsrc}_task{task_id}.h5'
    # with h5py.File(h5_output_file, "w") as h5f:
    #     h5_group = h5f.create_group(f'{channel}_psq{psq}')
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
                nt=nt
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