import h5py
import numpy as np
from typing import List,Dict 
import os
import argparse
#import matplotlib.pyplot as plt
import pickle 
import pandas as pd
import time 
import yaml

from gamma import gamma
import operator_factory
from ingest_data import load_elemental, load_peram, reverse_perambulator_time

timestr = time.strftime("%Y-%m-%d")
gamma_i = [gamma[1],gamma[2],gamma[3],gamma[4]]


def check_files(num_vecs,cfg_id,peram_dir,peram_strange_dir,meson_dir):
    
    peram_strange_filename = None 
    peram_strange_filename = f"peram_strange_nv{num_vecs}_cfg{cfg_id}.h5"
    for file in os.listdir(peram_strange_dir):
        if file == peram_strange_filename:
            peram_strange_file = os.path.join(peram_strange_dir, file)
            break

    peram_file = None
    peram_filename = f"peram_{num_vecs}_cfg{cfg_id}.h5"
    for file in os.listdir(peram_dir):
        if file == peram_filename:
            peram_file = os.path.join(peram_dir, file)
            break

    meson_file = None
    meson_filename = f"meson-{num_vecs}_cfg{cfg_id}.h5"
    for file in os.listdir(meson_dir):
        if file == meson_filename:
            meson_file = os.path.join(meson_dir, file)
            break

    if not peram_file:
        print(f"Peram file for configuration {cfg_id} not found in {peram_dir}. Skipping.")
        return False
    if not peram_strange_file:
        print(f"strange peram file for configuration {cfg_id} not found in {peram_strange_dir}. Skipping.")
        return False
    if not meson_file:
        print(f"Meson file for configuration {cfg_id} not found in {meson_dir}. Skipping.")
        return False

    print(f"Reading propagator file: {peram_file}")
    print(f"Reading strange perambulator file: {peram_strange_file}")

    print(f"Reading meson elementals file: {meson_file}")
    return peram_file,peram_strange_file,meson_file


def contract_local(meson_file,nt,nvec,operator, t):
    D0 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp')

    phi_0 = np.einsum("ij,ab->ijab", operator.gamma, D0[0])
    phi_t = np.einsum("ij,ab->ijab", operator.gamma, D0[t], optimize="optimal")
    return phi_0, phi_t

def contract_nabla(meson_file,nt,nvec,operator, t):
    '''compute single derivative interpolator at src and snk'''
    # load single disp. elementals
    D1 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_1')
    D2 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_2')
    D3 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_3')
    nabla_0 = sum(
        np.einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[0] if i == 0 else D2[0] if i == 1 else D3[0])
        for i in range(3)
    )
    nabla_t = sum(
        np.einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[t] if i == 0 else D2[t] if i == 1 else D3[t])
        for i in range(3)
    )
    return nabla_0, nabla_t

def contract_B_D(meson_file,nt,nvec,operator, t, add=True):
    """
    Compute the gixBi and gixBi_t terms for B or D operators.
    The 'add' parameter determines whether to sum or subtract terms (B: subtract, D: add).
    """
    coeff = 1 if add else -1
    # load elementals displaced with two covariant derivatives
    D1D2 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_1_2')
    D2D1 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_2_1')

    D1D3 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_1_3')
    D3D1 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_3_1')

    D2D3 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_2_3')
    D3D2 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_3_2')

    # src terms (t=0)
    D2D3_phi_0_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[0])
    D2D3_phi_0_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[0])  # Subtract this one

    D2D3_phi_0_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[0])
    D2D3_phi_0_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[0])  # Subtract this one

    D2D3_phi_0_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[0])
    D2D3_phi_0_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[0])  # Subtract this one

    gixBi = (D2D3_phi_0_1 - coeff * D2D3_phi_0_2 +
             D2D3_phi_0_3 - coeff * D2D3_phi_0_4 +
             D2D3_phi_0_5 - coeff * D2D3_phi_0_6)

    # snk terms (t)
    D2D3_phi_t_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[t])
    D2D3_phi_t_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[t])  # Subtract this one

    D2D3_phi_t_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[t])
    D2D3_phi_t_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[t])  # Subtract this one

    D2D3_phi_t_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[t])
    D2D3_phi_t_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[t])  # Subtract this one

    gixBi_t = (D2D3_phi_t_1 - coeff * D2D3_phi_t_2 +
               D2D3_phi_t_3 - coeff * D2D3_phi_t_4 +
               D2D3_phi_t_5 - coeff * D2D3_phi_t_6)

    return gixBi, gixBi_t

from operator_factory import QuantumNum
def correlator_matrix(
    task_id:str,
    use_pickle: bool,
    operators:List[QuantumNum],
    peram_dir,
    meson_dir,
    nt:int,
    channel: str,
    cfg_id, # each cfg is processed one by one maybe this can be parallelized
    ncfg:int,
    ntsrc:int,
    nvec:int,
    debug:bool):
    
    # load pickle files 
    if use_pickle:
        pick_light = 'peram_light_1001.pkl'
        pick_strange = 'peram_strange_1001.pkl'
        peram = pd.read_pickle(pick_light)
        peram_strange = pd.read_pickle(pick_strange)
        print(peram.shape,peram_strange.shape)
    else:
        peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
        for file in os.listdir(peram_dir):
            if file == peram_filename:
                peram_file = os.path.join(peram_dir, file)
                break
        peram = load_peram(peram_file, nt, nvec, ntsrc)

    # set meson elemental h5 path 
    meson_filename = f"meson-{nvec}_cfg{cfg_id}.h5"
    for file in os.listdir(meson_dir):
        if file == meson_filename:
            meson_file = os.path.join(meson_dir, file)
            break

    # different backward perambulator allows for different quark flavors eg. strange,charm 
    # operators = f'operator_factory.{channel}'
    for i, op in enumerate(operators):
            if operators[op].strange != 0: 
                peram_back = reverse_perambulator_time(peram_strange)
            else:
                peram_back = reverse_perambulator_time(peram)

    meson_matrix = np.zeros((len(operators),len(operators),nt),dtype=np.cdouble)
    with h5py.File(f"gevp_{channel}_{timestr}_{task_id}.h5", "w") as h5_group:
        for src_idx, (src_name, src_op) in enumerate(operators.items()):  # src_idx is an integer index
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):  # 
                for tsrc in range(ntsrc):
                        for t in range(nt):
                            tau = peram[tsrc, t, :, :, :, :]
                            tau_ = peram_back[tsrc, t, :, :, :, :]

                            if src_op.deriv is None:
                                phi_0, _ = contract_local(meson_file,nt,nvec,src_op, t)
                            elif src_op.deriv == "nabla":
                                phi_0, _ = contract_nabla(meson_file,nt,nvec,src_op, t)
                            elif src_op.deriv in ["B", "D"]:
                                phi_0, _ = contract_B_D(meson_file,nt,nvec,src_op, t, add=(src_op.deriv == "D"))
                            else:
                                continue
                            
                            if snk_op.deriv is None:
                                _, phi_t = contract_local(meson_file,nt,nvec,snk_op, t)
                            elif snk_op.deriv == "nabla":
                                _, phi_t = contract_nabla(meson_file,nt,nvec,snk_op, t)
                            elif snk_op.deriv in ["B", "D"]:
                                _, phi_t = contract_B_D(meson_file,nt,nvec,snk_op, t, add=(snk_op.deriv == "D"))
                            else:
                                continue
                            print('performing contraction for',(src_name, src_op),(snk_name, snk_op),tsrc,'timeslice',t)
                            
                            # Perform contraction
                            meson_matrix[src_idx, snk_idx, t] = np.einsum(
                                "ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize="optimal"
                            )
                        group_name = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        if group_name in h5_group:
                            del h5_group[group_name]  # Avoid overwriting existing datasets
                        h5_group.create_dataset(group_name, data=meson_matrix[src_idx, snk_idx,:])

                        # # Write out 2pt correlators for the current src-snk pair
                        # h5_group = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        # h5_group.create_dataset(data=meson_matrix[src_idx, snk_idx,:,:])

    print("HDF5 file successfully written with GEVP data.")
#---------------------------------------------------------------------------------------------------------#

def correlator_matrix_debug(
    use_pickle: bool,
    task_id:int,
    operators: List[QuantumNum],
    peram_dir,
    meson_dir,
    nt: int,
    channel: str,
    cfg_id,  # each cfg is processed one by one; maybe this can be parallelized
    ncfg: int,
    ntsrc: int,
    nvec: int,
    dry_run: bool = True,  # Enable dry-run mode
):
    # Load pickle files if specified
    if use_pickle:
        pick_light = 'peram_light_1001.pkl'
        pick_strange = 'peram_strange_1001.pkl'
        peram = pd.read_pickle(pick_light)
        peram_strange = pd.read_pickle(pick_strange)
        print(f"Loaded perambulators from pickle:")
        print(f"Light: {peram.shape}, Strange: {peram_strange.shape}")
    else:
        peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
        for file in os.listdir(peram_dir):
            if file == peram_filename:
                peram_file = os.path.join(peram_dir, file)
                break
        peram = load_peram(peram_file, nt, nvec, ntsrc)

    # Set meson HDF5 file path
    meson_filename = f"meson-{nvec}_cfg{cfg_id}.h5"
    for file in os.listdir(meson_dir):
        if file == meson_filename:
            meson_file = os.path.join(meson_dir, file)
            break

    # Handle backward perambulator based on strangeness
    for i, op in enumerate(operators):
        if operators[op].strange != 0:
            peram_back = reverse_perambulator_time(peram_strange)
        else:
            peram_back = reverse_perambulator_time(peram)

    if not dry_run:
        # Initialize meson matrix only in non-dry-run mode
        meson_matrix = np.zeros((len(operators), len(operators), ncfg, nt), dtype=np.cdouble)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(f"Dry-run mode: {dry_run}")
    print(f"Configuration ID: {cfg_id}, Channel: {channel}")
    print(f"Number of configurations: {ncfg}, Number of timeslices: {nt}, Source time slices: {ntsrc}\n")

    # Iterate through operators and print what would be done
    for src_idx, (src_name, src_op) in enumerate(operators.items()):
        for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
            for tsrc in range(ntsrc):
                for t in range(nt):
                    print(f"Would process:")
                    print(f"  Source operator: {src_name} ({src_op})")
                    print(f"  Sink operator: {snk_name} ({snk_op})")
                    print(f"  Time source slice: {tsrc}, Time slice: {t}")
                    print(f"  Expected dataset path: /{src_name}_{snk_name}/tsrc_{tsrc}/cfg_{cfg_id}")

                    # Debugging shapes of arrays (mocked here)
                    print("  Shapes:")
                    print(f"    tau: {peram[tsrc, t, :, :, :, :].shape}")
                    print(f"    tau_: {peram_back[tsrc, t, :, :, :, :].shape}")

                    if src_op.deriv is None:
                        print("    Source phi_0: Local contraction")
                    elif src_op.deriv == "nabla":
                        print("    Source phi_0: Nabla contraction")
                    elif src_op.deriv in ["B", "D"]:
                        print(f"    Source phi_0: {'B' if src_op.deriv == 'B' else 'D'} contraction")

                    if snk_op.deriv is None:
                        print("    Sink phi_t: Local contraction")
                    elif snk_op.deriv == "nabla":
                        print("    Sink phi_t: Nabla contraction")
                    elif snk_op.deriv in ["B", "D"]:
                        print(f"    Sink phi_t: {'B' if snk_op.deriv == 'B' else 'D'} contraction")

                    print("-" * 80)

    print("Dry-run completed. No computations or file writes performed.")

def load_op_map(channel:str):
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")

def main(in_file,cfg_ids,task_id):
    with open(in_file, 'r') as f:
        ini = yaml.safe_load(f)

    # cfg_ids = ini['cfg_ids']
    channel = ini['channel']
    nvec = ini['nvec']
    ntsrc = ini['ntsrc']
    # task_id = ini['task_id']
    h5_path = os.path.abspath(ini['h5_base_path'])
    nt = ini['nt']

    # Define derived paths
    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}', f'tsrc-{ntsrc}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{nvec}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')
    h5_output_file = f"gevp_{channel}_nvec_{nvec}_tsrc_{ntsrc}_task{task_id}.h5"

    # Use operators from operator factory (mocked here for illustration)
    operators = operator_factory.a1_mp

    # Process each configuration ID
    for cfg_id in cfg_ids:
        try:
            two_pt_matrix = correlator_matrix(
                use_pickle=False,
                task_id=task_id,
                operators=operators,
                channel=channel,
                cfg_id=cfg_id,
                nvec=nvec,
                ntsrc=ntsrc,
                ncfg=ini['ncfg'],
                peram_dir=peram_dir,
                meson_dir=meson_dir,
                nt=nt,
                debug=ini.get('debug', False)
            )
            if not two_pt_matrix:
                print(f"Skipping configuration {cfg_id}, file is missing.")
        except FileNotFoundError as e:
            print(e)

    print(f"All cfgs processed & saved to {h5_output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a YAML input file for configuration processing.")
    parser.add_argument('--ini', type=str, required=True, help="Path to the YAML input file.")
    parser.add_argument('--cfg_ids',type=str)
    parser.add_argument('--task_id',type=str)
    args = parser.parse_args()

    main(args.ini,args.cfg_ids,args.task_id)
