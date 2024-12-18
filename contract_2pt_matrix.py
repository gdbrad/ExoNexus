import h5py
import numpy as np
from typing import List,Dict 
import os
import argparse
#import matplotlib.pyplot as plt
import pickle 
import pandas as pd
import time 

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
    with h5py.File(f"gevp_{channel}_{timestr}.h5", "w") as h5_group:
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
                            meson_matrix[src_idx, snk_idx, :, t] = np.einsum(
                                "ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize="optimal"
                            )
                        group_name = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        if group_name in h5_group:
                            del h5_group[group_name]  # Avoid overwriting existing datasets
                        h5_group.create_dataset(group_name, data=meson_matrix[src_idx, snk_idx, :, :])

                        # # Write out 2pt correlators for the current src-snk pair
                        # h5_group = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                        # h5_group.create_dataset(data=meson_matrix[src_idx, snk_idx,:,:])

    print("HDF5 file successfully written with GEVP data.")
#---------------------------------------------------------------------------------------------------------#

def correlator_matrix_debug(
    use_pickle: bool,
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

def main(cfg_ids, channel,h5_dir, nvec, ntsrc,task_id,show_plot=False):
    h5_path = os.path.abspath('/p/scratch/exotichadrons/exolaunch')
    nt = 96  
    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}', f'tsrc-{ntsrc}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{nvec}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')
    h5_output_file = f'{channel}_nvec_{nvec}_tsrc_{ntsrc}_task{task_id}.h5'
    # h5_output_path = os.path.join(h5_dir,h5_output_file)
    with h5py.File(h5_output_file, "w") as h5f:
        for op in op_map:
            h5_group = h5f.create_group(op)
            for cfg_id in cfg_ids:
                try:
                    two_pt_matrix = correlator_matrix(
                    use_pickle=False,
                    h5_group=h5_group,
                    # h5_dir=h5_dir,
                    cfg_id=cfg_id,
                    nvec=nvec,
                    ntsrc=ntsrc,
                    ncfg=200,
                    peram_dir=peram_dir,
                    # peram_strange_dir=peram_strange_dir,
                    meson_dir=meson_dir,
                    op_map=op_map,
                    op_name=list(op_map.keys()),
                    nt = nt,
                )
                    if not two_pt_matrix:
                        print(f"Skipping configuration {cfg_id} file is missing")
                except FileNotFoundError as e:
                    print(e)
        print(f"All cfgs processed & saved to {h5_output_file}.")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for multiple configurations.")
    parser.add_argument('--cfg_ids', type=str, required=True, help="List of configuration IDs to process")
    parser.add_argument('--channel', type=str, required=True, help="JPC and irrep")
    parser.add_argument('--h5_dir', type=str, required=True, help="h5 output path")
    parser.add_argument('--nvec', type=int, required=True, help="Number of eigenvectors")
    parser.add_argument('--ntsrc', type=int, required=True, help="Number of source time slices")
    # parser.add_argument('--plot', action='store_true', help="Show plot of pion distribution")
    parser.add_argument('--task', type=int, required=True, help="SLURM array task ID or unique identifier for this run")


    args = parser.parse_args()
    # if args.cfg_ids is None:
    #     cfg_ids = list(range(11, 1992, 10))  # Generate 11, 21, 31, ..., 1991
    # else:
    cfg_ids =  [int(cfg) for cfg in args.cfg_ids.split(',')]

    main(cfg_ids=cfg_ids, channel=args.channel,h5_dir=args.h5_dir,nvec=args.nvec, ntsrc=args.ntsrc, task_id=args.task)