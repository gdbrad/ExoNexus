import h5py
import numpy as np
from typing import List,Dict 
from dataclasses import dataclass
import os
import argparse
import datetime
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from gamma import gamma
import time
#import matplotlib.pyplot as plt
import pickle 
import pandas as pd
import operator_factory

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

def contract_ops_matrix(
        h5_dir: str,
        pickle: bool, # use pickled h5 for ease of testing 
        channel: str, # eg. isovector_pp or isovector_mm 
        cfg_id, # for testing single cfg
        num_vecs:int,
        num_tsrcs:int,
        peram_dir, 
        peram_strange_dir, 
        meson_dir,
        op_map:Dict, # operator dict from operator_factory 
        op_name: List[str], # list of ops in a particular channel 
        Lt:int, # temporal extent of lattice 
        show_plot=False):
    '''
    C(t, 0) = Tr[phi(t)tau(t,0)phi(0)tau(0,t)]
    Calculate the two-point correlation function for a given set of operators; If given a list of operators, a correlation matrix will be built to be fed into GEVP solver. gauge covariant spatial derivatives are combined with a gamma matrix within a fermion bilinear. 

    Parameters:
    - op_map/op_name: List of OperatorFactory objects defining the interpolating fields.
    - elemental: Meson elemental object .
    - perambulator: perambulator (quark propagator) data.
    - timeslices: Iterable of time slices at which the correlation function is to be evaluated.
    - Lt: Temporal extent of the lattice.
    - numvecs: Number of eigenvectors used in the calculation (will be <<< distillation basis)
    
    Returns:
    - A NumPy array of shape (Nop, Lt) containing the two-point correlation function
      values for each operator and timeslice.
    Once the τ(perambulators) have been computed and stored, the correlation of any source and sink operators can be computed a posteriori. this is determined by the indicated PC value -> dim of lattice irep 
    '''
    # load pickle files 
    pick_light = 'peram_light_1001.pkl'
    pick_strange = 'peram_strange_1001.pkl'

    timestr = time.strftime("%Y%m%d-%H")
    h5_output_file = f'{channel}_nvec_{num_vecs}_tsrc_{num_tsrcs}_{timestr}.h5'
    h5_output_path = os.path.join(h5_dir,h5_output_file)
    nop = len(op_name)

    # perams dont have momentum projection
    if pickle: 
        peram = pd.read_pickle(pick_light)
        peram_strange = pd.read_pickle(pick_strange)
        print(peram.shape,peram_strange.shape)
    else:
        peram = load_peram(peram_file, Lt, num_vecs, num_tsrcs)

    # Load perambulator and meson elemental
    meson_filename = f"meson-{num_vecs}_cfg{cfg_id}.h5"
    for file in os.listdir(meson_dir):
        if file == meson_filename:
            meson_file = os.path.join(meson_dir, file)
            break

    meson = np.zeros((nop, Lt), dtype=np.cdouble)  # Reset for each tsrc
    # different backward quark propagator allows for different quark flavors eg. strange,charm 
    for i, op in enumerate(op_name):
            operator = op_map.get(op)
            if operator.strange != 0: 
                peram_back = reverse_perambulator_time(peram_strange)
            else:
                peram_back = reverse_perambulator_time(peram)

    # zero disp. elemental 
    D0 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp')

    # load single disp. elementals
    D1 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_1')
    D2 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_2')
    D3 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_3')

    # load elementals displaced with two covariant derivatives
    D1D2 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_1_2')
    D2D1 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_2_1')

    D1D3 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_1_3')
    D3D1 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_3_1')

    D2D3 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_2_3')
    D3D2 = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp_3_2')

    # Process each tsrc, for each time slice, for each cfg 
    for tsrc in range(num_tsrcs):
        for t in range(Lt):
            tau = peram[tsrc, t, :, :, :, :]
            tau_ = peram_back[tsrc, t, :, :, :, :]

            for i, op in enumerate(op_name):
                operator = op_map.get(op)

                if operator.deriv is None:
                    phi_0 = np.einsum("ij,ab->ijab", operator.gamma, D0[0])
                    phi_t = np.einsum("ij,ab->ijab", operator.gamma, D0[t], optimize='optimal')
                    pion = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')
                    meson[i, t] = pion

                elif operator.deriv == 'nabla': 
                    D1_phi_0 = np.einsum("ij,ab->ijab", operator.gamma@gamma_i[0], D1[0])
                    D2_phi_0 = np.einsum("ij,ab->ijab", operator.gamma@gamma_i[1], D2[0])
                    D3_phi_0 = np.einsum("ij,ab->ijab", operator.gamma@gamma_i[2], D3[0])
                    nabla_0 =  D1_phi_0 + D2_phi_0 + D3_phi_0
                    D1_phi_t = np.einsum("ij,ab->ijab", operator.gamma@gamma_i[0], D1[t])
                    D2_phi_t = np.einsum("ij,ab->ijab", operator.gamma@gamma_i[1], D2[t])
                    D3_phi_t = np.einsum("ij,ab->ijab", operator.gamma@gamma_i[2], D3[t])
                    nabla_t =  D1_phi_t + D2_phi_t + D3_phi_t
                    nabla = np.einsum("ijab,jkbc,klcd,lida", nabla_t, tau, nabla_0, tau_, optimize='optimal')
                    meson[i,t] = nabla

                elif operator.deriv == 'B':
                    # 3_2, 1_3, 2_1 for the B operator carry a -1 coeff
                    # dydz + -dzdy x gamma_1
                    D2D3_phi_0_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[0]) 
                    D2D3_phi_0_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[0]) #subract this one

                    D2D3_phi_0_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[0]) 
                    D2D3_phi_0_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[0]) #subtract this one 

                    D2D3_phi_0_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[0]) 
                    D2D3_phi_0_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[0]) #subtract this one 
                    
                    gixBi =  D2D3_phi_0_1 - D2D3_phi_0_2
                    gixBi += D2D3_phi_0_3 - D2D3_phi_0_4
                    gixBi += D2D3_phi_0_5 - D2D3_phi_0_6 

                    D2D3_phi_t_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[t]) 
                    D2D3_phi_t_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[t]) #subract this one

                    D2D3_phi_t_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[t]) 
                    D2D3_phi_t_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[t]) #subtract this one 

                    D2D3_phi_t_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[t]) 
                    D2D3_phi_t_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[t]) #subtract this one 
                    gixBi_t = D2D3_phi_t_1 - D2D3_phi_t_2
                    gixBi_t += D2D3_phi_t_3 - D2D3_phi_t_4
                    gixBi_t += D2D3_phi_t_5 - D2D3_phi_t_6 
                    B_1 = np.einsum("ijab,jkbc,klcd,lida", gixBi_t, tau, gixBi, tau_, optimize='optimal')
                    meson[i,t] = B_1

                elif operator.deriv == 'D':
                    # 3_2, 1_3, 2_1 for the B operator carry a -1 coeff
                    # dydz + -dzdy x gamma_1
                    D2D3_phi_0_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[0]) 
                    D2D3_phi_0_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[0]) #subract this one

                    D2D3_phi_0_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[0]) 
                    D2D3_phi_0_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[0]) #subtract this one 

                    D2D3_phi_0_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[0]) 
                    D2D3_phi_0_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[0]) #subtract this one 
                    
                    gixBi =  D2D3_phi_0_1 + D2D3_phi_0_2
                    gixBi += D2D3_phi_0_3 + D2D3_phi_0_4
                    gixBi += D2D3_phi_0_5 + D2D3_phi_0_6 

                    D2D3_phi_t_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[t]) 
                    D2D3_phi_t_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[t]) #subract this one

                    D2D3_phi_t_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[t]) 
                    D2D3_phi_t_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[t]) #subtract this one 

                    D2D3_phi_t_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[t]) 
                    D2D3_phi_t_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[t]) #subtract this one 
                    gixBi_t = D2D3_phi_t_1  + D2D3_phi_t_2
                    gixBi_t += D2D3_phi_t_3 + D2D3_phi_t_4
                    gixBi_t += D2D3_phi_t_5 + D2D3_phi_t_6 
                    D_1 = np.einsum("ijab,jkbc,klcd,lida", gixBi_t, tau, gixBi, tau_, optimize='optimal')
                    meson[i,t] = D_1
        
            # if os.path.exists(h5_output_file):
            #     os.remove(h5_output_file)
            #     print('removed previous h5 file')
        with h5py.File(h5_output_path, "a") as h5f:
            tsrc_group_name = f'tsrc_{tsrc}/cfg_1001'
            tsrc_group = h5f.create_group(tsrc_group_name)
            # Loop over operators and save their respective datasets
            for i, op in enumerate(op_name):
                operator_dataset_name = f'{op}'
                if operator_dataset_name in tsrc_group:
                    del tsrc_group[operator_dataset_name]  # Delete existing dataset to avoid errors
                tsrc_group.create_dataset(operator_dataset_name, data=meson[i, :])
        # print(f'pion for tsrc {tsrc}:', meson)

def load_op_map(channel:str):
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")

def main(cfg_ids, channel, h5_dir, num_vecs, num_tsrcs,op_map,task_id,show_plot=False):
    h5_path = os.path.abspath('/p/scratch/exotichadrons/exolaunch')
    Lt = 96  
    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{num_vecs}', f'tsrc-{num_tsrcs}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{num_vecs}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')
    op_map = load_op_map(channel)
    for cfg_id in cfg_ids:
        try:
            two_pt_matrix = contract_ops_matrix(
            pickle=True,
            h5_dir=h5_dir,
            channel='a1_mp',
            cfg_id=cfg_id,
            num_vecs=num_vecs,
            num_tsrcs=num_tsrcs,
            peram_dir=peram_dir,
            peram_strange_dir=peram_strange_dir,
            meson_dir=meson_dir,
            op_map=op_map,
            op_name=list(op_map.keys()),
            Lt = Lt,
        )
            if not two_pt_matrix:
                print(f"Skipping configuration {cfg_id} file is missing")
        except FileNotFoundError as e:
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for multiple configurations.")
    parser.add_argument('--cfg_ids', type=str, required=True, help="List of configuration IDs to process")
    parser.add_argument('--channel', type=str, required=True, help="JPC and irrep")
    parser.add_argument('--h5_dir', type=str, required=True, help="h5 output path")
    parser.add_argument('--nvec', type=int, required=True, help="Number of eigenvectors")
    parser.add_argument('--ntsrc', type=int, required=True, help="Number of source time slices")
    parser.add_argument('--plot', action='store_true', help="Show plot of pion distribution")
    parser.add_argument('--task', type=int, required=True, help="SLURM array task ID or unique identifier for this run")


    args = parser.parse_args()
    # if args.cfg_ids is None:
    #     cfg_ids = list(range(11, 1992, 10))  # Generate 11, 21, 31, ..., 1991
    # else:
    cfg_ids =  [int(cfg) for cfg in args.cfg_ids.split(',')]

    main(cfg_ids=cfg_ids, channel=args.channel,num_vecs=args.nvec, num_tsrcs=args.ntsrc, show_plot=args.plot,task_id=args.task)