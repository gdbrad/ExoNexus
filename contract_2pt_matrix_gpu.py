import h5py
import numpy as np
import os
import argparse
import pickle
import pandas as pd
import time
import yaml

from gamma import gamma
import operator_factory
from operator_factory import QuantumNum
from ingest_data import load_elemental, load_peram, reverse_perambulator_time

timestr = time.strftime("%Y-%m-%d")
gamma_i = [gamma[1], gamma[2], gamma[3], gamma[4]]


def use_gpu_einsum(gpu):
    """Returns the appropriate einsum function based on GPU flag."""
    if gpu:
        import cupy as cp
        return cp, cp.einsum
    else:
        return np, np.einsum


def contract_local(meson_file, nt, nvec, operator, t, gpu=False):
    xp, einsum = use_gpu_einsum(gpu)
    D0 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp')

    phi_0 = einsum("ij,ab->ijab", operator.gamma, D0[0])
    phi_t = einsum("ij,ab->ijab", operator.gamma, D0[t], optimize="optimal")
    return phi_0, phi_t


def contract_nabla(meson_file, nt, nvec, operator, t, gpu=False):
    xp, einsum = use_gpu_einsum(gpu)

    D1 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_1')
    D2 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_2')
    D3 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_3')

    nabla_0 = sum(
        einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[0] if i == 0 else D2[0] if i == 1 else D3[0])
        for i in range(3)
    )
    nabla_t = sum(
        einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[t] if i == 0 else D2[t] if i == 1 else D3[t])
        for i in range(3)
    )
    return nabla_0, nabla_t

def contract_B_D(meson_file,nt,nvec,operator, t, add=True,gpu=False):
    xp, einsum = use_gpu_einsum(gpu)

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
    D2D3_phi_0_1 = einsum("ij,ab->ijab", gamma_i[0], D2D3[0])
    D2D3_phi_0_2 = einsum("ij,ab->ijab", gamma_i[0], D3D2[0])  # Subtract this one

    D2D3_phi_0_3 = einsum("ij,ab->ijab", gamma_i[1], D3D1[0])
    D2D3_phi_0_4 = einsum("ij,ab->ijab", gamma_i[1], D1D3[0])  # Subtract this one

    D2D3_phi_0_5 = einsum("ij,ab->ijab", gamma_i[2], D1D2[0])
    D2D3_phi_0_6 = einsum("ij,ab->ijab", gamma_i[2], D2D1[0])  # Subtract this one

    gixBi = (D2D3_phi_0_1 - coeff * D2D3_phi_0_2 +
             D2D3_phi_0_3 - coeff * D2D3_phi_0_4 +
             D2D3_phi_0_5 - coeff * D2D3_phi_0_6)

    # snk terms (t)
    D2D3_phi_t_1 = einsum("ij,ab->ijab", gamma_i[0], D2D3[t])
    D2D3_phi_t_2 = einsum("ij,ab->ijab", gamma_i[0], D3D2[t])  # Subtract this one

    D2D3_phi_t_3 = einsum("ij,ab->ijab", gamma_i[1], D3D1[t])
    D2D3_phi_t_4 = einsum("ij,ab->ijab", gamma_i[1], D1D3[t])  # Subtract this one

    D2D3_phi_t_5 = einsum("ij,ab->ijab", gamma_i[2], D1D2[t])
    D2D3_phi_t_6 = einsum("ij,ab->ijab", gamma_i[2], D2D1[t])  # Subtract this one

    gixBi_t = (D2D3_phi_t_1 - coeff * D2D3_phi_t_2 +
               D2D3_phi_t_3 - coeff * D2D3_phi_t_4 +
               D2D3_phi_t_5 - coeff * D2D3_phi_t_6)

    return gixBi, gixBi_t


def correlator_matrix(task_id, use_pickle, operators, peram_dir, meson_dir, peram_strange_dir, nt, channel, cfg_id, ncfg, ntsrc, nvec, debug, strange,gpu=False):
    xp, einsum = use_gpu_einsum(gpu)

    if use_pickle:
        pick_light = 'peram_light_1001.pkl'
        pick_strange = 'peram_strange_1001.pkl'
        peram = pd.read_pickle(pick_light)
        peram_strange = pd.read_pickle(pick_strange)
    else:
        peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
        peram_file = os.path.join(peram_dir, peram_filename)
        peram = load_peram(peram_file, nt, nvec, ntsrc)

    peram_strange_filename = f"peram_strange_nv{nvec}_cfg{cfg_id}.h5"
    peram_strange_file = os.path.join(peram_strange_dir, peram_strange_filename)
    peram_strange = load_peram(peram_strange_file, nt, nvec, ntsrc)

    meson_filename = f"meson-{nvec}_cfg{cfg_id}.h5"
    meson_file = os.path.join(meson_dir, meson_filename)

    for i, op in enumerate(operators):
        if operators[op].strange != 0:
            peram_back = reverse_perambulator_time(peram_strange)
        else:
            peram_back = reverse_perambulator_time(peram)

    meson_matrix = xp.zeros((len(operators), len(operators), nt), dtype=xp.cdouble)
    if strange:
        h5_file = f"h5-out/gevp_strange_{channel}_{timestr}_{task_id}.h5" 
    else:
        h5_file = f"h5-out/gevp_light_{channel}_{timestr}_{task_id}.h5" 

    with h5py.File(h5_file, "w") as h5_group:
        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                for tsrc in range(ntsrc):
                    for t in range(nt):
                        tau = peram[tsrc, t, :, :, :, :]
                        tau_ = peram_back[tsrc, t, :, :, :, :]

                        if src_op.deriv is None:
                            phi_0, _ = contract_local(meson_file, nt, nvec, src_op, t, gpu=gpu)
                        elif src_op.deriv == "nabla":
                            phi_0, _ = contract_nabla(meson_file, nt, nvec, src_op, t, gpu=gpu)
                        elif src_op.deriv in ["B", "D"]:
                            phi_0, _ = contract_B_D(meson_file,nt,nvec,src_op, t, add=(src_op.deriv == "D"),gpu=gpu)
                        else:
                            continue

                        if snk_op.deriv is None:
                            _, phi_t = contract_local(meson_file, nt, nvec, snk_op, t, gpu=gpu)
                        elif snk_op.deriv == "nabla":
                            _, phi_t = contract_nabla(meson_file, nt, nvec, snk_op, t, gpu=gpu)
                        elif snk_op.deriv in ["B", "D"]:
                            _, phi_t = contract_B_D(meson_file,nt,nvec,snk_op, t, add=(snk_op.deriv == "D"),gpu=gpu)
                        else:
                            continue

                        print(f'Performing contraction for {src_name}-{snk_name}, timeslice {t}')

                        meson_matrix[src_idx, snk_idx, t] = einsum(
                            "ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize="optimal"
                        )

                    group_name = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                    if group_name in h5_group:
                        del h5_group[group_name]
                    h5_group.create_dataset(group_name, data=xp.asnumpy(meson_matrix[src_idx, snk_idx, :]))

    print("HDF5 file successfully written with GEVP data.")

def load_op_map(channel:str):
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")


def main(in_file, cfg_ids, task_id, gpu,strange):
    with open(in_file, 'r') as f:
        ini = yaml.safe_load(f)

    channel = ini['channel']
    nvec = ini['nvec']
    ntsrc = ini['ntsrc']
    h5_path = os.path.abspath(ini['h5_base_path'])
    nt = ini['nt']

    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{nvec}', f'tsrc-{ntsrc}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{nvec}')
    peram_strange_dir = os.path.join(h5_path, 'perams_strange_sdb')

    operators = load_op_map(channel)

    cfg_ids = list(map(int, cfg_ids.split(',')))

    for cfg_id in cfg_ids:
        try:
            correlator_matrix(
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
                peram_strange_dir=peram_strange_dir,
                nt=nt,
                debug=ini.get('debug', False),
                gpu=gpu,
                strange=strange
            )
        except FileNotFoundError as e:
            print(e)

    print("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a YAML input file.")
    parser.add_argument('--ini', type=str, required=True)
    parser.add_argument('--cfg_ids', type=str)
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--gpu', action='store_true', help="Enable GPU acceleration")
    parser.add_argument('--strange', action='store_true', help="strange operators")

    args = parser.parse_args()

    main(args.ini, args.cfg_ids, args.task_id, args.gpu,args.strange)

