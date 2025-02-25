import h5py
import numpy as np
import os
import argparse
import time
import yaml
import sys

from gamma import gamma
import operator_factory
from operator_factory import QuantumNum
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
from mpi4py import MPI
# from opt_einsum import contract
timestr = time.strftime("%Y-%m-%d")
gamma_i = [gamma[1], gamma[2], gamma[3], gamma[4]]

def contract_local(meson_file, nt, nvec, operator, t):
    D0 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp')

    phi_0 = np.einsum("ij,ab->ijab", operator.gamma, D0[0], optimize="optimal")
    phi_t = np.einsum("ij,ab->ijab", operator.gamma, D0[t], optimize="optimal")
    return phi_0, phi_t

def contract_nabla(meson_file, nt, nvec, operator, t):

    D1 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_1')
    D2 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_2')
    D3 = load_elemental(meson_file, nt, nvec, mom='mom_0_0_0', disp='disp_3')

    nabla_0 = sum(
    np.einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[0] if i == 0 else D2[0] if i == 1 else D3[0], optimize="optimal") 
    for i in range(3)
        )
    
    nabla_t = sum(
        np.einsum("ij,ab->ijab", operator.gamma @ gamma_i[i], D1[t] if i == 0 else D2[t] if i == 1 else D3[t], optimize="optimal")
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
    D2D3_phi_0_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[0], optimize="optimal")
    D2D3_phi_0_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[0], optimize="optimal")  # Subtract this one

    D2D3_phi_0_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[0], optimize="optimal")
    D2D3_phi_0_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[0], optimize="optimal")  # Subtract this one

    D2D3_phi_0_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[0], optimize="optimal")
    D2D3_phi_0_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[0], optimize="optimal")  # Subtract this one

    gixBi = (D2D3_phi_0_1 - coeff * D2D3_phi_0_2 +
             D2D3_phi_0_3 - coeff * D2D3_phi_0_4 +
             D2D3_phi_0_5 - coeff * D2D3_phi_0_6)

    # snk terms (t)
    D2D3_phi_t_1 = np.einsum("ij,ab->ijab", gamma_i[0], D2D3[t], optimize="optimal")
    D2D3_phi_t_2 = np.einsum("ij,ab->ijab", gamma_i[0], D3D2[t], optimize="optimal")  # Subtract this one

    D2D3_phi_t_3 = np.einsum("ij,ab->ijab", gamma_i[1], D3D1[t], optimize="optimal")
    D2D3_phi_t_4 = np.einsum("ij,ab->ijab", gamma_i[1], D1D3[t], optimize="optimal")  # Subtract this one

    D2D3_phi_t_5 = np.einsum("ij,ab->ijab", gamma_i[2], D1D2[t], optimize="optimal")
    D2D3_phi_t_6 = np.einsum("ij,ab->ijab", gamma_i[2], D2D1[t], optimize="optimal")  # Subtract this one

    gixBi_t = (D2D3_phi_t_1 - coeff * D2D3_phi_t_2 +
               D2D3_phi_t_3 - coeff * D2D3_phi_t_4 +
               D2D3_phi_t_5 - coeff * D2D3_phi_t_6)

    return gixBi, gixBi_t


def correlator_matrix(operators, peram_dir, meson_dir, peram_strange_dir, nt, channel, cfg_id, ntsrc, nvec, strange):
     
    peram_filename = f"peram_{nvec}_cfg{cfg_id}.h5"
    peram_file = os.path.join(peram_dir, peram_filename)
    peram = load_peram(peram_file, nt, nvec, ntsrc)

    # if strange:
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
    

    meson_matrix = np.zeros((len(operators), len(operators), nt), dtype=np.cdouble)
    if strange:
        h5_file = f"h5-kaon/test_gevp_strange_{channel}_{timestr}_{cfg_id}.h5" 
    else:
        h5_file = f"h5-pion/test_gevp_light_{channel}_{timestr}_{cfg_id}.h5" 

    with h5py.File(h5_file, "w") as h5_group:
        for src_idx, (src_name, src_op) in enumerate(operators.items()):
            for snk_idx, (snk_name, snk_op) in enumerate(operators.items()):
                for tsrc in range(ntsrc):
                    for t in range(nt):
                        tau = peram[tsrc, t, :, :, :, :]
                        tau_ = peram_back[tsrc, t, :, :, :, :]

                        if src_op.deriv is None:
                            phi_0, _ = contract_local(meson_file, nt, nvec, src_op, t)
                        elif src_op.deriv == "nabla":
                            phi_0, _ = contract_nabla(meson_file, nt, nvec, src_op, t)
                        elif src_op.deriv in ["B", "D"]:
                            phi_0, _ = contract_B_D(meson_file,nt,nvec,src_op, t, add=(src_op.deriv == "D"))
                        else:
                            continue

                        if snk_op.deriv is None:
                            _, phi_t = contract_local(meson_file, nt, nvec, snk_op, t)
                        elif snk_op.deriv == "nabla":
                            _, phi_t = contract_nabla(meson_file, nt, nvec, snk_op, t)
                        elif snk_op.deriv in ["B", "D"]:
                            _, phi_t = contract_B_D(meson_file,nt,nvec,snk_op, t, add=(snk_op.deriv == "D"))
                        else:
                            continue

                        print(f'Performing contraction for {src_name}-{snk_name}, tsrc {tsrc}, timeslice {t}')

                        meson_matrix[src_idx, snk_idx, t] = np.einsum(
                            "ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize="optimal"
                        )

                    group_name = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                    h5_group.create_dataset(group_name, data=meson_matrix[src_idx, snk_idx, :])

                    # h5_group = f"/{src_op.name}_{snk_op.name}/tsrc_{tsrc}/cfg_{cfg_id}"
                    # h5_group.create_dataset(data=meson_matrix[src_idx, snk_idx,:,:])

    print("HDF5 file successfully written with GEVP data.")

def load_op_map(channel:str):
    try:
        op_map = getattr(operator_factory, channel)
        if not isinstance(op_map, dict):
            raise TypeError(f"'{channel}' is not a dict")
        return op_map
    except AttributeError:
        raise AttributeError(f"'{channel}' not found in operator_factory")
    
invalid_cfgs = {
    21, 171, 1001, 1061, 1271, 1371, 1451, 1531, 
    1591, 1611, 1641, 1711, 1781, 1851, 
    1871, 1901, 1941, 1991
}

def get_valid_cfgs(start_cfg, end_cfg):
    """
    Generate a list of valid configurations that are not in the exclusion list.
    Distribute configurations among processes.
    """
    return [cfg for cfg in range(start_cfg, end_cfg, 10) if cfg not in invalid_cfgs]

def main(in_file, strange):
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    # print(f"Rank {rank}/{size}: Entering main()")  # Debug print

    # valid_cfgs = get_valid_cfgs(11, 1991)
    # cfg_id = (rank + 1) * 10 + 1

    # print(f"Rank {rank}: Processing cfg_id {cfg_id}") 
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    print(f"Getting configuration for rank {rank}",flush=False)
    invalid_cfgs = {21, 171, 1001, 1061, 1271, 1371, 1451, 1531, 1591, 1611, 1641, 1711, 1781, 1851, 1871, 1901, 1941, 1991}
    valid_cfgs = get_valid_cfgs(11,1991)
    cfg_id = valid_cfgs[rank % len(valid_cfgs)]
    # cfg_id=(rank+1)*10 + 1
    if (cfg_id in invalid_cfgs) or (cfg_id > 1991):
        # exit()
        print(f"Rank {rank}: Skipping invalid cfg_id {cfg_id}",flush=False)
        MPI.COMM_WORLD.Barrier()
        MPI.Finalize()
        sys.exit(0)
     # Debug print
    # invalid_cfgs = {21, 171, 1001, 1061, 1271, 1371, 1451, 1531,
    # 1591, 1611, 1641, 1711, 1781, 1851,
    # 1871, 1901, 1941, 1991
    # }
    
    # if cfg_id==1001:
    #     print(f"Rank {rank} skipping cfg_id {cfg_id}.")
    #     comm.Barrier() 
    #     # exit()
    #     return
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
    try:
        print(f"Rank {rank} will now read/write file {cfg_id}...",flush=False)
        print(f"Rank {rank} has finished reading/writing file {cfg_id}.",flush=False)
        correlator_matrix(
            operators=operators,
            channel=channel,
            cfg_id=cfg_id,
            nvec=nvec,
            ntsrc=ntsrc,
            peram_dir=peram_dir,
            meson_dir=meson_dir,
            peram_strange_dir=peram_strange_dir,
            nt=nt,
            strange=strange
        )
    except FileNotFoundError as e:
        print(e)

    print("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a YAML input file.")
    parser.add_argument('--ini', type=str, required=True)
    parser.add_argument('--strange', action='store_true', help="strange operators")
    args = parser.parse_args()
    main(args.ini, args.strange)

