import h5py
import numpy as np
import os 
from typing import Iterable,List,Dict,Union
from pathlib import Path
from gamma import gamma

def contract_ops_matrix(
        h5_dir: str,
        h5_group:str,
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
    if pickle:
        pick_light = 'peram_light_1001.pkl'
        pick_strange = 'peram_strange_1001.pkl'
        peram = pd.read_pickle(pick_light)
        peram_strange = pd.read_pickle(pick_strange)
        print(peram.shape,peram_strange.shape)
    else:
        # perams dont have momentum projection
        # peram_file = None
        peram_filename = f"peram_{num_vecs}_cfg{cfg_id}.h5"
        for file in os.listdir(peram_dir):
            if file == peram_filename:
                peram_file = os.path.join(peram_dir, file)
                break
        peram = load_peram(peram_file, Lt, num_vecs, num_tsrcs)
    nop = len(op_name)

    # Load perambulator and meson elemental
    meson_filename = f"meson-{num_vecs}_cfg{cfg_id}.h5"
    for file in os.listdir(meson_dir):
        if file == meson_filename:
            meson_file = os.path.join(meson_dir, file)
            break

    # meson = np.zeros((nop, Lt), dtype=np.cdouble)
    # different backward perambulator allows for different quark flavors eg. strange,charm 
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
        meson = np.zeros((nop, Lt), dtype=np.cdouble)
        corr_matrix = np.zeros((nop,nop, Lt), dtype=np.cdouble) #off diag 

        for t in range(Lt):
            tau = peram[tsrc, t, :, :, :, :]
            tau_ = peram_back[tsrc, t, :, :, :, :]

            for i, op in enumerate(op_name):
                operator = op_map.get(op)

                # (0,0)
                #(0,1) = phi_0 )(src) nabla_t (snk)
                #(0,2) = phi_0 )(src) (snk) gixBi_t


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
        
        # write out 2pt corrs for current tsrc in loop
        # Write out only the required operator to HDF5
        group_name = f'tsrc_{tsrc}/cfg_{cfg_id}'
        if group_name in h5_group:
            del h5_group[group_name]  # Clear existing data to prevent overwrites
        h5_group.create_dataset(group_name, data=meson[0, :])  # Save only the first operator

    print(f"Cfg {cfg_id} processed successfully for all tsrc.")
    #     h5_group.create_dataset(f'tsrc_{tsrc}/cfg_{cfg_id}', data=meson)
    
    # print(f"Cfg {cfg_id} processed successfully for all tsrc.")
    # return True
        # with h5py.File(h5_output_path, "w") as h5f:
        #     tsrc_group_name = f'tsrc_{tsrc}/cfg_{cfg_id}'
        #     tsrc_group = h5f.create_group(tsrc_group_name)
        #     # Loop over operators and save their respective datasets
        #     for i, op in enumerate(op_name):
        #         operator_dataset_name = f'{op}'
        #         if operator_dataset_name in tsrc_group:
        #             del tsrc_group[operator_dataset_name]  # Delete existing dataset to avoid errors
        #         tsrc_group.create_dataset(operator_dataset_name, data=meson[i, :])
        # # print(f'pion for tsrc {tsrc}:', meson)


def contract_pion_t(t: int, meson_elemental: np.ndarray, prop: np.ndarray, prop_back: np.ndarray, gamma: dict[int, np.ndarray]) -> np.cdouble:
    phi_0 = np.einsum("ij,ab->ijab", gamma[5], meson_elemental[0])
    phi_t = np.einsum("ij,ab->ijab", gamma[5], meson_elemental[t])
    tau = prop[t, :, :, :, :]
    tau_ = prop_back[t, :, :, :, :]
    p = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_)
    print(t, p)
    return p

def contract_vector_t(t: int, meson_elemental: np.ndarray, prop: np.ndarray, prop_back: np.ndarray, g_idx: int, gamma: dict[int, np.ndarray]) -> np.cdouble:
    phi_0 = np.einsum("ij,ab->ijab", gamma[g_idx], meson_elemental[0])
    phi_t = np.einsum("ij,ab->ijab", np.condjugate(np.transpose(gamma[g_idx])), meson_elemental[t])
    tau = prop[t, :, :, :, :]
    tau_ = prop_back[t, :, :, :, :]
    p = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_)
    print(t, p)
    return p

def contract(
            base_path: Path,
            Lt:int,
            num_vecs:int,
            mom_keys:Dict,
            disp_keys:Dict,
            configs:Union[int, Iterable[int], None] = None)-> Dict[int, Dict[int, np.ndarray]]:
    data = {num_vecs: {}}
    perams_path = os.path.join(base_path,'perams_sdb',f'numvec{num_vecs}')
    elemental_path = os.path.join(base_path,'meson_sdb',f'numvec{num_vecs}')
    output_path = os.path.join(base_path, 'pion_h5', f'numvec{num_vecs}')
    os.makedirs(output_path, exist_ok=True)

    if configs is None:
        configs = range(11, 1992, 10)
    elif isinstance(configs, List):
        configs = configs
    for config in configs:
        peram_file = os.path.join(perams_path, f'peram_{num_vecs}_cfg{config}.h5')
        elemental_file = os.path.join(elemental_path, f'meson-{num_vecs}_cfg{config}.h5')
            
        if os.path.isfile(peram_file) & os.path.isfile(elemental_file):
            print(f"Starting contraction for cfg {config}")
            # continue
        
        with h5py.File(peram_file, 'r') as peram_data:
            peram = np.zeros((Lt, Lt, 4, 4, num_vecs, num_vecs), dtype=np.cdouble)
            for t_source_idx in range(Lt):
                t_source_data = peram_data[f't_source_{t_source_idx}']
                for t_slice_idx in range(Lt):
                    t_slice_data = t_source_data[f't_slice_{t_slice_idx}']
                    for spin_src_idx in range(4):
                        spin_src_data = t_slice_data[f'spin_src_{spin_src_idx}']
                        for spin_snk_idx in range(4):
                            spin_snk_data = spin_src_data[f'spin_sink_{spin_snk_idx}']
                            peram[t_source_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :] = \
                                spin_snk_data['real'][:] + spin_snk_data['imag'][:] * 1j

        with h5py.File(elemental_file, 'r') as elemental_data:
            elemental = np.zeros((Lt,  len(mom_keys), len(disp_keys),num_vecs, num_vecs), dtype=np.cdouble)
            for t_slice_idx in range(0, Lt):
                t_slice_data = elemental_data[f't_slice_{t_slice_idx}']
                for mom_idx in range(len(mom_keys.keys())):
                    mom_data = t_slice_data[mom_keys[mom_idx]]
                    for disp_idx in range(0, len(disp_keys)):
                        disp_data = mom_data[disp_keys[disp_idx]]
                        elemental[t_slice_idx, mom_idx, disp_idx, :, :] = \
                    disp_data['real'][:] + disp_data['imag'][:] * 1j
               
                        elemental[t_slice_idx, mom_idx, disp_idx, :, :] = \
                        disp_data['real'][:] + disp_data['imag'][:] * 1j
        mom_keys_inv = {v: k for k, v in mom_keys.items()}
        disp_keys_inv = {v: k for k, v in disp_keys.items()}
    
        meson_elemental = np.squeeze(elemental[:, mom_keys_inv['mom_0_0_0'], disp_keys_inv['disp'], :, :])
        pion = np.zeros((Lt,Lt), dtype=np.cdouble)
        for t_src in range(0,Lt):
            phi_0 = np.einsum("ij,ab->ijab", DP.g5, meson_elemental[t_src])

            for t in range(0,Lt):
                phi_t = np.einsum("ij,ab->ijab", DP.g5, meson_elemental[t])
                tau = np.squeeze(peram[t_src, t, :, :, :, :])
                tau_ = np.squeeze(peram[t, t_src, :, :, :, :])
            
                pion[t_src,t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_)
                print(pion[t_src,t].shape, pion[t_src,t])
        pion = pion.real
        output_file = os.path.join(output_path, f'pion_nvec{num_vecs}_cfg{config}.h5')
        with h5py.File(output_file, 'w') as f5:
            f5.create_dataset('pion', data=pion)
        
        print(f"Pion data written to {output_file}")

    return data