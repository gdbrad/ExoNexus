import h5py
import numpy as np
import os
import argparse
import datetime
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
import gamma as gamma

def process_configuration(cfg_id, num_vecs, num_tsrcs, peram_dir, meson_dir, Lt, h5_group, show_plot=False):
    '''contractions of pion zero displacement operators 
    TODO rest of meson spectrum 
    '''
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
    if not meson_file:
        print(f"Meson file for configuration {cfg_id} not found in {meson_dir}. Skipping.")
        return False

    print(f"Reading propagator file: {peram_file}")
    print(f"Reading meson elementals file: {meson_file}")
    
    # Load perambulator and meson elemental
    peram = load_peram(peram_file, Lt, num_vecs, num_tsrcs)
    meson_elemental = load_elemental(meson_file, Lt, num_vecs, mom='mom_0_0_0', disp='disp')
    peram_back = reverse_perambulator_time(peram)
    meson = np.zeros(Lt, dtype=np.cdouble)  # Shape (96, 200) for each tsrc
    phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])

    for tsrc in range(num_tsrcs):
        for t in range(Lt):
            phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
            tau = peram[tsrc, t, :, :, :, :]
            tau_ = peram_back[tsrc, t, :, :, :, :]
            # Contract pion, assuming the 200 dimension comes from an appropriate contraction of indices
            contracted_result = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')
            
            # Store the contracted result in the pion array (Lt, 200)
            meson[t] = contracted_result  # Ensure this matches the dimension of 200.

        meson = meson.real
        h5_group.create_dataset(f'tsrc_{tsrc}/cfg_{cfg_id}', data=meson)

        if show_plot:
            plt.plot(np.arange(Lt), pion[:, 0], '.', label=f'Pion Distribution (first column) - tsrc {tsrc}, cfg {cfg_id}')
            plt.yscale('log')
            plt.legend()
            plt.savefig(f'pion-{cfg_id}-tsrc-{tsrc}-{num_vecs}-{datetime.datetime.today()}.pdf')

    print(f"Cfg {cfg_id} processed successfully.")
    return True

def main(cfg_ids, num_vecs, num_tsrcs, task_id,show_plot=False):
    h5_path = os.path.abspath('/p/scratch/exotichadrons/exolaunch')
    Lt = 96  
    peram_dir = os.path.join(h5_path, 'perams_sdb', f'numvec{num_vecs}', f'tsrc-{num_tsrcs}')
    meson_dir = os.path.join(h5_path, 'meson_sdb', f'numvec{num_vecs}')

    h5_output_file = f'pion_2pt_nvec_{num_vecs}_tsrc_{num_tsrcs}_task{task_id}.h5'
    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group('pion_000')
        for cfg_id in cfg_ids:
            try:
                processed = process_configuration(cfg_id, num_vecs, num_tsrcs, peram_dir, meson_dir, Lt, h5_group, show_plot)
                if not processed:
                    print(f"Skipping configuration {cfg_id} file is missing")
            except FileNotFoundError as e:
                print(e)

        print(f"All cfgs processed & saved to {h5_output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for multiple configurations.")
    parser.add_argument('--cfg_ids', type=str, required=True, help="List of configuration IDs to process")
    parser.add_argument('--nvec', type=int, required=True, help="Number of eigenvectors")
    parser.add_argument('--ntsrc', type=int, required=True, help="Number of source time slices")
    parser.add_argument('--plot', action='store_true', help="Show plot of pion distribution")
    parser.add_argument('--task', type=int, required=True, help="SLURM array task ID or unique identifier for this run")


    args = parser.parse_args()
    # if args.cfg_ids is None:
    #     cfg_ids = list(range(11, 1992, 10))  # Generate 11, 21, 31, ..., 1991
    # else:
    cfg_ids =  [int(cfg) for cfg in args.cfg_ids.split(',')]

    main(cfg_ids=cfg_ids, num_vecs=args.nvec, num_tsrcs=args.ntsrc, show_plot=args.plot,task_id=args.task)