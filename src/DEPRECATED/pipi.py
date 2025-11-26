import h5py
import numpy as np
import os
import argparse
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
import gamma as gamma
from typing import List

# Constants
NUM_VECS = 64
NUM_TSRCS = 24
LT = 64
BASE_PATH = os.path.abspath('/p/scratch/exotichadrons/exolaunch')
ENS = 'eric-L32T64'

def get_file_path(directory, filename, cfg_id):
    """Construct file path and check if it exists."""
    full_path = os.path.join(directory, filename.format(cfg_id=cfg_id))
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}. Skipping.")
        return None
    return full_path

def pipi_twopt(cfg_id, dirs, h5_group, flavor_contents: List[str], tsrc_avg=True,three_bar=False):
    """Process one configuration for given flavor systems, compute di-meson correlator with direct and crossing terms."""
    # file naming conventions for a single configuration
    file_specs = {
        'light': (dirs['light'], f"peram_{NUM_VECS}_cfg{{cfg_id}}.h5"),  # Light perambulator
        'meson': (dirs['meson'], f"meson-{NUM_VECS}_cfg{{cfg_id}}.h5"),
    }

    paths = {key: get_file_path(dir, template, cfg_id) for key, (dir, template) in file_specs.items()}
    if not paths['light'] or not paths['meson']:
        return False

    print(f"Reading light perambulator file: {paths['light']}")
    print(f"Reading meson elementals file: {paths['meson']}")

    # load common data
    meson_elemental = load_elemental(paths['meson'], LT, NUM_VECS, mom='mom_0_0_0', disp='disp')
    peram_light = load_peram(paths['light'], LT, NUM_VECS, NUM_TSRCS)  # Explicitly light

    # flavor combinations for perambulators
    flavor_map = {
        'light_light': (peram_light, peram_light, paths['light'], paths['light']),
    }

    # store the correlators and perambulator data for each flavor system
    """
    Direct:
    pion A (creation operator): <phi_t_A * tau_A * phi_0_A * tau_A_back>
    pion B (creation operator): <phi_t_B * tau_B * phi_0_B * tau_B_back>

    pi-pi creation operator = pion A x pion B 
    Di-meson direct: C_A(t) * C_B(t), where C_A and C_B are the individual meson correlators.
    """
    correlators = {}
    peram_data = {}
    for idx, flavor_content in enumerate(flavor_contents, 1):  # Start index at 1 for meson1, meson2
        if flavor_content not in flavor_map:
            print(f"Flavor '{flavor_content}' not recognized, defaulting to light_light.")
            flavor_content = 'light_light'

        peram, peram_back_data, peram_file, peram_back_file = flavor_map[flavor_content]
        if peram_back_data is None:
            print(f"Required {flavor_content} back perambulator file missing: {peram_back_file}. Skipping.")
            return False
        peram_back = reverse_perambulator_time(peram_back_data)
        print(f"Flavor '{flavor_content}' (meson {idx}):")
        print(f"  Perambulator loaded: {peram_file}")
        print(f"  Reverse perambulator loaded: {peram_back_file}")

        # store perambulator data for computation of crossing terms
        peram_data[flavor_content] = (peram, peram_back)

        # direct diagram
        meson_data = np.zeros((NUM_TSRCS, LT), dtype=np.cdouble)  # Store tsrc x Lt data
        phi_A_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])
        phi_B_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])


        for tsrc in range(NUM_TSRCS):
            for t in range(LT):
                phi_A_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                phi_B_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                tau_A = peram[tsrc, t, :, :, :, :]
                tau_B = peram[tsrc, t, :, :, :, :]
                tau_A_back = peram_back[tsrc, t, :, :, :, :]
                tau_B_back = peram_back[tsrc, t, :, :, :, :]
                pipi_A = np.einsum("ijab,jkbc,klcd,lida", phi_A_t, tau_A, phi_A_0, tau_A_back, optimize='optimal')
                pipi_B = np.einsum("ijab,jkbc,klcd,lida", phi_B_t, tau_B, phi_B_0, tau_B_back, optimize='optimal')

                meson_data[tsrc, t] = pipi_A * pipi_B 

        meson_data = meson_data.real

        # tsrc avg
        key_prefix = f'meson{idx}_{flavor_content}'  #  assign unique prefix for each meson
        if tsrc_avg:
            for tsrc in range(NUM_TSRCS):
                meson_data[tsrc] = np.roll(meson_data[tsrc], -4 * tsrc)  # shift tsrc to origin to get symmetric corr 
            meson_avg = meson_data.mean(axis=0)  # do tsrc avg here 
            correlators[flavor_content] = meson_avg
            h5_group.create_dataset(f'{key_prefix}/cfg_{cfg_id}_tsrc_avg', data=meson_avg)
        else:
            correlators[flavor_content] = meson_data
            for tsrc in range(NUM_TSRCS):
                h5_group.create_dataset(f'{key_prefix}/tsrc_{tsrc}/cfg_{cfg_id}', data=meson_data[tsrc])

        print(f"Correlator for {flavor_content} (meson {idx}) computed successfully{' with tsrc averaging' if tsrc_avg else ''}.")


    print(f"Cfg {cfg_id} processed successfully.")
    return True

def main(cfg_id: int, flavor_contents: List[str], task_id: int):
    """Process a single configuration for one or two flavor systems."""
    dirs = {
        'light': os.path.join(BASE_PATH, ENS, 'perams_sdb', f'numvec{NUM_VECS}'),
        'meson': os.path.join(BASE_PATH, ENS, 'meson_sdb', f'numvec{NUM_VECS}'),
    }

    # output file name based on flavor combination
    group_name = '_'.join(flavor_contents) if len(flavor_contents) > 1 else flavor_contents[0]
    # FIX THIS 
    h5_output_file = f'pipi_2pt_task{task_id}.h5'

    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f"{group_name}_000")
        try:
            if not pipi_twopt(cfg_id, dirs, h5_group, flavor_contents):
                print(f"Skipping configuration {cfg_id} due to missing files.")
        except FileNotFoundError as e:
            print(f"Error: {e}")

        print(f"Configuration {cfg_id} processed & saved to {h5_output_file} under group '{group_name}_000'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for one or two flavor systems.")
    parser.add_argument('--cfg_id', type=int, required=True, help="Single configuration ID to process")
    parser.add_argument('--flavor', type=str, required=True, help="Flavor content(s), comma-separated (e.g., light_charm,light_light)")
    parser.add_argument('--task', type=int, required=True, help="Task ID for this run")

    args = parser.parse_args()
    flavor_contents = args.flavor.split(',')
    if len(flavor_contents) > 2:
        raise ValueError("At most two flavor contents are supported (e.g., light_charm,light_light).")
    main(cfg_id=args.cfg_id, flavor_contents=flavor_contents, task_id=args.task)