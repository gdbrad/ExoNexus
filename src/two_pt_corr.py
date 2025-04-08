import h5py
import numpy as np
import os
import argparse
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
import gamma as gamma

# Constants
NUM_VECS = 96
NUM_TSRCS = 24
LT = 96
BASE_PATH = os.path.abspath('/p/scratch/exotichadrons/exolaunch')
ENS = 'gio-L32T96'

def get_file_path(directory, filename, cfg_id):
    """Construct file path and check if it exists."""
    full_path = os.path.join(directory, filename.format(cfg_id=cfg_id))
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}. Skipping.")
        return None
    return full_path

def process_configuration(cfg_id, dirs, h5_group, flavor_content, tsrc_avg=True):
    """Process one configuration, optionally averaging over tsrc."""
    # File templates for a single configuration
    file_specs = {
        'light': (dirs['light'], f"peram_{NUM_VECS}_cfg{{cfg_id}}.h5"),  # Light perambulator
        'meson': (dirs['meson'], f"meson-{NUM_VECS}_cfg{{cfg_id}}.h5"),
        'strange': (dirs['strange'], f"peram_strange_nv{NUM_VECS}_cfg{{cfg_id}}.h5"),
        'charm': (dirs['charm'], f"peram_charm_{NUM_VECS}_cfg{{cfg_id}}.h5")
    }

    # Get file paths
    paths = {key: get_file_path(dir, template, cfg_id) for key, (dir, template) in file_specs.items()}
    if not paths['light'] or not paths['meson']:
        return False

    print(f"Reading light perambulator file: {paths['light']}")
    print(f"Reading meson elementals file: {paths['meson']}")

    # Load data
    meson_elemental = load_elemental(paths['meson'], LT, NUM_VECS, mom='mom_0_0_0', disp='disp')
    peram_strange = load_peram(paths['strange'], LT, NUM_VECS, NUM_TSRCS) if paths['strange'] else None
    peram_charm = load_peram(paths['charm'], LT, NUM_VECS, NUM_TSRCS) if paths['charm'] else None
    peram_light = load_peram(paths['light'], LT, NUM_VECS, NUM_TSRCS)  # Explicitly light

    # Flavor-specific perambulator logic
    flavor_map = {
        'light_strange': (peram_light, peram_strange, paths['light'], paths['strange']),
        'light_charm': (peram_light, peram_charm, paths['light'], paths['charm']),
        'charm_strange': (peram_charm, peram_strange, paths['charm'], paths['strange'])
    }

    if flavor_content in flavor_map:
        peram, peram_back_data, peram_file, peram_back_file = flavor_map[flavor_content]
        if peram_back_data is None:
            print(f"Required {flavor_content} back perambulator file missing: {peram_back_file}. Skipping.")
            return False
        peram_back = reverse_perambulator_time(peram_back_data)
        print(f"Flavor '{flavor_content}':")
        print(f"  Perambulator loaded: {peram_file}")
        print(f"  Reverse perambulator loaded: {peram_back_file}")
    else:
        peram = peram_light
        peram_back = reverse_perambulator_time(peram_light)  # Default to light
        print("Flavor not specified or unrecognized, defaulting to light:")
        print(f"  Perambulator loaded: {paths['light']}")
        print(f"  Reverse perambulator loaded: {paths['light']}")

    # Contraction with source averaging
    meson_data = np.zeros((NUM_TSRCS, LT), dtype=np.cdouble)  # Store tsrc x Lt data
    phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])

    for tsrc in range(NUM_TSRCS):
        for t in range(LT):
            phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
            tau = peram[tsrc, t, :, :, :, :]
            tau_ = peram_back[tsrc, t, :, :, :, :]
            meson_data[tsrc, t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')

    meson_data = meson_data.real

    # Source averaging
    if tsrc_avg:
        for tsrc in range(NUM_TSRCS):
            meson_data[tsrc] = np.roll(meson_data[tsrc], -4 * tsrc)  # Shift each tsrc
        meson_avg = meson_data.mean(axis=0)  # Average over tsrc
        h5_group.create_dataset(f'cfg_{cfg_id}_tsrc_avg', data=meson_avg)
    else:
        for tsrc in range(NUM_TSRCS):
            h5_group.create_dataset(f'tsrc_{tsrc}/cfg_{cfg_id}', data=meson_data[tsrc])

    print(f"Cfg {cfg_id} processed successfully{' with tsrc averaging' if tsrc_avg else ''}.")
    return True

def main(cfg_id, flavor_content, task_id):
    """Process a single configuration and save results."""
    dirs = {
        'light': os.path.join(BASE_PATH, ENS, 'perams_sdb', f'numvec{NUM_VECS}', f'tsrc-{NUM_TSRCS}'),
        'meson': os.path.join(BASE_PATH, ENS, 'meson_sdb', f'numvec{NUM_VECS}','h5'),
        'strange': os.path.join(BASE_PATH, ENS, 'perams_strange_sdb'),
        'charm': os.path.join(BASE_PATH, ENS, 'perams_charm_sdb', f'numvec{NUM_VECS}')
    }

    h5_output_file = f'{flavor_content}_2pt_nvec_{NUM_VECS}_tsrc_{NUM_TSRCS}_task{task_id}.h5'
    with h5py.File(h5_output_file, "w") as h5f:
        # Use flavor_content in group name, default to 'light' if None or unrecognized
        group_name = f"{flavor_content if flavor_content in ['light_strange', 'light_charm', 'charm_strange'] else 'light'}_000"
        h5_group = h5f.create_group(group_name)
        try:
            if not process_configuration(cfg_id, dirs, h5_group, flavor_content):
                print(f"Skipping configuration {cfg_id} due to missing files.")
        except FileNotFoundError as e:
            print(f"Error: {e}")

        print(f"Configuration {cfg_id} processed & saved to {h5_output_file} under group '{group_name}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single peram and meson file with tsrc averaging.")
    parser.add_argument('--cfg_id', type=int, required=True, help="Single configuration ID to process")
    parser.add_argument('--flavor', type=str, help="Flavor content (e.g., light_strange)")
    parser.add_argument('--task', type=int, required=True, help="Task ID for this run")

    args = parser.parse_args()
    main(cfg_id=args.cfg_id, flavor_content=args.flavor, task_id=args.task)