import h5py
import numpy as np
import os
import argparse
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
import gamma as gamma
from typing import List

# Constants
NUM_VECS = 64
NUM_TSRCS = 16
LT = 64
BASE_PATH = os.path.abspath('/p/scratch/exotichadrons/su3-distillation')
ENS = 'b3.4_s32t64'

def get_file_path(directory, filename, cfg_id):
    """Construct file path and check if it exists."""
    full_path = os.path.join(directory, filename.format(cfg_id=cfg_id))
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}. Skipping.")
        return None
    return full_path

def dpi_twopt(cfg_id, dirs, h5_group, flavor_contents: List[str], tsrc_avg=True,three_bar=False):
    """Process one configuration for given flavor systems, compute di-meson correlator with direct and crossing terms."""
    # file naming conventions for a single configuration
    file_specs = {
        'light': (dirs['light'], f"peram_{NUM_VECS}_cfg{{cfg_id}}.h5"),  # Light perambulator
        'meson': (dirs['meson'], f"meson-{NUM_VECS}_cfg{{cfg_id}}.h5"),
        'strange': (dirs['strange'], f"peram_strange_nv{NUM_VECS}_cfg{{cfg_id}}.h5"),
        'charm': (dirs['charm'], f"peram_charm_nv{NUM_VECS}_cfg{{cfg_id}}.h5")
    }

    paths = {key: get_file_path(dir, template, cfg_id) for key, (dir, template) in file_specs.items()}
    if not paths['light'] or not paths['meson']:
        return False

    print(f"Reading light perambulator file: {paths['light']}")
    if paths['charm']:
        print(f"Reading charm perambulator file: {paths['charm']}")
    print(f"Reading meson elementals file: {paths['meson']}")

    # load common data
    meson_elemental = load_elemental(paths['meson'], LT, NUM_VECS, mom='mom_0_0_0', disp='disp')
    peram_strange = load_peram(paths['strange'], LT, NUM_VECS, NUM_TSRCS) if paths['strange'] else None
    peram_charm = load_peram(paths['charm'], LT, NUM_VECS, NUM_TSRCS) if paths['charm'] else None
    peram_light = load_peram(paths['light'], LT, NUM_VECS, NUM_TSRCS)  # Explicitly light

    # flavor combinations for perambulators
    flavor_map = {
        'light_strange': (peram_light, peram_strange, paths['light'], paths['strange']),
        'light_light': (peram_light, peram_light, paths['light'], paths['light']),
        'light_charm': (peram_light, peram_charm, paths['light'], paths['charm']),
        'charm_strange': (peram_charm, peram_strange, paths['charm'], paths['strange'])
    }

    # store the correlators and perambulator data for each flavor system
    """
    direct and crossing graph topology for the [15] must be subtracted 
        to account for all possible Wick contraction topologies for a di-meson correlator eg. [15] or [6], need to compute these contributions separately for each flavor system ->>> form the final di-meson correlator as the difference between the direct and crossing terms (i.e., direct - crossing)
        Direct:
        Meson A: <phi_t_A * tau_A * phi_0_A * tau_A_back>
        Meson B: <phi_t_B * tau_B * phi_0_B * tau_B_back>
        Di-meson direct: C_A(t) * C_B(t), where C_A and C_B are the individual meson correlators.
        Crossing:
        <phi_t_A * tau_A * phi_0_B * tau_B_back> * <phi_t_B * tau_B * phi_0_A * tau_A_back>
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
        phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])

        for tsrc in range(NUM_TSRCS):
            for t in range(LT):
                phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                tau = peram[tsrc, t, :, :, :, :]
                tau_ = peram_back[tsrc, t, :, :, :, :]
                meson_data[tsrc, t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')

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

    # di-meson correlator if multiple flavors are provided eg. light-charm, light-light
    if len(flavor_contents) > 1:
        flavor1, flavor2 = flavor_contents
        group_name = f'{flavor1}_{flavor2}'

        # direct di-meson correlator: C1(t) * C2(t)
        direct_correlator = np.prod([correlators[flavor] for flavor in flavor_contents], axis=0)

        # init crossing di-meson correlator, swap backward propagating perambulators
        crossing_data = np.zeros((NUM_TSRCS, LT), dtype=np.cdouble)
        disconnected_data = np.zeros((NUM_TSRCS, LT), dtype=np.cdouble)

        peram1, peram_back1 = peram_data[flavor1]
        peram2, peram_back2 = peram_data[flavor2]

        # crossing term: meson1 uses peram1 and peram_back2, meson2 uses peram2 and peram_back1
        for tsrc in range(NUM_TSRCS):
            for t in range(LT):
                phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0], optimize= 'optimal')

                # meson1: forward = flavor1, backward = flavor2
                tau1 = peram1[tsrc, t, :, :, :, :]
                tau1_back = peram_back2[tsrc, t, :, :, :, :]
                meson1_cross = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau1, phi_0, tau1_back, optimize='optimal')

                # meson2: forward = flavor2, backward = flavor1
                tau2 = peram2[tsrc, t, :, :, :, :]
                tau2_back = peram_back1[tsrc, t, :, :, :, :]
                meson2_cross = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau2, phi_0, tau2_back, optimize='optimal')

                crossing_data[tsrc, t] = meson1_cross * meson2_cross

                # disconnected piece for \bar{3} ; approximate quark loop for meson2, connected for meson1
                # meson1: connected correlator, the same same as direct for flavor1
                meson1_conn = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau1, phi_0, peram_back1[tsrc, t, :, :, :, :], optimize='optimal')

                # meson2: quark loop approximation using peram2 * peram_back2 at same t
                loop = np.einsum("siab,sjcd->abcd", peram2[tsrc, t, :, :, :, :], peram_back2[tsrc, t, :, :, :, :], optimize='optimal')
                meson2_loop = np.einsum("ijab,abcd->", phi_t, loop, optimize='optimal')

                disconnected_data[tsrc, t] = meson1_conn * meson2_loop



        crossing_data = crossing_data.real

        # tsrc averaging for crossing term
        if tsrc_avg:
            for tsrc in range(NUM_TSRCS):
                crossing_data[tsrc] = np.roll(crossing_data[tsrc], -4 * tsrc)
            crossing_avg = crossing_data.mean(axis=0)
        else:
            crossing_avg = crossing_data

        # [15] correlator
        correlator_15 = direct_correlator - crossing_avg
        # [6] correlator
        correlator_6 = direct_correlator + crossing_avg
        # [\bar{3}] is more complicated.. need to add disconnected term (last one)
        if three_bar:
            correlator_3_bar = direct_correlator + (1/3 * crossing_avg) - 8/3 * disconnected_data
        else: 
            pass

        if tsrc_avg:
            h5_group.create_dataset(f'{group_name}/direct/cfg_{cfg_id}_tsrc_avg', data=direct_correlator)
            h5_group.create_dataset(f'{group_name}/crossing/cfg_{cfg_id}_tsrc_avg', data=crossing_avg)
            h5_group.create_dataset(f'{group_name}/15/cfg_{cfg_id}_tsrc_avg', data=correlator_15)
            h5_group.create_dataset(f'{group_name}/6/cfg_{cfg_id}_tsrc_avg', data=correlator_6)
            if three_bar:
                h5_group.create_dataset(f'{group_name}/3_bar/cfg_{cfg_id}_tsrc_avg', data=correlator_3_bar)


        else:
            for tsrc in range(NUM_TSRCS):
                tsrc_direct = np.prod([correlators[flavor][tsrc] for flavor in flavor_contents], axis=0)
                h5_group.create_dataset(f'{group_name}/direct/tsrc_{tsrc}/cfg_{cfg_id}', data=tsrc_direct)
                h5_group.create_dataset(f'{group_name}/crossing/tsrc_{tsrc}/cfg_{cfg_id}', data=crossing_data[tsrc])
                h5_group.create_dataset(f'{group_name}/15/tsrc_{tsrc}/cfg_{cfg_id}', data=tsrc_direct - crossing_data[tsrc])
                h5_group.create_dataset(f'{group_name}/6/tsrc_{tsrc}/cfg_{cfg_id}', data=tsrc_direct + crossing_data[tsrc])


        print(f"Di-meson correlator for {group_name} computed: direct - crossing saved.")

    print(f"Cfg {cfg_id} processed successfully.")
    return True

def main(cfg_id: int, flavor_contents: List[str]):
    """Process a single configuration for one or two flavor systems."""
    dirs = {
        'light': os.path.join(BASE_PATH, ENS, 'perams_sdb', f'numvec{NUM_VECS}'),
        'meson': os.path.join(BASE_PATH, ENS, 'meson_sdb'),
        'strange': os.path.join(BASE_PATH, ENS, 'perams_strange_sdb'),
        'charm': os.path.join(BASE_PATH, ENS, 'perams_charm_sdb')
    }

    # output file name based on flavor combination
    group_name = '_'.join(flavor_contents) if len(flavor_contents) > 1 else flavor_contents[0]
    # FIX THIS 
    h5_output_file = f'Dpi_all_tsrc_{group_name}_2pt.h5'

    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f"{group_name}_000")
        try:
            if not dpi_twopt(cfg_id, dirs, h5_group, flavor_contents):
                print(f"Skipping configuration {cfg_id} due to missing files.")
        except FileNotFoundError as e:
            print(f"Error: {e}")

        print(f"Configuration {cfg_id} processed & saved to {h5_output_file} under group '{group_name}_000'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for one or two flavor systems.")
    parser.add_argument('--cfg_id', type=int, required=True, help="Single configuration ID to process")
    parser.add_argument('--flavor', type=str, required=True, help="Flavor content(s), comma-separated (e.g., light_charm,light_light)")

    args = parser.parse_args()
    flavor_contents = args.flavor.split(',')
    if len(flavor_contents) > 2:
        raise ValueError("At most two flavor contents are supported (e.g., light_charm,light_light).")
    main(cfg_id=args.cfg_id, flavor_contents=flavor_contents)