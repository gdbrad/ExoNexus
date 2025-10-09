import h5py
import numpy as np
import os
import argparse
from ingest_data import load_elemental, load_peram, reverse_perambulator_time
import gamma as gamma
from typing import List
from datetime import date

# Constants
MOM_KEYS = {0: 'mom_-1_0_0', 1: 'mom_-2_0_0', 2: 'mom_-3_0_0', 3: 'mom_0_-1_0', 4: 'mom_0_-2_0', 5: 'mom_0_-3_0', 6: 'mom_0_0_-1', 7: 'mom_0_0_-2', 8: 'mom_0_0_-3', 9: 'mom_0_0_0', 10: 'mom_0_0_1', 11: 'mom_0_0_2', 12: 'mom_0_0_3', 13: 'mom_0_1_0', 14: 'mom_0_2_0', 15: 'mom_0_3_0', 16: 'mom_1_0_0', 17: 'mom_2_0_0', 18: 'mom_3_0_0'}
MOM_KEYS_INV = {v: k for k, v in MOM_KEYS.items()}
BASE_PATH = os.path.abspath('/p/scratch/exotichadrons/su3-distillation')

MESON_NAME_MAP = {
    'light_light': 'pi',
    'light_charm': 'D',
    'light_strange': 'K',
    'charm_strange': 'Ds'
}

DI_MESON_NAME_MAP = {
    ('light_light', 'light_light'): 'pipi',
    ('light_charm', 'light_light'): 'Dpi',
    ('light_light', 'light_strange'): 'piK',
    ('light_charm', 'light_strange'): 'DK',
    ('charm_strange', 'light_light'): 'Dspi',
    ('charm_strange', 'light_strange'): 'DsK',
    ('light_strange', 'light_strange'): 'KK',
    ('light_strange', 'light_charm'): 'KD',
    ('light_light', 'light_charm'): 'piD',
}

def get_meson_system_name(flavor_contents: List[str]) -> str:
    if len(flavor_contents) == 1:
        return MESON_NAME_MAP.get(flavor_contents[0], flavor_contents[0])
    else:
        flavor_pair = tuple(flavor_contents)
        return DI_MESON_NAME_MAP.get(flavor_pair, '_'.join(flavor_contents))

def get_file_path(directory, filename, cfg_id):
    """Construct file path and check if it exists."""
    full_path = os.path.join(directory, filename.format(cfg_id=cfg_id))
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}. Skipping.")
        return None
    return full_path



def two_pt(nvecs:int,
           LT:int,
           cfg_id,
           dirs,
           h5_group, 
           flavor_contents: List[str],
           num_tsrc:int, 
           tsrc_step:int,
           tsrc_avg=False, 
           three_bar=False):
    
    """Process one configuration for given flavor systems, compute di-meson correlator with proper interpolators."""
    # determine required perambulators based on flavor content
    required_flavors = set()
    for flavor_content in flavor_contents:
        if flavor_content == 'light_light':
            required_flavors.add('light')
        elif flavor_content == 'light_strange':
            required_flavors.update(['light', 'strange'])
        elif flavor_content == 'light_charm':
            required_flavors.update(['light', 'charm'])
        elif flavor_content == 'charm_strange':
            required_flavors.update(['charm', 'strange'])
        elif flavor_content == 'charm_charm':
            required_flavors.update(['charm', 'charm'])
        else:
            print(f"Flavor '{flavor_content}' not recognized, defaulting to light_light.")
            required_flavors.add('light')

    # file naming conventions for a single cfg
    file_specs = {
        'light': (dirs['light'], f"peram_{nvecs}_cfg{{cfg_id}}.h5"),
        'meson': (dirs['meson'], f"meson-{nvecs}_cfg{{cfg_id}}.h5"),
        'strange': (dirs['strange'], f"peram_strange_nv{nvecs}_cfg{{cfg_id}}.h5"),
        'charm': (dirs['charm'], f"peram_charm_nv{nvecs}_cfg{{cfg_id}}.h5")
    }

    # only check paths for required flavors and meson elemental
    paths = {'meson': get_file_path(*file_specs['meson'], cfg_id)}
    for flavor in required_flavors:
        paths[flavor] = get_file_path(*file_specs[flavor], cfg_id)

    if not paths['meson'] or not all(paths.get(flavor) for flavor in required_flavors):
        return False

    print(f"Reading meson elementals file: {paths['meson']}")
    for flavor in required_flavors:
        print(f"Reading {flavor} perambulator file: {paths[flavor]}")

    # load common data
    meson_elemental = load_elemental(paths['meson'], LT, nvecs, mom='mom_0_0_0', disp='disp')
    peram_light = load_peram(paths['light'], LT, nvecs, num_tsrc) if 'light' in required_flavors else None
    peram_strange = load_peram(paths['strange'], LT, nvecs, num_tsrc) if 'strange' in required_flavors else None
    peram_charm = load_peram(paths['charm'], LT, nvecs, num_tsrc) if 'charm' in required_flavors else None

    # construct flavor_map based on available perambulators
    flavor_map = {}
    for flavor_content in flavor_contents:
        if flavor_content not in {'light_light', 'light_strange', 'light_charm', 'charm_strange'}:
            print(f"Flavor '{flavor_content}' not recognized, defaulting to light_light.")
            flavor_content = 'light_light'

        if flavor_content == 'light_light':
            flavor_map[flavor_content] = (peram_light, peram_light, paths.get('light'), paths.get('light'))
        elif flavor_content == 'light_strange' and 'strange' in required_flavors:
            flavor_map[flavor_content] = (peram_light, peram_strange, paths.get('light'), paths.get('strange'))
        elif flavor_content == 'light_charm' and 'charm' in required_flavors:
            flavor_map[flavor_content] = (peram_light, peram_charm, paths.get('light'), paths.get('charm'))
        elif flavor_content == 'charm_strange' and 'charm' in required_flavors and 'strange' in required_flavors:
            flavor_map[flavor_content] = (peram_charm, peram_strange, paths.get('charm'), paths.get('strange'))
        else:
            print(f"Required perambulators for {flavor_content} not available. Defaulting to light_light.")
            flavor_map[flavor_content] = (peram_light, peram_light, paths.get('light'), paths.get('light'))

    # Store perambulator data for each flavor system
    peram_data = {}
    for idx, flavor_content in enumerate(flavor_contents, 1): 
        peram, peram_back_data, peram_file, peram_back_file = flavor_map[flavor_content]
        if peram_back_data is None:
            print(f"Required {flavor_content} back perambulator file missing: {peram_back_file}. Skipping.")
            return False
        peram_back = reverse_perambulator_time(peram_back_data)
        print(f"Flavor '{flavor_content}' (meson {idx}):")
        print(f"  Perambulator loaded: {peram_file}")
        print(f"  Reverse perambulator loaded: {peram_back_file}")

        # Store perambulator data for di-meson contractions
        peram_data[flavor_content] = (peram, peram_back)

    # Compute di-meson correlator if multiple flavors are provided
    if len(flavor_contents) > 1:
        flavor1, flavor2 = flavor_contents
        group_name = get_meson_system_name(flavor_contents)
        is_identical = flavor1 == flavor2  

        #direct_data = np.zeros((num_tsrc, LT), dtype=np.cdouble)
        direct_data = np.zeros(LT, dtype=np.cdouble)

        crossing_data = np.zeros((num_tsrc, LT), dtype=np.cdouble) if not is_identical else None
        disconnected_data = np.zeros((num_tsrc, LT), dtype=np.cdouble) if three_bar else None

        peram1, peram_back1 = peram_data[flavor1]
        peram2, peram_back2 = peram_data[flavor2]

        for tsrc in range(num_tsrc):
            for t in range(LT):
                phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0], optimize='optimal')

                # Direct term: <phi_A(t) tau_A phi_A(0) tau_A^dagger> * <phi_B(t) tau_B phi_B(0) tau_B^dagger>
                tau1 = peram1[tsrc, t, :, :, :, :]
                tau1_back = peram_back1[tsrc, t, :, :, :, :]
                meson1_direct = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau1, phi_0, tau1_back, optimize='optimal')

                tau2 = peram2[tsrc, t, :, :, :, :]
                tau2_back = peram_back2[tsrc, t, :, :, :, :]
                meson2_direct = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau2, phi_0, tau2_back, optimize='optimal')

                direct_data[t] = meson1_direct * meson2_direct

                # Crossing term (only for non-identical mesons)
                if not is_identical:
                    # Meson 1: <phi_A(t) tau_A phi_B(0) tau_B^dagger>
                    meson1_cross = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau1, phi_0, peram_back2[tsrc, t, :, :, :, :], optimize='optimal')
                    # Meson 2: <phi_B(t) tau_B phi_A(0) tau_A^dagger>
                    meson2_cross = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau2, phi_0, peram_back1[tsrc, t, :, :, :, :], optimize='optimal')
                    crossing_data[tsrc, t] = meson1_cross * meson2_cross

                # Disconnected term for \bar{3} (if enabled)
                if three_bar:
                    # Meson 1: connected correlator
                    meson1_conn = meson1_direct  # Reuse direct term
                    # Meson 2: quark loop approximation at t
                    loop = np.einsum("siab,sjcd->abcd", peram2[tsrc, t, :, :, :, :], peram_back2[tsrc, t, :, :, :, :], optimize='optimal')
                    meson2_loop = np.einsum("ijab,abcd->", phi_t, loop, optimize='optimal')
                    disconnected_data[tsrc, t] = meson1_conn * meson2_loop

        direct_data = direct_data.real
        if not is_identical:
            crossing_data = crossing_data.real
        if three_bar:
            disconnected_data = disconnected_data.real

        # t_src averaging
        if tsrc_avg:
            for tsrc in range(num_tsrc):
                t = int(tsrc.split('_'[1]))
                direct_data[tsrc] = np.roll(direct_data[tsrc], -tsrc_step * tsrc)
                if not is_identical:
                    crossing_data[tsrc] = np.roll(crossing_data[tsrc], -tsrc_step * tsrc)
                if three_bar:
                    disconnected_data[tsrc] = np.roll(disconnected_data[tsrc], -tsrc_step * tsrc)
            direct_avg = direct_data.mean(axis=0)
            crossing_avg = crossing_data.mean(axis=0) if not is_identical else None
            disconnected_avg = disconnected_data.mean(axis=0) if three_bar else None
        else:
            direct_avg = direct_data
            crossing_avg = crossing_data if not is_identical else None
            disconnected_avg = disconnected_data if three_bar else None

        # Compute representations
        if is_identical:
            # For pipi, only direct term (single Wick topology)
            correlator_15 = direct_avg  # No crossing term
            correlator_6 = direct_avg   # No crossing term
            if three_bar:
                correlator_3_bar = direct_avg - (8/3) * disconnected_avg
        else:
            # [15]: direct - crossing
            correlator_15 = direct_avg - crossing_avg
            # [6]: direct + crossing
            correlator_6 = direct_avg + crossing_avg
            # \bar{3}: direct + (1/3)crossing - (8/3)disconnected
            if three_bar:
                correlator_3_bar = direct_avg + (1/3) * crossing_avg - (8/3) * disconnected_avg

        # Save results
        if tsrc_avg:
            h5_group.create_dataset(f'{group_name}/direct/cfg_{cfg_id}_tsrc_avg', data=direct_avg)
            if not is_identical:
                h5_group.create_dataset(f'{group_name}/crossing/cfg_{cfg_id}_tsrc_avg', data=crossing_avg)
            h5_group.create_dataset(f'{group_name}/15/cfg_{cfg_id}_tsrc_avg', data=correlator_15)
            h5_group.create_dataset(f'{group_name}/6/cfg_{cfg_id}_tsrc_avg', data=correlator_6)
            if three_bar:
                h5_group.create_dataset(f'{group_name}/disconnected/cfg_{cfg_id}_tsrc_avg', data=disconnected_avg)
                h5_group.create_dataset(f'{group_name}/3_bar/cfg_{cfg_id}_tsrc_avg', data=correlator_3_bar)
        else:
            for tsrc in range(num_tsrc):
                h5_group.create_dataset(f'{group_name}/direct/tsrc_{tsrc}/cfg_{cfg_id}', data=direct_data[tsrc])
                if not is_identical:
                    h5_group.create_dataset(f'{group_name}/crossing/tsrc_{tsrc}/cfg_{cfg_id}', data=crossing_data[tsrc])
                h5_group.create_dataset(f'{group_name}/15/tsrc_{tsrc}/cfg_{cfg_id}', 
                                       data=direct_data[tsrc] if is_identical else direct_data[tsrc] - crossing_data[tsrc])
                h5_group.create_dataset(f'{group_name}/6/tsrc_{tsrc}/cfg_{cfg_id}', 
                                       data=direct_data[tsrc] if is_identical else direct_data[tsrc] + crossing_data[tsrc])
                if three_bar:
                    h5_group.create_dataset(f'{group_name}/disconnected/tsrc_{tsrc}/cfg_{cfg_id}', data=disconnected_data[tsrc])
                    h5_group.create_dataset(f'{group_name}/3_bar/tsrc_{tsrc}/cfg_{cfg_id}', 
                                           data=direct_data[tsrc] - (8/3) * disconnected_data[tsrc] if is_identical 
                                           else direct_data[tsrc] + (1/3) * crossing_data[tsrc] - (8/3) * disconnected_data[tsrc])

        # Save individual meson correlators
        for idx, flavor_content in enumerate(flavor_contents, 1):
            key_prefix = f'meson{idx}_{flavor_content}'
            meson_data = np.zeros(LT, dtype=np.cdouble)
            peram, peram_back = peram_data[flavor_content]
            phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0])

            for tsrc in range(num_tsrc):
                for t in range(LT):
                    phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                    tau = peram[tsrc, t, :, :, :, :]
                    tau_ = peram_back[tsrc, t, :, :, :, :]
                    meson_data[t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')

                meson_data = meson_data.real
                meson_group = h5_group.require_group(f'{key_prefix}/cfg_{cfg_id}')

                if tsrc_avg:
                    for tsrc in range(num_tsrc):
                        meson_data[tsrc] = np.roll(meson_data[tsrc], -tsrc_step * tsrc)
                    meson_avg = meson_data.mean(axis=0)
                    h5_group.create_dataset('tsrc_avg', data=meson_avg)
                else:
                    meson_group.create_dataset(f'tsrc_{tsrc}', data=meson_data)

                print(f"Single meson correlator for {flavor_content} (meson {idx}) computed successfully{' with tsrc averaging' if tsrc_avg else ''}.")

        print(f"Di-meson correlator for {group_name} computed: representations [15], [6]{', \\bar{{3}}' if three_bar else ''} saved.")
############################################################################
    # Single meson case: compute two-point correlator
    else:
        flavor = flavor_contents[0]
        group_name = get_meson_system_name(flavor_contents)
        key_prefix = f'meson1_{flavor}'
        meson_data = np.zeros((num_tsrc,LT), dtype=np.cdouble)
        peram, peram_back = peram_data[flavor]
        #phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0],optimize='optimal')

        for tsrc_idx in range(num_tsrc-1):
            tsrc=tsrc_idx*4
            #for t in range(LT):
            for delta_t in range(LT):
                print('Lt',LT)
                t = tsrc + delta_t
                if t>=LT:
                    t = t-LT
                phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[tsrc_idx],optimize='optimal')
                
                phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                tau = peram[tsrc_idx, t, :, :, :, :]
                tau_ = peram_back[tsrc_idx, t, :, :, :, :]
                meson_data[tsrc_idx,delta_t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')
                with open("test_test.txt","a") as f:
                    print("tsrc",tsrc , "delta_t",delta_t,"t",t, meson_data[tsrc_idx,delta_t], file=f)

            #meson_data = meson_data.real
        

        if tsrc_avg:
            for tsrc in range(num_tsrc-1):
                meson_data[tsrc] = np.roll(meson_data[tsrc], tsrc * -4)
            meson_avg = meson_data.mean(axis=0)
            h5_group.create_dataset(f'{key_prefix}/cfg_{cfg_id}_tsrc_avg', data=meson_avg)
        else:
            h5_group.create_dataset(f'{key_prefix}/tsrc_{tsrc}/cfg_{cfg_id}', data=meson_data[tsrc_idx])



        print(f"Single meson correlator for {flavor} computed successfully{' with tsrc averaging' if tsrc_avg else ''}.")

       

    print(f"Cfg {cfg_id} processed successfully.")
    return True
############################################################################

def main(nvecs:int,
         LT: int,
         ens:str,
         cfg_id: int, 
         flavor_contents: List[str],
         num_tsrc:int,
         tsrc_step: int,
         three_bar: bool = False):
    """Process a single configuration for one or two flavor systems."""
    dirs = {
        'light': os.path.join(BASE_PATH, ens, 'perams_sdb', f'numvec{nvecs}'),
        'meson': os.path.join(BASE_PATH, ens, 'meson_sdb'),
        'strange': os.path.join(BASE_PATH, ens, 'perams_strange_sdb'),
        'charm': os.path.join(BASE_PATH, ens, 'perams_charm_sdb')
    }

    # Output file name based on meson system
    system_name = get_meson_system_name(flavor_contents)
    h5_output_file = f'{system_name}_cfg{cfg_id}_2pt_nvec_{nvecs}_tsrc_{num_tsrc}_test.h5'

    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f"{system_name}_000")
        try:
            if not two_pt(nvecs,LT,cfg_id, dirs, h5_group, flavor_contents,num_tsrc,tsrc_step, three_bar=three_bar):
                print(f"Skipping configuration {cfg_id} due to missing files.")
        except FileNotFoundError as e:
            print(f"Error: {e}")

        print(f"Configuration {cfg_id} processed & saved to {h5_output_file} under group '{system_name}_000'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for one or two flavor systems.")
    parser.add_argument('--nvecs', type=int, required=True, help="num vectors used")
    parser.add_argument('--lt', type=int, required=True, help="temporal extent")

    parser.add_argument('--ens',type=str,required=True)
    parser.add_argument('--cfg_id', type=int, required=True, help="Single configuration ID to process")
    parser.add_argument('--flavor', type=str, required=True, help="Flavor content(s), comma-separated (e.g., light_charm,light_light)")
    parser.add_argument('--ntsrc', type=int, required=True, help="number of tsrc insertions")
    parser.add_argument('--tsrc_step',type=int,required=False)
    parser.add_argument('--three_bar', action='store_true', help="Compute the 3-bar representation with disconnected term")

    args = parser.parse_args()
    flavor_contents = args.flavor.split(',')
    print(flavor_contents)
    print(args.ntsrc)
    if len(flavor_contents) > 2:
        raise ValueError("only mesons are currently supported (e.g., light_charm,light_light).")
    main(LT=args.lt,nvecs=args.nvecs,ens=args.ens,cfg_id=args.cfg_id, flavor_contents=flavor_contents,num_tsrc=args.ntsrc, tsrc_step=args.tsrc_step,three_bar=args.three_bar)
