import h5py
import numpy as np
import os
import argparse
from insertion_factory import gamma
from typing import List
from file_io import FileIO

def _tsrc_avg(group_name:str):
    pass
    

def two_pt(nvecs: int,
           LT: int,
           cfg_id: int,
           file_io: FileIO,
           h5_group,
           flavor_contents: List[str],
           num_tsrc: int,
           tsrc_step: int,
           tsrc_avg: bool = False,
           three_bar: bool = False) -> bool:
    """
    Process one configuration for given flavor systems, compute di-meson correlator with proper interpolators.
    """
    # Get required flavors and file specifications using FileIO
    required_flavors = file_io.get_required_flavors
    file_specs = file_io.file_specs()

    # Check paths for required flavors and meson elemental
    paths = {'meson': file_io.get_file_path('meson')}
    for flavor in required_flavors:
        if flavor in file_specs:
            paths[flavor] = file_io.get_file_path(flavor)

    if not paths['meson'] or not all(paths.get(flavor) for flavor in required_flavors):
        print(f"Missing required files for cfg {cfg_id}. Skipping.")
        return False

    print(f"Reading meson elementals file: {paths['meson']}")
    for flavor in required_flavors:
        print(f"Reading {flavor} perambulator file: {paths[flavor]}")

    # Use preloaded data from FileIO
    if file_io.meson_elemental is None:
        print(f"Meson elemental data not loaded for cfg {cfg_id}. Skipping.")
        return False
    meson_elemental = file_io.meson_elemental
    peram_light = file_io.peram_light
    peram_strange = file_io.peram_strange
    peram_charm = file_io.peram_charm

    # Store perambulator data for each flavor system
    peram_data = file_io.peram_data()
#-------------------------------------------------------------# 
    # Compute di-meson correlator if multiple flavors are provided
    if len(flavor_contents) > 1:
        flavor1, flavor2 = flavor_contents
        group_name = file_io.get_meson_system_name()
        is_identical = flavor1 == flavor2

        direct_data = np.zeros((num_tsrc, LT), dtype=np.cdouble)
        crossing_data = np.zeros((num_tsrc, LT), dtype=np.cdouble) if not is_identical else None
        disconnected_data = np.zeros((num_tsrc, LT), dtype=np.cdouble) if three_bar else None

        peram1, peram_back1 = peram_data[flavor1]
        peram2, peram_back2 = peram_data[flavor2]

        for tsrc in range(num_tsrc):
            for t in range(LT):
                phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0], optimize='optimal')

                # Direct term
                tau1 = peram1[tsrc, t, :, :, :, :]
                tau1_back = peram_back1[tsrc, t, :, :, :, :]
                meson1_direct = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau1, phi_0, tau1_back, optimize='optimal')

                tau2 = peram2[tsrc, t, :, :, :, :]
                tau2_back = peram_back2[tsrc, t, :, :, :, :]
                meson2_direct = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau2, phi_0, tau2_back, optimize='optimal')

                direct_data[tsrc, t] = meson1_direct * meson2_direct

                # Crossing term (only for non-identical mesons)
                if not is_identical:
                    meson1_cross = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau1, phi_0, peram_back2[tsrc, t, :, :, :, :], optimize='optimal')
                    meson2_cross = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau2, phi_0, peram_back1[tsrc, t, :, :, :, :], optimize='optimal')
                    crossing_data[tsrc, t] = meson1_cross * meson2_cross

                # Disconnected term for \bar{3} (if enabled)
                if three_bar:
                    meson1_conn = meson1_direct
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
            correlator_15 = direct_avg
            correlator_6 = direct_avg
            if three_bar:
                correlator_3_bar = direct_avg - (8/3) * disconnected_avg
        else:
            correlator_15 = direct_avg - crossing_avg
            correlator_6 = direct_avg + crossing_avg
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
            meson_data = np.zeros((num_tsrc, LT), dtype=np.cdouble)
            peram, peram_back = peram_data[flavor_content]
            phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0], optimize='optimal')

            for tsrc in range(num_tsrc):
                for t in range(LT):
                    phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[t], optimize='optimal')
                    tau = peram[tsrc, t, :, :, :, :]
                    tau_ = peram_back[tsrc, t, :, :, :, :]
                    meson_data[tsrc, t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')

                meson_data = meson_data.real
                meson_group = h5_group.require_group(f'{key_prefix}/cfg_{cfg_id}')

                if tsrc_avg:
                    for tsrc_idx in range(num_tsrc):
                        meson_data[tsrc_idx] = np.roll(meson_data[tsrc_idx], -tsrc_step * tsrc_idx)
                    meson_avg = meson_data.mean(axis=0)
                    meson_group.create_dataset('tsrc_avg', data=meson_avg)
                else:
                    meson_group.create_dataset(f'tsrc_{tsrc}', data=meson_data[tsrc])

                print(f"Single meson correlator for {flavor_content} (meson {idx}) computed successfully{' with tsrc averaging' if tsrc_avg else ''}.")

        print(f"Di-meson correlator for {group_name} computed: representations [15], [6]{', \\bar{{3}}' if three_bar else ''} saved.")

    # Single meson case: compute two-point correlator
    else:
        flavor = flavor_contents[0]
        group_name = file_io.get_meson_system_name()
        key_prefix = f'meson1_{flavor}'
        meson_data = np.zeros((num_tsrc, LT), dtype=np.cdouble)
        peram, peram_back = peram_data[flavor]
        phi_0 = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[0], optimize='optimal')
        

        for tsrc_idx in range(num_tsrc):
            tsrc = tsrc_idx * tsrc_step
            for delta_t in range(LT):
                t = (tsrc + delta_t) % LT
                phi_t = np.einsum("ij,ab->ijab", gamma.gamma[5], meson_elemental[delta_t], optimize='optimal')
                tau = peram[tsrc_idx, delta_t, :, :, :, :]
                tau_ = peram_back[tsrc_idx, delta_t, :, :, :, :]
                meson_data[tsrc_idx, delta_t] = np.einsum("ijab,jkbc,klcd,lida", phi_t, tau, phi_0, tau_, optimize='optimal')
                with open("test_test.txt", "a") as f:
                    print(f"tsrc {tsrc}, delta_t {delta_t}, t {t}, {meson_data[tsrc_idx, delta_t]}", file=f)

            meson_data = meson_data.real

            if tsrc_avg:
                for tsrc_idx in range(num_tsrc):
                    meson_data[tsrc_idx] = np.roll(meson_data[tsrc_idx], -tsrc_step * tsrc_idx)
                meson_avg = meson_data.mean(axis=0)
                h5_group.create_dataset(f'{key_prefix}/cfg_{cfg_id}_tsrc_avg', data=meson_avg)
            else:
                h5_group.create_dataset(f'{key_prefix}/tsrc_{tsrc_idx}/cfg_{cfg_id}', data=meson_data[tsrc_idx])

        print(f"Single meson correlator for {flavor} computed successfully{' with tsrc averaging' if tsrc_avg else ''}.")

    print(f"Cfg {cfg_id} processed successfully.")
    return True

def main(nvecs: int,
         LT: int,
         ens: str,
         cfg_id: int,
         flavor_contents: List[str],
         num_tsrc: int,
         tsrc_step: int,
         data1: bool,
         three_bar: bool = False,
         ):
    """
    Process a single configuration for one or two flavor systems.
    """
    # Instantiate FileIO
    file_io = FileIO(
        flavor_contents=flavor_contents,
        cfg_id=cfg_id,
        ens=ens,
        nvecs=nvecs,
        LT=LT,
        num_tsrc=num_tsrc,
        tsrc_step=tsrc_step,
        data1=data1
    )

    # Output file name based on meson system
    system_name = file_io.get_meson_system_name()
    h5_output_file = f'{ens}_{system_name}_cfg{cfg_id}_2pt_nvec_{nvecs}_tsrc_{num_tsrc}_test.h5'

    with h5py.File(h5_output_file, "w") as h5f:
        h5_group = h5f.create_group(f"{system_name}_000")
        try:
            if not two_pt(nvecs, LT, cfg_id, file_io, h5_group, flavor_contents, num_tsrc, tsrc_step, three_bar=three_bar):
                print(f"Skipping configuration {cfg_id} due to missing files.")
        except FileNotFoundError as e:
            print(f"Error: {e}")

        print(f"Configuration {cfg_id} processed & saved to {h5_output_file} under group '{system_name}_000'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process peram and meson files for one or two flavor systems.")
    parser.add_argument('--nvecs', type=int, required=True, help="num vectors used")
    parser.add_argument('--lt', type=int, required=True, help="temporal extent")
    parser.add_argument('--ens', type=str, required=True)
    parser.add_argument('--cfg_id', type=int, required=True, help="Single configuration ID to process")
    parser.add_argument('--flavor', type=str, required=True, help="Flavor content(s), comma-separated (e.g., light_charm,light_light)")
    parser.add_argument('--ntsrc', type=int, required=True, help="number of tsrc insertions")
    parser.add_argument('--tsrc_step', type=int, required=False, default=1, help="step size for tsrc")
    parser.add_argument('--data1',action='store_true',help="is meson data in the data1 dir")
    parser.add_argument('--three_bar', action='store_true', help="Compute the 3-bar representation with disconnected term")

    args = parser.parse_args()
    flavor_contents = args.flavor.split(',')
    if len(flavor_contents) > 2:
        raise ValueError("Only mesons are currently supported (e.g., light_charm,light_light).")
    main(
        nvecs=args.nvecs,
        LT=args.lt,
        ens=args.ens,
        cfg_id=args.cfg_id,
        flavor_contents=flavor_contents,
        num_tsrc=args.ntsrc,
        tsrc_step=args.tsrc_step,
        three_bar=args.three_bar,
        data1=args.data1
    )