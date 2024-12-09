import pathlib
import h5py
import numpy as np
import os 
import tqdm
import joblib 
from gamma import gamma
import pickle

mom_keys = {0: 'mom_-1_0_0', 1: 'mom_-2_0_0', 2: 'mom_-3_0_0', 3: 'mom_0_-1_0', 4: 'mom_0_-2_0', 5: 'mom_0_-3_0', 6: 'mom_0_0_-1', 7: 'mom_0_0_-2', 8: 'mom_0_0_-3', 9: 'mom_0_0_0', 10: 'mom_0_0_1', 11: 'mom_0_0_2', 12: 'mom_0_0_3', 13: 'mom_0_1_0', 14: 'mom_0_2_0', 15: 'mom_0_3_0', 16: 'mom_1_0_0', 17: 'mom_2_0_0', 18: 'mom_3_0_0'}
mom_keys_inv = {v: k for k, v in mom_keys.items()}

disp_keys = {0: 'disp', 1: 'disp_1', 2: 'disp_1_1', 3: 'disp_1_2', 4: 'disp_1_3', 5: 'disp_2', 6: 'disp_2_1', 7: 'disp_2_2', 8: 'disp_2_3', 9: 'disp_3', 10: 'disp_3_1', 11: 'disp_3_2', 12: 'disp_3_3'}
disp_keys_inv = {v: k for k, v in disp_keys.items()}

def load_peram(file: str, max_t: int, n_vecs: int, num_tsrcs: int = 24) -> np.ndarray:
    """
    Reads an HDF5 file written by chroma containing the distilled perambulator data from multiple t_sources.

    Parameters
    ----------
    file : str
        The path to the hdf5 file.
    max_t : int
        Temporal extent of the lattice (usually Lt).
    n_vecs : int
        Number of distillation basis vectors.
    num_tsrcs : int
        Number of t_sources (default is 24).

    Returns
    -------
    perambulator : np.ndarray
        A numpy array containing the perambulators. The shape is (num_tsrcs, max_t, 4, 4, n_vecs, n_vecs).
    """
    print(f"Reading propagator file: {file}")
    pickle_file = "/home/grant/exotraction/peram_1001.pkl"  # Cache file name based on input parameters
    # if os.path.exists(pickle_file):
        # print('found pickle file')
        # with open(pickle_file,'rb') as pf:
        #     f = pickle.load(pf)
    # else: 
    # with h5py.File(file,'r') as hf:
    #     f = {key: hf[key][:] for key in hf.keys()}

    # if pathlib.Path(cache_file).is_file():
    #     print(f"Loading cached meson data from {cache_file}")
    #     return joblib.load(cache_file)
    
    # Initialize the array for all t_sources
    peram = np.zeros((num_tsrcs, max_t, 4, 4, n_vecs, n_vecs), dtype=np.cdouble)

    # Open the file for reading
    with h5py.File(file, 'r') as f:
    # Loop over all t_sources (t_source_0, t_source_4, ..., t_source_92)
        for t_src_idx in tqdm.trange(0, num_tsrcs, leave=False, desc="Loading t_sources"):
            t_src_group_name = f't_source_{t_src_idx * 4}'  # Assuming the t_sources are spaced by 4 (e.g., 0, 4, 8, ...)
            if t_src_group_name not in f:
                print(f"Warning: {t_src_group_name} not found in {file}. Skipping...")
                continue
            t_source_data = f[t_src_group_name]

            # Loop over all t_slices for this t_source
            for t_slice_idx in tqdm.trange(0, max_t, leave=False, desc="Loading t_slices"):
                t_slice_data = t_source_data[f't_slice_{t_slice_idx}']
                
                # Loop over the spin sources and spin sinks
                for spin_src_idx in range(0, 4):
                    spin_src_data = t_slice_data[f'spin_src_{spin_src_idx}']
                    for spin_snk_idx in range(0, 4):
                        spin_snk_data = spin_src_data[f'spin_sink_{spin_snk_idx}']
                        peram[t_src_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :] = \
                            spin_snk_data['real'][:] + spin_snk_data['imag'][:] * 1j

    # print(f"Caching peram data to {cache_file}")
    # joblib.dump(peram, cache_file)
    # if pickle:
    #     print("pickling file")
    #     with open('peram_light_1001.pkl', 'wb') as f:
    #         pickle.dump(peram, f)

    return peram


def load_elemental(file: pathlib.Path | str, max_t: int, n_vecs: int, mom: str | None = None, disp: str | None = None) -> np.ndarray:
    """Reads an HDF5 file written by chroma containing the meson elemental data. By default it reads all momenta and displacements, \
        unless specified by `mom` and `disp` strings

    Parameters
    ----------
    file : `pathlib.Path | str`
        The path to the hdf5 file
    max_t : `int`
        Temporal extent of the lattice
    n_vecs : `int`
        Number of distillation basis vectors
    mom : `str | None`, optional
        Selected momentum string key (default: None)
    disp : `str | None`, optional
        Selected displacement string key (default: None)

    Returns
    -------
    meson : `numpy.ndarray`
        A numpy array containing the meson elementals, the size is (max_t, n_mom, n_disp, n_vecs, n_vecs) if \
        no specific key is given for momentum or displacement, otherwise reduntant axes are dropped
    """

    # cache_file = f"/home/grant/exotraction/{os.path.basename(file)}_cache_.joblib"  # Cache file name based on input parameters

    # # Check if the cache file exists
    # if pathlib.Path(cache_file).is_file():
    #     print(f"Loading cached meson data from {cache_file}")
    #     return joblib.load(cache_file)
    
    print(f"Reading meson elementals file: {file}")
    meson_data = h5py.File(file, 'r')
    n_mom = len(meson_data['t_slice_0'].keys())
    n_disp = len(meson_data['t_slice_0']['mom_0_0_0'].keys())

    if not mom and not disp:
        meson = np.zeros((max_t, n_mom, n_disp, n_vecs, n_vecs), dtype=np.cdouble)
        for t_slice_idx in tqdm.trange(0, max_t, leave=False):
            t_slice_data = meson_data[f't_slice_{t_slice_idx}']
            for mom_idx in range(0, 19):
                mom_data = t_slice_data[mom_keys[mom_idx]]
                for disp_idx in range(0, 13):
                    disp_data = mom_data[disp_keys[disp_idx]]
                    meson[t_slice_idx, mom_idx, disp_idx, :, :] = \
                        disp_data['real'][:] + disp_data['imag'][:] * 1j
                    
    elif not mom and disp:
        meson = np.zeros((max_t, n_mom, n_vecs, n_vecs), dtype=np.cdouble)
        for t_slice_idx in tqdm.trange(0, max_t, leave=False):
            t_slice_data = meson_data[f't_slice_{t_slice_idx}']
            for mom_idx in range(0, 19):
                mom_data = t_slice_data[mom_keys[mom_idx]]
                disp_data = mom_data[disp]
                meson[t_slice_idx, mom_idx, :, :] = \
                    disp_data['real'][:] + disp_data['imag'][:] * 1j


    elif mom and disp:
        meson = np.zeros((max_t, n_vecs, n_vecs), dtype=np.cdouble)
        for t_slice_idx in tqdm.trange(0, max_t, leave=False):
            t_slice_data = meson_data[f't_slice_{t_slice_idx}']
            mom_data = t_slice_data[mom]
            disp_data = mom_data[disp]
            meson[t_slice_idx, :, :] = \
                disp_data['real'][:] + disp_data['imag'][:] * 1j
            
    # print(f"Caching meson data to {cache_file}")
    # joblib.dump(meson, cache_file)
    # print("pickling file")
    # with open('meson_1001.pkl', 'wb') as f:
    #     pickle.dump(meson, f)

    return meson

def reverse_perambulator_time(peram: np.ndarray) -> np.ndarray:
    """
    Calculates the time-reversed perambulator from the forward one for all t_sources.

    Parameters
    ----------
    peram : numpy.ndarray
        The forward perambulator matrix of shape (num_tsrcs, max_t, 4, 4, n_vecs, n_vecs).

    Returns
    -------
    peram_reverse : np.ndarray
        A numpy array containing the time-reversed perambulator, the size is (num_tsrcs, max_t, 4, 4, n_vecs, n_vecs).
    """
    print("Computing the time-reversed perambulator")

    num_tsrcs, max_t, _, _, n_vecs, _ = peram.shape
    peram_reverse = np.zeros_like(peram)
    # Initialize the reverse perambulator array
    peramb_reverse = np.zeros((num_tsrcs, max_t, 4, 4, n_vecs, n_vecs), dtype=np.cdouble)

    # Loop over all time sources (tsrc)
    for tsrc_idx in tqdm.trange(num_tsrcs, leave=False, desc="Reversing time for all t_sources"):
        for t_slice_idx in tqdm.trange(max_t, leave=False, desc=f"t_slice for tsrc {tsrc_idx}"):
            for spin_src_idx in range(4):
                for spin_snk_idx in range(4):
                    # Reverse time slice by taking conjugate transpose
                    peramb_reverse[tsrc_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :] = \
                        np.transpose(np.conjugate(peram[tsrc_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :]))

    # Apply gamma_5 to reverse time and space directions as per the required transformation
#     return np.einsum("ab,tbcij,cd->tdaij", gamma[5], peramb_reverse, gamma[5], optimize='optimal')
    print("gamma[5] shape:", gamma[5].shape)  # Should be (4, 4)
    print("peramb_reverse shape:", peramb_reverse.shape)  # Should be (nu
    peramb_reverse = np.einsum(
        "ab,tsbcij,cd->tsdaij",  # Modified einsum for all t_sources
        gamma[5], peramb_reverse, gamma[5],
        optimize='optimal'
    )
    return peramb_reverse
