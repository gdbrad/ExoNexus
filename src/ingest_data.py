import pathlib
import h5py
import numpy as np
import os 
from gamma import gamma
import pickle

mom_keys = {0: 'mom_-1_0_0', 1: 'mom_-2_0_0', 2: 'mom_-3_0_0', 3: 'mom_0_-1_0', 4: 'mom_0_-2_0', 5: 'mom_0_-3_0', 6: 'mom_0_0_-1', 7: 'mom_0_0_-2', 8: 'mom_0_0_-3', 9: 'mom_0_0_0', 10: 'mom_0_0_1', 11: 'mom_0_0_2', 12: 'mom_0_0_3', 13: 'mom_0_1_0', 14: 'mom_0_2_0', 15: 'mom_0_3_0', 16: 'mom_1_0_0', 17: 'mom_2_0_0', 18: 'mom_3_0_0'}
mom_keys_inv = {v: k for k, v in mom_keys.items()}

disp_keys = {0: 'disp', 1: 'disp_1', 2: 'disp_1_1', 3: 'disp_1_2', 4: 'disp_1_3', 5: 'disp_2', 6: 'disp_2_1', 7: 'disp_2_2', 8: 'disp_2_3', 9: 'disp_3', 10: 'disp_3_1', 11: 'disp_3_2', 12: 'disp_3_3'}
disp_keys_inv = {v: k for k, v in disp_keys.items()}

def load_peram(file: str, max_t: int, n_vecs: int, num_tsrcs: int | None = None,tsrc_step=None) -> np.ndarray:
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
    tsrc_step : int
        Step size between t_sources (default is 4).

    Returns
    -------
    perambulator : np.ndarray
        A numpy array containing the perambulators. The shape is (num_tsrcs, max_t, 4, 4, n_vecs, n_vecs).
    """
    
    # Initialize the array for all t_sources
    peram = np.zeros((num_tsrcs, max_t, 4, 4, n_vecs, n_vecs), dtype=np.cdouble)
    
    # Open the file for reading
    with h5py.File(file, 'r') as f:
        tsrc_keys = sorted([k for k in f.keys()if k.startswith("t_source_")])

        if not tsrc_keys:
            raise RuntimeError(f"No t_source_* groups found in {file}")

        # Step 2: Auto-detect number of sources if requested
        if num_tsrcs is None:
            num_tsrcs = len(tsrc_keys)
            print(f"[LOAD_PERAM] Auto-detected {num_tsrcs} time sources in {file}")
        else:
            print(f"[LOAD_PERAM] Requested {num_tsrcs} time sources from {file} ({len(tsrc_keys)} available)")

        # Loop over all t_sources (t_source_0, t_source_4, ..., t_source_92)
        for t_src_idx in range(0, num_tsrcs):
            t_src_group_name = f't_source_{t_src_idx*tsrc_step}'  # Using tsrc_step for spacing
            if t_src_group_name not in f:
                print(f"Warning: {t_src_group_name} not found in {file}. Skipping...")
                continue
            t_source_data = f[t_src_group_name]

            # Loop over all t_slices for this t_source
            for t_slice_idx in range(0, max_t):
                t_slice_data = t_source_data[f't_slice_{t_slice_idx}']
                
                # Loop over the spin sources and spin sinks
                for spin_src_idx in range(0, 4):
                    spin_src_data = t_slice_data[f'spin_src_{spin_src_idx}']
                    for spin_snk_idx in range(0, 4):
                        spin_snk_data = spin_src_data[f'spin_sink_{spin_snk_idx}']
                        peram[t_src_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :] = \
                            spin_snk_data['real'][:] + spin_snk_data['imag'][:] * 1j

    return peram

# def load_peram(path: str, max_t: int, n_vecs: int, num_tsrcs: int | None = None, tsrc_step: int | None = None) -> np.ndarray:
#     """
#     Load a Chroma perambulator HDF5 file into a dense numpy array.
    
#     Parameters
#     ----------
#     path : str
#         Path to the .h5 perambulator file
#     max_t : int
#         Temporal lattice extent (Lt)
#     n_vecs : int
#         Number of distillation vectors
#     num_tsrcs : int or None
#         Number of time sources to load. If None → auto-detect all available
#     tsrc_step : int or None
#         Expected step between time sources (only used for warning messages)
    
#     Returns
#     -------
#     np.ndarray
#         Shape: (num_tsrcs, max_t, 4, 4, n_vecs, n_vecs), dtype=complex128
#     """
#     import h5py
#     import numpy as np

#     with h5py.File(path, 'r') as f:
#         # Step 1: Find all existing t_source_XXX groups
#         tsrc_keys = sorted([
#             k for k in f.keys()
#             if k.startswith("t_source_")])

#         if not tsrc_keys:
#             raise RuntimeError(f"No t_source_* groups found in {path}")

#         # Step 2: Auto-detect number of sources if requested
#         if num_tsrcs is None:
#             num_tsrcs = len(tsrc_keys)
#             print(f"[LOAD_PERAM] Auto-detected {num_tsrcs} time sources in {path}")
#         else:
#             print(f"[LOAD_PERAM] Requested {num_tsrcs} time sources from {path} ({len(tsrc_keys)} available)")

#         # Limit to available sources
#         num_tsrcs = min(num_tsrcs, len(tsrc_keys))
#         if num_tsrcs < len(tsrc_keys):
#             print(f"    → Only loading first {num_tsrcs} sources")

#         # Step 3: Pre-allocate array
#         peram = np.zeros((num_tsrcs, max_t, 4, 4, n_vecs, n_vecs), dtype=np.cdouble)

#         # Step 4: Fill it — using actual existing keys (robust!)
#         loaded = 0
#         for idx, key in enumerate(tsrc_keys[:num_tsrcs]):
#             t_src_group = f[key]

#             for t_slice_idx in range(max_t):
#                 t_slice_name = f"t_slice_{t_slice_idx}"
#                 if t_slice_name not in t_src_group:
#                     continue  # some files may be incomplete

#                 t_slice_group = t_src_group[t_slice_name]

#                 for spin_src in range(4):
#                     spin_src_name = f"spin_src_{spin_src}"
#                     if spin_src_name not in t_slice_group:
#                         continue
#                     spin_src_group = t_slice_group[spin_src_name]

#                     for spin_snk in range(4):
#                         spin_snk_name = f"spin_sink_{spin_snk}"
#                         if spin_snk_name not in spin_src_group:
#                             continue
#                         block = spin_src_group[spin_snk_name]

#                         # Chroma usually stores real/imag separately
#                         if 'real' in block and 'imag' in block:
#                             real_part = block['real'][:]
#                             imag_part = block['imag'][:]
#                             peram[idx, t_slice_idx, spin_src, spin_snk, :, :] = real_part + 1j * imag_part
#                         else:
#                             # fallback: maybe it's already complex
#                             data = block[...]
#                             peram[idx, t_slice_idx, spin_src, spin_snk, :, :] = data

#             loaded += 1

#         print(f"[LOAD_PERAM] Successfully loaded {loaded}/{num_tsrcs} time sources from {path}")

#     return peram

def load_elemental(file: str, max_t: int, n_vecs: int, mom: str | None = None, disp: str | None = None) -> np.ndarray:
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
    
    # print(f"Reading meson elementals file: {file}")
    meson_data = h5py.File(file, 'r')
    n_mom = len(meson_data['t_slice_0'].keys())
    n_disp = len(meson_data['t_slice_0']['mom_0_0_0'].keys())

    if not mom and not disp:
        meson = np.zeros((max_t, n_mom, n_disp, n_vecs, n_vecs), dtype=np.cdouble)
        for t_slice_idx in range(0, max_t):
            t_slice_data = meson_data[f't_slice_{t_slice_idx}']
            for mom_idx in range(0, 19):
                mom_data = t_slice_data[mom_keys[mom_idx]]
                for disp_idx in range(0, 13):
                    disp_data = mom_data[disp_keys[disp_idx]]
                    meson[t_slice_idx, mom_idx, disp_idx, :, :] = \
                        disp_data['real'][:] + disp_data['imag'][:] * 1j
                    
    elif not mom and disp:
        meson = np.zeros((max_t, n_mom, n_vecs, n_vecs), dtype=np.cdouble)
        for t_slice_idx in range(0, max_t):
            t_slice_data = meson_data[f't_slice_{t_slice_idx}']
            for mom_idx in range(0, 19):
                mom_data = t_slice_data[mom_keys[mom_idx]]
                disp_data = mom_data[disp]
                meson[t_slice_idx, mom_idx, :, :] = \
                    disp_data['real'][:] + disp_data['imag'][:] * 1j


    elif mom and disp:
        meson = np.zeros((max_t, n_vecs, n_vecs), dtype=np.cdouble)
        for t_slice_idx in range(0, max_t):
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
    # print("Computing the time-reversed perambulator")

    num_tsrcs, max_t, _, _, n_vecs, _ = peram.shape
    peram_reverse = np.zeros_like(peram)
    # Initialize the reverse perambulator array
    peramb_reverse = np.zeros((num_tsrcs, max_t, 4, 4, n_vecs, n_vecs), dtype=np.cdouble)

    # Loop over all time sources (tsrc)
    for tsrc_idx in range(num_tsrcs):
        for t_slice_idx in range(max_t):
            for spin_src_idx in range(4):
                for spin_snk_idx in range(4):
                    # Reverse time slice by taking conjugate transpose
                    peramb_reverse[tsrc_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :] = \
                        np.transpose(np.conjugate(peram[tsrc_idx, t_slice_idx, spin_src_idx, spin_snk_idx, :, :]))

    # Apply gamma_5 to reverse time and space directions as per the required transformation
#     return np.einsum("ab,tbcij,cd->tdaij", gamma[5], peramb_reverse, gamma[5], optimize='optimal')
    # print("gamma[5] shape:", gamma[5].shape)  # Should be (4, 4)
    # print("peramb_reverse shape:", peramb_reverse.shape)  # Should be (nu
    peramb_reverse = np.einsum(
        "ab,tsbcij,cd->tsdaij",  # Modified einsum for all t_sources
        gamma[5], peramb_reverse, gamma[5],
        optimize='optimal'
    )
    return peramb_reverse
