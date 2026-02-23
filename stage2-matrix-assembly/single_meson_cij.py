import numpy as np
import h5py
import glob
import os


# ---------------------------------------------------------
# Discover operator basis
# ---------------------------------------------------------

def discover_operator_basis(example_file, irrep):
    with h5py.File(example_file, "r") as f:
        pairs = list(f[irrep].keys())

    ops = set()
    for name in pairs:
        op1, op2 = name.split("__X__")
        ops.add(op1)
        ops.add(op2)

    return sorted(list(ops))


# ---------------------------------------------------------
# Build C_ij matrix (single meson version)
# ---------------------------------------------------------

def build_Cij(stage1_dir, irrep, output_file):

    cfg_files = sorted(glob.glob(os.path.join(stage1_dir, "cfg*.h5")))
    ncfg = len(cfg_files)

    if ncfg == 0:
        raise RuntimeError("No cfg files found.")

    # ---------------------------------------------
    # Discover operator basis
    # ---------------------------------------------

    operator_list = discover_operator_basis(cfg_files[0], irrep)
    nops = len(operator_list)
    op_index = {op: i for i, op in enumerate(operator_list)}

    # ---------------------------------------------
    # Get Ntsrc and Lt
    # ---------------------------------------------

    with h5py.File(cfg_files[0], "r") as f:
        first_dataset = next(iter(f[irrep].values()))
        Ntsrc, LT = first_dataset.shape

    print(f"Found {ncfg} configs")
    print(f"Operators: {nops}")
    print(f"Ntsrc: {Ntsrc}, Lt: {LT}")

    # ---------------------------------------------
    # Allocate full matrix
    # ---------------------------------------------

    C = np.zeros((ncfg, Ntsrc, nops, nops, LT), dtype=np.float64)

    # ---------------------------------------------
    # Loop over configurations
    # ---------------------------------------------

    for cfg_idx, fname in enumerate(cfg_files):

        print(f"[{cfg_idx+1}/{ncfg}] {fname}")

        with h5py.File(fname, "r") as f:

            for pair_name, dataset in f[irrep].items():

                op1, op2 = pair_name.split("__X__")
                i = op_index[op1]
                j = op_index[op2]

                # Dataset shape: (Ntsrc, LT)
                C[cfg_idx, :, i, j, :] = dataset[:]

    # ---------------------------------------------
    # Save output
    # ---------------------------------------------

    with h5py.File(output_file, "w") as f:

        f.create_dataset("Cij", data=C, compression="gzip")

        f.create_dataset(
            "operators",
            data=np.array(operator_list, dtype="S")
        )

    print("Stage 2 complete (single meson).")