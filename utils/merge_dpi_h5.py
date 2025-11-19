# merge_dpi_configs_full.py
# Merges ALL datasets from per-config files → one perfect averaged HDF5 file

import h5py
import numpy as np
import gvar as gv
from pathlib import Path
from src.dimeson_factory import DiMesonOperator
import argparse
from datetime import datetime
from glob import glob

parser = argparse.ArgumentParser(description="Merge ALL Dπ correlator datasets across configurations")
parser.add_argument("files", nargs="+", help="Input .h5 files (supports wildcards in quotes)")
parser.add_argument("--output", "-o", default=None, help="Output filename (auto-generated if not given)")
args = parser.parse_args()

files = []
for pattern in args.files:
    files.extend(glob(pattern))
files = sorted(set(files))
if not files:
    raise FileNotFoundError("No input files found!")

print(f"[MERGE] Found {len(files)} configuration files")
for f in files[:5]: print("   ", Path(f).name)
if len(files) > 5: print("   ...")

# ------------------------------------------------------------------
N = DiMesonOperator.num_operators()
Lt = 64
print(f"[INFO] Operator basis: {N}x{N} = {N*N} operators")

# Datasets we want to merge
datasets_to_merge = ["15", "6", "direct"]
optional_datasets = ["crossing"]  # only if not identical particles

# Containers: dict of (n_cfg, N, N, Lt) or (n_cfg, Lt)
raw_data = {}
for ds in datasets_to_merge + optional_datasets:
    raw_data[ds] = np.zeros((len(files), N, N, Lt), dtype=np.float64)
raw_data["D_correlator"]  = np.zeros((len(files), Lt), dtype=np.float64)
raw_data["pi_correlator"] = np.zeros((len(files), Lt), dtype=np.float64)

# ------------------------------------------------------------------
print("[LOAD] Loading all datasets from all configurations...")
for cfg_idx, fpath in enumerate(files):
    print(f"   [{cfg_idx+1:3d}/{len(files)}] {Path(fpath).name}")
    with h5py.File(fpath, "r") as f:
        if "Ptot_000_a1p" not in f:
            print(f"      Warning: no Ptot_000_a1p group in {fpath}")
            continue
        block = f["Ptot_000_a1p"]

        # Load di-meson matrix datasets
        for grp_name in block.keys():
            if not grp_name.startswith("op"):
                continue
            try:
                src_str, snk_str = grp_name.split("_X_")
                i = int(src_str.replace("op", ""))
                j = int(snk_str.replace("op", ""))

                for ds_name in datasets_to_merge:
                    if ds_name in block[grp_name]:
                        data = block[grp_name][ds_name][:]
                        if data.shape == (Lt,):
                            raw_data[ds_name][cfg_idx, i, j] = data
                        else:
                            raw_data[ds_name][cfg_idx, i, j] = np.mean(data, axis=0)

                if "crossing" in block[grp_name]:
                    data = block[grp_name]["crossing"][:]
                    
                    if data.shape == (Lt,):
                        raw_data["crossing"][cfg_idx, i, j] = data
                    else:
                        raw_data["crossing"][cfg_idx, i, j] = np.mean(data, axis=0)
            except Exception as e:
                print(f"      Warning: failed on {grp_name}: {e}")

        # Load individual mesons
        for name, key in [("D_correlator", "D_correlator"), ("pi_correlator", "pi_correlator")]:
            if key in block:
                data = block[key][:]
                if data.shape == (Lt,):
                    raw_data[name][cfg_idx] = data
                else:
                    raw_data[name][cfg_idx] = np.mean(data, axis=0)

print("[AVERAGE] Computing gvar averages with full covariance...")

# ------------------------------------------------------------------
# Final averaged gvars
# ------------------------------------------------------------------
averaged = {}

# Di-meson matrices
for ds in datasets_to_merge:
    flat = raw_data[ds].reshape(len(files), -1)
    averaged[ds] = gv.dataset.avg_data(flat).reshape(N, N, Lt)

if np.any(raw_data["crossing"] != 0):
    flat = raw_data["crossing"].reshape(len(files), -1)
    averaged["crossing"] = gv.dataset.avg_data(flat).reshape(N, N, Lt)
else:
    averaged["crossing"] = None

# Individual mesons
averaged["D_correlator"]  = gv.dataset.avg_data(raw_data["D_correlator"])
averaged["pi_correlator"] = gv.dataset.avg_data(raw_data["pi_correlator"])

# ------------------------------------------------------------------
# Write final merged file
# ------------------------------------------------------------------
outname = args.output or f"merged_Dpi_Ptot000_a1p_{len(files)}cfg_full.h5"
print(f"[WRITE] Writing complete merged file → {outname}")

with h5py.File(outname, "w") as f:
    grp = f.create_group("Ptot_000_a1p")

    # Write all matrix datasets
    for src in range(N):
        for snk in range(N):
            gname = f"op{src:02d}_X_op{snk:02d}"
            g = grp.create_group(gname)
            g.attrs["src_index"] = src
            g.attrs["snk_index"] = snk
            g.attrs["src_name"]  = DiMesonOperator.index_to_name(src)
            g.attrs["snk_name"]  = DiMesonOperator.index_to_name(snk)

            for ds in datasets_to_merge:
                mean = gv.mean(averaged[ds][src, snk])
                err  = gv.sdev(averaged[ds][src, snk])
                g.create_dataset(ds,           data=mean)
                g.create_dataset(ds + "_err",   data=err)

            if averaged["crossing"] is not None:
                mean = gv.mean(averaged["crossing"][src, snk])
                err  = gv.sdev(averaged["crossing"][src, snk])
                g.create_dataset("crossing",     data=mean)
                g.create_dataset("crossing_err", data=err)

    # Write individual mesons
    for name in ["D_correlator", "pi_correlator"]:
        mean = gv.mean(averaged[name])
        err  = gv.sdev(averaged[name])
        grp.create_dataset(name,          data=mean)
        grp.create_dataset(name + "_err", data=err)

    # Metadata
    f.attrs["n_configurations"] = len(files)
    f.attrs["merged_datasets"]  = datasets_to_merge + (["crossing"] if averaged["crossing"] else [])
    f.attrs["merge_date"]       = datetime.now().isoformat()
    f.attrs["operator_basis"]   = N
    f.attrs["creator"]          = "merge_dpi_configs_full.py"

print(f"\n[SUCCESS] FULLY MERGED FILE CREATED: {outname}")
print("   Contains: 15, 6, direct, crossing (if present), D_correlator, pi_correlator")
print("   Every dataset has _err companion")
print("   Ready for final GEVP, fits, plots — forever.")
print("\n   Now just do:")
print(f"   python analyze_dpi_matrix.py --file {outname}")