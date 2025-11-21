# # merge_dpi_h5.py
# import sys
# from pathlib import Path
# SRC_DIR = Path(__file__).resolve().parent.parent / "src"
# sys.path.insert(0, str(SRC_DIR))

# # Now these imports work
# import h5py
# import numpy as np
# from pathlib import Path
# import argparse
# from dimeson_factory import DiMesonOperator



# def merge_per_pair_files(input_dir: str, output_file: str = "merged_Dpi.h5"):
#     input_dir = Path(input_dir)
#     files = sorted(input_dir.glob("*.h5"))
#     if not files:
#         print(f"[ERROR] No files found in {input_dir}")
#         return

#     print(f"[MERGE] Found {len(files)} configuration files")

#     # Get operator basis
#     dimesons = DiMesonOperator.generate_operators()
#     N = len(dimesons.keys())
#     print(f"[INFO] Operator basis: {N} × {N}")
#     files = sorted(input_dir.glob("*.h5"))
#     if not files:
#         raise FileNotFoundError("No files found")

#     print(f"[MERGE] {len(files)} files → averaging tsrc on-the-fly (no memory explosion)")

#     # Operator basis

#     # Load first file to get Lt and ntsrc
#     with h5py.File(files[0], "r") as f:
#         sample = f["Ptot_000_a1p/op00_X_op00/15"]
#         Lt = sample.shape[-1]
#         ntsrc = sample.shape[0] if sample.ndim == 2 else 1

#     # ONLY store averaged over tsrc → tiny memory
#     C15 = np.zeros((len(files), N, N, Lt))
#     C6 = np.zeros((len(files), N, N, Lt))
#     C_direct = np.zeros((len(files), N, N, Lt))

#     for cfg_idx, fpath in enumerate(files):
#         print(f"  [{cfg_idx+1}/{len(files)}] {fpath.name}")
#         with h5py.File(fpath, "r") as f:
#             block = f["Ptot_000_a1p"]
#             for key in block.keys():
#                 if not key.startswith("op"):
#                     continue
#                 src = int(key.split("_X_")[0][2:])
#                 snk = int(key.split("_X_")[1][2:])

#                 for ds_name, target in [("15", C15), ("6", C6), ("direct", C_direct)]:
#                     if ds_name in block[key]:
#                         raw = np.array(block[key][ds_name])
#                         if raw.shape == (Lt, ntsrc):
#                             raw = raw.T  # (ntsrc, Lt)
#                         # tsrc average + rolling
#                         avg = np.zeros(Lt)
#                         for k in range(ntsrc):
#                             avg += np.roll(raw[k], -k * 8, axis=-1)
#                         target[cfg_idx, src, snk] = avg / ntsrc

#     # Final average over configs
#     C15 = C15.mean(axis=0)
#     C6 = C6.mean(axis=0)
#     C_direct = C_direct.mean(axis=0)

#     # Write
#     with h5py.File(output_file, "w") as f:
#         grp = f.create_group("Ptot_000_a1p")
#         grp.create_dataset("15", data=C15)
#         grp.create_dataset("6", data=C6)
#         grp.create_dataset("direct", data=C_direct)
#         grp.attrs["n_configurations"] = len(files)
#         grp.attrs["tsrc_averaged"] = True

#     print(f"\n[DONE] Merged {len(files)} configs → {output_file}")
#     print(f"       Final matrix shape: {N}×{N}×{Lt}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input_dir")
#     parser.add_argument("--output", "-o", default="merged_Dpi.h5")
#     args = parser.parse_args()
#     merge_per_pair_files(args.input_dir, args.output)

# merge_dpi_h5_minimal.py


# ZERO AVERAGING — just copy everything into cfg groups

# merge_dpi_h5_FINAL.py
#!/usr/bin/env python3
"""
Merge all per-config files into one clean file with:
merged_Dpi.h5
└── Ptot_000_a1p
    ├── cfg_3000
    ├── cfg_3040
    └── ...
No averaging, no memory explosion, no corruption.
"""

import h5py
from pathlib import Path
import argparse

def merge_configs(input_dir: str, output_file: str):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No Dpi_cfg*_matrix_*.h5 files in {input_dir}")

    print(f"[MERGE] Found {len(files)} config files → merging into {output_file}")

    # Create output file
    with h5py.File(output_file, "w", libver='latest') as fout:
        # Create parent group
        ptot_group = fout.create_group("Ptot_000_a1p")
        ptot_group.attrs["description"] = "All configs under Ptot_000_a1p"

        for i, fpath in enumerate(files):
            cfg_id = fpath.name.split("cfg")[1].split("_")[0]
            print(f"  [{i+1:2d}/{len(files)}] cfg {cfg_id} ← {fpath.name}")

            # Open source file, copy entire content under cfg_XXXX
            with h5py.File(fpath, "r") as fin:
                # Copy the whole Ptot_000_a1p group from the source file
                fin.copy("Ptot_000_a1p", ptot_group, name=f"cfg_{cfg_id}")

            # Force flush every few configs to prevent corruption
            if (i + 1) % 5 == 0:
                fout.flush()
                print(f"    → flushed to disk at cfg {cfg_id}")

        fout.flush()

    print(f"\n[SUCCESS] Merged {len(files)} configs into {output_file}")
    print("   Structure: /Ptot_000_a1p/cfg_XXXX/Ptot_000_a1p/op00_X_op00/...")
    print("   Ready for your GEVP analysis!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing Dpi_cfg*_matrix_*.h5 files")
    parser.add_argument("--output", "-o", default="merged_Dpi.h5", help="Output filename")
    args = parser.parse_args()

    merge_configs(args.input_dir, args.output)