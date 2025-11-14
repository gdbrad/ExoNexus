# merge_per_pair_h5.py
import h5py
import os
import argparse
from typing import List, Dict

def merge_per_pair_files(cfg_id: int, output_dir: str = "."):
    """
    Merge all per-pair HDF5 files for a given cfg into one single organized HDF5 file
    
    Structure:
    Dpi_cfg3000_combined.h5
        /D_000_rho_nabla_a1p_X_pi_000_rho_nabla_a1p/
            /direct
            /crossing
            /15
            /6
            /3_bar  (if exists)
            /disconnected  (if exists)
        /D_000_rho_nabla_a1p_X_pi_001_rho_nabla_a1p/
            ...
        ...
    
    Args:
        cfg_id: Configuration ID (e.g., 3000)
        output_dir: Directory where the per-pair files are located (default: current dir)
    """
    print(f"[MERGE] Starting merge for cfg{cfg_id:04d} in {output_dir}")

    # Find all per-pair files
    pattern = f"Dpi_cfg{cfg_id:04d}_*_X_*.h5"
    pair_files = [f for f in os.listdir(output_dir) if f.startswith(f"Dpi_cfg{cfg_id:04d}_") and f.endswith('.h5') and '_X_' in f]
    if not pair_files:
        print(f"[ERROR] No per-pair files found for cfg{cfg_id:04d}")
        return

    print(f"[MERGE] Found {len(pair_files)} pair files to merge")

    # Output file
    out_file = f"Dpi_cfg{cfg_id:04d}_combined.h5"
    out_path = os.path.join(output_dir, out_file)
    if os.path.exists(out_path):
        print(f"[WARN] Output file {out_file} already exists – overwriting")
        os.remove(out_path)

    with h5py.File(out_path, "w") as f_out:
        for pair_file in pair_files:
            pair_path = os.path.join(output_dir, pair_file)
            pair_name = pair_file.replace(f"Dpi_cfg{cfg_id:04d}_", "").replace(".h5", "")
            print(f"  → Merging {pair_name} from {pair_file}")

            with h5py.File(pair_path, "r") as f_in:
                if 'direct' in f_in:
                    f_out.create_dataset(f"{pair_name}/direct", data=f_in['direct'][()])
                if 'crossing' in f_in:
                    f_out.create_dataset(f"{pair_name}/crossing", data=f_in['crossing'][()])
                if '15' in f_in:
                    f_out.create_dataset(f"{pair_name}/15", data=f_in['15'][()])
                if '6' in f_in:
                    f_out.create_dataset(f"{pair_name}/6", data=f_in['6'][()])
                if '3_bar' in f_in:
                    f_out.create_dataset(f"{pair_name}/3_bar", data=f_in['3_bar'][()])
                if 'disconnected' in f_in:
                    f_out.create_dataset(f"{pair_name}/disconnected", data=f_in['disconnected'][()])

    print(f"[DONE] All pairs merged into {out_file} ({os.path.getsize(out_path) / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-pair HDF5 files into one combined file.")
    parser.add_argument("--cfg_id", type=int, required=True, help="Configuration ID (e.g., 3000)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory with per-pair files")
    args = parser.parse_args()
    merge_per_pair_files(args.cfg_id, args.output_dir)