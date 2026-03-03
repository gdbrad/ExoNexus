
"""
Merge all per-config files into one clean file with:
merged_Dpi.h5
└── Ptot_000_a1p
    ├── cfg_1000
    ├── cfg_1080
    └── ...
"""

import h5py
from pathlib import Path
import argparse
import shutil
import os 

def merge_configs(input_dir: str, output_file: str,move:bool=True):
    input_dir = Path(input_dir)
    output_file = os.path.join(input_dir,'merged_Dpi.h5')
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
    print("   Structure: /Ptot_000_a1p/cfg_XXXX/op00_X_op00/...")
    print("   Ready to be tsrc and gauge averaged!")

    if move:
        processed_dir = input_dir / "processed_configs_h5"
        processed_dir.mkdir(exist_ok=True)
        
        moved = 0
        for fpath in files:
            shutil.move(str(fpath), str(processed_dir / fpath.name))
            moved += 1
        
        print(f"[CLEANUP] Moved {moved} individual config files to {processed_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing Dpi_cfg*_matrix_*.h5 files")
    parser.add_argument("--output", "-o", default="merged_Dpi.h5", help="Output filename")
    args = parser.parse_args()

    merge_configs(args.input_dir, args.output)