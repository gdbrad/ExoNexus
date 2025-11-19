# merge_per_pair_h5.py
#!/usr/bin/env python3
"""
Correctly merge per-pair files into one combined file with 6 source groups.
Your filenames are:
  Dpi_cfg3000_D_000_pi_none_a1pXpi_000_pi_none_a1p_X_D_100_pi2_none_a1pXpi_-100_pi2_none_a1p.h5
"""

import h5py
import os
import argparse
from pathlib import Path

def extract_clean_dimeson_name(raw_part: str) -> str:
    """
    From a raw part like:
      D_000_pi_none_a1pXpi_000_pi_none_a1p
    or
      D_100_pi2_none_a1pXpi_-100_pi2_none_a1p
    return the clean DiMeson name without momentum digits at the end of the pi part.
    """
    # Split into D part and pi part
    if "Xpi_" not in raw_part:
        return raw_part  # fallback

    d_part, pi_part = raw_part.split("Xpi_", 1)
    # Remove momentum from pi part (everything after the last underscore before momentum)
    # pi part looks like: 000_pi_none_a1p  or  -100_pi2_none_a1p
    if "_" in pi_part:
        pi_base = pi_part.split("_", 1)[1]  # drop the momentum prefix
    else:
        pi_base = pi_part

    return f"{d_part}Xpi_{pi_base}"

def merge_per_pair_files(cfg_id: int, input_dir: str = ".", output_dir: str = "."):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"Dpi_cfg{cfg_id:04d}_*.h5"
    pair_files = sorted(input_dir.glob(pattern))
    if not pair_files:
        print(f"[ERROR] No files found for cfg {cfg_id}")
        return

    print(f"[MERGE] Found {len(pair_files)} per-pair files")

    out_file = output_dir / f"Dpi_cfg{cfg_id:04d}_combined.h5"
    if out_file.exists():
        print(f"[INFO] Removing existing {out_file.name}")
        out_file.unlink()

    with h5py.File(out_file, "w") as f_out:
        big_group = f_out.create_group("Ptot_000_a1p")
        print("[MERGE] Building 6 source → 6 sink hierarchy...")

        for pf in pair_files:
            raw = pf.name.replace(f"Dpi_cfg{cfg_id:04d}_", "").replace(".h5", "")

            # Split on the single "_X_" that separates source and sink
            if "_X_" not in raw:
                print(f"[WARN] No '_X_' in {pf.name} – skipping")
                continue

            src_raw, snk_raw = raw.split("_X_", 1)

            src_name = extract_clean_dimeson_name(src_raw)
            snk_name = extract_clean_dimeson_name(snk_raw)

            print(f"  → {pf.name}")
            print(f"      src = {src_name}")
            print(f"      snk = {snk_name}")

            # src_group = f_out.require_group(f"src_{src_name}")
            # snk_group = src_group.require_group(f"snk_{snk_name}")
            pair_group = big_group.require_group(f"{src_name}_X_{snk_name}")

            with h5py.File(pf, "r") as f_in:
                for ds_name in f_in.keys():
                    f_in.copy(f_in[ds_name], pair_group, name=ds_name)

    size_mb = out_file.stat().st_size / (1024**2)
    print(f"\n[DONE] Combined file: {out_file} ({size_mb:.1f} MB)")
    print("   6 source groups, each with 6 sink subgroups → 36 datasets total")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-pair files (6 src × 6 snk)")
    parser.add_argument("--cfg_id", type=int, required=True)
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    merge_per_pair_files(args.cfg_id, args.input_dir, args.output_dir)