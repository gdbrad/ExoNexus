import h5py
import os
import glob
import re
import argparse

def merge_h5_files(source_file, addon_file, output_file):
    """
    Merges momentum groups from addon_file into source_file under matching t_slice groups.
    - source_file: Path to the original meson file
    - addon_file: Path to the meson2 file
    - output_file: Path to the merged output file
    """
    with h5py.File(source_file, 'r') as src, h5py.File(addon_file, 'r') as add:
        with h5py.File(output_file, 'w') as out:
            # Copy everything from source to output
            for key in src.keys():
                src.copy(key, out)
            
            # Add momentum groups from addon, per t_slice
            for t_slice in add.keys():
                if t_slice not in out:
                    print(f"Warning: t_slice '{t_slice}' not found in source for {source_file}. Skipping t_slice.")
                    continue
                
                src_t_group = out[t_slice]
                add_t_group = add[t_slice]
                
                for mom_group in add_t_group.keys():
                    if mom_group in src_t_group:
                        print(f"Warning: Momentum group '{mom_group}' already exists in {t_slice} for {source_file}. Skipping.")
                        continue
                    
                    # Copy the entire momentum group
                    add_t_group.copy(mom_group, src_t_group)
    
    print(f"Merge complete: {output_file}")

def find_matching_configs(directory, ensemble):
    """
    Finds configs in directory where both meson-{ensemble}_cfg*.h5 and meson2-{ensemble}_cfg*.h5 exist.
    Returns a list of configs (e.g., '1000').
    """
    meson_pattern = os.path.join(directory, f"meson-{ensemble}_cfg*.h5")
    meson2_pattern = os.path.join(directory, f"meson2-{ensemble}_cfg*.h5")
    
    meson_files = glob.glob(meson_pattern)
    meson2_files = glob.glob(meson2_pattern)
    
    config_regex = re.compile(r'cfg(\d+)\.h5$')
    meson_configs = {config_regex.search(f).group(1) for f in meson_files if config_regex.search(f)}
    meson2_configs = {config_regex.search(f).group(1) for f in meson2_files if config_regex.search(f)}
    matching_configs = sorted(meson_configs.intersection(meson2_configs))
    print(f"Found {len(matching_configs)} matching configs for ensemble '{ensemble}': {matching_configs}")
    
    return matching_configs

def main():
    parser = argparse.ArgumentParser(description="Merge meson and meson2 H5 files for an ensemble.")
    parser.add_argument("ensemble", help="Ensemble name (e.g., '64')")
    parser.add_argument("--directory", default="dpi-data", help="Directory containing the H5 files (default: dpi-data)")
    parser.add_argument("--output_dir", default=".", help="Output directory for merged files (default: current dir)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' not found.")
        return
    
    configs = find_matching_configs(args.directory, args.ensemble)
    
    for config in configs:
        source = os.path.join(args.directory, f"meson-{args.ensemble}_cfg{config}.h5")
        addon = os.path.join(args.directory, f"meson2-{args.ensemble}_cfg{config}.h5")
        output = os.path.join(args.output_dir, f"merged_meson-{args.ensemble}_cfg{config}.h5")
        
        if os.path.exists(source) and os.path.exists(addon):
            merge_h5_files(source, addon, output)
        else:
            print(f"Skipping config {config}: Files not found.")

if __name__ == "__main__":
    main()