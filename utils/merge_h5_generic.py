
import h5py
import numpy as np
import argparse
import glob
import os

def merge_h5_files(input_files, output_file):
    """
    Merges multiple HDF5 files into a single output file by extracting only the real part of datasets.
    Each configuration gets its own dataset inside the appropriate group.
    
    Parameters:
        input_files (list): List of input HDF5 file paths.
        output_file (str): Path to the merged output file.
    """
    if not input_files:
        print("No input files provided. Exiting.")
        return

    with h5py.File(output_file, 'w') as out_f:
        for h5_file in input_files:
            cfg_id = int(os.path.basename(h5_file).split('_')[1].replace('cfg', ''))  # Extract cfg ID from filename

            with h5py.File(h5_file, 'r') as in_f:
                copy_and_store_groups(in_f, out_f, cfg_id)

    print(f"Merged {len(input_files)} files into {output_file}")

def copy_and_store_groups(in_f, out_f, cfg_id, path="/"):
    """
    Recursively copies groups and creates separate datasets for each configuration.

    Parameters:
        in_f (h5py.File): Input HDF5 file object.
        out_f (h5py.File): Output HDF5 file object.
        cfg_id (int): Configuration ID extracted from filename.
        path (str): Current path in the HDF5 file structure.
    """
    for key in in_f[path]:
        in_obj = in_f[path][key]
        out_path = os.path.join(path, key)  

        if isinstance(in_obj, h5py.Group):
            if out_path not in out_f:
                out_f.create_group(out_path)
            copy_and_store_groups(in_f, out_f, cfg_id, out_path)
        
        elif isinstance(in_obj, h5py.Dataset):
            real_data = in_obj[()]

            # if real_data.shape[-1] != 96:
            #     print(f"Warning: Unexpected dataset shape {real_data.shape} in {out_path}, skipping.")
            #     return
            
            cfg_dataset_name = f"{out_path}/cfg{cfg_id}"
            if cfg_dataset_name not in out_f:
                out_f.create_dataset(cfg_dataset_name, data=real_data, dtype=real_data.dtype)
            else:
                print(f"Warning: Dataset {cfg_dataset_name} already exists. Skipping duplicate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge HDF5 files while extracting only the real part of data.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing HDF5 files")
    parser.add_argument("--output", type=str, required=True, help="Output merged HDF5 file")
    args = parser.parse_args()
    
    input_files = sorted(glob.glob(os.path.join(args.input, "*.h5"))) 
    if not input_files:
        print("No HDF5 files found in the input directory.")
    else:
        merge_h5_files(input_files, args.output)
