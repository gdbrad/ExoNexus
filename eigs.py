import numpy as np
import re
from pathlib import Path
import glob

def parse_eigenvalues(file_path):
    """Parse eigenvalues from chroma stdout into a 64x96 array (timeslices x eigenvalues).
    modified cpp routine is here https://github.com/gdbrad/chroma_sdb-h5-convert/blob/main/inline_create_colorvecs_superb.cc
    """
    eigenvalues = []
    pattern = r"Eigenvalue for t=\s*\d+\s*:\s*(-?\d*\.?\d+)"
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                eigenvalues.append(np.abs(float(match.group(1))))
    
    n_timeslices = 64
    n_eigenvalues = 96
    expected_length = n_timeslices * n_eigenvalues
    
    if len(eigenvalues) != expected_length:
        raise ValueError(f"Expected {expected_length} eigenvalues in {file_path}, got {len(eigenvalues)}")
    
    return np.array(eigenvalues).reshape(n_timeslices, n_eigenvalues)

def save_all_configs(file_paths, output_file="all_configs_eigenvalues.npy"):
    """
    Process multiple config files and save all eigenvalues into a 3D numpy array.
    Shape: (n_configs, n_timeslices, n_eigenvalues)
    """
    all_configs = []
    n_timeslices = 64
    n_eigenvalues = 96
    
    for file_path in sorted(file_paths): 
        try:
            config_eigs = parse_eigenvalues(file_path)
            all_configs.append(config_eigs)
            print(f"processed {file_path}: {config_eigs.shape}")
        except Exception as e:
            print(f"error processing {file_path}: {str(e)}")
    
    if not all_configs:
        print("no valid data")
        return
    
    # convert to array w size  (n_configs, 64, 96)
    result_array = np.array(all_configs)
    
    # save to .npy file
    np.save(output_file, result_array)
    print(f"Saved eigenvalues for {len(all_configs)} configs to {output_file}")
    print(f"Output shape: {result_array.shape}")
    print(f"Sample (first config, first timeslice, first 5 values): {result_array[0, 0, :5]}")

if __name__ == "__main__":
    file_paths = glob.glob("eigs/eigs*.out")
    if not file_paths:
        print("No .out files found in /p/scratch/exotichadrons/profiles/")
    else:
        save_all_configs(file_paths, output_file="all_configs_eigenvalues.npy")