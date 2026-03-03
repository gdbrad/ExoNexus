import numpy as np
import h5py
import re
from pathlib import Path
import glob

def parse_eigenvalues(file_path):
    eigenvalues = []
    pattern = r"Eigenvalue for t=\s*\d+\s*:\s*(-?\d*\.?\d+)"
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                eigenvalues.append(float(match.group(1)))
    
    eig_array = np.array(eigenvalues)
    #eig_array_sorted = np.sort(eig_array)[::-1]
    return eig_array

def save_to_numpy(file_paths, output_dir="numpy_output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for file_path in file_paths:
        try:
            eigenvalues = parse_eigenvalues(file_path)
            output_file = Path(output_dir) / f"{Path(file_path).stem}_eigenvalues.npy"
            if len(eigenvalues)<6144:
                print('skipping')
            else:
                np.save(output_file, eigenvalues)
                print(f"Saved {len(eigenvalues)} eigenvalues to {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    file_paths = glob.glob("eigs/*.out")
    if not file_paths:
        print("No .out files found")
    else:
        save_to_numpy(file_paths, output_dir="numpy_output")