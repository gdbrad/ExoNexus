import numpy as np
import h5py
import glob
import os
import argparse
from pathlib import Path

def discover_operator_basis(example_file, irrep_name):
    with h5py.File(example_file, "r") as f:
        pairs = list(f[irrep_name].keys())

    ops = set()
    for name in pairs:
        if "__X__" not in name:
            continue
        op1, op2 = name.split("__X__")
        ops.add(op1)
        ops.add(op2)

    return sorted(list(ops))

def build_Cij(stage1_dir, irrep_name, output_file):
    # Match prefixes before 'cfg' (e.g. b3.4_s32t64_cfg0900.h5)
    cfg_files = sorted(glob.glob(os.path.join(stage1_dir, "*cfg*.h5")))
    
    ncfg = len(cfg_files)
    if ncfg == 0:
        raise RuntimeError(f"No *cfg*.h5 files found in {stage1_dir}")

    operator_list = discover_operator_basis(cfg_files[0], irrep_name)
    nops = len(operator_list)
    op_index = {op: i for i, op in enumerate(operator_list)}

    with h5py.File(cfg_files[0], "r") as f:
        first_dataset = next(v for k, v in f[irrep_name].items() if "__X__" in k)
        Ntsrc, LT = first_dataset.shape
        
    print(f"Found {ncfg} configs")
    print(f"Operators: {nops}")
    
    C = np.zeros((ncfg, Ntsrc, nops, nops, LT), dtype=np.complex128)

    for cfg_idx, fname in enumerate(cfg_files):
        print(f"[{cfg_idx+1}/{ncfg}] {fname}")
        with h5py.File(fname, "r") as f:
            for pair_name, dataset in f[irrep_name].items():
                if "__X__" not in pair_name:
                    continue
                op1, op2 = pair_name.split("__X__")
                i = op_index[op1]
                j = op_index[op2]
                
                data = dataset[:]  # (Ntsrc, Lt)
                C[cfg_idx, :, i, j, :] = data
                
                # Reconstruct full hermitian matrix (conjugate the symmetric element)
                if i != j:
                    C[cfg_idx, :, j, i, :] = np.conjugate(data)

    with h5py.File(output_file, "w") as f:
        f.create_dataset("Cij", data=C)
        f.create_dataset("operators", data=np.array(operator_list, dtype="S"))
        
    print(f"Stage 2 complete for block {irrep_name}.")

def discover_blocks(example_file):
    """
    Return list of block names in a Stage-1 file.
    """
    with h5py.File(example_file, "r") as f:
        return list(f.keys())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", required=True,
                        help="Directory containing *cfg*.h5 files")
    parser.add_argument("--outdir", required=True,
                        help="Output dir for Stage-2 files")
    parser.add_argument("--block", default=None,
                        help="Specific block (irrep_name) to process if null then process all blocks")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_files = sorted(glob.glob(os.path.join(args.stage1_dir, "*cfg*.h5")))
    if len(cfg_files) == 0:
        raise RuntimeError(f"No *cfg*.h5 files found in {args.stage1_dir}")
        
    example_file = str(cfg_files[0])

    if args.block is not None:
        blocks = [args.block]
    else:
        blocks = discover_blocks(example_file)
        
    print("Blocks to process:")
    for b in blocks:
        print("  ", b)

    for block_name in blocks:
        print("\n=====================================")
        print(f"Processing block: {block_name}")
        print("=====================================")
        output_file = outdir / f"Cij_{block_name}.h5"
        
        build_Cij(
            stage1_dir=str(stage1_dir),
            irrep_name=block_name,
            output_file=str(output_file)
        )
        
    print("\nStage 2 complete.")

if __name__ == "__main__":
    main()