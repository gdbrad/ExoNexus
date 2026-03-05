import argparse
import h5py
import yaml
from pathlib import Path
from datetime import datetime
from distillation_data import DistillationData
from single_meson_corr import SingleMesonCorrelator
from meson_factory import MesonFactory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True, help="Path to ensemble YAML file")
    parser.add_argument("--cfg_id", type=int, required=True, help="Configuration ID")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for correlator HDF5 files")
    args = parser.parse_args()

    # Load YAML
    yaml_path = Path(args.yaml_file).resolve()  # ensure absolute path
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    # Top-level key is ensemble name
    ensemble_name = list(yaml_data.keys())[0]
    ensemble_cfg = yaml_data[ensemble_name]
    operators = ensemble_cfg.get("operators", {})
    if not operators:
        raise ValueError("No operators found in YAML file under ensemble key")

    # Base path for temporary outputs
    base_path = Path(ensemble_cfg["paths"]["base_path"])
    ensemble_short = ensemble_cfg["ensemble"]["short"]

    # Initialize distillation data
    proc = DistillationData(ensemble_short, args.yaml_file, args.cfg_id)
    proc.load_single_meson()

    # Use the output directory provided by the launcher
    corr_dir = Path(args.outdir)
    corr_dir.mkdir(parents=True, exist_ok=True)

    # HDF5 output path
    outfile = corr_dir / f"{ensemble_short}_cfg{args.cfg_id:04d}.h5"
    if outfile.exists():
        print(f"[SKIP] {outfile} already exists")
        return

    print(f"[INFO] Writing single-meson matrix to {outfile}")

    with h5py.File(outfile, "w") as f_cfg:
        f_cfg.attrs["ensemble"] = ensemble_short
        f_cfg.attrs["cfg_id"] = args.cfg_id
        f_cfg.attrs["nvecs"] = proc.nvecs
        f_cfg.attrs["nt"] = proc.lt
        f_cfg.attrs["ntsrc"] = proc.ntsrc

        for irrep_name, irrep_settings in operators.items():
            irrep = irrep_settings["irrep"]
            mesons = irrep_settings["mesons"]
            insertions = irrep_settings["insertions"]

            print(f"[INFO] Processing irrep: {irrep}")
            factory = MesonFactory()
            ops = factory.generate(
                meson=mesons,
                insertions=insertions,
                momentum=(0, 0, 0),
                irrep=irrep
            )

            n_ops = len(ops)
            print(f"[INFO] {irrep}: {n_ops} operators")
            print(f"[INFO] Computing Hermitian-reduced {n_ops}x{n_ops} matrix")

            grp_irrep = f_cfg.create_group(irrep_name)
            grp_irrep.attrs["irrep"] = irrep
            grp_irrep.attrs["n_ops"] = n_ops
            for op in ops:
                grp_irrep.attrs[f"op_{op.short}"] = op.name

            # Hermitian-reduced matrix
            for i, op_A in enumerate(ops):
                for j in range(i, n_ops):
                    op_B = ops[j]
                    dataset_name = f"{op_A.short}__X__{op_B.short}"
                    try:
                        C = SingleMesonCorrelator.two_pt_corr(
                            proc=proc,
                            op_src=op_A,
                            op_snk=op_B
                        )
                        grp_irrep.create_dataset(dataset_name, data=C)
                    except Exception as e:
                        print(f"[ERROR] {dataset_name}: {e}")

    print(f"[DONE] cfg {args.cfg_id:04d} complete in {corr_dir}")
    print(f"[INFO] Correlators successfully saved in {corr_dir}")


if __name__ == "__main__":
    main()