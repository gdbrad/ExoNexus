import argparse
import h5py
import yaml
from pathlib import Path
from datetime import datetime
from distillation_data import DistillationData
from single_meson_corr import SingleMesonCorrelator
from meson_factory import MesonFactory

def make_run_dir(base_dir: Path, ensemble: str, mode: str) -> Path:
    """Create a timestamped run directory with attempt suffixes to avoid overwriting."""
    base_run_dir = base_dir / ensemble / mode
    base_run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = base_run_dir / f"run-{date_str}"
    attempt = 1
    while run_dir.exists():
        run_dir = base_run_dir / f"run-{date_str}_{attempt}"
        attempt += 1

    run_dir.mkdir()
    return run_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True, help="Path to ensemble YAML file")
    parser.add_argument("--cfg_id", type=int, required=True, help="Configuration ID")
    parser.add_argument("--tmp_base", default="./tmp", help="Base temporary directory")
    args = parser.parse_args()

    # Load YAML
    yaml_path = Path(args.yaml_file)
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    ensemble_name = list(yaml_data.keys())[0]
    ensemble_cfg = yaml_data[ensemble_name]
    operators = ensemble_cfg["operators"]

    # Init correlator factory
    proc = DistillationData(ensemble_name, args.cfg_id,args.yaml_file)
    proc.load_single_meson()

    # Create run directories
    tmp_base = Path(args.tmp_base)
    run_dir = make_run_dir(tmp_base, ensemble_name, "contractions")
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    outfile = run_dir / f"{ensemble_name}_cfg{args.cfg_id:04d}.h5"
    if outfile.exists():
        print(f"[SKIP] {outfile} already exists")
        return

    print(f"[INFO] Writing single-meson matrix to {outfile}")
    with h5py.File(outfile, "w") as f_cfg:
        f_cfg.attrs["ensemble"] = ensemble_name
        f_cfg.attrs["cfg_id"] = args.cfg_id
        f_cfg.attrs["nvecs"] = proc.nvecs
        f_cfg.attrs["nt"] = proc.lt
        f_cfg.attrs["ntsrc"] = proc.ntsrc

        for irrep_name, irrep_settings in operators.items():
            irrep = irrep_settings["irrep"]
            mesons = irrep_settings["mesons"]
            insertions = irrep_settings["insertions"]

            print(f"[INFO] {irrep}")
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

    print(f"[DONE] cfg {args.cfg_id:04d} complete in {run_dir}")

if __name__ == "__main__":
    main()