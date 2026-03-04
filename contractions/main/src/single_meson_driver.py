import argparse
import h5py
import yaml
from pathlib import Path
from datetime import datetime
from distillation_data import DistillationData
from single_meson_corr import SingleMesonCorrelator
from meson_factory import MesonFactory

def make_run_dir(base_dir: Path, ensemble_short: str, mode: str = "contractions") -> Path:
    """
    Create a timestamped run directory with attempt suffixes to avoid overwriting.
    If base_dir already ends with ensemble_short, do not append it again.
    """
    base_dir = base_dir.resolve()
    if base_dir.name != ensemble_short:
        base_run_dir = base_dir / ensemble_short / mode
    else:
        base_run_dir = base_dir / mode

    base_run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = base_run_dir / f"run-{date_str}"
    attempt = 1
    while run_dir.exists():
        run_dir = base_run_dir / f"run-{date_str}_{attempt}"
        attempt += 1

    # Create subdirs
    run_dir.mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "correlators").mkdir()
    (run_dir / "scripts").mkdir()  # optional

    return run_dir
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True, help="Path to ensemble YAML file")
    parser.add_argument("--cfg_id", type=int, required=True, help="Configuration ID")
    parser.add_argument("--tmp_base", default="/p/project1/exflash/contractions-tmp",
                        help="Base directory for temporary outputs")
    args = parser.parse_args()

    # Resolve YAML path to absolute
    yaml_path = Path(args.yaml_file).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load YAML
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    # Determine ensemble short name
    # Supports both flat YAML or nested under ensemble_short
    if "ensemble" in yaml_data and "short" in yaml_data["ensemble"]:
        ensemble_short = yaml_data["ensemble"]["short"]
        operators = yaml_data["operators"]
        paths = yaml_data.get("paths", {})
    else:
        # Nested under ensemble name
        ensemble_short = list(yaml_data.keys())[0]
        ensemble_cfg = yaml_data[ensemble_short]
        operators = ensemble_cfg["operators"]
        paths = ensemble_cfg.get("paths", {})

    # Base path for run directories
    base_path = Path(paths.get("base_path", args.tmp_base)).resolve()
    run_dir = make_run_dir(base_path, ensemble_short, mode="contractions")
    log_dir = run_dir / "logs"
    corr_dir = run_dir / "correlators"

    # Initialize distillation data
    proc = DistillationData(ensemble_short, str(yaml_path), args.cfg_id)
    proc.load_single_meson()

    # HDF5 output file
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
    print(f"[INFO] Correlators written to: {corr_dir}")
    print(f"[INFO] Logs in: {log_dir}")

if __name__ == "__main__":
    main()