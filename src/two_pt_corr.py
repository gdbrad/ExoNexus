# two_pt_corr.py 
import argparse
import h5py
import yaml
from pathlib import Path
from datetime import datetime
import os

from correlator_factory import CorrelatorFactory
from dimeson_factory import DiMesonOperator

def build_processor(yaml_data: dict, cfg_id: int):
    ens = list(yaml_data.keys())[0]
    settings = yaml_data[ens]
    params = settings["params"]
    global insertions_D, insertions_pi, moms_list
    ins_D  = settings["ins_D"]
    ins_pi = settings["ins_pi"]
    mom_pairs = [tuple(tuple(p) for p in pair) for pair in settings["mom_pairs"]]
    # generate the operator basis first 
    print(f"[OP] Generating operators: "
          f"{len(ins_D)} D × {len(ins_pi)} π × {len(mom_pairs)} moms = "
          f"{len(ins_D)*len(ins_pi)*len(mom_pairs)} total")
    DiMesonOperator.generate_operators(ins_D, ins_pi, mom_pairs)


    proc = CorrelatorFactory(
        ens=ens,
        cfg_id=cfg_id,
        flavor_contents=params["flavor_contents"]
    )

    three_bar = params.get("three_bar", False)
    tsrc_avg  = params.get("tsrc_avg", False)

    print(f"[BUILD] Loading distillation data for cfg {cfg_id} ...")
    proc.get_contraction_params()
    proc.load_for_system("Dpi")

    return proc, three_bar, tsrc_avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Dπ two-point correlator matrices.")
    parser.add_argument("--yaml_file", type=str, required=True, help="Path to ensemble YAML")
    parser.add_argument("--cfg_id",    type=int, required=True, help="Configuration number")
    parser.add_argument("--outdir",    type=str, default="results-multi-run", help="Output directory (default: results/)")
    args = parser.parse_args()

    yaml_path = Path(args.yaml_file)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with yaml_path.open() as f:
        yaml_data = yaml.safe_load(f)

    proc, three_bar, tsrc_avg = build_processor(yaml_data, args.cfg_id)

    # ------------------------------------------------------------------
    # Robust output directory + filename
    # ------------------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)          

    timestamp = datetime.now().strftime("%Y-%m-%d")
    ens = proc.ens

    out_file = outdir / (
        f"{ens}_Dpi_cfg{args.cfg_id:04d}_single_"
        f"n{proc.nvecs}_ntsrc{proc.ntsrc}_{timestamp}.h5"
    )

    print(f"[WRITE] Saving correlator matrix to:")
    print(f"        {out_file.resolve()}")

    # ------------------------------------------------------------------
    # Compute and write
    # ------------------------------------------------------------------
    with h5py.File(out_file, "w", libver='latest') as f:
        grp = f.create_group("Ptot_000_a1p")   
        success = CorrelatorFactory.two_pt(
            proc=proc,
            h5_group=grp,
            tsrc_avg=tsrc_avg,
            three_bar=three_bar,
        )
        if success:
            print(f"SUCCESS: Correlator matrix written to {out_file}")
        else:
            print("ERROR: Correlator computation failed!")

    # Optional: create a symlink "latest.h5" for convenience
    latest_link = outdir / "latest.h5"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(out_file.name)
    print(f"Symlink → {latest_link}")


if __name__ == "__main__":
    main()
