# two_pt_corr.py
import argparse
import h5py
import yaml
from pathlib import Path
from datetime import datetime

from file_io import DistillationObjectsIO         
from correlator_factory import CorrelatorFactory


# ----------------------------------------------------------------------
# Helper: build a CorrelatorFactory from the YAML + cfg_id
# ----------------------------------------------------------------------
def build_processor(yaml_data: dict, cfg_id: int) -> tuple[CorrelatorFactory, bool, bool]:
    """
    Returns (proc, three_bar, tsrc_avg)
    """
    # ------------------------------------------------------------------
    # 1. Extract ensemble name and all parameters from the YAML
    # ------------------------------------------------------------------
    ens = list(yaml_data.keys())[0]                
    settings = yaml_data[ens]
    params = settings["params"]
    flavor_contents = params["flavor_contents"]
    three_bar       = params.get("three_bar", False)
    tsrc_avg        = params.get("tsrc_avg",   False)

    
    # Build the processor that the correlator code expects
    # ------------------------------------------------------------------
    proc = CorrelatorFactory(
        ens=ens,
        cfg_id=cfg_id,
        flavor_contents=flavor_contents,
        nvecs=params["nvecs"],
        lt=params["lt"],
        ntsrc=params["ntsrc"],
        tsrc_step=params.get("tsrc_step", 8),
        data1=False,
)
    # Load data into the SAME object
    print(f"[BUILD] Loading data into proc (cfg {cfg_id})...")
    proc.get_contraction_params()   # ensures consistency
    proc.load_for_system("Dpi")     

    return proc, three_bar, tsrc_avg


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Dπ two-point correlator matrices."
    )
    parser.add_argument(
        "--yaml_file", type=str, required=True,
        help="Path to the *.ini.yml that defines the ensemble."
    )
    parser.add_argument(
        "--cfg_id", type=int, required=True,
        help="Configuration index (e.g. 100)."
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml_file)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with yaml_path.open() as f:
        yaml_data = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Build the processor (all heavy lifting happens here)
    # ------------------------------------------------------------------
    proc, three_bar, tsrc_avg = build_processor(yaml_data, args.cfg_id)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    out_file = (
        f"{proc.ens}_Dpi_cfg{args.cfg_id}_corrmatrix_"
        f"nvec_{proc.nvecs}_tsrc_{proc.ntsrc}_{timestamp}.h5"
    )

    # ------------------------------------------------------------------
    # Run the correlator and write the HDF5
    # ------------------------------------------------------------------
    with h5py.File(out_file, "w") as f:
        #h5_group = h5py.create_group('Dpi_a1u')
        grp = f.create_group("Dpi-corrmatrix")
        CorrelatorFactory.two_pt(
            proc,
            grp,
            tsrc_avg=tsrc_avg,
            three_bar=three_bar,
        )

    print(f"Finished - results written to {out_file}")


if __name__ == "__main__":
    main()