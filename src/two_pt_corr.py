import argparse
import h5py 
import yaml
from pathlib import Path 
from correlator_factory import CorrelatorFactory
import file_io
# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------
def correlator_processor(yaml_data:dict,cfg_id:int) -> CorrelatorFactory:
    ens = list(yaml_data.keys())[0]
    settings = yaml_data[ens]
    params = settings["params"]
    flavor_contents = params["flavor_contents"]
    three_bar = params.get("three_bar", False)
    tsrc_avg = params.get("tsrc_avg", False)
    
    tmp = file_io.DistillationObjectsIO(ens=ens)
    tmp.get_contraction_params()

    return CorrelatorFactory(
            ens=ens,
            cfg_id=cfg_id,
            flavor_contents=flavor_contents,
            nvecs=tmp.nvecs,
            lt=tmp.lt,
            ntsrc=tmp.ntsrc,
            tsrc_step=tmp.tsrc_step,
            data1=False,  # Default; can be added to YAML if needed
        ), three_bar, tsrc_avg

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute single- or di-meson two-point correlators."
    )
    parser.add_argument("--yaml_file", type=str, required=True, help="Path to the input YAML file")
    parser.add_argument("--cfg_id", type=int, required=True, help="Configuration index")
    args = parser.parse_args()
    
    # Load YAML data
    yaml_path = Path(args.yaml_file)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No YAML file found at {yaml_path}")
    with yaml_path.open() as f:
        yaml_data = yaml.safe_load(f)
    
    proc, three_bar, tsrc_avg = correlator_processor(yaml_data, args.cfg_id)
    system_name = proc.get_meson_system_name()
    out_file = f"{proc.ens}_{system_name}_cfg{args.cfg_id}_2pt_nvec_{proc.nvecs}_tsrc_{proc.ntsrc}_test.h5"
    
    with h5py.File(out_file, "w") as f:
        grp = f.create_group(f"{system_name}_000")
        CorrelatorFactory.two_pt(
            proc,
            grp,
            tsrc_avg=tsrc_avg,
            three_bar=three_bar,
        )
    print(f"Finished - results in {out_file}")

if __name__ == "__main__":
    main()