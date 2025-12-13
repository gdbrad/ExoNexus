#!/usr/bin/env python3
import argparse
import h5py
import yaml
from pathlib import Path

from correlator_factory import CorrelatorFactory
from dimeson_factory import DiMesonFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--cfg_id", type=int, required=True)
    parser.add_argument("--outdir", default="results_per_op")
    args = parser.parse_args()

    yaml_path = Path(args.yaml_file)
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    ens = list(yaml_data.keys())[0]
    settings = yaml_data[ens]
    params = settings["params"]
    system = settings["system"]

    proc = CorrelatorFactory(
        ens=ens,
        cfg_id=args.cfg_id,
        flavor_contents=params["flavor_contents"],
        nvecs=params["nvecs"],
        lt=params["lt"],
        ntsrc=params["ntsrc"],
        tsrc_step=params.get("tsrc_step", 8),
    )
    proc.load_for_system(system)

    outdir = Path(args.outdir) / f"cfg{args.cfg_id:04d}"
    outdir.mkdir(parents=True, exist_ok=True)
    print("DEBUG: mom_pairs from YAML:", settings["mom_pairs"])
    print("DEBUG: type:", type(settings["mom_pairs"]))
    print("DEBUG: len:", len(settings["mom_pairs"]))

    momentum_pairs = []
    for pair in settings["mom_pairs"]:
        p1 = tuple(pair[0])   # e.g. [0,0,0] → (0,0,0)
        p2 = tuple(pair[1])   # e.g. [0,0,0] → (0,0,0)
        momentum_pairs.append((p1, p2))

    print(f"DEBUG: Converted {len(momentum_pairs)} momentum pairs")

    # Generate operators
    factory = DiMesonFactory()
    factory.generate(
        meson1_list=settings.get("meson1", []),
        meson2_list=settings.get("meson2", []),
        insertions1=settings.get("ins_1", []),      
        insertions2=settings.get("ins_2", []),     
        momentum_pairs=momentum_pairs,
        irrep=settings.get("irrep", "a1u")
    )

    # Save lookup table
    lookup = {pair_short: pair_full for _, _, pair_short, pair_full in factory.pairs}
    yaml.dump(lookup, open(outdir / "operator_lookup.yaml", "w"), default_flow_style=False)

    print(f"[INFO] Computing {len(factory.pairs)} operator pairs")

    for op1, op2, pair_short, pair_full in factory.pairs:
        filename = f"{pair_short}.h5"  # → op000Dxop000PI.h5
        outfile = outdir / filename

        with h5py.File(outfile, "w") as f:
            grp = f.create_group("corr")

            success = CorrelatorFactory.two_pt_single_dimeson(
                proc=proc,
                op1_src=op1,
                op2_src=op2,
                op1_snk=op1,
                op2_snk=op2,
                h5_group=grp,
                tsrc_avg=False
            )

            if success:
                grp.attrs["pair"] = pair_short
                grp.attrs["full_name"] = pair_full
                print(f"→ {filename}")
            else:
                print(f"→ [FAILED] {filename}")
                outfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()