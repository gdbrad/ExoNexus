import argparse
import h5py
import yaml
from pathlib import Path

from correlator_factory import CorrelatorFactory
from meson_factory import MesonFactory
from dimeson_factory import DiMesonFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--cfg_id", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    # -----------------------------------------------------
    # Load YAML
    # -----------------------------------------------------
    yaml_path = Path(args.yaml_file)
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    ens = list(yaml_data.keys())[0]
    settings = yaml_data[ens]

    params = settings["params"]
    system = settings["system"]
    operators = settings["operators"]

    # -----------------------------------------------------
    # Initialize correlator factory
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # One file per configuration
    # -----------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"cfg{args.cfg_id:04d}.h5"

    if outfile.exists():
        print(f"[SKIP] {outfile} already exists")
        return

    print(f"[INFO] Writing full matrix to {outfile}")

    # =====================================================
    # Open configuration file
    # =====================================================
    with h5py.File(outfile, "w") as f_cfg:

        f_cfg.attrs["ensemble"] = ens
        f_cfg.attrs["cfg_id"] = args.cfg_id
        f_cfg.attrs["nvecs"] = params["nvecs"]
        f_cfg.attrs["nt"] = params["lt"]
        f_cfg.attrs["ntsrc"] = params["ntsrc"]

        # =====================================================
        # Loop over irrep blocks
        # =====================================================
        for ptot_irrep in operators:

            ptot_irrep_settings = operators[ptot_irrep]

            irrep = ptot_irrep_settings["irrep"]
            ptot = int(ptot_irrep_settings["ptot"])

            if ptot != 0:
                raise NotImplementedError("Only P=0 supported")

            meson1 = ptot_irrep_settings["meson1"]
            meson2 = ptot_irrep_settings["meson2"]
            ins1 = ptot_irrep_settings["ins_1"]
            ins2 = ptot_irrep_settings["ins_2"]

            # -------------------------------------------------
            # Extract single-meson momenta
            # -------------------------------------------------
            raw_momenta = [
                tuple(mom_pair[0])
                for mom_pair in ptot_irrep_settings["mom_pairs"]
            ]

            print(f"[INFO] {irrep}: raw momenta count = {len(raw_momenta)}")

            # -------------------------------------------------
            # Generate projected operators
            # -------------------------------------------------
            factory = DiMesonFactory()

            factory.generate_projected_zero_momentum(
                meson1_list=meson1,
                meson2_list=meson2,
                insertions1=ins1,
                insertions2=ins2,
                momentum_list=raw_momenta,
                irrep=irrep
            )

            pairs = factory.pairs
            n_ops = len(pairs)

            print(f"[INFO] {irrep}: {n_ops} projected operators")
            print(f"[INFO] Computing Hermitian-reduced {n_ops}x{n_ops} matrix")

            # -------------------------------------------------
            # Create irrep group
            # -------------------------------------------------
            grp_irrep = f_cfg.create_group(irrep)
            grp_irrep.attrs["ptot"] = ptot
            grp_irrep.attrs["n_ops"] = n_ops

            # Store operator lookup inside H5
            for _, _, short, full in pairs:
                grp_irrep.attrs[f"op_{short}"] = full

            # -------------------------------------------------
            # Hermitian-reduced matrix (upper triangle only)
            # -------------------------------------------------
            for i, A in enumerate(pairs):
                for j in range(i, n_ops):

                    B = pairs[j]

                    op1_src, op2_src, short_A, full_A = A
                    op1_snk, op2_snk, short_B, full_B = B

                    dataset_name = f"{short_A}__X__{short_B}"

                    try:
                        results = CorrelatorFactory.two_pt_dimeson(
                        proc=proc,
                        op1_src=op1_src,
                        op2_src=op2_src,
                        op1_snk=op1_snk,
                        op2_snk=op2_snk
                    )

                        pair_group = grp_irrep.create_group(dataset_name)

                        for key, array in results.items():
                            pair_group.create_dataset(
                                key,
                                data=array,
                                compression="gzip"
                            )


                    except Exception as e:
                        print(f"[ERROR] {dataset_name}")
                        print(e)

    print(f"[DONE] cfg {args.cfg_id:04d} complete.")


if __name__ == "__main__":
    main()
