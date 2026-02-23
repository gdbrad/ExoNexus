import argparse
import h5py
import yaml
from pathlib import Path

from meson_correlator_factory import SingleMesonCorrelator
from meson_factory import MesonFactory


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

    proc.load_single_mesons()

    # -----------------------------------------------------
    # Output file per configuration
    # -----------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"cfg{args.cfg_id:04d}.h5"

    if outfile.exists():
        print(f"[SKIP] {outfile} already exists")
        return

    print(f"[INFO] Writing single-meson matrix to {outfile}")

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
        # Loop over irreps
        # =====================================================
        for irrep_name in operators:

            irrep_settings = operators[irrep_name]

            irrep = irrep_settings["irrep"]
            mesons = irrep_settings["mesons"]
            insertions = irrep_settings["insertions"]

            print(f"[INFO] {irrep}")

            # -------------------------------------------------
            # Generate projected operators
            # -------------------------------------------------
            factory = MesonFactory()

            factory.generate_projected_zero_momentum(
                meson_list=mesons,
                insertions=insertions,
                irrep=irrep
            )

            ops = factory.operators
            n_ops = len(ops)

            print(f"[INFO] {irrep}: {n_ops} operators")
            print(f"[INFO] Computing Hermitian-reduced {n_ops}x{n_ops} matrix")

            # -------------------------------------------------
            # Create irrep group
            # -------------------------------------------------
            grp_irrep = f_cfg.create_group(irrep)
            grp_irrep.attrs["n_ops"] = n_ops

            # Store operator lookup
            for short, full in ops:
                grp_irrep.attrs[f"op_{short}"] = full

            # -------------------------------------------------
            # Hermitian-reduced matrix
            # -------------------------------------------------
            for i, A in enumerate(ops):
                for j in range(i, n_ops):

                    short_A, op_A = A
                    short_B, op_B = ops[j]

                    dataset_name = f"{short_A}__X__{short_B}"

                    try:
                        C = CorrelatorFactory.two_pt_single(
                            proc=proc,
                            op_src=op_A,
                            op_snk=op_B
                        )

                        grp_irrep.create_dataset(
                            dataset_name,
                            data=C,
                            compression="gzip"
                        )

                    except Exception as e:
                        print(f"[ERROR] {dataset_name}")
                        print(e)

    print(f"[DONE] cfg {args.cfg_id:04d} complete.")


if __name__ == "__main__":
    main()