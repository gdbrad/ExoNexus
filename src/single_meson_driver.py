import argparse
import h5py
import yaml
from pathlib import Path

from correlator_factory import CorrelatorFactory
from single_meson_corr import SingleMesonCorrelator
from meson_factory import MesonFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--cfg_id", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    # -----------------------------------------------------
    # Load YAML (operator definitions only)
    # -----------------------------------------------------
    yaml_path = Path(args.yaml_file)
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    ens = list(yaml_data.keys())[0]
    settings = yaml_data[ens]

    operators = settings["operators"]

    proc = CorrelatorFactory(
        ens=ens,
        cfg_id=args.cfg_id,
    )
    proc.load(system_name="single")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"cfg{args.cfg_id:04d}.h5"

    if outfile.exists():
        print(f"[SKIP] {outfile} already exists")
        return

    print(f"[INFO] Writing single-meson matrix to {outfile}")

    with h5py.File(outfile, "w") as f_cfg:

        f_cfg.attrs["ensemble"] = ens
        f_cfg.attrs["cfg_id"] = args.cfg_id
        f_cfg.attrs["nvecs"] = proc.nvecs
        f_cfg.attrs["nt"] = proc.lt
        f_cfg.attrs["ntsrc"] = proc.ntsrc

        for irrep_name in operators:
            irrep_settings = operators[irrep_name]

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

            # -------------------------------------------------
            # Create irrep group
            # -------------------------------------------------
            #grp_irrep = f_cfg.create_group(irrep)
            grp_irrep = f_cfg.create_group(irrep_name)
            grp_irrep.attrs["irrep"] = irrep
            grp_irrep.attrs["n_ops"] = n_ops

            # Store operator lookup
            for op in ops:
                grp_irrep.attrs[f"op_{op.short}"] = op.name

            # -------------------------------------------------
            # Hermitian-reduced matrix
            # -------------------------------------------------
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