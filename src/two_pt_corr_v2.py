import argparse
import h5py
import yaml
from pathlib import Path

from correlator_factory import CorrelatorFactory
from meson_factory import MesonFactory
from dimeson_factory import DiMesonFactory


# =========================================================
# Single Meson Sector
# =========================================================
def run_single_meson_sector(proc, f_cfg, sector_name, sector_settings):

    print(f"\n[INFO] Running single-meson sector: {sector_name}")

    ptot = sector_settings.get("ptot", [0, 0, 0])
    if ptot != [0, 0, 0]:
        raise NotImplementedError("Only P=0 supported for now")

    grp_sector = f_cfg.create_group(f"single_mesons/{sector_name}")

    for irrep, irrep_data in sector_settings["irreps"].items():

        print(f"[INFO]  Irrep: {irrep}")

        operators = irrep_data["operators"]

        factory = MesonFactory()

        ops = factory.generate(
            operators=operators,
            irrep=irrep
        )

        n_ops = len(ops)

        grp_irrep = grp_sector.create_group(irrep)
        grp_irrep.attrs["n_ops"] = n_ops

        # Hermitian-reduced matrix
        for i, op_i in enumerate(ops):
            for j in range(i, n_ops):

                op_j = ops[j]

                dataset_name = f"{op_i.short}__X__{op_j.short}"

                try:
                    corr = CorrelatorFactory.two_pt_single_meson(
                        proc=proc,
                        op_src=op_i,
                        op_snk=op_j
                    )

                    grp_irrep.create_dataset(
                        dataset_name,
                        data=corr,
                        #compression="gzip"
                    )

                except Exception as e:
                    print(f"[ERROR] {dataset_name}")
                    print(e)


# =========================================================
# Two Meson Sector
# =========================================================
def run_two_meson_sector(proc, f_cfg, sector_name, sector_settings):

    print(f"\n[INFO] Running two-meson sector: {sector_name}")

    irrep = sector_settings["irrep"]
    ptot = sector_settings["ptot"]

    if ptot != [0, 0, 0]:
        raise NotImplementedError("Only P=0 supported")

    meson1 = sector_settings["meson1"]
    meson2 = sector_settings["meson2"]
    ins1 = sector_settings["ins_1"]
    ins2 = sector_settings["ins_2"]

    raw_momenta = [
        tuple(mom_pair[0])
        for mom_pair in sector_settings["mom_pairs"]
    ]

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

    grp_sector = f_cfg.create_group(f"two_mesons/{sector_name}")
    grp_sector.attrs["irrep"] = irrep
    grp_sector.attrs["n_ops"] = n_ops

    for _, _, short, full in pairs:
        grp_sector.attrs[f"op_{short}"] = full

    # Hermitian reduced matrix
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

                pair_group = grp_sector.create_group(dataset_name)

                for key, array in results.items():
                    pair_group.create_dataset(
                        key,
                        data=array,
                        compression="gzip"
                    )

            except Exception as e:
                print(f"[ERROR] {dataset_name}")
                print(e)


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--cfg_id", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    # -----------------------------------------------------
    # Load YAML
    # -----------------------------------------------------
    with open(args.yaml_file) as f:
        yaml_data = yaml.safe_load(f)

    ens = list(yaml_data.keys())[0]
    settings = yaml_data[ens]

    params = settings["params"]
    sectors = settings["sectors"]

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

    proc.load_for_system()

    # -----------------------------------------------------
    # Output file
    # -----------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"cfg{args.cfg_id:04d}.h5"

    if outfile.exists():
        print(f"[SKIP] {outfile} exists")
        return

    print(f"[INFO] Writing to {outfile}")

    with h5py.File(outfile, "w") as f_cfg:

        f_cfg.attrs["ensemble"] = ens
        f_cfg.attrs["cfg_id"] = args.cfg_id

        # -------------------------
        # Dispatch sectors
        # -------------------------
        if "single_mesons" in sectors:
            for name, sec in sectors["single_mesons"].items():
                run_single_meson_sector(proc, f_cfg, name, sec)

        if "two_mesons" in sectors:
            for name, sec in sectors["two_mesons"].items():
                run_two_meson_sector(proc, f_cfg, name, sec)

    print(f"[DONE] cfg {args.cfg_id:04d} complete.")


if __name__ == "__main__":
    main()