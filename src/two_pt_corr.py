import argparse
from correlator_factory import CorrelatorFactory
import file_io
# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------
def build_processor(args: argparse.Namespace) -> CorrelatorFactory:
    tmp = file_io.DistillationObjectsIO(ens=args.ens)
    tmp.get_contraction_params()

    return CorrelatorFactory(
        ens=args.ens,
        cfg_id=args.cfg_id,
        flavor_contents=args.flavor.split(","),
        nvecs=args.nvecs or tmp.nvecs,
        lt=args.lt or tmp.lt,
        ntsrc=args.ntsrc or tmp.ntsrc,
        tsrc_step=args.tsrc_step or tmp.tsrc_step,
        data1=args.data1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute single- or di-meson two-point correlators."
    )
    parser.add_argument("--nvecs", type=int, help="Number of distillation vectors")
    parser.add_argument("--lt", type=int, help="Temporal lattice size")
    parser.add_argument("--ens", type=str, required=True, help="Ensemble name")
    parser.add_argument("--cfg_id", type=int, required=True, help="Configuration index")
    parser.add_argument(
        "--flavor",
        type=str,
        required=True,
        help="Comma-separated flavour contents, e.g. light_light,light_charm",
    )
    parser.add_argument("--ntsrc", type=int, help="Number of t_src insertions")
    parser.add_argument(
        "--tsrc_step", type=int, default=1, help="Step between successive t_src"
    )
    parser.add_argument("--data1", action="store_true", help="meson data lives in data1/")
    parser.add_argument("--three_bar", action="store_true", help="Compute the bar{3} irrep")
    parser.add_argument("--tsrc_avg", action="store_true", help="Average over source times")

    args = parser.parse_args()
    if len(args.flavor.split(",")) > 2:
        raise ValueError("Only up to two flavours are supported.")

    proc = build_processor(args)

    system_name = proc.get_meson_system_name()
    out_file = f"{args.ens}_{system_name}_cfg{args.cfg_id}_2pt_nvec_{proc.nvecs}_tsrc_{proc.ntsrc}_test.h5"

    with h5py.File(out_file, "w") as f:
        grp = f.create_group(f"{system_name}_000")
        CorrelatorFactory.two_pt(
            proc,
            grp,
            tsrc_avg=args.tsrc_avg,
            three_bar=args.three_bar,
        )

    print(f"Finished - results in {out_file}")


if __name__ == "__main__":
    main()