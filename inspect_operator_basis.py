import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import yaml
from src.dimeson_factory import DiMesonFactory
def main():
    yaml_file = Path("b3.4-s24t64/b3.4-s24t64.ini.yml")  # ← change if needed
    out_txt = Path("operator_basis.txt")

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    ens = list(data.keys())[0]
    settings = data[ens]

    # Extract parameters
    meson1 = settings.get("meson1", [])
    meson2 = settings.get("meson2", [])
    ins1 = settings.get("ins_1", [])
    ins2 = settings.get("ins_2", [])
    mom_pairs = settings["mom_pairs"]
    irrep = settings.get("irrep", "a1u")

    # Convert strings to lists
    m1_list = [meson1] if isinstance(meson1, str) else meson1
    m2_list = [meson2] if isinstance(meson2, str) else meson2

    print(f"Generating operator basis for {ens} → {len(m1_list)}×{len(m2_list)} mesons")
    print(f"  D operators : {m1_list}")
    print(f"  π operators : {m2_list}")
    print(f"  Insertions D: {ins1}")
    print(f"  Insertions π: {ins2}")
    print(f"  Mom pairs   : {len(mom_pairs)}")
    print(f"  Irrep       : {irrep}")
    print(f"Writing full list → {out_txt}")

    factory = DiMesonFactory()
    factory.generate(
        meson1_list=m1_list,
        meson2_list=m2_list,
        insertions1=ins1,
        insertions2=ins2,
        momentum_pairs=[tuple(tuple(p) for p in pair) for pair in mom_pairs],
        irrep=irrep
    )

    with open(out_txt, "w") as f:
        f.write(f"# Operator basis for {ens} — {len(factory.pairs)} di-meson operators\n")
        f.write(f"# meson1: {m1_list} | meson2: {m2_list} | irrep: {irrep}\n")
        f.write(f"# Generated: {Path(__file__).name}\n")
        f.write("#\n")
        f.write("# Format: index | short_pair | D_operator → π_operator | full_name\n")
        f.write("#" + "-"*120 + "\n")

        for idx, (op1, op2, short_pair, full_name) in enumerate(factory.pairs, 1):
            f.write(f"{idx:4d} | {short_pair:20s} | "
                    f"{op1.short:8s} → {op2.short:8s} | {full_name}\n")

    print(f"Done! Full basis written to: {out_txt}")
    print(f"   Total operators: {len(factory.pairs)}")
    print("   Example lines:")
    print(open(out_txt).read().splitlines()[:15])


if __name__ == "__main__":
    main()