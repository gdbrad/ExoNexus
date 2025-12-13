# dimeson_factory.py — FINAL: D_000_pi_none_a1uxPI_000_pi_none_a1u.h5
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np 
I = np.identity(4)
from gamma import gamma

gamma_insertion_dict = {
    'a0': I, 'pi': gamma[5], 'pi2': gamma[4] @ gamma[5], 'b0': gamma[4],
    'rho': I, 'rho2': gamma[4], 'a1': gamma[5], 'b1': gamma[4] @ gamma[5],
}

@dataclass
class BareOperator:
    meson: str
    mom: Tuple[int,int,int]
    ins: str
    irrep: str
    name: str
    short: str = ""

    @property
    def base_gamma(self):
        gname, _ = self.ins.split('_')
        return gamma_insertion_dict[gname]

    @property
    def gamma_i(self) -> bool:
        gname, _ = self.ins.split('_')
        return gname in {'rho', 'rho2', 'a1', 'b1'}

    @property
    def derivative(self):
        _, d = self.ins.split('_')
        return d if d != 'none' else None

class DiMesonFactory:
    def __init__(self):
        self.pairs = []

    @staticmethod
    def mom_to_str(m: Tuple[int,int,int]) -> str:
        """Convert momentum to string like mom_0_0_-1"""
        return f"mom_{m[0]}_{m[1]}_{m[2]}"

    @staticmethod
    def mom_to_short(m: Tuple[int,int,int]) -> str:
        """Convert (0,0,-1) → '00m1', (1,0,0) → '100'"""
        return "".join(str(x) if x >= 0 else f"m{abs(x)}" for x in m)

    def generate(self, meson1_list, meson2_list, insertions1, insertions2,
                 momentum_pairs, irrep="a1u"):

        m1_list = [meson1_list] if isinstance(meson1_list, str) else meson1_list
        m2_list = [meson2_list] if isinstance(meson2_list, str) else meson2_list

        self.pairs = []

        for (p1_raw, p2_raw) in momentum_pairs:
            p1 = tuple(p1_raw)
            p2 = tuple(p2_raw)

            if tuple(a + b for a, b in zip(p1, p2)) != (0, 0, 0):
                continue

            p1_short = self.mom_to_short(p1)
            p2_short = self.mom_to_short(p2)

            for m1 in m1_list:
                prefix1 = "D" if m1 == "D" else "PI" if m1 in {"p", "pion"} else m1[:2].upper()
                for ins1 in insertions1:
                    full1 = f"{m1}_mom_{p1[0]}_{p1[1]}_{p1[2]}_{ins1}_{irrep}"
                    short1 = f"{prefix1}_{p1_short}_{ins1}"

                    op1 = BareOperator(meson=m1, mom=p1, ins=ins1, irrep=irrep,
                                       name=full1, short=short1)

                    for m2 in m2_list:
                        prefix2 = "D" if m2 == "D" else "PI" if m2 in {"p", "pion"} else m2[:2].upper()
                        for ins2 in insertions2:
                            full2 = f"{m2}_mom_{p2[0]}_{p2[1]}_{p2[2]}_{ins2}_{irrep}"
                            short2 = f"{prefix2}_{p2_short}_{ins2}"

                            op2 = BareOperator(meson=m2, mom=p2, ins=ins2, irrep=irrep,
                                               name=full2, short=short2)

                            pair_short = f"{short1}x{short2}"
                            pair_full = f"{full1}X{full2}"
                            self.pairs.append((op1, op2, pair_short, pair_full))

        print(f"[DiMesonFactory] Generated {len(self.pairs)} di-meson operators")
        return self.pairs