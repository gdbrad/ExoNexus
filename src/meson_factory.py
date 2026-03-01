"""
meson_factory.py

Single-meson operator construction.

- No cubic orbit projection
- Fixed momentum
- One operator per interpolator
- Designed for use with CorrelatorFactory.two_pt_meson()

SU(3) light sector + charm
"""

from dataclasses import dataclass
from typing import Tuple, List
from gamma import gamma
import numpy as np


# ---------------------------------------------------------
# Gamma insertion dictionary
# ---------------------------------------------------------

I = np.identity(4)

gamma_insertion_dict = {
    "a0": I,                      # 0+
    "pi": gamma[5],# 0-
    "pi_2": gamma[5]@gamma[4],              
    "rho": I,                     # 1- (handled via gamma_i flag)
    "a1": gamma[5],               # 1+ (gamma_i @ gamma5)
}


# ---------------------------------------------------------
# Bare single-meson operator
# ---------------------------------------------------------

@dataclass
class MesonOperator:
    """
    Single meson operator.

    meson  : flavor label ("light", "charm_light", "charm_charm")
    mom    : fixed momentum tuple (px,py,pz)
    ins    : insertion string e.g. "pi_none"
    irrep  : lattice irrep
    name   : full name
    short  : short name
    """

    meson: str
    mom: Tuple[int, int, int]
    ins: str
    irrep: str
    name: str
    short: str = ""

    @property
    def base_gamma(self):
        gname, _ = self.ins.split("_")
        return gamma_insertion_dict[gname]

    @property
    def gamma_i(self) -> bool:
        gname, _ = self.ins.split("_")
        return gname in {"rho", "a1"}

    @property
    def derivative(self):
        _, d = self.ins.split("_")
        return d if d != "none" else None


# ---------------------------------------------------------
# MesonFactory
# ---------------------------------------------------------

class MesonFactory:
    """
    Generates single-meson operators.

    Usage:
        mf = MesonFactory()
        ops = mf.generate(
            meson="charm_light",
            insertions=["pi_none", "rho_none"],
            momentum=(0,0,0),
            irrep="a1u"
        )
    """

    def __init__(self):
        self.ops: List[MesonOperator] = []

    @staticmethod
    def mom_to_str(mom: Tuple[int, int, int]) -> str:
        px, py, pz = mom
        return f"mom_{px}_{py}_{pz}"

    def generate(
        self,
        meson,
        insertions,
        momentum=(0, 0, 0),
        irrep="a1u"
    ) -> List[MesonOperator]:

        mlist = [meson] if isinstance(meson, str) else meson

        self.ops = []

        for m in mlist:
            for ins in insertions:

                full = f"{m}_p{momentum}_{ins}_{irrep}"
                short = f"{m}_{ins}"

                op = MesonOperator(
                    meson=m,
                    mom=momentum,
                    ins=ins,
                    irrep=irrep,
                    name=full,
                    short=short,
                )

                self.ops.append(op)

        print(f"[MesonFactory] Generated {len(self.ops)} operators")
        return self.ops