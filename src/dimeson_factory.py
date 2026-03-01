# dimeson_factory.py
"""
Refactored to:
 - Remove redundant cubic-rotated momentum copies
 - Build ONE operator per |p|^2 shell
 - Store full cubic orbit inside operator
 - Prepare for orbit-projected contraction

Zero total momentum only (P=0 case).
"""

from dataclasses import dataclass
from typing import Tuple, List
from itertools import permutations, product
import numpy as np
from gamma import gamma

I = np.identity(4)

gamma_insertion_dict = {
    'a0': I,
    'pi': gamma[5],
    'pi2': gamma[4] @ gamma[5],
    'b0': gamma[4],
    'rho': I,
    'rho2': gamma[4],
    'a1': gamma[5],
    'b1': gamma[4] @ gamma[5],
}


# ---------------------------------------------------------
# Helper: Build full cubic orbit of a momentum vector
# ---------------------------------------------------------
def cubic_orbit(p: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    Generate all cubic rotations and reflections of p.
    Used to build momentum shells at P=0.
    """
    px, py, pz = p
    orbit = set()

    for perm in set(permutations([px, py, pz])):
        for signs in product([-1, 1], repeat=3):
            orbit.add(tuple(s * x for s, x in zip(signs, perm)))

    return list(orbit)


# ---------------------------------------------------------
# Bare operator
# ---------------------------------------------------------
@dataclass
class BareOperator:
    """
    Modified:
    - mom can be None (projected operator)
    - orbit stores cubic orbit momenta
    """
    meson: str
    mom: Tuple[int, int, int] | None
    ins: str
    irrep: str
    name: str
    short: str = ""
    orbit: List[Tuple[int, int, int]] | None = None

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
    
# ---------------------------------------------------------
# DiMesonFactory
# ---------------------------------------------------------
class DiMesonFactory:
    """
    Refactored:
      - Generates ONE operator per |p|^2 shell
      - Stores full cubic orbit internally
      - Eliminates redundant momentum combinations
    """

    def __init__(self):
        self.pairs = []

    @staticmethod
    def mom_to_str(mom):
        px, py, pz = mom
        return f"mom_{px}_{py}_{pz}"


    def generate_projected_zero_momentum(
        self,
        meson1_list,
        meson2_list,
        insertions1,
        insertions2,
        momentum_list,
        irrep="a1u"
    ):
        """
        Build ONE operator per |p|^2 shell.
        Assumes p2 = -p1 (P=0).
        """

        m1_list = [meson1_list] if isinstance(meson1_list, str) else meson1_list
        m2_list = [meson2_list] if isinstance(meson2_list, str) else meson2_list

        # -------------------------------------------------
        # Step 1: Build unique |p|^2 shells
        # -------------------------------------------------
        shells = {}

        for p in momentum_list:
            p = tuple(p)
            p2 = sum(x * x for x in p)

            if p2 == 0:
                continue

            if p2 not in shells:
                shells[p2] = cubic_orbit(p)

        # -------------------------------------------------
        # Step 2: Create projected operators
        # -------------------------------------------------
        self.pairs = []

        for p2_shell, orbit in shells.items():

            for m1 in m1_list:
                for ins1 in insertions1:

                    full1 = f"{m1}_shell{p2_shell}_{ins1}_{irrep}"
                    short1 = f"{m1}_p{p2_shell}_{ins1}"

                    op1 = BareOperator(
                        meson=m1,
                        mom=None,
                        orbit=orbit,
                        ins=ins1,
                        irrep=irrep,
                        name=full1,
                        short=short1
                    )

                    for m2 in m2_list:
                        for ins2 in insertions2:

                            full2 = f"{m2}_shell{p2_shell}_{ins2}_{irrep}"
                            short2 = f"{m2}_p{p2_shell}_{ins2}"

                            op2 = BareOperator(
                                meson=m2,
                                mom=None,
                                orbit=[tuple(-x for x in p) for p in orbit],
                                ins=ins2,
                                irrep=irrep,
                                name=full2,
                                short=short2
                            )

                            pair_short = f"{short1}x{short2}"
                            pair_full = f"{full1}X{full2}"

                            self.pairs.append((op1, op2, pair_short, pair_full))

        print(f"[DiMesonFactory] Generated {len(self.pairs)} projected operators")
        return self.pairs
