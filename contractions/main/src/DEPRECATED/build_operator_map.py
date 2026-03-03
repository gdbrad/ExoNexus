'''extract projected operator coeffs from ``operator_factory'' and build a master operator map using CG coeffs'''
from dataclasses import dataclass
import numpy as np 
import sympy 
from sympy.algebras.quaternion import * 
from sympy import pi 
from itertools import product 
from sympy import S
from numpy import load 



def subduce():
    '''subduce operators from definite continuum spin into definite cubic irreps of the octahedral group to restore rotational symmetry '''



#--------------------------------------# 
def helicity_ops(psq:int):
    '''form helicity operators from little groups of Oh using at rest operators '''
    if not psq.isinstance(tuple): 
        raise(TypeError,'psq not a tuple')
    
    psq_instance = MomentaProjection()
    ksq = psq_instance.list_from_mom2_max(psq)

    
lookup_table = LG

#------------------------------------------------------# 
## single cover cubic grp irreps (P^2 = 0)

A1Rep = CubicRep(dim=1,G=0,group="Oh",rep="A1")
A2Rep = CubicRep(dim=1,G=0,group="Oh",rep="A2")
T1Rep = CubicRep(dim=3,G=0,group="Oh",rep="T1")
T2Rep = CubicRep(dim=3,G=0,group="Oh",rep="T2")
ERep = CubicRep(dim=2,G=0,group="Oh",rep="E")

## -------------------------------------------------- ## 



def read_operator_map():
    '''reads operator map from file. should be txt '''



def read_kaon_operator_map():
    '''sign of odd charge-conj operators need a flip!'''


## ------------------- MASTER LOOKUP TABLE ------------------------------- ## 


@dataclass
class IrrepNames:
    name: str
    wp: str
    np: str
    ferm: bool  # double cover
    lg: bool
    dim: int
    G: int

@dataclass
class CubicRep:
    dim: int
    G: int
    group: str
    rep: str

@dataclass
class LG(CubicRep):
    dim: int
    G: int
    group: str
    rep: str
    char: np.cdouble
    matrix: np.ndarray

@dataclass
class HelicityOperator(LG):
    mom: str 
    dim: int
    G: int
    group: str
    rep: str
    char: np.cdouble
    matrix: np.ndarray

# Hardcoded lookup table for LG instances (Cubic group O, example irreps)
"""
Each meson operator contains a compound index \( l \) comprised of:
\\begin{itemize}
    \\item Three-momentum \( \\mathbf{p} \): The three-momentum of the meson.
    \\item Irrep \( \\Lambda \): The irreducible representation of the little group of \( \\mathbf{p} \).
    \\item Row \( \\lambda \): The row of the irreducible representation.
    \\item Total isospin \( I \): The total isospin quantum number.
    \\item Isospin projection \( I_3 \): The isospin projection quantum number.
    \\item Strangeness \( S \): The strangeness quantum number.
    \\item Operator identifier (str): A string identifier for the operator.
\\end{itemize}
"""

'''
D_4 [0,0,n]
Momentum;  {n, 0, 0};  {0, n, 0};  {0, 0, n};  {-n, 0, 0};  {0, -n, 0};  {0, 0, -n}
Rotation:
    {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}}
    {{0, -1, 0}, {0, 0, 1}, {-1, 0, 0}}
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
    {{0, 0, -1}, {0, -1, 0}, {-1, 0, 0}} 
    {{0, 1, 0}, {0, 0, -1}, {-1, 0, 0}}
    {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}}
'''
LG_TABLE = {

    LG(
        dim=1,
        G=0,  # Order of group O
        group="Oh",
        rep="A1",
        char=np.cdouble(1.0),  # Character for identity element
        matrix=np.array([[1.0]], dtype=np.cdouble)  # 1x1 identity matrix
    ),
    LG(
        dim=1,
        G=0,
        group="Oh",
        rep="A2",
        char=np.cdouble(1.0),  # Placeholder character
        matrix=np.array([[1.0]], dtype=np.cdouble)  # 1x1 matrix
    ),
    LG(
        dim=2,
        G=0,
        group="Oh",
        rep="E",
        char=np.cdouble(2.0),  # Trace of 2x2 identity
        matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.cdouble)  # 2x2 identity
    ),
    LG(
        dim=3,
        G=0,
        group="Oh",
        rep="T1",
        char=np.cdouble(3.0),  # Trace of 3x3 identity
        matrix=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.cdouble)  # 3x3 identity matrix
    ),
    LG(
        dim=3,
        G=1,
        group="Oh",
        rep="T2",
        char=np.cdouble(3.0),  # Placeholder character
        matrix=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.cdouble)  # 3x3 identity matrix
    ),
}

# Example function to access the table
def get_lg_rep(group: str, rep: str) -> LG:
    for entry in LG_TABLE:
        if entry.group == group and entry.rep == rep:
            return entry
    raise ValueError(f"No representation found for group {group} and rep {rep}")

# Example usage
if __name__ == "__main__":
    # Get A1 representation for group O
    a1_rep = get_lg_rep("O", "A1")
    print(f"Representation {a1_rep.rep}: dim={a1_rep.dim}, char={a1_rep.char}")
    print(f"Matrix:\n{a1_rep.matrix}")


    