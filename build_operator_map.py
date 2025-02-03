'''extract projected operator coeffs from ``operator_factory'' and build a master operator map using CG coeffs'''
from dataclasses import dataclass
import numpy as np 
import sympy 
from sympy.algebras.quaternion import * 
from sympy import pi 

# ------------ Momenta ---------------------# 

def order_momenta(mom):
    '''put momentum 3-tuple into canonical order '''


def cm_to_lab(mom):
    '''transformation from COM frame to lab frame'''
    

def operator_list_irrepmom(irrep):
    pass 

@dataclass
class IrrepNames: 
    name: str 
    wp: str 
    np: str
    ferm: bool #double cover 
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
    matrix: np.ndarray[np.cdouble]


@dataclass
class CubicHelicityRep(LG):
    twoHelicity: int 
    helicityRep: str 


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

    