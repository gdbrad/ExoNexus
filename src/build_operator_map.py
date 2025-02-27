'''extract projected operator coeffs from ``operator_factory'' and build a master operator map using CG coeffs'''
from dataclasses import dataclass
import numpy as np 
import sympy 
from sympy.algebras.quaternion import * 
from sympy import pi 
from itertools import product 
from sympy import S
from numpy import load 

# ------------ Momenta ---------------------# 

def order_momenta(mom):
    '''put momentum 3-tuple into canonical order '''


def cm_to_lab(mom):
    '''transformation from COM frame to lab frame'''
    

def operator_list_irrepmom(irrep):
    pass 

class MomentaProjection: 
    def __init__(self):
        self.data_path = '/home/grant/exotraction/data'

    '''
    Here we define various 3-momenta conventions since angular momentum takes on a different form on the lattice and ceases to be a good quantum number
    Essentially, the same operator comes out for different relative momenta. In each dict, labeled by the total momentum P^2, all relative momenta satisfying this criteria
    are generated. Computes projected operator coefficients for non-zero total momenta
    '''

    def load_coeffs(self):
        data = load(self.data_path + f'/Oh/T1_minus.npz')
        lst = data.files
        for item in lst:
            return data[item]

    def list_from_mom2_max(self,n):
        '''generate relative momenta for a given total momenta'''
        imax = int(np.sqrt(n))
        mom = []
        # print('P^2='f"{n}")
        for it in product(range(-imax, imax + 1), repeat=3):
            i, j, k = it
            if i**2 + j**2 + k**2 <= n:
                if i**2 + j**2 + k**2 == n:
                    mom.append([S(k), S(j), S(i)])
                    #print(f"mom list:({k},{j},{i})")
        return mom

    def mom_list_to_dict(self,mom_list):
        mom_dict = {k: " ".join(map(str,v)) for k, v in enumerate(mom_list)}
        return mom_dict

    def mom_list_to_numpy_array(self,mom_list):
        '''convert list of momenta to a numpy array'''
        return np.array([mom_list])

def subduce():
    '''subduce operators from definite continuum spin into definite cubic irreps of the octahedral group to restore rotational symmetry '''



#--------------------------------------# 
def helicity_ops(psq:int):
    '''form helicity operators from little groups of Oh using at rest operators '''
    if not psq.isinstance(tuple): 
        raise(TypeError,'psq not a tuple')
    
    psq_instance = MomentaProjection()
    ksq = psq_instance.list_from_mom2_max(psq)

    


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

    