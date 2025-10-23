from dataclasses import dataclass
from gamma import gamma
import numpy as np 
from itertools import product 
from numpy import load 
from collections import List 

# mom= 'mom_0_0_0'
I = np.array([[1, 0],
              [0, 1]])
IDEN = np.identity(4)

"""Define operator basis. We employ the same basis as arXiv:0707.4162 [hep-lat, physics:hep-ph]
See app. B

should handle momentum and irrep subduction at the operator level

The objects that inherit ``QuantumNum" are devoid of any insertion yet eg. gamma or displacement structure, these just point to the flavor of perambulator that should be called at the source and sink. The insertion and derivative operator should result from parsing of the ``str`` of operator name 

Here we need to
1. describe correct quark content 
2. generate the desired quantum numbers of state

"""

@dataclass
class QuantumNum:
    '''
    Houses all operator information into mom/disp/gamma structure
    An insertion requires gamma structure, derivative(disp) and a projection name corresponding to a SU2 CG coeff -> subduction coeff, that is, if the momentum tuple or list of tuples provided is non-zero. Flavor is a single ``str" value as the operator keys correspond to a src or snk operator, the same operator can be both at the src and snk. 

    gamma: single or list 
    disp: null vector, 1-tuple, 2-tuple -> displacement three vector; apply covariant derivative operator in a specified spatial direction 
    Attributes:
        name: Operator name
        had: 1 for meson (no parity in name), 2 for baryon (parity in name)
        flavor: Flavor of the particle ('light', 'strange', 'charm', etc.)
        twoI: 2 * Isospin
        S: Spin
        P: Rest-frame parity
        C: Charge conjugation parity
        gamma: Base gamma structure (single or list)
        gamma_i: Whether gamma_i is summed over [g1, g2, g3, g4]
        deriv: Covariant derivative type (None, 'nabla', 'B', 'D', etc.)
        mom: Momentum as a string (e.g., 'mom_0_0_0')
    '''
    name:str
    had: int   
    F: str      
    flavor: str 
    twoI: int   
    S: int      
    P: int      
    C: int      
    gamma: str  
    gamma_i: bool 
    deriv: str
    mom: str 

class MHICollection:
    '''collect multiple two-hadron correlators together in a matrix'''




class MHI(List[List[QuantumNum]]):
    '''construct multi-hadron operators out of single hadron operators using nested lists of `QuantumNum` objects '''

    pair_name: str
    total_mom: int
    irrep: str 
    irrep_row: str
    flavor_structure: List[str,str] # eg. 'light_charm'

    if total_mom != 0:
        descent_of_symmetry = subduction.lookup() # follows altmann ref 

        

    



def get_dim_channel(channel:dict): 
    dim = sum(map(len,channel.values()))
    return dim 

# type operator_list_t = tuple[str,str] 

# def project_op_weights(channel: str, L: int, irreps:list):
#     '''extract projected operator coefficients'''
#     ops_map = {}

#     return 
# 

"""interpolating operators for Dpi and D*pi meson
We must tie together a light and charm perambulator with some gamma structure and projection operator for non-zero momentum, so we will have two sets of fwd/backward perambulators and 4 elementals 

"""

DD_a1m = {
    "D": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None),
    "D_2": QuantumNum(name='D_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
    "D_rhoxB": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=True,deriv="B"),
    "D_nabla": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=True,deriv="nabla"),
    "D_Bxnabla": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=True,deriv="nabla"),


    "D_rho2xB": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=True,deriv="B"),

    "D_star": QuantumNum(name='pion_2',had=1, F="T1m", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv=None),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B")
}

DD_t1p = {
    "pion_2": QuantumNum(name='pion_2',had=1, F="A1", flavor='light',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B")

}


a1_p_dpi_nomix = {
    "pion_2": QuantumNum(name='pion_2',had=1, F="A1", flavor='light',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
   }



a1_mp_nomix_charm = {
    "D_2": QuantumNum(name='D_2',had=1, F="A1", flavor='charm',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
    "pion": QuantumNum(name='pion',had=1, F="A1",flavor='light',twoI=1,S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None),
}

''' construct interpolators for 4 J^PC values '''
# this is test case see charmonium spectrum paper by hadspec 
#  
# kaon_single = {
#     "kaon": QuantumNum(name="kaon",had=1, F="A1", twoI=1,strange=-1, S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False),
#     # "kaon_2": QuantumNum(name="kaon_2",had=1, F="A1", twoI=1, strange=-1,S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None),
#     # "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
# }

## MAYBE DONT MIX TIME REVERSAL ## 
## EXCLUDE G5 
## gevp no longer hyperbolic cos behavior so cant mix with periodic bdy conditions

a1_mp_nomix = {
    "pion_2": QuantumNum(name='pion_2',had=1, F="A1", flavor='light',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
   }
a1_mp = {
    "pion": QuantumNum(name='pion',had=1, F="A1",flavor='light',twoI=1,S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None),
    "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
    "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1",flavor='light',twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B"),
}
a1_mp_strange_nomix = {
    "kaon_2": QuantumNum(name="kaon_2",had=1, F="A1", twoI=1, flavor='strange',S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='strange',twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
}
a1_mp_strange = {
    "kaon": QuantumNum(name="kaon",had=1, F="A1", twoI=1,flavor='strange', S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False),
    "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", flavor='strange',twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
    "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1", flavor='strange',twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B"),
}


# }

# light_isovector = {
#      "rho":  QuantumNum(name='rho',had=1, F="T1", twoI=1, strange=0,S=0, P=-1, C=-1, gamma=IDEN,gamma_i=True,deriv=None),
#     "rho_2":  QuantumNum(name='rho_2',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv=None),
#     "pion": QuantumNum(name='pion',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=False,deriv=None),
#     "pion_2": QuantumNum(name='pion_2',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
# }


# j_m_m = {
#     'a1xNABLA_A1': QuantumNum(name='a1xNABLA_A1',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=True,deriv="nabla"),
#     "rho":  QuantumNum(name='rho',had=1, F="T1", twoI=1, strange=0,S=0, P=-1, C=-1, gamma=I,gamma_i=True,deriv=None),
#     "rho_2":  QuantumNum(name='rho_2',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv=None),
#     "a0xNABLA":  QuantumNum(name='a0xNABLA',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=I,gamma_i=False,deriv="nabla"),
#     "a1xNABLA_T2":  QuantumNum(name='a1xNABLA_T2',had=1, F="T2", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=False,deriv="nabla"),
#     "rhoxD_A2":  QuantumNum(name='rhoxD_A2',had=1, F="A2", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv="D"),
#     # "rhoxD_A2":  QuantumNum(name='rhoxD_A2',had=1, F="A2", twoI=1, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv="D"),

# }

# # j_p_p = {
# #     "a0": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=False,deriv=None),
# #     "a1": QuantumNum(had=1, F="T1", twoI=1, S=0, P=1, C=1, gamma=gamma[5],gamma_i=True,deriv=None),
# #     "rhoxNABLA_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=True,deriv="nabla"),
# #     "rhoxNABLA_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=True,deriv="nabla"),
# #     "b_1xB_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="B"),

# # }

# j_p_m = {


# }


@dataclass
class ProjectedOperator:
    dir: str
    irrep: str # A1,T1
    mom: str #tuple of ints without commas "000" "001" etc 
    t0: int 
    tz: int
    states: list[int]




