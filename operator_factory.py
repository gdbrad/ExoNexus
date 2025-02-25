from dataclasses import dataclass
from gamma import gamma
import numpy as np 
from itertools import product 
from numpy import load 

# mom= 'mom_0_0_0'
I = np.array([[1, 0],
              [0, 1]])
IDEN = np.identity(4)
@dataclass
class QuantumNum:
    '''
    Houses all operator information into mom/disp/gamma structure
    An insertion requires gamma structure, derivative(disp) and a projection name corresponding to a SU2 CG coeff -> subduction coeff, that is, if the momentum tuple or list of tuples provided is non-zero

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

def get_dim_channel(channel:dict): 
    dim = sum(map(len,channel.values()))
    return dim 

# type operator_list_t = tuple[str,str] 

# def project_op_weights(channel: str, L: int, irreps:list):
#     '''extract projected operator coefficients'''
#     ops_map = {}

#     return  



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

# a1_mp_strange = {
#     "kaon": QuantumNum(name="kaon",had=1, F="A1", twoI=1,strange=-1, S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False),
#     "kaon_2": QuantumNum(name="kaon_2",had=1, F="A1", twoI=1, strange=-1,S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None),
#     "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
#     "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B"),
#     "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
   
    
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




