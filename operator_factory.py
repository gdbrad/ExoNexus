from dataclasses import dataclass
from gamma import gamma
import numpy as np 
from itertools import product 
from sympy import S

mom= 'mom_0_0_0'
I = np.array([[1, 0],
              [0, 1]])
IDEN = np.identity(4)
@dataclass
class QuantumNum:
    '''
    Houses all operator information into mom/disp/gamma structure 
    gamma: single or list 
    disp: null vector, 1-tuple, 2-tuple -> displacement three vector; apply covariant derivative operator in a specified spatial direction 
    '''
    name:str
    had: int    # 1->meson (no parity in op name), 2->baryon (parity is in op name)
    F: str      # Flavor irrep
    twoI: int   # 2*Isospin
    strange: int # strangeness +1/-1 or 0
    S: int      # spin 
    P: int      # Rest-frame parity
    C: int      # charge conjugation
    gamma: str  # base gamma structure without gamma_i or derivatives
    gamma_i: bool # [g1,g2,g3,g4] summed over 
    deriv: str  # covariant derivative eg. g_i
    mom: str 


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
a1_mp = {
    "pion": QuantumNum(name='pion',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom=mom),
    "pion_2": QuantumNum(name='pion_2',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None,mom=mom),
    "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", strange=0,twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla",mom=mom),
    "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1", strange=0,twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B",mom=mom),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1", strange=0,twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B",mom=mom),
}

strange_a1_m_p = {
    "kaon": QuantumNum(name="kaon",had=1, F="A1", twoI=1,strange=-1, S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False,mom=mom),
    "kaon_2": QuantumNum(name="kaon_2",had=1, F="A1", twoI=1, strange=-1,S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None,mom=mom),
    "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla",mom=mom),
    "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B",mom=mom),
    "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B",mom=mom),
   
    
}
'a1_pionx_g5_000'
'a1_pion_2x_g5_000'


light_isovector = {
     "rho":  QuantumNum(name='rho',had=1, F="T1", twoI=1, strange=0,S=0, P=-1, C=-1, gamma=IDEN,gamma_i=True,deriv=None,mom=mom),
    "rho_2":  QuantumNum(name='rho_2',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv=None,mom=mom),
    "pion": QuantumNum(name='pion',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=False,deriv=None,mom=mom),
    "pion_2": QuantumNum(name='pion_2',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None,mom=mom),
}


j_m_m = {
    'a1xNABLA_A1': QuantumNum(name='a1xNABLA_A1',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=True,deriv="nabla",mom=mom),
    "rho":  QuantumNum(name='rho',had=1, F="T1", twoI=1, strange=0,S=0, P=-1, C=-1, gamma=I,gamma_i=True,deriv=None,mom=mom),
    "rho_2":  QuantumNum(name='rho_2',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv=None,mom=mom),
    "a0xNABLA":  QuantumNum(name='a0xNABLA',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=I,gamma_i=False,deriv="nabla",mom=mom),
    "a1xNABLA_T2":  QuantumNum(name='a1xNABLA_T2',had=1, F="T2", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=False,deriv="nabla",mom=mom),
    "rhoxD_A2":  QuantumNum(name='rhoxD_A2',had=1, F="A2", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv="D",mom=mom),
    # "rhoxD_A2":  QuantumNum(name='rhoxD_A2',had=1, F="A2", twoI=1, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv="D",mom=mom),

}

# j_p_p = {
#     "a0": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=False,deriv=None,mom=mom),
#     "a1": QuantumNum(had=1, F="T1", twoI=1, S=0, P=1, C=1, gamma=gamma[5],gamma_i=True,deriv=None,mom=mom),
#     "rhoxNABLA_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=True,deriv="nabla",mom=mom),
#     "rhoxNABLA_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=True,deriv="nabla",mom=mom),
#     "b_1xB_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="B",mom=mom),

# }

j_p_m = {


}

@dataclass
class ProjectedOperator:
    dir: str
    irrep: str # A1,T1
    mom: str #tuple of ints without commas "000" "001" etc 
    t0: int 
    tz: int
    states: list[int]

class MomentaProjection: 
    def __init__(self):
        pass

    '''
    Here we define various 3-momenta conventions since angular momentum takes on a different form on the lattice and ceases to be a good quantum number
    Essentially, the same operator comes out for different relative momenta. In each dict, labeled by the total momentum P^2, all relative momenta satisfying this criteria
    are generated. Computes projected operator coefficients for non-zero total momenta
    '''

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


print(list_from_mom2_max(n=1))