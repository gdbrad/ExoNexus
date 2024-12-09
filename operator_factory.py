from dataclasses import dataclass
from gamma import gamma
import numpy as np 

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
    LOOK UP CHROMA CONVENTION FOR THIS ^ i think disp_1: disp(1,0,0)
    mom: List of 3-tuples 
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

j_m_p = {
    "kaon_A1": QuantumNum(name="kaon_A1",had=1, F="A1", twoI=1,strange=1, S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False,mom=mom),
    "kaon_2_A1": QuantumNum(name="kaon_2_A1",had=1, F="A1", twoI=1, strange=1,S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None,mom=mom),
    "kaon_A1_001": QuantumNum(name="kaon_A1",had=1, F="A1", twoI=1,strange=1, S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False,mom='mom_0_0_0'),
    "kaon_2_A1_001": QuantumNum(name="kaon_2_A1",had=1, F="A1", twoI=1, strange=1,S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_0'),


    
}
# class OperatorInsertion:
#     '''forms the Hermitian operators to construct the correlator matrix'''
#     def __init__(
#             self,
#             gamma:str, # see Gamma class 
#             derivative: str, # see Deriv class 
#             projection: str, # see Irrep class 
#             mom_dict = Dict[int,str]) -> None: # see Mom class of tuples
    
#     def _get_parity(self):
#         self.parity = gamma_parity * displacement_parity 
        
#     def _get_charge_conj(self):

#     def helicity_ops():
#     '''op mom row'''

#     mx,my,mz = mom[0]mom[1]mom[2]

#     get_helicity_op(pattern,row,row)

#     def spin_1_mat():
#         eps = eps3d(mom,true)
#         Gamma(4,4)


