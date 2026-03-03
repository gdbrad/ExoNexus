from dataclasses import dataclass
from insertion_factory.gamma import gamma,gamma_i
from insertion_factory.insertion import gamma_insertion_dict,flavor_dict,derivative_dict

import numpy as np 
from itertools import product 
from numpy import load 
from typing import List,Dict,Union
import re

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
class BareOperator:
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
    F: str      
    flavor: str 
    twoI: int      
    mom: str 

"""interpolating operators for Dpi and D*pi meson
We must tie together a light and charm perambulator with some gamma structure and projection operator for non-zero momentum, so we will have two sets of fwd/backward perambulators and 4 elementals 

"""

my_list = ["D_000_nabla_pi2_a1p","pi_001_none_rho2_a1p","D_-10-1_B_pi2_a1p"]

# find all signed single-digit integers
def mommy(s:str):
    nums = re.findall(r'-?\d', s)
    return "mom_" + "_".join(nums)

def parse_op(op:str):
    keys = op.split('_')

    # return QuantumNum(name=op,
    #             F=keys[-1],
    #             flavor=flavor_dict[keys[0]],
    #             twoI=0, 
    #             gamma=gamma_insertion_dict[keys[3]],
    #             deriv=keys[2] if keys[2]!='none' else None,
    #             mom=mommy(keys[1]),

    return BareOperator(name=op,
            F=keys[-1],
            flavor=flavor_dict[keys[0]],
            twoI=0,
            mom=mommy(keys[1]),
            )
                
new_DD_a1m = {key: parse_op(key) for key in my_list}

print(new_DD_a1m)

#-----------------------Monomial insertions(covariant derivatives)-------------------------#
# see table II https://arxiv.org/pdf/1204.5425

gamma_insertion_dict = {
    'a0': I,
    'pi': gamma[5],
    'pi2': gamma[4]@gamma[5],
    'b0': gamma[4],
    'rho': gamma_i,
    'rho2': gamma[4]@gamma_i,
    'a1': gamma[5]@gamma_i,
    'b1': gamma[4]@gamma[5]@gamma_i,

}

derivative_dict = {
    'none': None,
    'nabla': 'nabla',
    'B': 'B',
    'D':'D'
}

flavor_dict = {'D':'charm_light',
                   'pi':'light_light'}

@dataclass
class MonomialInsertion:
    name:str
    J: int 
    P: str #'plus'/'minus'
    C: str #'plus'/'minus'
    deriv_op: str
    gamma_matrix:str
    irrep: str 

a1p_dict = {'rhoxnabla':'0pp',
            'rho2xnabla':'0pp',
            'a1xnabla':'0mm',
            'b1xnabla':'0mp',
            'rhoxB':'0mp',
            'rho2xB':'0mp',
            'a1xB':'0pm',
            'b1xB':'0pp'} 

def parse_insertion(insertion:str):
    for k,v in a1p_dict.keys():
        keys = k.split('x')
        return MonomialInsertion(name=insertion,
                                 J=v.split[0],
                                 P=v.split[1],
                                 C=v.split[2],
                                 deriv_op=k.split('x')[0],
                                 gamma_matrix=gamma_insertion_dict[k.split('x')[0]],
                                 irrep='a1',
                                 )
#------------------------------------------------------------------------------#


#--------------------Momentum-----------------------------------------_# 

momentum_dict = Dict[int,str]

#--------------------Construct dimeson operators ------------------------------#
@dataclass
class DiMesonOperator:
    op1: BareOperator
    op2: BareOperator
    name: str 
    F: str 







class MHI(List[List[QuantumNum]]):
    '''construct multi-hadron operators out of single hadron operators using nested lists of `QuantumNum` objects '''
    def __init__(self):
        op1_name: str
        op2_name: str
        irrep: str 
        irrep_row: str
        flavor_structure: List[str,str] # eg. 'light_charm'

    def make_mhi(self):
        pass

    def _combine_pair(self):
        return self.op1_name + "_"+ self.op2_name 



    # if total_mom != 0:
    #     descent_of_symmetry = subduction.lookup() # follows altmann ref 


def get_dim_channel(channel:dict): 
    dim = sum(map(len,channel.values()))
    return dim 


class MHICollection:
    '''collect multiple two-hadron correlators together in a matrix'''

    pass 



# type operator_list_t = tuple[str,str] 

# def project_op_weights(channel: str, L: int, irreps:list):
#     '''extract projected operator coefficients'''
#     ops_map = {}

#     return 
# 




