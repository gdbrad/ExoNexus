from dataclasses import dataclass
from insertion_factory.gamma import gamma,gamma_i
import numpy as np 
from itertools import product 
from numpy import load 

# mom= 'mom_0_0_0'
I = np.array([[1, 0],
              [0, 1]])
IDEN = np.identity(4)
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
parse_insertion

# a1p_000 = {
#     MonomialInsertion(
#         'rhoxnabla',
#         J=0,
#         P=1,
#         C=1,
#         deriv_op='nabla',
#         gamma_matrix=gamma_insertion_dict['rho'],
#         irrep='a1'
        
#     )
# }