from dataclasses import dataclass
import numpy as np
from itertools import product
import re
from typing import List, Dict, Union, Tuple, Any

I = np.identity(4)
from gamma import gamma,gamma_i

"""see https://arxiv.org/pdf/1607.07093 table 5

"""
# these get coupled together in an insertion monomial
#----------------------------------------------------#
gamma_insertion_dict = {
    'a0': I,
    'pi': gamma[5],
    'pi2': gamma[4]@gamma[5],
    'b0': gamma[4],
    'rho': I, #gi
    'rho2': gamma[4],#gi
    'a1': gamma[5],#gi
    'b1': gamma[4]@gamma[5]#gi

}

derivative_dict = {
    'none': None,
    'nabla': 'nabla',
    'B': 'B',
    'D': 'D'
}
#----------------------------------------------------#

flavor_dict = {'D': 'charm_light', 'pion': 'light_light'}

@dataclass
class BareOperator:
    name: str
    F: str
    flavor: str
    twoI: int
    mom: str
    gamma: Any
    gamma_i: bool
    deriv: Union[str, None]

def mommy(s: str) -> str:
    nums = re.findall(r'-?\d', s)
    return "mom_" + "_".join(nums)

def parse_op(op: str) -> BareOperator:
    keys = op.split('_')
    gamma_mat = gamma_insertion_dict[keys[2]]
    # if gamma_mat in ['rho','rho2','a1','b1']:
    #     gamma_i= True
    # else:
    #     gamma_i = False
    deriv = derivative_dict.get(keys[3], keys[3]) if keys[3] != 'none' else None
    return BareOperator(
        name=op,
        F=keys[-1],
        flavor=flavor_dict[keys[0]],
        twoI=0,
        mom=mommy(keys[1]),
        gamma=gamma_mat,
        gamma_i=gamma_i,
        deriv=deriv
    )

#t1p_dict

a1p_dict = {
    'pi_none': '0mp',
    'pi2_none': '0mp',
    'rho_nabla': '0pp',
    #'rho2_nabla': '0pp',
    #'a1_B': '0pm',
    #'b1_B': '0pp',
    #'b1_nabla': '0mp'
    
}

possible_insertions = list(a1p_dict.keys())

def mom_to_str(m: Tuple[int, int, int]) -> str:
    return ''.join(str(c) if c >= 0 else f"-{abs(c)}" for c in m)

#moms_list = list(product([-1, 0, 1], repeat=3))
moms_list = [
    [(0, 0, 0),(0, 0, 0)],
    [(0, 0, 1),(0, 0, -1)],
    [(0, 1, 0),(0, -1, 0)],
    [(1, 0, 0),(-1, 0, 0)]
]

#print(moms_list)
#moms_list = 

class DiMesonFactory:
    def __init__(self):
        self.operators = []
        self.name
    op1: BareOperator
    op2: BareOperator
    name: str
    F: str

    @classmethod
    def generate_operators(
        cls,
        insertions_D: List[str],
        insertions_pi: List[str],
        momentum_pairs: List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]) -> Dict[str, 'DiMesonOperator']:
            """
            Generate all possible DiMesonOperator combinations for D-pi in a1p irrep.
            Returns a dict keyed by operator name.
            """
            operators = {}
            idx = 0
            #di_mesons: List['DiMesonOperator'] = []
            for pair in momentum_pairs:
                mom1, mom2 = pair
                mom1_str = mom_to_str(mom1)
                mom2_str = mom_to_str(mom2)
                for ins1 in insertions_D:
                    g1, d1 = ins1.split('_')
                    op1_str = f"D_{mom1_str}_{g1}_{d1}_a1p"
                    op1 = parse_op(op1_str)
                    for ins2 in insertions_pi:
                        g2, d2 = ins2.split('_')
                        op2_str = f"pion_{mom2_str}_{g2}_{d2}_a1p"
                        op2 = parse_op(op2_str)

                        dimeson_op_name = f"{op1.name}X{op2.name}"
                        op = cls(op1=op1, op2=op2, name=dimeson_op_name, F='a1p')
                        operators[dimeson_op_name] = op
                        idx += 1
                        #di_mesons.append(dim)
                # Register ordered list for integer indexing
            ordered = list(operators.values())
            cls._ordered = ordered
            cls._name_to_idx = {op.name: i for i, op in enumerate(ordered)}
            cls._idx_to_name = {i: op.name for i, op in enumerate(ordered)}

            print(f"[DiMeson] Generated {len(ordered)} operators:")
            for i, op in enumerate(ordered):
                print(f"  op {i+1:2d}: {op.name}")
            return operators
    
    @classmethod
    def get_ordered(cls):
        return cls._ordered

    @classmethod
    def name_to_index(cls, name: str) -> int:
        return cls._name_to_idx[name]

    @classmethod
    def index_to_name(cls, idx: int) -> str:
        return cls._idx_to_name[idx]
