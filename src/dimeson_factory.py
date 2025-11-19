from dataclasses import dataclass
import numpy as np
from itertools import product
import re
from typing import List, Dict, Union, Tuple, Any

I = np.identity(4)
from insertion_factory.gamma import gamma,gamma_i

"""see https://arxiv.org/pdf/1607.07093 table 5

"""

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
    #'rho_nabla': '0pp',
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

@dataclass
class DiMesonOperator:
    op1: BareOperator
    op2: BareOperator
    name: str
    F: str

    @classmethod
    def generate_operators(cls) -> Dict[str, 'DiMesonOperator']:
            """
            Generate all possible DiMesonOperator combinations for D-pi in a1p irrep.
            Returns a dict keyed by operator name.
            """
            di_mesons: List['DiMesonOperator'] = []
            for pair in moms_list:
                mom1, mom2 = pair
                mom1_str = mom_to_str(mom1)
                mom2_str = mom_to_str(mom2)
                for ins1 in possible_insertions:
                    g1, d1 = ins1.split('_')
                    op1_str = f"D_{mom1_str}_{g1}_{d1}_a1p"
                    op1 = parse_op(op1_str)
                    for ins2 in possible_insertions:
                        g2, d2 = ins2.split('_')
                        op2_str = f"pion_{mom2_str}_{g2}_{d2}_a1p"
                        op2 = parse_op(op2_str)
                        dim_name = f"{op1.name}X{op2.name}"
                        dim = cls(op1=op1, op2=op2, name=dim_name, F='a1p')
                        di_mesons.append(dim)
            print(f"Total number of DiMeson operators created: {len(di_mesons)}")
            for i, dim in enumerate(di_mesons):
                print(f"op {i+1}: {dim.name}")
            operators: Dict[str, 'DiMesonOperator'] = {dim.name: dim for dim in di_mesons}
            # ADD THIS: ordered list + name lookup
            cls._ordered_names = list(operators.keys())        # e.g. ["D_000_pi_none_a1pXpi_000_pi_none_a1p", ...]
            cls._name_to_idx = {name: i for i, name in enumerate(cls._ordered_names)}
            cls._idx_to_name = {i: name for i, name in enumerate(cls._ordered_names)}

            print(f"DiMesonOperator: registered {len(cls._ordered_names)} operators with integer mapping")
            return operators

    @classmethod
    def name_to_index(cls, name: str) -> int:
        return cls._name_to_idx[name]

    @classmethod
    def index_to_name(cls, idx: int) -> str:
        return cls._idx_to_name[idx]

    @classmethod
    def num_operators(cls) -> int:
        return len(cls._ordered_names)
