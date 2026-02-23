"""multi-particle operators projected onto an irrep"""
from dataclasses import dataclass
import numpy as np 
from itertools import product 
from typing import List 

IDEN = np.identity(4)
"""here we want to make a single operator by first applying Wigner D-matrices to single operators 
then `MultiHadronOperator` contains the product of single operators. 
"""

@dataclass
class SingleHadronOperator:
    r"""
    :math:`i`, :math:`J_i`, :math:`M_i`, :math:`\vec{p}`.
    """
    name: str  # eg. "pion"
    smear_label: str # SS,PS
    mom_class: list[int] # canonical mom type  eg. [1 1 1]
    disp_list: list[int] 

@dataclass 
class ProjectedOperator(SingleHadronOperator): 
    irrep: str 
    flavor: str 
    mom_class: list[int]

@dataclass
class CGPair: 
    left: str 
    right: str 
    target: ProjectedOperator

@dataclass
class MultiHadronOperator:
    creation_operator: bool
    operators: List[ProjectedOperator]
    cg_pairs: List[CGPair]
    flavor: str
    ir_mom: str
    op: str


def main(mhi_string:str)-> MultiHadronOperator:
    
    



if __name__== "__main__": 
    '''get attributes from lookup table and momentum average equal and opposite momenta
    produces projected operator at each timeslice 
    
    '''
    ex = "XXpion_pionxD0_J0__J0_D2A2__-101xxpion_pionxD0_J0__J0_D2A2__110__F3,1_D2B1P,1__011XXpion_pionxD0_J0__J0_D2A2__0-1-1__F5,1_T1mM,1__000"

    pion_0 = 'pion_proj0_p100_H0D4A2'
    pion_1 = 'pion_proj1_p010_H0D4A2'
    pion_2 = 'pion_proj2_p001_H0D4A2'






    main(ex)

