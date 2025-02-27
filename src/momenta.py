'''momenta manipulation utils'''
from dataclasses import dataclass
import numpy as np 
import sympy 

def canonical_order(mom: str, return_as_array: bool = False):
    """
    Canonically order a momentum string of the form 'mom_a_b_c'.
    Returns either a string (default) or an array of 3 integers if `return_as_array` is True.
    """
    components = list(map(int, mom.split('_')[1:]))
    components = [abs(x) for x in components]
    components.sort(reverse=True)
    if return_as_array:
        return components
    else:
        return "mom_" + "_".join(map(str, components))

# automated CG coeffs # 

@dataclass
class CubicCanonicalRotation: 
    alpha: np.cdouble
    beta: np.cdouble 
    gamma: np.cdouble

def return_rotation_euler(angles:list[sympy.re],seq='xyz'):
    '''return rotation matrix D_mumu calculated from the Euler angles; 
    {phi,theta,psi} in units of pi
    
    '''
    rot_matrix = sympy.algebras.Quaternion.from_euler(angles,seq)

    return rot_matrix 