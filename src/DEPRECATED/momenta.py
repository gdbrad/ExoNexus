'''momenta manipulation utils'''
from dataclasses import dataclass
import numpy as np
from itertools import product


# ------------ Momenta ---------------------# 

def order_momenta(mom):
    '''put momentum 3-tuple into canonical order '''


def cm_to_lab(mom):
    '''transformation from COM frame to lab frame'''
    

def operator_list_irrepmom(irrep):
    pass 

class MomentaProjection: 
    def __init__(self):
        self.data_path = '/home/grant/exotraction/data'

    '''
    Here we define various 3-momenta conventions since angular momentum takes on a different form on the lattice and ceases to be a good quantum number
    Essentially, the same operator comes out for different relative momenta. In each dict, labeled by the total momentum P^2, all relative momenta satisfying this criteria
    are generated. Computes projected operator coefficients for non-zero total momenta
    '''

    def load_coeffs(self):
        data = load(self.data_path + f'/Oh/T1_minus.npz')
        lst = data.files
        for item in lst:
            return data[item]

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
    
print(canonical_order('mom_0_-2_-3'))



# automated CG coeffs # 

@dataclass
class CubicCanonicalRotation: 
    alpha: np.cdouble
    beta: np.cdouble 
    gamma: np.cdouble

# def return_rotation_euler(angles:list[sympy.re],seq='xyz'):
#     '''return rotation matrix D_mumu calculated from the Euler angles; 
#     {phi,theta,psi} in units of pi
    
#     '''
#     rot_matrix = sympy.algebras.Quaternion.from_euler(angles,seq)

#     return rot_matrix 