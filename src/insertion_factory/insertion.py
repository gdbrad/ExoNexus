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

insertion_dict = {
    'a0': I,
    'pi': gamma[5],
    'pi2': gamma[4]@gamma[5],
    'b0': gamma[4],
    'rho': gamma_i,
    'rho2': gamma[4]@gamma_i,
    'a1': gamma[5]@gamma_i,
    'b1': gamma[4]@gamma[5]@gamma_i,

}

flavor_dict = {'D':'charm_light',
                   'pi':'light_light'}