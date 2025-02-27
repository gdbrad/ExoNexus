import numpy 
from numpy import load
from pathlib import Path
# path = '/home/grant/external/mhi/test/data/Oh'
path = '/home/grant/external/mhi/test/data/spinless'

data = load(path+'/basis_100.npz')
# lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])

from operator_factory import MomentaProjection
mom = MomentaProjection()
# print(MomentaProjection.load_coeffs(mom='000'))
print(mom.load_coeffs())

def parse_cgs():
    
