import sys
sys.path.append('../')
sys.path.append('../../')
import h5py
import numpy as np
import matplotlib.pyplot as plt


file = 'pipi_2pt_nvec_64_tsrc_24_task1.h5'
with h5py.File(file) as f:
    data = f['/pipi_000/pipi/direct/cfg_400_tsrc_avg'][:]

print(data)
import matplotlib.pyplot
plt.plot(np.arange(64), data, '.', )
plt.yscale('log')
plt.legend()
plt.savefig('pipi.jpg')

# compute eff mass 

