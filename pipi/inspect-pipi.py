import sys
sys.path.append('../')
sys.path.append('../../')
import h5py
import numpy as np
import matplotlib.pyplot as plt


file = '/p/scratch/exflash/dpi-contractions/b3.4-s24t64/results-multi-run/b3.4-s24t64_Dpi_cfg1000_matrix_n64_ntsrc16_2025-11-24.h5'
with h5py.File(file) as f:
    data = f['/Ptot_000_a1p/pi_correlator'][:]

print(data)
import matplotlib.pyplot
plt.plot(np.arange(64), data[0], '.', )
plt.yscale('log')
plt.legend()
plt.savefig('pi-test.jpg')

# compute eff mass 

