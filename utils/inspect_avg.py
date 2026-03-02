import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
file = Path("Dpi_gauge_averaged.h5")
plot_dir = file.parent / "inspection_plots"
plot_dir.mkdir(exist_ok=True)

print(f"Loading {file.name}")

with h5py.File(file, "r") as f:
    g = f["Ptot_000_a1p"]
    ncfg = g.attrs["n_configurations"]
    N    = g.attrs["operator_basis_size"]
    Lt   = g.attrs["Lt"]
    t    = np.arange(Lt)

    # Load means and errors (already computed by your averaging script)
    D      = np.array(g["D_correlator"])
    D_err  = np.array(g["D_correlator_err"])
    pi     = np.array(g["pi_correlator"])
    pi_err = np.array(g["pi_correlator_err"])
    C15    = np.array(g["15"])        # shape (N, N, Lt)
    C15_err= np.array(g["15_err"])

# ==============================================================
plt.figure(figsize=(15, 5))

# 1. D meson
plt.subplot(1, 3, 1)
plt.errorbar(t, D, yerr=D_err, fmt='o', capsize=4, color='tab:blue', label='D')
plt.yscale('log')
plt.ylim(bottom=1e-3)
plt.title(f"D meson correlator\n{ncfg} configurations")
plt.xlabel("t/a")
plt.grid(alpha=0.3, which='both', ls=':')
plt.legend()

# 2. π meson
plt.subplot(1, 3, 2)
plt.errorbar(t, pi, yerr=pi_err, fmt='s', capsize=4, color='tab:red', label='π')
plt.yscale('log')
plt.ylim(bottom=1e-11)
plt.title(f"π meson correlator\n{ncfg} configurations")
plt.xlabel("t/a")
plt.grid(alpha=0.3, which='both', ls=':')
plt.legend()

# 3. Diagonal elements of the 15 matrix
plt.subplot(1, 3, 3)
for i in range(N):
    label = f"op{i:02d}" if i % max(1, N//8) == 0 else None
    plt.errorbar(t, C15[i,i], yerr=C15_err[i,i],
                 fmt='o', alpha=0.7, markersize=4, capsize=2, label=label)
plt.yscale('log')
plt.title(f"Diagonal correlators (15 irrep)\n{N}×{N} basis")
plt.xlabel("t/a")
plt.grid(alpha=0.3, which='both', ls=':')
if N <= 16:
    plt.legend(fontsize=8, ncol=2)

plt.suptitle(f"Gauge-averaged Dπ correlators — {ncfg} configurations — pure numpy", 
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(plot_dir / "gauge_averaged_correlators_SIMPLE.pdf", dpi=200, bbox_inches='tight')
plt.close()

print(f"DONE → {plot_dir / 'gauge_averaged_correlators_SIMPLE.pdf'}")
print("   • Pure numpy, no gvar")
print("   • Log scale with proper error bars")
print("   • Ready to show anyone")