# inspect_full_matrix.py
# Full inspection of a single-config Dπ correlator file
# Shows: D_corr, pi_corr, 15 and 6 for ALL operator pairs

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

# ==============================================================
input_file = Path("/p/scratch/exflash/dpi-contractions/b3.4-s24t64/results-multi-run/b3.4-s24t64_Dpi_cfg2840_matrix_n64_ntsrc16_2025-11-27.h5") 
plot_dir   = input_file.parent / "inspection_plots"
#plot_dir.mkdir(exist_ok=True)
# ==============================================================

#!/usr/bin/env python3
# quick_inspect.py
# Clean, robust, publication-ready inspection script
# • tsrc + rolling averaged D and π
# • Only markers, no lines
# • Works from anywhere

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# ==============================================================

with h5py.File(input_file, "r") as f:
    ptot = f["Ptot_000_a1p"]

    # --- Operator basis ---
    op_keys = [k for k in ptot.keys() if k.startswith("op") and "_X_" in k]
    if not op_keys:
        raise RuntimeError("No operator pairs found!")

    src_ids = [int(k.split("_X_")[0][2:]) for k in op_keys]
    snk_ids = [int(k.split("_X_")[1][2:]) for k in op_keys]
    N = max(max(src_ids), max(snk_ids)) + 1
    Lt = ptot[op_keys[0]]["15"].shape[-1]
    print(f"[INFO] {N}×{N} operator basis, Lt = {Lt}")

    # --- Load and tsrc-average D and π correlators ---
    tsrc_step = 4

    def tsrc_average(data):
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ntsrc = data.shape[0]
        avg = np.zeros(Lt)
        for k in range(ntsrc):
            avg += np.roll(data[k], -k * tsrc_step)
        return avg / ntsrc

    D_raw  = np.array(ptot.get("D_correlator",  np.zeros(Lt)))
    pi_raw = np.array(ptot.get("pi_correlator", np.zeros(Lt)))

    D_corr  = tsrc_average(D_raw)
    pi_corr = tsrc_average(pi_raw)

    t = np.arange(Lt)

    # --- Load full 15 and 6 matrices (tsrc-averaged) ---
    C15 = np.zeros((N, N, Lt))
    C6  = np.zeros((N, N, Lt))

    print(f"[PROC] Loading {len(op_keys)} operator pairs...")
    for key in op_keys:
        src = int(key.split("_X_")[0][2:])
        snk = int(key.split("_X_")[1][2:])
        C15[src, snk] = tsrc_average(np.array(ptot[key]["15"]))
        if "6" in ptot[key]:
            C6[src, snk] = tsrc_average(np.array(ptot[key]["6"]))

# ==============================================================
# PLOTS — only markers, no lines
# ==============================================================

# 1. Single mesons — tsrc averaged
plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.plot(t, D_corr, 'o', markersize=5, color='tab:blue', alpha=0.9)
plt.yscale('log')
plt.title("D correlator (tsrc+roll averaged)")
plt.xlabel("t/a"); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t, pi_corr, 's', markersize=5, color='tab:red', alpha=0.9)
plt.yscale('log')
plt.title("π correlator (tsrc+roll averaged)")
plt.xlabel("t/a"); plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / "single_mesons_averaged.pdf", dpi=180)
plt.close()

# # 2. Heatmap at t=6
# t_show = 6
# fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

# im1 = ax[0].imshow(C15[:,:,t_show], cmap="viridis", norm=plt.pcolor.LogNorm(vmin=1e-16))
# ax[0].set_title(f"15 (A1⁺) at t = {t_show}")
# plt.colorbar(im1, ax=ax[0])

# im2 = ax[1].imshow(C6[:,:,t_show], cmap="plasma", norm=plt.pcolor.LogNorm(vmin=1e-20))
# ax[1].set_title(f"6 at t = {t_show}")
# plt.colorbar(im2, ax=ax[1])

# for a in ax:
#     a.set_xlabel("sink operator index")
# ax[0].set_ylabel("source operator index")
# plt.tight_layout()
# plt.savefig(plot_dir / f"heatmap_t{t_show}.pdf", dpi=180)
# plt.close()

# 3. All diagonal 15 correlators
plt.figure(figsize=(12, 7))
for i in range(N):
    plt.plot(t, C15[i,i], 'o', markersize=4, alpha=0.8,
             label=f"op{i:02d}" if i % max(1, N//10) == 0 else "")
plt.yscale('log')
plt.xlabel("t/a"); plt.ylabel("C_{ii}(t)  [15]")
plt.title(f"Diagonal correlators — {N}×{N} basis")
plt.grid(alpha=0.3)
plt.legend(fontsize=9, ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig(plot_dir / "diagonal_15.pdf", dpi=180)
plt.close()

# # 4. Effective mass from diagonal 15
# meff = np.log(C15[:,:,:,:-1] / C15[:,:,:,1:])
# diag_meff = meff[np.arange(N), np.arange(N)]

# plt.figure(figsize=(11, 6))
# for i in range(0, N, max(1, N//10)):
#     plt.plot(t[1:], diag_meff[i], 'o', markersize=5, label=f"op{i:02d}")
# plt.xlabel("t/a"); plt.ylabel("a m_eff(t)")
# plt.title("Effective mass — diagonal 15 correlators")
# plt.grid(alpha=0.3)
# plt.legend(fontsize=9, ncol=2)
# plt.ylim(bottom=0)
# plt.tight_layout()
# plt.savefig(plot_dir / "effmass_diagonal.pdf", dpi=180)
# plt.close()

print(f"\n[SUCCESS] All plots saved (markers only, no lines)")
print(f"    → {plot_dir.resolve()}")
print("    Files:")
print("      • single_mesons_averaged.pdf")
print("      • heatmap_t6.pdf")
print("      • diagonal_15.pdf")
print("      • effmass_diagonal.pdf")
print("\nReady for GEVP!")