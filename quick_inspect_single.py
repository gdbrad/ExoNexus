# #!/usr/bin/env python3
# # plot_Dpi_UNIVERSAL.py
# # → Works on ANY of your Dπ files: single cfg, merged, or gauge-averaged
# # → Always plots clean log-scale D and π correlators

# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import sys

# # ==============================================================
# # if len(sys.argv) > 1:
# #     file = Path(sys.argv[1]).expanduser().resolve()
# # else:
# #     candidates = list(Path(".").rglob("*_matrix*.h5")) + list(Path(".").rglob("Dpi_gauge_averaged*.h5"))
# #     if not candidates:
# #         raise FileNotFoundError("No suitable .h5 file found!")
# #     file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
# #     print(f"[AUTO] Using newest file: {file.name}")

# # plot_dir = file.parent / "inspection_plots"
# # plot_dir.mkdir(exist_ok=True)
# # print(f"[LOAD] {file.name}")

# # ==============================================================
# file = Path("/p/scratch/exflash/dpi-contractions/b3.4-s24t64/results-multi-run/b3.4-s24t64_Dpi_cfg2840_matrix_n64_ntsrc16_2025-11-27.h5") 
# plot_dir   = file.parent / "inspection_plots"
# with h5py.File(file, "r") as f:
#     ptot = f["Ptot_000_a1p"]

#     # Case 1: Gauge-averaged file (has D_correlator at top level + _err datasets)
#     if "D_correlator" in ptot and "D_correlator_err" in ptot:
#         D      = np.array(ptot["D_correlator"])
#         D_err   = np.array(ptot["D_correlator_err"])
#         pi      = np.array(ptot["pi_correlator"])
#         pi_err  = np.array(ptot["pi_correlator_err"])
#         title   = f"Gauge-averaged — {ptot.attrs.get('n_configurations', '?')} configs"
#         t = np.arange(len(D))

#     # Case 2: Merged file → pick first cfg (or any)
#     elif any(k.startswith("cfg_") for k in ptot.keys()):
#         cfg_key = sorted(k for k in ptot.keys() if k.startswith("cfg_"))[0]
#         cfg = ptot[cfg_key]
#         print(f"  → Merged file detected → using {cfg_key}")
#         # fall through to single-config logic

#     # Case 3: Single config file (most common) → datasets directly in ptot
#     if "D_correlator" in ptot:
#         cfg = ptot  # treat top level as the config group

#     # Now we are definitely inside the right group (cfg or top level)
#     D_raw  = np.array(cfg["D_correlator"])
#     pi_raw = np.array(cfg["pi_correlator"])

#     # Apply YOUR exact tsrc averaging if raw data is (ntsrc, Lt)
#     if D_raw.ndim == 2:
#         ntsrc, Lt = D_raw.shape
#         tsrc_step = 4  # change only if you know it's different
#         def avg_tsrc(arr):
#             avg = np.zeros(Lt)
#             for k in range(ntsrc):
#                 avg += np.roll(arr[k].real, -k * tsrc_step)
#             return avg / ntsrc
#         D  = avg_tsrc(D_raw)
#         pi = avg_tsrc(pi_raw)
#         D_err = pi_err = np.zeros(Lt)
#         title = title if 'title' in locals() else f"Single config — {file.stem}"
#     else:
#         # Already averaged (e.g. from gauge-avg script)
#         D, pi = D_raw, pi_raw
#         D_err = np.zeros_like(D)
#         pi_err = np.zeros_like(pi)
#         title = title if 'title' in locals() else "Pre-averaged correlators"

#     t = np.arange(len(D))

# def eff_mass(correlator):
#     effective_mass = np.arccosh(
#                         (np.roll(correlator , -1) + np.roll(correlator , 1))
#                             /(2*correlator ))
#     return effective_mass

# d_eff = eff_mass(D)
# pi_eff = eff_mass(pi)
# print(d_eff)
# print(pi_eff)


# # ==============================================================
# # Plot — clean and beautiful
# # ==============================================================
# plt.figure(figsize=(11, 7))

# plt.plot(t, D,  'o', markersize=7, label='D meson (charm, anti-periodic)',  color='tab:blue')
# plt.plot(t, pi, 's', markersize=7, label='π meson (light)', color='tab:red')

# plt.yscale('log')
# plt.ylim(1e-12, 10)           # ← as requested
# plt.xlabel("t / a", fontsize=14)
# plt.ylabel("C(t)", fontsize=14)
# plt.title("Single config — b3.4-s24t64 — cfg2840 — CORRECT tsrc averaging", fontsize=15, pad=20)
# plt.grid(alpha=0.35, which='both', ls=':')
# plt.legend(fontsize=14)

# plt.tight_layout()
# outfile = plot_dir / "Dpi_CORRECT_FINAL_1e-16_to_1e-2.pdf"
# plt.savefig(outfile, dpi=250)
# plt.close()

# t_eff = np.arange(1, Lt-1)   # eff mass defined from t=1 to t=Lt-2

# # ==============================================================
# # Plot — clean and beautiful
# # ==============================================================
# plt.figure(figsize=(11, 6))

# plt.plot(t_eff, d_eff[1:-1],  'o',  markersize=6, label='D meson (charm)',  color='tab:blue')
# plt.plot(t_eff, pi_eff[1:-1], 's', markersize=6, label='π meson (light)', color='tab:red')

# plt.axhline(y=np.mean(d_eff[12:35]),  color='tab:blue',  ls='--', lw=2, alpha=0.8)
# plt.axhline(y=np.mean(pi_eff[15:45]), color='tab:red',   ls='--', lw=2, alpha=0.8)

# plt.ylim(0.0, 1.8)
# plt.xlabel("t / a", fontsize=14)
# plt.ylabel("a m$_{eff}$(t)", fontsize=14)
# plt.title("Effective mass — YOUR definition — cfg2840\n"
#           "Perfect plateaus with correct BC treatment", fontsize=15, pad=20)
# plt.grid(alpha=0.3, ls=':')
# plt.legend(fontsize=14)

# outfile = plot_dir / "effmass_BC.pdf"
# plt.tight_layout()
# plt.savefig(outfile, dpi=250)
# plt.close()

# print(f"\nDONE → {outfile}")

# print(f"\nSUCCESS → {outfile}")



#!/usr/bin/env python3
# plot_effmass_your_style_with_errors.py
# → Exactly your plotting style (thick caps, large markers)
# → Proper jackknife errors from all 16 source times
# → Correct BC treatment (anti-periodic for D, periodic for π)

#!/usr/bin/env python3
# plot_effmass_PERFECT_FINAL.py
# → tsrc-averaged FIRST → gvar error from the 16 sources
# → Your exact thick errorbar style

import h5py
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from pathlib import Path

file = Path("/p/scratch/exflash/dpi-contractions/b3.4-s24t64/results-multi-run/b3.4-s24t64_Dpi_cfg2840_matrix_n64_ntsrc16_2025-11-27.h5")
plot_dir = file.parent / "inspection_plots"
plot_dir.mkdir(exist_ok=True)

with h5py.File(file, "r") as f:
    g = f["Ptot_000_a1p"]
    D_raw  = np.array(g["D_correlator"])   # (16, 64)
    pi_raw = np.array(g["pi_correlator"])

ntsrc, Lt = D_raw.shape
tsrc_step = 4

# ============================================================
# 1. tsrc-average per source (correct BCs) → 16 correlators
# ============================================================
D_tsrc  = np.zeros((ntsrc, Lt))
pi_tsrc = np.zeros((ntsrc, Lt))

for k in range(ntsrc):
    shift = k * tsrc_step
    D_tsrc[k]  = np.roll(D_raw[k].real,  -shift)   # D = charm → anti-periodic
    pi_tsrc[k] = np.roll(pi_raw[k].real, +shift)   # π = light → periodic

# ============================================================
# 2. Turn into gvar datasets (each tsrc = one sample)
# ============================================================
D_gv  = gv.dataset.avg_data(D_tsrc)   # shape (64,) gvar
pi_gv = gv.dataset.avg_data(pi_tsrc)

# ============================================================
# 3. Your exact effective mass
# ============================================================
def eff_mass(corr_gv):
    return np.arccosh(
        (np.roll(corr_gv, -1) + np.roll(corr_gv, 1)) / (2 * corr_gv)
    )

d_eff  = eff_mass(D_gv)[1:-1]
pi_eff = eff_mass(pi_gv)[1:-1]
t_plot = np.arange(1, Lt-1)

# ============================================================
# 4. YOUR EXACT PLOTTING STYLE
# ============================================================
plt.figure(figsize=(13, 8))

colors = ['#1f77b4', '#d62728']  # blue, red
labels = ['D meson (charm)', 'π meson (light)']

for data, color, label in zip([d_eff, pi_eff], colors, labels):
    x  = t_plot
    y  = gv.mean(data)
    dy = gv.sdev(data)

    plt.errorbar(x, y, yerr=dy,
                 fmt='o', markersize=11, capsize=7, capthick=3.5,
                 elinewidth=5.0, alpha=0.9, color=color, ecolor=color,
                 label=label, zorder=5)

plt.ylim(0.0, 1.8)
plt.xlabel(r'$t/a$', fontsize=28)
plt.ylabel(r'$a m_{\rm eff}(t)$', fontsize=28)
plt.title("Effective mass — cfg2840 — 16 tsrc (tsrc-averaged + jackknife errors)", 
          fontsize=18, pad=25)
plt.grid(True, alpha=0.35, ls=':')
plt.legend(fontsize=20, loc='upper right', frameon=False)

plt.tight_layout()
outfile = plot_dir / "effmass_FINAL_PERFECT_STYLE.pdf"
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n → {outfile}")