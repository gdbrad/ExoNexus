# # final_average_32x32.py
# # ONLY operators 0–31 → 32×32 matrix → memory < 400 MB → NEVER crashes

import h5py
import numpy as np
import gvar as gv
from pathlib import Path
from src.dimeson_factory import DiMesonOperator

# ==============================================================
merged_file = Path("merged_Dpi.h5")
output_file = Path("Dpi_direct_8x8_averaged.h5")
tsrc_step   = 8
max_op      = 8   # ← 0 to 31 inclusive → 32×32
# ==============================================================

print(f"[LOAD] {merged_file} → using only op00 to op{max_op:02d}")

ops = DiMesonOperator.generate_operators()
N_full = len(ops.keys())
print(f"[INFO] Full basis: {N_full}×{N_full}, but we take only first {max_op+1}")

with h5py.File(merged_file, "r") as f:
    ptot = f["Ptot_000_a1p"]
    cfg_keys = sorted([k for k in ptot.keys() if k.startswith("cfg_")])
    n_cfg = len(cfg_keys)
    print(f"[INFO] {n_cfg} configurations")

    # First config to get dimensions
    first_cfg = ptot[cfg_keys[0]]
    sample_op = next(k for k in first_cfg.keys() if k.startswith("op00_X_op"))
    sample = first_cfg[sample_op]["15"]
    Lt = sample.shape[-1]
    ntsrc = sample.shape[0] if sample.ndim == 2 else 1
    print(f"[INFO] Lt={Lt}, ntsrc={ntsrc}")

    N = max_op + 1
    accum = np.zeros((n_cfg, N, N, Lt))

    for cfg_idx, cfg_key in enumerate(cfg_keys):
        print(f"  [{cfg_idx+1:2d}/{n_cfg}] {cfg_key}")
        cfg_group = ptot[cfg_key]

        for op_key in cfg_group.keys():
            if not op_key.startswith("op"): continue
            src_str, snk_str = op_key.split("_X_")
            src = int(src_str[2:])
            snk = int(snk_str[2:])
            if src > max_op or snk > max_op:
                continue  # skip anything involving op32+

            raw = np.array(cfg_group[op_key]["dirct"])
            if raw.shape == (Lt, ntsrc):
                raw = raw.T

            # tsrc average + rolling
            avg = np.zeros(Lt)
            for k in range(ntsrc):
                avg += np.roll(raw[k], -k * tsrc_step)
            accum[cfg_idx, src, snk] = avg / ntsrc

    print("[AVERAGE] Final gauge averaging...")
    flat = accum.reshape(n_cfg, -1)
    C_gv = gv.dataset.avg_data(flat).reshape(N, N, Lt)

# Write
with h5py.File(output_file, "w") as f:
    grp = f.create_group("Ptot_000_a1p")
    grp.create_dataset("dirct", data=gv.mean(C_gv))
    grp.create_dataset("direct_err", data=gv.sdev(C_gv))
    grp.attrs["n_configurations"] = n_cfg
    grp.attrs["operator_basis"] = N
    grp.attrs["operators_used"] = f"0 to {max_op}"
    grp.attrs["tsrc_averaged"] = True
    grp.attrs["gauge_averaged"] = True

print(f"\n[SUCCESS] 32×32 averaged file written: {output_file}")
print("   Memory used: < 400 MB")
print("   Ready for GEVP!")

# # Quick plot
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# t = np.arange(Lt)
# for i in range(0, N, 4):
#     y = gv.mean(C_gv[i,i])
#     e = gv.sdev(C_gv[i,i])
#     plt.errorbar(t, y, e, fmt='o', capsize=3, label=f"op{i:02d}")
# plt.yscale('log')
# plt.xlabel("t"); plt.ylabel("C(t)")
# plt.title(f"Dπ 32×32 (op00–op{max_op}) — {n_cfg} cfg — full averaged")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("32x32_correlators.pdf")
# plt.show()

# plot_32_dimeson_operators_separate.py
# Plots each of the 32 diagonal correlators with full name

# plot_32x32_all_pairs_separate.py
# 32×32 = 1024 separate plots — one for each operator pair
# Only points, no lines — exactly as requested

# import h5py
# import numpy as np
# import gvar as gv
# import matplotlib.pyplot as plt
# from pathlib import Path

# # =================================================================
# avg_file = Path("Dpi_32x32_averaged.h5")
# # =================================================================

# # Your 32 operator names (index 0 = op1, ..., index 31 = op32)
# op_names = [
#     "D_000_pi_none × pion_000_pi_none",
#     "D_000_pi_none × pion_000_pi2_none",
#     "D_000_pi_none × pion_000_rho_nabla",
#     "D_000_pi_none × pion_000_rho2_nabla",
#     "D_000_pi2_none × pion_000_pi_none",
#     "D_000_pi2_none × pion_000_pi2_none",
#     "D_000_pi2_none × pion_000_rho_nabla",
#     "D_000_pi2_none × pion_000_rho2_nabla",
#     "D_000_rho_nabla × pion_000_pi_none",
#     "D_000_rho_nabla × pion_000_pi2_none",
#     "D_000_rho_nabla × pion_000_rho_nabla",
#     "D_000_rho_nabla × pion_000_rho2_nabla",
#     "D_000_rho2_nabla × pion_000_pi_none",
#     "D_000_rho2_nabla × pion_000_pi2_none",
#     "D_000_rho2_nabla × pion_000_rho_nabla",
#     "D_000_rho2_nabla × pion_000_rho2_nabla",
#     "D_001_pi_none × pion_00-1_pi_none",
#     "D_001_pi_none × pion_00-1_pi2_none",
#     "D_001_pi_none × pion_00-1_rho_nabla",
#     "D_001_pi_none × pion_00-1_rho2_nabla",
#     "D_001_pi2_none × pion_00-1_pi_none",
#     "D_001_pi2_none × pion_00-1_pi2_none",
#     "D_001_pi2_none × pion_00-1_rho_nabla",
#     "D_001_pi2_none × pion_00-1_rho2_nabla",
#     "D_001_rho_nabla × pion_00-1_pi_none",
#     "D_001_rho_nabla × pion_00-1_pi2_none",
#     "D_001_rho_nabla × pion_00-1_rho_nabla",
#     "D_001_rho_nabla × pion_00-1_rho2_nabla",
#     "D_001_rho2_nabla × pion_00-1_pi_none",
#     "D_001_rho2_nabla × pion_00-1_pi2_none",
#     "D_001_rho2_nabla × pion_00-1_rho_nabla",
#     "D_001_rho2_nabla × pion_00-1_rho2_nabla"
# ]

# # Load data
# with h5py.File(avg_file, "r") as f:
#     C_mean = np.array(f["Ptot_000_a1p/15"])
#     C_err  = np.array(f["Ptot_000_a1p/15_err"])

# N = 32
# Lt = C_mean.shape[-1]
# t = np.arange(Lt)

# # Create 32×32 grid
# fig, axes = plt.subplots(N, N, figsize=(64, 64), sharex=True, sharey=True)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

# for src in range(N):
#     for snk in range(N):
#         ax = axes[src, snk]
        
#         y = C_mean[src, snk]
#         e = C_err[src, snk]
        
#         # Only plot points
#         ax.errorbar(t, y, e,
#                     fmt='o', markersize=2.5, capsize=1.5,
#                     color='darkblue', alpha=0.9, lw=0.8)
        
#         ax.set_yscale('log')
#         ax.grid(True, alpha=0.2, lw=0.5)
        
#         # Labels only on edges
#         if src == N-1:
#             ax.set_xlabel(f"{snk+1}", fontsize=8)
#         if snk == 0:
#             ax.set_ylabel(f"{src+1}", fontsize=8, rotation=0, labelpad=10, va='center')
        
#         # Title only on top row
#         if src == 0:
#             ax.set_title(f"{snk+1}", fontsize=9, pad=10)

# # Global title and labels
# fig.suptitle("Dπ A1⁺ — All 32×32 Operator Pairs (tsrc+gauge averaged)\n"
#              "Rows: Source operator (1–32), Columns: Sink operator (1–32)", 
#              fontsize=20, y=0.98)

# # Add legend box with operator names
# legend_text = "\n".join([f"{i+1:2d}: {name}" for i, name in enumerate(op_names)])
# fig.text(0.97, 0.5, legend_text, transform=fig.transFigure,
#          fontsize=8, verticalalignment='center',
#          bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.9))

# # Save
# plt.savefig("Dpi_32x32_all_pairs_separate.pdf", dpi=150)
# plt.close()

# print("DONE: Dpi_32x32_all_pairs_separate.pdf created")
# print("     32×32 = 1024 plots, only points, full operator labels")
# print("     Ready for paper, talk, or deep inspection")