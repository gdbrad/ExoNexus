# plot_32x32_all_pairs_PERFECT.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# =================================================================
#avg_file   = Path("Dpi_32x32_averaged.h5")
avg_file   = Path("Dpi_direct_8x8_averaged.h5")

output_pdf = Path("Dpi_8x8_direct.pdf")
# =================================================================

with h5py.File(avg_file, "r") as f:
    C_mean = np.array(f["Ptot_000_a1p/dirct"])
    C_err  = np.array(f["Ptot_000_a1p/direct_err"])

N = 10
Lt = C_mean.shape[-1]
t = np.arange(Lt)

# Operator names (shortened for clarity)
op_names = [
    "D000 π000 π",    "D000 π000 π2",   "D000 π000 ρ∇",   "D000 π000 ρ2∇",
    "D000 π2000 π",   "D000 π2000 π2",  "D000 π2000 ρ∇",  "D000 π2000 ρ2∇",
    "D000 ρ∇000 π",   "D000 ρ∇000 π2",  "D000 ρ∇000 ρ∇",  "D000 ρ∇000 ρ2∇",
    "D000 ρ2∇000 π",  "D000 ρ2∇000 π2", "D000 ρ2∇000 ρ∇", "D000 ρ2∇000 ρ2∇",
    "D001 π00-1 π",   "D001 π00-1 π2",  "D001 π00-1 ρ∇",  "D001 π00-1 ρ2∇",
    "D001 π200-1 π",  "D001 π200-1 π2", "D001 π200-1 ρ∇", "D001 π200-1 ρ2∇",
    "D001 ρ∇00-1 π",  "D001 ρ∇00-1 π2", "D001 ρ∇00-1 ρ∇", "D001 ρ∇00-1 ρ2∇",
    "D001 ρ2∇00-1 π", "D001 ρ2∇00-1 π2","D001 ρ2∇00-1 ρ∇","D001 ρ2∇00-1 ρ2∇"
]

with PdfPages(output_pdf) as pdf:
    for src in range(N):
        # One row = one source operator, 8 columns (4 pages per source)
        for page in range(4):
            start_snk = page * 8
            end_snk   = start_snk + 8

            fig, axes = plt.subplots(2, 4, figsize=(28, 12), sharex=True)
            fig.suptitle(f"Source op {src+1:02d}: {op_names[src]}  →  Sink ops {start_snk+1}-{end_snk}",
                         fontsize=24, y=0.96)

            for i, snk in enumerate(range(start_snk, end_snk)):
                ax = axes[i//4, i%4]
                y = C_mean[src, snk]
                e = C_err[src, snk]

                ax.errorbar(t, y, e,
                            fmt='o', markersize=10, capsize=6, capthick=2,
                            color='darkblue', alpha=0.95, mew=2, mec='black',
                            elinewidth=2)

                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-13)
                ax.grid(True, alpha=0.4, lw=1)
                ax.set_title(f"Sink {snk+1:02d}: {op_names[snk]}", fontsize=14, pad=10)

            # Labels
            for ax in axes[-1, :]:
                ax.set_xlabel("Euclidean time t", fontsize=14)
            for ax in axes[:, 0]:
                ax.set_ylabel("C(t)", fontsize=14)

            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            pdf.savefig(fig, dpi=200)
            plt.close(fig)

            print(f"Source {src+1:02d} — page {page+1}/4 saved")

print(f"\nPERFECTION ACHIEVED → {output_pdf}")
print("   • 32 sources × 4 pages = 128 total pages")