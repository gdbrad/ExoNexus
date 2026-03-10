"""https://arxiv.org/pdf/1803.09673 describes dividing the original 5x5 operator basis into two subsets of 3 and 2 operators, where the 3-operator subset has a dominant positive-definite contribution to the correlator matrix at t0. This script implements a similar strategy to identify and drop noisy operators that lead to indefinite correlator matrices, ensuring a more stable GEVP analysis."""


import os
import argparse
import numpy as np
import h5py
import gvar as gv
import lsqfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gevp_spec
import gevp_nolan

def log_m_eff(C):
    """Standard logarithmic effective mass: ln(C(t)/C(t+1))"""
    ratio = C[:-1] / C[1:]
    meff = []
    for r in ratio:
        if r.mean > 0:
            meff.append(gv.log(r))
        else:
            meff.append(gv.gvar(np.nan, np.nan))
    return np.array(meff)

def scan_fit_windows(lam0, tmin_list, tmax, prior, fcn):
    results = []
    for tmin in tmin_list:
        t = np.arange(tmin, tmax)
        try:
            fit = lsqfit.nonlinear_fit(
                data=(t, lam0[tmin:tmax]),
                prior=prior,
                fcn=fcn,
                svdcut=1e-2
            )
            results.append({
                "tmin": tmin,
                "E": fit.p["E0"],
                "chi2dof": fit.chi2 / fit.dof,
                "Q": fit.Q
            })
        except Exception:
            pass
    return results

def two_exp(t, p, t0):
    """Standard two-state decaying exponential"""
    E0 = p["E0"]
    E1 = p["E0"] + p["dE1"]  
    # Using (t - t0) anchors the amplitudes A0 and A1 to O(1) for priors
    return p["A0"] * gv.exp(-E0 * (t - t0)) + p["A1"] * gv.exp(-E1 * (t - t0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specfile", required=True)
    parser.add_argument("--t0", type=int, default=7)
    parser.add_argument("--tmin", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=24)
    args = parser.parse_args()

    # lattice scale for beta=3.4, 32^3x64 ensemble
    # (Update this a^-1 explicitly based on your specific ensemble)
    a_inv_MeV = 2000.0  # Placeholder: ~0.1 fm

    base_dir = os.path.dirname(os.path.abspath(args.specfile))
    plots_dir = os.path.join(base_dir, "plots")
    fits_dir = os.path.join(base_dir, "fits")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(fits_dir, exist_ok=True)

    with h5py.File(args.specfile, "r") as f:
        channels = list(f.keys())
        for channel in channels:
            for flavor in f[channel]:
                channel_plots_dir = os.path.join(plots_dir, channel, flavor)
                channel_fits_dir = os.path.join(fits_dir, channel, flavor)
                os.makedirs(channel_plots_dir, exist_ok=True)
                os.makedirs(channel_fits_dir, exist_ok=True)

                print(f"Channel: {channel} | Flavor: {flavor}")
                
                try: 
                    dset = f[channel][flavor]["t0avg"]["Matrix"]
                except KeyError: 
                    continue
                
                Cjk_full = dset[:] 
                ops_full = [o.decode() for o in f[channel][flavor]["t0avg"]["operators"][:]]
                
                Cjk_full = 0.5 * (Cjk_full + np.swapaxes(Cjk_full.conj(), 2, 3))
                
                C0_mean = np.mean(Cjk_full[:, args.t0, :, :], axis=0).real
                diags = np.diag(C0_mean)
                
                # Split into positive and negative metric bases
                keep_pos = [i for i, d in enumerate(diags) if d > 1e-12]
                keep_neg = [i for i, d in enumerate(diags) if d < -1e-12]
                
                def prune_indefinite(keep_list, sign):
                    keep_current = list(keep_list)
                    while len(keep_current) > 1:
                        sub_C = C0_mean[np.ix_(keep_current, keep_current)] * sign
                        evals, _ = np.linalg.eigh(sub_C)
                        if np.min(evals) > 1e-11:
                            break
                        sub_diags = np.abs(np.diag(sub_C))
                        worst_local_idx = np.argmin(sub_diags)
                        print(f"  -> Subset indefinite. Dropping noisy operator {keep_current[worst_local_idx]}")
                        keep_current.pop(worst_local_idx)
                    return keep_current
                
                groups = []
                pos_pruned = prune_indefinite(keep_pos, 1)
                if pos_pruned:
                    groups.append(("pos", pos_pruned, 1))
                
                neg_pruned = prune_indefinite(keep_neg, -1)
                if neg_pruned:
                    groups.append(("neg", neg_pruned, -1))
                    
                print(f"Divided into {len(groups)} GEVP subsets:")
                for gn, k, s in groups:
                    print(f"  - {gn} metric subset: {[ops_full[i] for i in k]}")
                
                for group_name, keep, dom_sign in groups:
                    print(f"\n--- Processing {group_name.upper()} subset ---")
                    Cjk = Cjk_full[:, :, keep, :][:, :, :, keep]
                    ops = [ops_full[i] for i in keep]
                    Nbins, Lt, Nops, _ = Cjk.shape

                    Cjk = Cjk * dom_sign
                    C_scale = np.max(np.abs(np.diag(np.mean(Cjk[:, args.t0, :, :], axis=0))))
                    regulator = max(1e-9 * C_scale, 1e-12)
                    Cjk += regulator * np.eye(Nops)[np.newaxis, np.newaxis, :, :]

                    raw_correlators = {}
                    part_name = f"{channel}_{flavor}_{group_name}"
                    
                    for i, op_snk in enumerate(ops):
                        for j, op_src in enumerate(ops):
                            corr_2d = Cjk[:, :, i, j]
                            key = (part_name, (j, i))
                            raw_correlators[key] = corr_2d

                    nolan_gevp = gevp_nolan.GEVP(
                        raw_correlators, 
                        t0=args.t0, 
                        td=args.t0 + 1
                    )

                    opt_corrs_raw = nolan_gevp._diagonalize(
                        diagonal_only=True, as_gvar=False
                    )
                    
                    N_kept = len([k for k in opt_corrs_raw.keys() if k[0] == part_name])
                    
                    lam_jk_0 = opt_corrs_raw[(part_name, (0, 0))]
                    lam0 = gevp_spec.jack_to_gvar(lam_jk_0)

                    # 1b. Plot raw diagonal correlators C_ii(t)
                    colors = plt.cm.tab10.colors
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for i, op in enumerate(ops):
                        C_ii = gevp_spec.jack_to_gvar(Cjk[:, :, i, i])
                        t_range = np.arange(Lt)
                        valid_i = np.array([not np.isnan(gv.mean(C_ii[t])) and gv.mean(C_ii[t]) > 0
                                            for t in range(Lt)])
                        ax.errorbar(t_range[valid_i], gv.mean(C_ii[valid_i]), yerr=gv.sdev(C_ii[valid_i]),
                                    fmt='o', markersize=4, capsize=3, color=colors[i % 10],
                                    label=op)
                    ax.axvline(args.t0, color='gray', linestyle='--', lw=1, label=f"$t_0={args.t0}$")
                    ax.set_yscale("log")
                    ax.set_xlim(0, Lt // 2)
                    ax.set_xlabel("$t/a$")
                    ax.set_ylabel("$C_{ii}(t)$")
                    ax.legend(fontsize=7, loc="upper right")
                    ax.set_title(f"{channel}  {flavor}  ({group_name})  —  diagonal correlators")
                    fig.tight_layout()
                    fig.savefig(os.path.join(channel_plots_dir, f"{group_name}_diag_corr.png"), dpi=150)
                    plt.close(fig)

                    # 2. Plot Principals (zoomed to fit window)
                    colors = plt.cm.tab10.colors
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for n in range(N_kept):
                        lam_gv = gevp_spec.jack_to_gvar(opt_corrs_raw[(part_name, (n, n))])
                        t_range = np.arange(Lt)
                        valid_n = np.array([not np.isnan(gv.mean(lam_gv[t])) and gv.mean(lam_gv[t]) > 0
                                            for t in range(Lt)])
                        ax.errorbar(t_range[valid_n], gv.mean(lam_gv[valid_n]), yerr=gv.sdev(lam_gv[valid_n]),
                                    fmt='o', markersize=4, capsize=3, color=colors[n % 10], label=f"state {n}")
                    ax.axvline(args.t0, color='gray', linestyle='--', lw=1, label=f"$t_0={args.t0}$")
                    ax.set_yscale("log")
                    ax.set_xlim(max(0, args.t0 - 2), min(Lt, args.tmax + 4))
                    ax.set_xlabel("$t/a$")
                    ax.set_ylabel("$\\lambda(t, t_0)$")
                    ax.legend(fontsize=8)
                    ax.set_title(f"{channel}  {flavor}  ({group_name})  —  principal correlators")
                    fig.tight_layout()
                    fig.savefig(os.path.join(channel_plots_dir, f"{group_name}_principal.png"), dpi=150)
                    plt.close(fig)
                    
                    # 3. Fit Ground State 
                    prior = gv.BufferDict()
                    try:
                        val1 = gv.mean(lam0[args.tmin])
                        val2 = gv.mean(lam0[args.tmin + 1])
                        guess_E0 = np.log(val1 / val2)
                        if guess_E0 <= 0 or np.isnan(guess_E0):
                            guess_E0 = 0.5 
                    except:
                        guess_E0 = 0.5

                    prior["E0"] = gv.gvar(guess_E0, guess_E0 * 0.5)
                    prior["A0"] = gv.gvar(1.0, 1.0)
                    prior["A1"] = gv.gvar(0.1, 1.0)
                    prior["dE1"] = gv.gvar(0.5, 0.5) 
                    
                    t = np.arange(args.tmin, args.tmax)
                    fit = lsqfit.nonlinear_fit(
                        data=(t, lam0[args.tmin:args.tmax]),
                        prior=prior,
                        fcn=lambda t, p: two_exp(t, p, args.t0),
                        svdcut=1e-2
                    )
                    
                    E_lat = fit.p['E0']
                    E_phys = E_lat * a_inv_MeV
                    
                    print(f"Ground state E0 = {E_lat} lat")
                    print(f"Physical Mass   = {E_phys} MeV")
                    
                    fit_filename = os.path.join(channel_fits_dir, f"{group_name}_fit.txt")
                    with open(fit_filename, 'w') as fit_file:
                        fit_file.write(str(fit.format(maxline=True)) + "\n\n")
                        fit_file.write(f"Ground state E0 = {E_lat}\n")
                        fit_file.write(f"Physical Mass = {E_phys} MeV\n")

                    # 4. Smooth continuous effective mass from two-exponential fit
                    t_dense = np.linspace(args.tmin - 0.5, args.tmax - 0.5, 300)
                    p_fit = fit.p
                    E0_p = p_fit["E0"]
                    E1_p = p_fit["E0"] + p_fit["dE1"]
                    A0_p = p_fit["A0"]
                    A1_p = p_fit["A1"]
                    dt_arr = t_dense - args.t0
                    exp0 = np.array([gv.exp(-E0_p * float(ti)) for ti in dt_arr])
                    exp1 = np.array([gv.exp(-E1_p * float(ti)) for ti in dt_arr])
                    meff_dense = (A0_p * E0_p * exp0 + A1_p * E1_p * exp1) / (A0_p * exp0 + A1_p * exp1)

                    # 5. Calculate Data Effective Mass
                    m_eff = log_m_eff(lam0)
                    t_eff = np.arange(Lt - 1) + 0.5

                    # 6. Plot M_eff + smooth fit overlay (zoomed to fit window)
                    xlim_lo = max(0.0, args.tmin - 2.0)
                    xlim_hi = min(float(Lt), args.tmax + 2.0)

                    window_data = [m_eff[t] for t in range(args.tmin, args.tmax - 1)
                                   if t < len(m_eff) and not np.isnan(gv.mean(m_eff[t]))]
                    if len(window_data) > 0:
                        yw = np.array([gv.mean(m) for m in window_data])
                        ye = np.array([gv.sdev(m) for m in window_data])
                        y_pad = (np.max(yw + ye) - np.min(yw - ye)) * 0.3
                        ylim_lo = max(0.0, np.min(yw - ye) - y_pad)
                        ylim_hi = 3.0
                        #ylim_hi = np.max(yw + ye) + y_pad
                    else:
                        ylim_lo, ylim_hi = None, None

                    fig, ax = plt.subplots(figsize=(8, 5))
                    valid = np.array([not np.isnan(gv.mean(m)) for m in m_eff])
                    ax.errorbar(t_eff[valid], gv.mean(m_eff[valid]), yerr=gv.sdev(m_eff[valid]),
                                fmt='o', color='steelblue', markersize=4, capsize=3,
                                label="$m_\\mathrm{eff}$")

                    meff_dn_mean = gv.mean(meff_dense)
                    meff_dn_sdev = gv.sdev(meff_dense)
                    ax.plot(t_dense, meff_dn_mean, color='red', lw=1.5, label="fit")
                    ax.fill_between(t_dense, meff_dn_mean - meff_dn_sdev, meff_dn_mean + meff_dn_sdev,
                                    color='red', alpha=0.25)

                    ax.axhline(gv.mean(E_lat), linestyle="--", color='black', lw=1,
                               label=f"$E_0 = {E_lat}$")
                    ax.fill_between([xlim_lo, xlim_hi],
                                    gv.mean(E_lat) - gv.sdev(E_lat),
                                    gv.mean(E_lat) + gv.sdev(E_lat),
                                    color='black', alpha=0.1)

                    ax.axvline(args.tmin - 0.5, color='gray', linestyle=':', lw=1)
                    ax.axvline(args.tmax - 0.5, color='gray', linestyle=':', lw=1, label="fit window")

                    if ylim_lo is not None:
                        ax.set_ylim(ylim_lo, ylim_hi)
                    ax.set_xlim(xlim_lo, xlim_hi)
                    ax.set_xlabel("$t/a$")
                    ax.set_ylabel("$m_\\mathrm{eff}(t)$")
                    ax.legend(fontsize=8, loc="upper right")
                    ax.set_title(f"{channel}  {flavor}  ({group_name})  —  $aE_0 = {E_lat}$")
                    fig.tight_layout()
                    fig.savefig(os.path.join(channel_plots_dir, f"{group_name}_meff_fit.png"), dpi=150)
                    plt.close(fig)

                    # 7. Stability Scans — chi2/dof coloured scatter
                    tmin_list = range(3, 10)
                    results = scan_fit_windows(lam0, tmin_list, args.tmax, prior, lambda t, p: two_exp(t, p, args.t0))
                    if len(results) > 0:
                        tmins  = [r["tmin"] for r in results]
                        E_means = [gv.mean(r["E"]) for r in results]
                        E_sdevs = [gv.sdev(r["E"]) for r in results]
                        chi2s  = [r["chi2dof"] for r in results]

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.errorbar(tmins, E_means, yerr=E_sdevs,
                                    fmt='none', color='gray', capsize=3, zorder=2)
                        sc = ax.scatter(tmins, E_means, c=chi2s, cmap='RdYlGn_r',
                                        vmin=0, vmax=2, zorder=3, s=50)
                        plt.colorbar(sc, ax=ax, label="$\\chi^2$/dof")

                        ax.axvline(args.tmin, color='steelblue', linestyle='--', lw=1,
                                   label=f"selected $t_{{\\min}}={args.tmin}$")
                        ax.axhline(gv.mean(E_lat), color='black', linestyle='--', lw=1)
                        ax.fill_between([min(tmins) - 0.5, max(tmins) + 0.5],
                                        gv.mean(E_lat) - gv.sdev(E_lat),
                                        gv.mean(E_lat) + gv.sdev(E_lat),
                                        color='black', alpha=0.12, label="$E_0$ (selected fit)")

                        ax.set_xlabel("$t_{\\min}$")
                        ax.set_ylabel("$E_0$")
                        ax.legend(fontsize=8)
                        ax.set_title(f"{channel}  {flavor}  ({group_name})  —  stability")
                        fig.tight_layout()
                        fig.savefig(os.path.join(channel_plots_dir, f"{group_name}_stability.png"), dpi=150)
                        plt.close(fig)

if __name__ == "__main__":
    main()