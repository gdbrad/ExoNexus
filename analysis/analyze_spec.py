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
    """Standard two-state decaying exponential implicitly anchored to t0."""
    E0 = p["E0"]
    E1 = p["E0"] + p["dE1"]  
    # Using (t - t0) cleanly anchors the amplitudes A0 and A1 to O(1) for priors
    return p["A0"] * gv.exp(-E0 * (t - t0)) + p["A1"] * gv.exp(-E1 * (t - t0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specfile", required=True)
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--tmin", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=24)
    args = parser.parse_args()

    # Define lattice scale for beta=3.4, 32^3x64 ensemble
    # (Update this a^-1 explicitly based on your specific ensemble standard)
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
                print(f"\n====================================")
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

                    # 2. Plot Principals
                    plt.figure()
                    for n in range(N_kept):
                        lam_gv = gevp_spec.jack_to_gvar(opt_corrs_raw[(part_name, (n, n))])
                        plt.errorbar(range(Lt), gv.mean(lam_gv), yerr=gv.sdev(lam_gv), fmt='o', label=f"state {n}")
                    plt.yscale("log")
                    plt.title(f"{channel} {flavor} ({group_name}) principal correlators")
                    plt.legend()
                    plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_{group_name}_principal.png"))
                    plt.close()
                    
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
                    
                    fit_filename = os.path.join(fits_dir, f"{channel}_{flavor}_{group_name}_fit.txt")
                    with open(fit_filename, 'w') as fit_file:
                        fit_file.write(str(fit.format(maxline=True)) + "\n\n")
                        fit_file.write(f"Ground state E0 = {E_lat}\n")
                        fit_file.write(f"Physical Mass = {E_phys} MeV\n")

                    # 4. Predict Effective Mass Fit Curve
                    t_fit = np.arange(args.tmin, args.tmax)
                    y_fit = two_exp(t_fit, fit.p, args.t0)
                    meff_fit_curve = log_m_eff(y_fit)
                    t_meff_fit = t_fit[:-1] + 0.5  # shift halfway between points

                    # 5. Calculate Data Effective Mass
                    m_eff = log_m_eff(lam0)
                    t_eff = np.arange(Lt - 1) + 0.5

                    # 6. Plot M_eff + Exponential Curve Fit Band (Zoomed In)
                    plt.figure()
                    valid = [not np.isnan(gv.mean(m)) for m in m_eff]
                    plt.errorbar(t_eff[valid], gv.mean(m_eff[valid]), yerr=gv.sdev(m_eff[valid]), fmt='o', label="data m_eff")
                    
                    # Plot the curved approaching fit band
                    plt.plot(t_meff_fit, gv.mean(meff_fit_curve), label="fit curve", color='red')
                    plt.fill_between(t_meff_fit, 
                                     gv.mean(meff_fit_curve) - gv.sdev(meff_fit_curve), 
                                     gv.mean(meff_fit_curve) + gv.sdev(meff_fit_curve), 
                                     color='red', alpha=0.3)
                    
                    # Plot horizontal asymptotic limit
                    plt.axhline(gv.mean(E_lat), linestyle="--", color='black', label="E0 asymptote")
                    plt.fill_between([0, Lt], gv.mean(E_lat) - gv.sdev(E_lat), gv.mean(E_lat) + gv.sdev(E_lat), color='black', alpha=0.1)
                    
                    # ZOOM IN dynamically to the fit window data
                    window_data = [m_eff[t] for t in range(args.tmin, args.tmax-1) if t < len(m_eff) and not np.isnan(gv.mean(m_eff[t]))]
                    if len(window_data) > 0:
                        y_min = np.min([gv.mean(m) - gv.sdev(m) for m in window_data])
                        y_max = np.max([gv.mean(m) + gv.sdev(m) for m in window_data])
                        y_pad = (y_max - y_min) * 0.5
                        plt.ylim(max(0, y_min - y_pad), y_max + y_pad)
                    
                    plt.xlim(max(0, args.tmin - 2), min(Lt, args.tmax + 4))
                    
                    plt.legend()
                    plt.title(f"{channel} {flavor} ({group_name})\nM_eff (E = {E_phys.mean:.1f} MeV)")
                    plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_{group_name}_meff_fit.png"))
                    plt.close()

                    # 7. Stability Scans (using simple exp function)
                    tmin_list = range(3, 10)
                    results = scan_fit_windows(lam0, tmin_list, args.tmax, prior, lambda t, p: two_exp(t, p, args.t0))
                    if len(results) > 0:
                        plt.figure()
                        plt.errorbar([r["tmin"] for r in results], 
                                     [gv.mean(r["E"]) for r in results], 
                                     yerr=[gv.sdev(r["E"]) for r in results], fmt='o')
                        plt.xlabel("tmin")
                        plt.ylabel("E0")
                        plt.title(f"{channel} {flavor} ({group_name}) stability")
                        plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_{group_name}_stability.png"))
                        plt.close()

if __name__ == "__main__":
    main()