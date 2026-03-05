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
                "E": fit.p["E"],
                "chi2dof": fit.chi2 / fit.dof,
                "Q": fit.Q
            })
        except Exception as e:
            print(f"Fit failed for tmin = {tmin}: {e}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specfile", required=True, help="Path to the resampled h5 spec file")
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--tmin", type=int, default=4)
    parser.add_argument("--tmax", type=int, default=18)
    args = parser.parse_args()

    # Automatically create output directories next to the specfile
    base_dir = os.path.dirname(os.path.abspath(args.specfile))
    plots_dir = os.path.join(base_dir, "plots")
    fits_dir = os.path.join(base_dir, "fits")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(fits_dir, exist_ok=True)

    def one_exp(t, p, Lt):
        """Symmetric cosh-like single exponential."""
        return p["A"] * gv.exp(-p["E"] * t) + p["A"] * gv.exp(-p["E"] * (Lt - t))

    with h5py.File(args.specfile, "r") as f:
        channels = list(f.keys())
        for channel in channels:
            for flavor in f[channel]:
                print("\n====================================")
                print(f"Channel: {channel} | Flavor: {flavor}")
                
                try:
                    dset = f[channel][flavor]["t0avg"]["Matrix"]
                except KeyError:
                    print(f"Skipping {channel}/{flavor} - missing 't0avg/Matrix'.")
                    continue
                
                # Cjk shape: (Nbins, Lt, Nops, Nops)
                Cjk = dset[:]
                Cjk = Cjk.real
                ops = [o.decode() for o in f[channel][flavor]["t0avg"]["operators"][:]]
                
                # Restrict basis natively
                keep = [i for i, op in enumerate(ops)]
                Cjk = Cjk[:, :, keep, :][:, :, :, keep]
                ops = [ops[i] for i in keep]
                Nbins, Lt, Nops, _ = Cjk.shape
                
                print(f"Operators: {ops}")
                print(f"Jackknife shape: {Cjk.shape}")

                # 1. Plot Diagonals
                plt.figure()
                for i in range(Nops):
                    # Average diag over jackknife bins to properly plot
                    diag_jk = Cjk[:, :, i, i]
                    C_diag = gevp_spec.jack_to_gvar(diag_jk)
                    plt.errorbar(range(Lt), gv.mean(C_diag), yerr=gv.sdev(C_diag), fmt='o', label=ops[i])
                plt.yscale("log")
                plt.title(f"{channel} {flavor} diags")
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_diag.png"))
                plt.close()

                # 2. Solve GEVP & Plot Principals
                lam_jk = gevp_spec.solve_gevp_jack(Cjk, args.t0)
                
                # Use correct jackknife covariance translation for the ground state
                lam0 = gevp_spec.jack_to_gvar(lam_jk[:, :, 0])
                
                plt.figure()
                for n in range(Nops):
                    lam_gv = gevp_spec.jack_to_gvar(lam_jk[:, :, n])
                    plt.errorbar(range(Lt), gv.mean(lam_gv), yerr=gv.sdev(lam_gv), fmt='o', label=f"state {n}")
                plt.yscale("log")
                plt.title(f"{channel} {flavor} principal correlators")
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_principal.png"))
                plt.close()

                # 3. Effective Mass
                m_eff = gevp_spec.effective_mass(lam0)
                plt.figure()
                plt.errorbar(range(len(m_eff)), gv.mean(m_eff), yerr=gv.sdev(m_eff), fmt='o')
                plt.title(f"{channel} {flavor} effective mass")
                plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_meff.png"))
                plt.close()

                # 4. Fit Ground State
                prior = gv.BufferDict()
                prior["A"] = gv.gvar(1.0, 2.0)
                prior["E"] = gv.gvar(0.35, 0.3)
                t = np.arange(args.tmin, args.tmax)

                fit = lsqfit.nonlinear_fit(
                    data=(t, lam0[args.tmin:args.tmax]),
                    prior=prior,
                    fcn=lambda t, p: one_exp(t, p, Lt),
                    svdcut=1e-2
                )
                
                print(f"Ground state E = {fit.p['E']}")
                fit_filename = os.path.join(fits_dir, f"{channel}_{flavor}_fit.txt")
                with open(fit_filename, 'w') as fit_file:
                    fit_file.write(str(fit.format(maxline=True)) + "\n\n")
                    fit_file.write(f"Ground state E = {fit.p['E']}\n")

                # 5. Plot Principal + Fit Band
                plt.figure()
                plt.errorbar(range(Lt), gv.mean(lam0), yerr=gv.sdev(lam0), fmt='o', label="principal")
                t_fit = np.arange(args.tmin, args.tmax)
                y_fit = one_exp(t_fit, fit.p, Lt)
                plt.plot(t_fit, gv.mean(y_fit), label="fit")
                plt.fill_between(t_fit, gv.mean(y_fit) - gv.sdev(y_fit), gv.mean(y_fit) + gv.sdev(y_fit), alpha=0.3)
                plt.yscale("log")
                plt.legend()
                plt.title(f"{channel} {flavor} principal + fit")
                plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_principal_fit.png"))
                plt.close()

                # 6. Plot Eff Mass + Fit Band
                plt.figure()
                plt.errorbar(range(len(m_eff)), gv.mean(m_eff), yerr=gv.sdev(m_eff), fmt='o', label="m_eff")
                Efit = fit.p["E"]
                plt.axhline(gv.mean(Efit), linestyle="--", label="fit E")
                plt.fill_between([0, Lt], gv.mean(Efit) - gv.sdev(Efit), gv.mean(Efit) + gv.sdev(Efit), alpha=0.2)
                plt.legend()
                plt.title(f"{channel} {flavor} m_eff + fit")
                plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_meff_fit.png"))
                plt.close()

                # 7. Stability Scans
                tmin_list = range(3, 10)
                results = scan_fit_windows(lam0, tmin_list, args.tmax, prior, lambda t, p: one_exp(t, p, Lt))
                if len(results) > 0:
                    plt.figure()
                    plt.errorbar([r["tmin"] for r in results], 
                                 [gv.mean(r["E"]) for r in results], 
                                 yerr=[gv.sdev(r["E"]) for r in results], fmt='o')
                    plt.xlabel("tmin")
                    plt.ylabel("E")
                    plt.title(f"{channel} {flavor} stability")
                    plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_stability.png"))
                    plt.close()

                    plt.figure()
                    plt.plot([r["tmin"] for r in results], [r["chi2dof"] for r in results], 'o-')
                    plt.axhline(1.0, linestyle="--", color='grey')
                    plt.xlabel("tmin")
                    plt.ylabel("chi2/dof")
                    plt.title(f"{channel} {flavor} chi2 scan")
                    plt.savefig(os.path.join(plots_dir, f"{channel}_{flavor}_chi2scan.png"))
                    plt.close()

if __name__ == "__main__":
    main()