import numpy as np
import gvar as gv
from scipy import linalg as la

def bootstrap_to_gvar(Cboot):
    """
    Convert bootstrap samples to a gvar array.
    Cboot shape: (Nboot, ...)
    """
    mean = np.mean(Cboot, axis=0)
    flat = Cboot.reshape(Cboot.shape[0], -1)
    cov = np.cov(flat, rowvar=False)
    return gv.gvar(mean, cov)

def jack_to_gvar(Cjk):
    """
    Convert Jackknife samples to a gvar array using the correct (N-1) scaling.
    Cjk shape: (Ncfg, ...)
    """
    N = Cjk.shape[0]
    mean = np.mean(Cjk, axis=0)
    flat = Cjk.reshape(N, -1)
    cov = np.cov(flat, rowvar=False) * (N - 1)
    
    # Check for scalar/1D vs multidimensional shapes when creating gvar
    if mean.shape == ():
        return gv.gvar(mean, np.sqrt(cov))
    return gv.gvar(mean, cov)

def effective_mass(C):
    """Calculate the simple effective mass: ln( C(t) / C(t+1) )."""
    return gv.log(C[:-1] / C[1:])

def solve_gevp_jack(Cjk, t0, tol_rel=1e-6):
    """
    Solve the Generalized Eigenvalue Problem (GEVP) on Jackknife blocks.
    Uses the ensemble mean of C(t0) to define the metric and applies a 
    singular value cutoff (tol_rel) to project out noisy directions.
    
    Cjk shape: (Ncfg, Lt, N, N)
    Returns lam: (Ncfg, Lt, N_kept) principal correlators.
    """
    Ncfg, Lt, N, _ = Cjk.shape
    
    # 1. Construct metric from the ensemble average at t0
    C0 = np.mean(Cjk[:, t0, :, :], axis=0)
    C0 = 0.5 * (C0 + C0.conj().T)  # Ensure hermiticity
    evals, evecs = la.eigh(C0)
    
    # 2. Project onto active non-noisy subspace
    tol = tol_rel * np.max(evals)
    keep = evals > tol
    if not np.any(keep):
        raise RuntimeError("All C(t0) eigenvalues are below tolerance, matrix is too noisy.")
        
    evals = evals[keep]
    evecs = evecs[:, keep]
    N_kept = len(evals)
    print(f"GEVP: keeping {N_kept} / {N} modes at t0={t0}.")
    
    # Compute C0^{-1/2} restricted to the active subspace
    C0_inv_sqrt = evecs @ np.diag(evals**-0.5) @ evecs.conj().T

    # 3. Solve GEVP block-by-block
    lam = np.zeros((Ncfg, Lt, N_kept))
    
    for k in range(Ncfg):
        Cb = Cjk[k]
        for t in range(Lt):
            Ct = 0.5 * (Cb[t] + Cb[t].conj().T)
            
            # Map generalized eigenproblem to standard eigenproblem
            M = C0_inv_sqrt @ Ct @ C0_inv_sqrt.conj().T
            w, _ = la.eigh(M)
            
            # Sort descending (largest eigenvalue == ground state principal)
            w_sorted = np.sort(w)[::-1].real
            lam[k, t, :len(w_sorted)] = w_sorted

    return lam

def solve_gevp_bootstrap(Cboot, t0):
    """
    Standard GEVP solver for bootstrap blocks using Cholesky decomposition.
    Cboot shape: (Nboot, Lt, N, N)
    """
    Nboot, Lt, N, _ = Cboot.shape
    lam_boot = np.zeros((Nboot, Lt, N))

    for b in range(Nboot):
        Cb = Cboot[b]
        C0 = Cb[t0]
        C0 = 0.5 * (C0 + C0.conj().T)
        
        try:
            L = la.cholesky(C0, lower=True)
            Linv = la.inv(L)
        except la.LinAlgError:
            # Fallback for poorly conditioned matrices
            C0 += 1e-8 * np.eye(N)
            L = la.cholesky(C0, lower=True)
            Linv = la.inv(L)

        for t in range(Lt):
            Ct = 0.5 * (Cb[t] + Cb[t].conj().T)
            M = Linv @ Ct @ Linv.T
            w, _ = la.eigh(M)
            lam_boot[b, t] = np.sort(w)[::-1].real
            
    return lam_boot