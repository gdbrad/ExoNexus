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

def solve_gevp_jack(Cjk, t0, td=None):
    """
    Fixed-Basis Variational GEVP (Robust Two-Step Method).
    Safely projects out noisy/negative eigenvectors in C(t0) before solving,
    strictly maintaining the safe subspace dimension N_kept.
    """
    Ncfg, Lt, N, _ = Cjk.shape
    
    if td is None:
        td = t0 + 1 
        
    # 1. Ensemble average to find stable eigenvectors
    C_mean = np.mean(Cjk, axis=0)
    
    C0 = 0.5 * (C_mean[t0] + C_mean[t0].conj().T)
    Cd = 0.5 * (C_mean[td] + C_mean[td].conj().T)
    
    # 2. Diagonalize C0 first to find the positive-definite subspace
    evals0, evecs0 = la.eigh(C0)
    
    # Drop zero or negative eigenvalues caused by noise/linear dependence
    tol = 1e-8 * np.max(evals0)
    keep = evals0 > tol
    if not np.any(keep):
        raise RuntimeError(f"All C(t0) eigenvalues are <= 0. Matrix is totally noisy at t0={t0}.")
        
    evals0 = evals0[keep]
    evecs0 = evecs0[:, keep]
    N_kept = len(evals0)
    
    # Form the projection matrix W (Maps N -> N_kept)
    W = evecs0 @ np.diag(evals0**-0.5)
    
    # 3. Form standard eigenvalue problem STRICTLY in the N_kept subspace
    M = W.conj().T @ Cd @ W
    M = 0.5 * (M + M.conj().T) # Re-enforce hermiticity
    
    evals_d, U = la.eigh(M)
    
    # Map back to generalized eigenvectors: v = W * U  (shape: N x N_kept)
    vecs = W @ U
    
    # Sort by largest eigenvalue (slowest decay = ground state)
    idx = np.argsort(evals_d)[::-1]
    vecs_sorted = vecs[:, idx]
    
    # Normalize to strictly equal 1 at t0
    for i in range(N_kept):
        v = vecs_sorted[:, i]
        norm = np.einsum('i,ij,j->', v.conj(), C0, v).real
        if norm > 0:
            vecs_sorted[:, i] = v / np.sqrt(norm)
    
    # 4. Project the matrix at all times on all jackknife bins
    lam = np.zeros((Ncfg, Lt, N_kept))
    
    for k in range(Ncfg):
        for t in range(Lt):
            Ct = 0.5 * (Cjk[k, t] + Cjk[k, t].conj().T)
            # Decorrelate / diagonalize
            C_rotated = vecs_sorted.conj().T @ Ct @ vecs_sorted
            lam[k, t, :] = np.diag(C_rotated).real

    return lam