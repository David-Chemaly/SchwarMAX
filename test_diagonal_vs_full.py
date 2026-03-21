"""
Compare full eigendecomposition vs diagonal approximation for weight marginalization.
All computations on CPU (numpy) for determinism.
"""

import pickle
import numpy as np
from scipy.stats import qmc as _qmc

path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
lib_file = 'orbital_library_adaptive_Nmax1e4.pkl'

with open(path_data + lib_file, 'rb') as f:
    orb_lib = pickle.load(f)

(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = [
    np.asarray(x, dtype=np.float64) for x in orb_lib[:18]
]

# ---- NNLS solve (numpy, deterministic) ----
eps = 1e-8
y_xy_safe = np.where(np.abs(y_xy) > eps, y_xy, 1.0)

U_rz  = A_Rzphi / (sig_Rzphi[:, None] + eps)
U_xy  = A_xy / (sig_xy[:, None] + eps)
U_h1  = (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps)
U_h2  = (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps)
U_h3  = (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps)
U_h4  = (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps)
U = np.vstack([U_rz, U_xy, U_h1, U_h2, U_h3, U_h4])
y_rz  = y_Rzphi / (sig_Rzphi + eps)
y_xy_obs = y_xy / (sig_xy + eps)
y_h1_obs = y_h1 / (sig_A1 + eps)
y_h2_obs = y_h2 / (sig_A2 + eps)
y_h3_obs = y_h3 / (sig_A3 + eps)
y_h4_obs = y_h4 / (sig_A4 + eps)
y_vec = np.concatenate([y_rz, y_xy_obs, y_h1_obs, y_h2_obs, y_h3_obs, y_h4_obs])

n_orb = U.shape[1]
lambda_reg = 1.0
reg = lambda_reg / n_orb

# Solve NNLS via ADMM on CPU
Q = U.T @ U + reg * np.eye(n_orb)
c = -U.T @ y_vec
rho_admm = np.trace(Q) / n_orb

# ADMM iterations
L_factor = np.linalg.cholesky(Q + rho_admm * np.eye(n_orb))
w = np.zeros(n_orb)
z = np.zeros(n_orb)
u = np.zeros(n_orb)
for _ in range(300):
    rhs = -c + rho_admm * (z - u)
    w = np.linalg.solve(L_factor, rhs)
    w = np.linalg.solve(L_factor.T, w)
    z = np.maximum(w + u, 0)
    u = u + w - z
weights = z.copy()

print(f"N orbits: {n_orb}")
print(f"N active (w > 1e-6): {np.sum(weights > 1e-6)}")
print(f"N active (w > 1e-4): {np.sum(weights > 1e-4)}")
print(f"Top 10 weights: {np.sort(weights)[-10:][::-1]}")

# ---- Compute logL_best ----
def compute_logl(w):
    res_rz = ((A_Rzphi @ w - y_Rzphi) / (sig_Rzphi + eps))**2
    res_den = ((A_xy @ w - y_xy) / (sig_xy + eps))**2
    h1m = (A_h1 * A_xy) @ w / y_xy_safe
    h2m = (A_h2 * A_xy) @ w / y_xy_safe
    h3m = (A_h3 * A_xy) @ w / y_xy_safe
    h4m = (A_h4 * A_xy) @ w / y_xy_safe
    h1m = np.clip(h1m, -10, 10)
    h2m = np.clip(h2m, -10, 10)
    h3m = np.clip(h3m, -10, 10)
    h4m = np.clip(h4m, -10, 10)
    res_h1 = np.where(np.abs(h1m) < 9.9, ((h1m - y_h1) / (sig_A1 + eps))**2, 0)
    res_h2 = np.where(np.abs(h2m) < 9.9, ((h2m - y_h2) / (sig_A2 + eps))**2, 0)
    res_h3 = np.where(np.abs(h3m) < 9.9, ((h3m - y_h3) / (sig_A3 + eps))**2, 0)
    res_h4 = np.where(np.abs(h4m) < 9.9, ((h4m - y_h4) / (sig_A4 + eps))**2, 0)
    return -0.5 * (np.nansum(res_rz) + np.nansum(res_den) +
                   np.nansum(res_h1) + np.nansum(res_h2) +
                   np.nansum(res_h3) + np.nansum(res_h4))

logl_best = compute_logl(weights)
print(f"\nlogL_best = {logl_best:.4f}")

# ---- Setup: active set and Hessian ----
N_ACTIVE = 1000
W_THRESHOLD = 1e-6
EIG_THRESHOLD = 0.01
N_QMC = 256  # more samples for accurate comparison

sorted_idx = np.argsort(-weights)[:N_ACTIVE]
w_active = weights[sorted_idx]
active_mask = w_active > W_THRESHOLD
n_active = int(np.sum(active_mask))
print(f"N_active (in top {N_ACTIVE}): {n_active}")

U_active = U[:, sorted_idx] * active_mask[None, :]
Q_active = U_active.T @ U_active + reg * np.eye(N_ACTIVE)

# ---- Full eigendecomposition ----
eigvals, eigvecs = np.linalg.eigh(Q_active)
keep = eigvals > EIG_THRESHOLD
n_keep = int(np.sum(keep))
print(f"N_keep (eig > {EIG_THRESHOLD}): {n_keep}")

# ---- Diagonal elements ----
Q_diag = np.diag(Q_active)
print(f"\n--- Hessian structure ---")
print(f"Q_diag range: [{Q_diag[active_mask].min():.4f}, {Q_diag[active_mask].max():.4f}]")
print(f"Eigenvalue range (kept): [{eigvals[keep].min():.4f}, {eigvals[keep].max():.4f}]")

# Off-diagonal strength: ratio of off-diag norm to diag norm
Q_offdiag = Q_active - np.diag(Q_diag)
offdiag_norm = np.linalg.norm(Q_offdiag[np.ix_(active_mask, active_mask)])
diag_norm = np.linalg.norm(Q_diag[active_mask])
print(f"||Q_offdiag|| / ||Q_diag|| = {offdiag_norm / diag_norm:.4f}")

# Hadamard ratio: log(prod(Q_ii)) - log(det(Q))
# = sum(log(Q_ii)) - sum(log(eigvals))
log_prod_diag = np.sum(np.log(Q_diag[active_mask]))
log_det_full = np.sum(np.log(eigvals[keep]))
hadamard_gap = 0.5 * (log_prod_diag - log_det_full)
print(f"Hadamard gap (diag overcount): {hadamard_gap:.2f}")

# ---- QMC samples ----
sampler = _qmc.Sobol(d=N_ACTIVE, scramble=True, seed=42)
U_sobol = np.clip(sampler.random(N_QMC), 1e-6, 1 - 1e-6)
from scipy.stats import norm
Z_qmc = norm.ppf(U_sobol)

# ---- Method 1: Full eigendecomposition sampling ----
sigma_eig = np.where(keep, 1.0 / np.sqrt(eigvals), 0.0)
z_full = Z_qmc * keep[None, :] * sigma_eig[None, :]
delta_w_full = z_full @ eigvecs.T
w_samples_full = np.maximum(w_active[None, :] + delta_w_full, 0.0)

w_full_all = np.tile(weights[None, :], (N_QMC, 1))
w_full_all[:, sorted_idx] = w_samples_full

logl_full = np.array([compute_logl(w_full_all[i]) for i in range(N_QMC)])
logl_max_full = np.max(logl_full)
logl_marg_full = logl_max_full + np.log(np.mean(np.exp(logl_full - logl_max_full)))

# ---- Method 2: Diagonal approximation sampling ----
sigma_diag = np.where(active_mask, 1.0 / np.sqrt(Q_diag), 0.0)
# Use same QMC samples but perturb independently per weight
delta_w_diag = Z_qmc * active_mask[None, :] * sigma_diag[None, :]
w_samples_diag = np.maximum(w_active[None, :] + delta_w_diag, 0.0)

w_diag_all = np.tile(weights[None, :], (N_QMC, 1))
w_diag_all[:, sorted_idx] = w_samples_diag

logl_diag = np.array([compute_logl(w_diag_all[i]) for i in range(N_QMC)])
logl_max_diag = np.max(logl_diag)
logl_marg_diag = logl_max_diag + np.log(np.mean(np.exp(logl_diag - logl_max_diag)))

# ---- Method 3: Analytic (full eigen) ----
# logL_marg = logL_best - 0.5 * sum_k log(2 - reg/lambda_k)
analytic_terms = np.where(keep, np.log(2.0 - reg / eigvals), 0.0)
logl_marg_analytic = logl_best - 0.5 * np.sum(analytic_terms)

# ---- Method 4: Analytic (diagonal) ----
analytic_diag_terms = np.where(active_mask, np.log(2.0 - reg / Q_diag), 0.0)
logl_marg_analytic_diag = logl_best - 0.5 * np.sum(analytic_diag_terms)

# ---- Results ----
print(f"\n{'='*60}")
print(f"{'Method':<35} {'logL_marg':>12} {'Penalty':>10}")
print(f"{'='*60}")
print(f"{'logL_best (no marg)':<35} {logl_best:>12.2f} {'0.00':>10}")
print(f"{'Full eigen sampling (N=256)':<35} {logl_marg_full:>12.2f} {logl_marg_full - logl_best:>10.2f}")
print(f"{'Diagonal sampling (N=256)':<35} {logl_marg_diag:>12.2f} {logl_marg_diag - logl_best:>10.2f}")
print(f"{'Full eigen analytic':<35} {logl_marg_analytic:>12.2f} {logl_marg_analytic - logl_best:>10.2f}")
print(f"{'Diagonal analytic':<35} {logl_marg_analytic_diag:>12.2f} {logl_marg_analytic_diag - logl_best:>10.2f}")
print(f"{'='*60}")

# ---- Distribution of sampled logL ----
print(f"\n--- logL sample distributions ---")
print(f"Full eigen:  mean={np.mean(logl_full):.2f}  std={np.std(logl_full):.2f}  min={np.min(logl_full):.2f}  max={np.max(logl_full):.2f}")
print(f"Diagonal:    mean={np.mean(logl_diag):.2f}  std={np.std(logl_diag):.2f}  min={np.min(logl_diag):.2f}  max={np.max(logl_diag):.2f}")

# ---- Correlation structure analysis ----
print(f"\n--- Correlation analysis ---")
# Compute correlation matrix of active weights from Q_active
Q_sub = Q_active[np.ix_(active_mask, active_mask)]
D_inv = np.diag(1.0 / np.sqrt(np.diag(Q_sub)))
corr = D_inv @ Q_sub @ D_inv
offdiag_corr = corr - np.eye(corr.shape[0])
print(f"Max |off-diag correlation|: {np.max(np.abs(offdiag_corr)):.4f}")
print(f"Mean |off-diag correlation|: {np.mean(np.abs(offdiag_corr)):.4f}")
print(f"Fraction |corr| > 0.5: {np.mean(np.abs(offdiag_corr) > 0.5):.4f}")
print(f"Fraction |corr| > 0.1: {np.mean(np.abs(offdiag_corr) > 0.1):.4f}")
