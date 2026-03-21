"""
Hybrid marginalization: QMC for bottom-K eigendirections + analytic for the rest.
All 596 non-zero orbits included, but only K dimensions sampled.
"""

import pickle
import time
import numpy as np
from scipy.stats import qmc as _qmc
from scipy.stats import norm
from functools import partial

import jax
import jax.numpy as jnp

from model_bar import solve_nnls_admm, _compute_logl_from_weights
from constants import EPSILON

path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
lib_file = 'orbital_library_adaptive_Nmax1e4.pkl'

with open(path_data + lib_file, 'rb') as f:
    orb_lib = pickle.load(f)

(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = orb_lib[:18]

weights = solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
)
weights.block_until_ready()

logl_best = float(_compute_logl_from_weights(
    weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4))
print(f"logL_best = {logl_best:.2f}")

# Build design matrix
eps = 1e-8
w_np = np.asarray(weights)
n_orb = w_np.shape[0]
reg = 1.0 / n_orb
y_xy_safe_np = np.where(np.abs(np.asarray(y_xy)) > eps, np.asarray(y_xy), 1.0)

U_rz  = np.asarray(A_Rzphi) / (np.asarray(sig_Rzphi)[:, None] + eps)
U_xy  = np.asarray(A_xy) / (np.asarray(sig_xy)[:, None] + eps)
U_h1  = (np.asarray(A_h1) * np.asarray(A_xy)) / y_xy_safe_np[:, None] / (np.asarray(sig_A1)[:, None] + eps)
U_h2  = (np.asarray(A_h2) * np.asarray(A_xy)) / y_xy_safe_np[:, None] / (np.asarray(sig_A2)[:, None] + eps)
U_h3  = (np.asarray(A_h3) * np.asarray(A_xy)) / y_xy_safe_np[:, None] / (np.asarray(sig_A3)[:, None] + eps)
U_h4  = (np.asarray(A_h4) * np.asarray(A_xy)) / y_xy_safe_np[:, None] / (np.asarray(sig_A4)[:, None] + eps)
U = np.vstack([U_rz, U_xy, U_h1, U_h2, U_h3, U_h4])

# All non-zero weight orbits
W_THRESHOLD = 1e-6
EIG_THRESHOLD = 0.01
nonzero_mask = w_np > W_THRESHOLD
nonzero_idx = np.where(nonzero_mask)[0]
n_nonzero = len(nonzero_idx)
w_active = w_np[nonzero_idx]

# Full Hessian for all non-zero orbits
U_active = U[:, nonzero_idx]
Q_active = U_active.T @ U_active + reg * np.eye(n_nonzero)
eigvals_np, eigvecs_np = np.linalg.eigh(Q_active)
keep = eigvals_np > EIG_THRESHOLD
n_keep = int(np.sum(keep))

print(f"n_nonzero = {n_nonzero}, n_keep = {n_keep}")
print(f"eigenvalue range (kept): [{eigvals_np[keep].min():.4f}, {eigvals_np[keep].max():.4f}]")

# ---- Full analytic penalty (reference) ----
analytic_full = -0.5 * np.sum(np.where(keep, np.log(2.0 - reg / eigvals_np), 0.0))
print(f"Full analytic penalty (all {n_keep} eigdirs): {analytic_full:.2f}")

# ---- Simulate GPU non-determinism ----
N_TRIALS = 10

def random_eigvecs(eigvecs, eigvals, seed):
    rng = np.random.default_rng(seed)
    V = eigvecs.copy()
    V = V * rng.choice([-1, 1], size=V.shape[1])[None, :]
    sorted_eigs = np.sort(eigvals)
    i = 0
    while i < len(sorted_eigs):
        j = i + 1
        while j < len(sorted_eigs) and abs(sorted_eigs[j] - sorted_eigs[i]) / (abs(sorted_eigs[i]) + 1e-10) < 0.01:
            j += 1
        if j - i > 1:
            idx = np.where((eigvals >= sorted_eigs[i] * 0.99) & (eigvals <= sorted_eigs[j-1] * 1.01))[0]
            if len(idx) > 1:
                Q_rand, _ = np.linalg.qr(rng.standard_normal((len(idx), len(idx))))
                V[:, idx] = V[:, idx] @ Q_rand
        i = j
    return V

# ---- JAX functions ----
def _logl_single(w_i):
    return _compute_logl_from_weights(
        w_i, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

_logl_batch = jax.vmap(_logl_single)
weights_j = jnp.array(weights)

# ---- Hybrid marginalization ----
# For each K_sample: QMC over bottom-K eigendirections + analytic for the rest
#
# Bottom-K = smallest eigenvalues = largest perturbation variance = most affected by truncation
# Top-(n_keep - K) = largest eigenvalues = small perturbation = analytic log(2-reg/λ) is accurate

@partial(jax.jit, static_argnames=('n_qmc', 'n_active', 'k_sample'))
def hybrid_marginalize_all_trials(all_eigvecs, eigvals, keep, Z_all,
                                  weights, sorted_idx, w_active,
                                  n_qmc, n_active, k_sample,
                                  analytic_penalty_rest):
    """
    Hybrid: QMC for bottom-K eigdirs + analytic penalty for the rest.
    all_eigvecs: (N_TRIALS, n_active, n_active)
    """
    Z = Z_all[:n_qmc, :k_sample]   # (n_qmc, k_sample)

    # Bottom-K eigenvalues (smallest, most uncertain)
    # eigvals are sorted ascending from eigh, so bottom-K = first K that are kept
    kept_indices = jnp.where(keep, jnp.arange(n_active), n_active)
    kept_sorted = jnp.sort(kept_indices)[:k_sample]  # indices of bottom-K kept eigvals

    sigma_bottom = 1.0 / jnp.sqrt(eigvals[kept_sorted])  # (k_sample,)

    def _one_trial(eigvecs_i):
        # Extract bottom-K eigenvectors
        V_bottom = eigvecs_i[:, kept_sorted]  # (n_active, k_sample)

        # QMC perturbations in bottom-K eigenbasis
        z_scaled = Z * sigma_bottom[None, :]  # (n_qmc, k_sample)
        delta_w = z_scaled @ V_bottom.T       # (n_qmc, n_active)
        w_samples = jnp.maximum(w_active[None, :] + delta_w, 0.0)

        w_full = jnp.tile(weights[None, :], (n_qmc, 1))
        w_full = w_full.at[:, sorted_idx].set(w_samples)

        logls = _logl_batch(w_full)
        logl_max = jnp.max(logls)
        logl_marg_qmc = logl_max + jnp.log(jnp.mean(jnp.exp(logls - logl_max)))

        # Add analytic penalty for remaining directions
        return logl_marg_qmc + analytic_penalty_rest

    return jax.vmap(_one_trial)(all_eigvecs)


# Pre-generate QMC
MAX_QMC = 4096
MAX_K = 400
sampler = _qmc.Sobol(d=MAX_K, scramble=True, seed=42)
Z_all_np = norm.ppf(np.clip(sampler.random(MAX_QMC), 1e-6, 1 - 1e-6)).astype(np.float32)
Z_all_j = jnp.array(Z_all_np)

eigvals_j = jnp.array(eigvals_np, dtype=jnp.float32)
keep_j = jnp.array(keep)
sorted_idx_j = jnp.array(nonzero_idx)
w_active_j = jnp.array(w_active, dtype=jnp.float32)

# ---- Test different K_sample values ----
K_values = [50, 100, 150, 200, 300, 420]
N_QMC_values = [64, 256, 1024, 4096]

# Pre-compute analytic penalties for the "rest" directions
# eigvals are sorted ascending by eigh
kept_eig_indices = np.where(keep)[0]
kept_eigvals = eigvals_np[kept_eig_indices]
# Sort by eigenvalue (ascending = most uncertain first)
sort_order = np.argsort(kept_eigvals)
kept_eigvals_sorted = kept_eigvals[sort_order]

print(f"\n{'#'*115}")

for K in K_values:
    K_actual = min(K, n_keep)

    # Analytic penalty for directions K_actual..n_keep (the well-constrained ones)
    rest_eigvals = kept_eigvals_sorted[K_actual:]
    analytic_rest = -0.5 * np.sum(np.log(2.0 - reg / rest_eigvals)) if len(rest_eigvals) > 0 else 0.0
    analytic_qmc_part = -0.5 * np.sum(np.log(2.0 - reg / kept_eigvals_sorted[:K_actual]))

    all_eigvecs = np.stack([random_eigvecs(eigvecs_np, eigvals_np, seed=t * 1000 + 7)
                            for t in range(N_TRIALS)])
    all_eigvecs_j = jnp.array(all_eigvecs, dtype=jnp.float32)

    print(f"\n{'='*115}")
    print(f"K_sample = {K_actual} (QMC dims) | rest = {n_keep - K_actual} (analytic)")
    print(f"  analytic penalty for rest: {analytic_rest:.2f}")
    print(f"  analytic penalty for QMC part (if used analytic): {analytic_qmc_part:.2f}")
    print(f"  total analytic (reference): {analytic_rest + analytic_qmc_part:.2f}")
    print(f"  eigval range in QMC part: [{kept_eigvals_sorted[:K_actual].min():.4f}, {kept_eigvals_sorted[:K_actual].max():.4f}]")
    print(f"{'='*115}")

    # Warmup
    _ = hybrid_marginalize_all_trials(
        all_eigvecs_j, eigvals_j, keep_j, Z_all_j,
        weights_j, sorted_idx_j, w_active_j,
        n_qmc=64, n_active=n_nonzero, k_sample=K_actual,
        analytic_penalty_rest=jnp.float32(analytic_rest))

    print(f"{'N_QMC':>8} | {'mean':>10} | {'std':>8} | {'Range':>8} | {'Penalty':>10} | {'Time':>6}")
    print(f"{'-'*115}")

    for nq in N_QMC_values:
        _ = hybrid_marginalize_all_trials(
            all_eigvecs_j, eigvals_j, keep_j, Z_all_j,
            weights_j, sorted_idx_j, w_active_j,
            n_qmc=nq, n_active=n_nonzero, k_sample=K_actual,
            analytic_penalty_rest=jnp.float32(analytic_rest))

        t0 = time.time()
        results = hybrid_marginalize_all_trials(
            all_eigvecs_j, eigvals_j, keep_j, Z_all_j,
            weights_j, sorted_idx_j, w_active_j,
            n_qmc=nq, n_active=n_nonzero, k_sample=K_actual,
            analytic_penalty_rest=jnp.float32(analytic_rest))
        results.block_until_ready()
        dt = time.time() - t0

        vals = np.asarray(results)
        rng_val = vals.max() - vals.min()
        penalty = vals.mean() - logl_best
        print(f"{nq:>8} | {vals.mean():>10.2f} | {vals.std():>8.2f} | {rng_val:>8.2f} | {penalty:>10.2f} | {dt:>5.2f}s")

# Also run full QMC (all 420 dims, no analytic) as reference
print(f"\n{'='*115}")
print(f"Reference: Full QMC all {n_keep} eigdirs (no hybrid)")
print(f"{'='*115}")

all_eigvecs = np.stack([random_eigvecs(eigvecs_np, eigvals_np, seed=t * 1000 + 7)
                        for t in range(N_TRIALS)])
all_eigvecs_j = jnp.array(all_eigvecs, dtype=jnp.float32)

@partial(jax.jit, static_argnames=('n_qmc', 'n_active'))
def full_marginalize_all_trials(all_eigvecs, eigvals, keep, Z_all,
                                weights, sorted_idx, w_active, n_qmc, n_active):
    Z = Z_all[:n_qmc, :n_active]
    sigma_eig = jnp.where(keep, 1.0 / jnp.sqrt(eigvals), 0.0)
    z_scaled = Z * keep[None, :] * sigma_eig[None, :]

    def _one_trial(eigvecs_i):
        delta_w = z_scaled @ eigvecs_i.T
        w_samples = jnp.maximum(w_active[None, :] + delta_w, 0.0)
        w_full = jnp.tile(weights[None, :], (n_qmc, 1))
        w_full = w_full.at[:, sorted_idx].set(w_samples)
        logls = _logl_batch(w_full)
        logl_max = jnp.max(logls)
        return logl_max + jnp.log(jnp.mean(jnp.exp(logls - logl_max)))

    return jax.vmap(_one_trial)(all_eigvecs)

# Need Sobol with n_nonzero dims for full test
sampler_full = _qmc.Sobol(d=n_nonzero, scramble=True, seed=42)
Z_full_np = norm.ppf(np.clip(sampler_full.random(4096), 1e-6, 1 - 1e-6)).astype(np.float32)
Z_full_j = jnp.array(Z_full_np)

for nq in [64, 256, 1024, 4096]:
    _ = full_marginalize_all_trials(all_eigvecs_j, eigvals_j, keep_j, Z_full_j,
                                    weights_j, sorted_idx_j, w_active_j, n_qmc=nq, n_active=n_nonzero)
    t0 = time.time()
    results = full_marginalize_all_trials(all_eigvecs_j, eigvals_j, keep_j, Z_full_j,
                                          weights_j, sorted_idx_j, w_active_j, n_qmc=nq, n_active=n_nonzero)
    results.block_until_ready()
    dt = time.time() - t0
    vals = np.asarray(results)
    print(f"  N_QMC={nq:>5}: mean={vals.mean():.2f}  std={vals.std():.2f}  range={vals.max()-vals.min():.2f}  penalty={vals.mean()-logl_best:.2f}  time={dt:.2f}s")

print(f"\n{'#'*115}")
print(f"logL_best = {logl_best:.2f}")
print(f"Full analytic penalty = {analytic_full:.2f}")
