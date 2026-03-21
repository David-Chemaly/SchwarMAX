"""
Hybrid marginalization: split orbits by SNR = w_i / sigma_i = w_i * sqrt(Q_ii).
- Low-SNR orbits (near truncation boundary w=0): QMC sampling on their sub-Hessian
- High-SNR orbits (far from boundary): analytic penalty on their sub-Hessian
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
Q_diag = np.diag(Q_active)
sigma_w = 1.0 / np.sqrt(Q_diag)
snr = w_active / sigma_w  # = w_i * sqrt(Q_ii)

print(f"\nn_nonzero = {n_nonzero}")
print(f"SNR range: [{snr.min():.4f}, {snr.max():.4f}]")
print(f"SNR distribution: <1: {np.sum(snr<1)}, <2: {np.sum(snr<2)}, <3: {np.sum(snr<3)}, <5: {np.sum(snr<5)}")
print(f"sigma_w range: [{sigma_w.min():.4f}, {sigma_w.max():.4f}]")
print(f"weight range (active): [{w_active.min():.6f}, {w_active.max():.4f}]")

# Full eigendecomposition as reference
eigvals_full, eigvecs_full = np.linalg.eigh(Q_active)
keep_full = eigvals_full > EIG_THRESHOLD
n_keep_full = int(np.sum(keep_full))
analytic_full = -0.5 * np.sum(np.where(keep_full, np.log(2.0 - reg / eigvals_full), 0.0))
print(f"\nFull Hessian: n_keep = {n_keep_full}, analytic penalty = {analytic_full:.2f}")

# ---- JAX functions ----
def _logl_single(w_i):
    return _compute_logl_from_weights(
        w_i, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

_logl_batch = jax.vmap(_logl_single)
weights_j = jnp.array(weights)

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


# ---- SNR-based hybrid marginalization ----
# Split: low-SNR orbits → QMC on their sub-Hessian
#         high-SNR orbits → analytic on their sub-Hessian

print(f"\n{'#'*115}")
print(f"SNR-based Hybrid: QMC for low-SNR orbits + analytic for high-SNR orbits")
print(f"{'#'*115}")

# Sort orbits by SNR (ascending = lowest SNR first)
snr_order = np.argsort(snr)

N_QMC_values = [64, 256, 1024, 4096]
N_LOW_SNR_values = [50, 100, 150, 200, 300]

MAX_QMC = 4096
MAX_DIM = 300

for n_low in N_LOW_SNR_values:
    # Low-SNR orbits (QMC)
    low_snr_local_idx = snr_order[:n_low]   # indices into the active set
    low_snr_global_idx = nonzero_idx[low_snr_local_idx]  # indices into full weight vector

    # High-SNR orbits (analytic)
    high_snr_local_idx = snr_order[n_low:]
    high_snr_global_idx = nonzero_idx[high_snr_local_idx]

    # Sub-Hessians
    U_low = U[:, low_snr_global_idx]
    Q_low = U_low.T @ U_low + reg * np.eye(n_low)
    w_low = w_np[low_snr_global_idx]

    U_high = U[:, high_snr_global_idx]
    n_high = len(high_snr_local_idx)
    Q_high = U_high.T @ U_high + reg * np.eye(n_high)

    # Eigendecomposition of sub-Hessians
    eigvals_low, eigvecs_low = np.linalg.eigh(Q_low)
    keep_low = eigvals_low > EIG_THRESHOLD
    n_keep_low = int(np.sum(keep_low))

    eigvals_high, _ = np.linalg.eigh(Q_high)
    keep_high = eigvals_high > EIG_THRESHOLD
    n_keep_high = int(np.sum(keep_high))

    # Analytic penalty for high-SNR block
    analytic_high = -0.5 * np.sum(np.where(keep_high, np.log(2.0 - reg / eigvals_high), 0.0))

    snr_low_range = snr[low_snr_local_idx]
    snr_high_range = snr[high_snr_local_idx]

    print(f"\n{'='*115}")
    print(f"N_low_SNR = {n_low} (QMC) | N_high_SNR = {n_high} (analytic)")
    print(f"  Low-SNR range:  [{snr_low_range.min():.4f}, {snr_low_range.max():.4f}]")
    print(f"  High-SNR range: [{snr_high_range.min():.4f}, {snr_high_range.max():.4f}]")
    print(f"  Low block: n_keep = {n_keep_low}, eigval range = [{eigvals_low[keep_low].min():.4f}, {eigvals_low[keep_low].max():.4f}]")
    print(f"  High block analytic penalty: {analytic_high:.2f} (n_keep={n_keep_high})")
    print(f"{'='*115}")

    # Pre-generate QMC for this dimensionality
    sampler = _qmc.Sobol(d=n_low, scramble=True, seed=42)
    Z_np = norm.ppf(np.clip(sampler.random(MAX_QMC), 1e-6, 1 - 1e-6)).astype(np.float32)

    # Generate trial eigenvectors (simulating GPU non-determinism)
    all_eigvecs_low = np.stack([random_eigvecs(eigvecs_low, eigvals_low, seed=t * 1000 + 7)
                                for t in range(N_TRIALS)])

    # JAX arrays
    Z_j = jnp.array(Z_np)
    eigvals_low_j = jnp.array(eigvals_low, dtype=jnp.float32)
    keep_low_j = jnp.array(keep_low)
    all_eigvecs_low_j = jnp.array(all_eigvecs_low, dtype=jnp.float32)
    low_snr_global_idx_j = jnp.array(low_snr_global_idx)
    w_low_j = jnp.array(w_low, dtype=jnp.float32)

    @partial(jax.jit, static_argnames=('n_qmc', 'n_dim'))
    def snr_hybrid_trials(all_eigvecs, eigvals, keep, Z,
                          weights, low_idx, w_low,
                          analytic_rest, n_qmc, n_dim):
        Z_use = Z[:n_qmc, :n_dim]
        sigma_eig = jnp.where(keep, 1.0 / jnp.sqrt(eigvals), 0.0)
        z_scaled = Z_use * keep[None, :n_dim] * sigma_eig[None, :n_dim]

        def _one_trial(eigvecs_i):
            delta_w = z_scaled @ eigvecs_i[:n_dim, :n_dim].T
            w_samples = jnp.maximum(w_low[None, :] + delta_w, 0.0)

            w_full = jnp.tile(weights[None, :], (n_qmc, 1))
            w_full = w_full.at[:, low_idx].set(w_samples)

            logls = _logl_batch(w_full)
            logl_max = jnp.max(logls)
            logl_marg_qmc = logl_max + jnp.log(jnp.mean(jnp.exp(logls - logl_max)))
            return logl_marg_qmc + analytic_rest

        return jax.vmap(_one_trial)(all_eigvecs)

    # Warmup
    _ = snr_hybrid_trials(
        all_eigvecs_low_j, eigvals_low_j, keep_low_j, Z_j,
        weights_j, low_snr_global_idx_j, w_low_j,
        jnp.float32(analytic_high), n_qmc=64, n_dim=n_low)

    print(f"{'N_QMC':>8} | {'mean':>10} | {'std':>8} | {'Range':>8} | {'Penalty':>10} | {'Time':>6}")
    print(f"{'-'*80}")

    for nq in N_QMC_values:
        # Warmup this N_QMC
        _ = snr_hybrid_trials(
            all_eigvecs_low_j, eigvals_low_j, keep_low_j, Z_j,
            weights_j, low_snr_global_idx_j, w_low_j,
            jnp.float32(analytic_high), n_qmc=nq, n_dim=n_low)

        t0 = time.time()
        results = snr_hybrid_trials(
            all_eigvecs_low_j, eigvals_low_j, keep_low_j, Z_j,
            weights_j, low_snr_global_idx_j, w_low_j,
            jnp.float32(analytic_high), n_qmc=nq, n_dim=n_low)
        results.block_until_ready()
        dt = time.time() - t0

        vals = np.asarray(results)
        rng_val = vals.max() - vals.min()
        penalty = vals.mean() - logl_best
        print(f"{nq:>8} | {vals.mean():>10.2f} | {vals.std():>8.2f} | {rng_val:>8.2f} | {penalty:>10.2f} | {dt:>5.2f}s")


# ---- Also compare with previous approach (hybrid by eigenvalue) ----
print(f"\n{'#'*115}")
print(f"Reference: Previous hybrid (QMC bottom-K eigdirs of full Hessian + analytic rest)")
print(f"{'#'*115}")

kept_eig_indices = np.where(keep_full)[0]
kept_eigvals = eigvals_full[kept_eig_indices]
sort_order_eig = np.argsort(kept_eigvals)
kept_eigvals_sorted = kept_eigvals[sort_order_eig]

for K in [50, 100]:
    K_actual = min(K, n_keep_full)
    rest_eigvals = kept_eigvals_sorted[K_actual:]
    analytic_rest_eig = -0.5 * np.sum(np.log(2.0 - reg / rest_eigvals)) if len(rest_eigvals) > 0 else 0.0

    all_eigvecs = np.stack([random_eigvecs(eigvecs_full, eigvals_full, seed=t * 1000 + 7)
                            for t in range(N_TRIALS)])

    # Use the full Hessian eigenvectors, but only QMC over bottom-K
    print(f"\n  K_sample = {K_actual} eigdirs (QMC) + {n_keep_full - K_actual} (analytic)")
    print(f"  analytic rest penalty: {analytic_rest_eig:.2f}")

    # For brevity, just do N_QMC=4096
    sampler_ref = _qmc.Sobol(d=n_nonzero, scramble=True, seed=42)
    Z_ref_np = norm.ppf(np.clip(sampler_ref.random(4096), 1e-6, 1 - 1e-6)).astype(np.float32)
    Z_ref_j = jnp.array(Z_ref_np)
    all_eigvecs_j = jnp.array(all_eigvecs, dtype=jnp.float32)
    eigvals_full_j = jnp.array(eigvals_full, dtype=jnp.float32)
    keep_full_j = jnp.array(keep_full)
    sorted_idx_j = jnp.array(nonzero_idx)
    w_active_j = jnp.array(w_active, dtype=jnp.float32)

    @partial(jax.jit, static_argnames=('n_qmc', 'n_active', 'k_sample'))
    def eigval_hybrid_trials(all_eigvecs, eigvals, keep, Z_all,
                             weights, sorted_idx, w_active,
                             n_qmc, n_active, k_sample,
                             analytic_penalty_rest):
        Z = Z_all[:n_qmc, :k_sample]
        kept_indices = jnp.where(keep, jnp.arange(n_active), n_active)
        kept_sorted = jnp.sort(kept_indices)[:k_sample]
        sigma_bottom = 1.0 / jnp.sqrt(eigvals[kept_sorted])

        def _one_trial(eigvecs_i):
            V_bottom = eigvecs_i[:, kept_sorted]
            z_scaled = Z * sigma_bottom[None, :]
            delta_w = z_scaled @ V_bottom.T
            w_samples = jnp.maximum(w_active[None, :] + delta_w, 0.0)
            w_full = jnp.tile(weights[None, :], (n_qmc, 1))
            w_full = w_full.at[:, sorted_idx].set(w_samples)
            logls = _logl_batch(w_full)
            logl_max = jnp.max(logls)
            return logl_max + jnp.log(jnp.mean(jnp.exp(logls - logl_max))) + analytic_penalty_rest

        return jax.vmap(_one_trial)(all_eigvecs)

    _ = eigval_hybrid_trials(all_eigvecs_j, eigvals_full_j, keep_full_j, Z_ref_j,
                             weights_j, sorted_idx_j, w_active_j,
                             n_qmc=4096, n_active=n_nonzero, k_sample=K_actual,
                             analytic_penalty_rest=jnp.float32(analytic_rest_eig))
    t0 = time.time()
    results = eigval_hybrid_trials(all_eigvecs_j, eigvals_full_j, keep_full_j, Z_ref_j,
                                   weights_j, sorted_idx_j, w_active_j,
                                   n_qmc=4096, n_active=n_nonzero, k_sample=K_actual,
                                   analytic_penalty_rest=jnp.float32(analytic_rest_eig))
    results.block_until_ready()
    dt = time.time() - t0
    vals = np.asarray(results)
    print(f"  N_QMC=4096: mean={vals.mean():.2f}  range={vals.max()-vals.min():.2f}  penalty={vals.mean()-logl_best:.2f}  time={dt:.2f}s")


# ---- Per-orbit truncation analysis ----
print(f"\n{'#'*115}")
print(f"Per-orbit truncation analysis")
print(f"{'#'*115}")
# For each orbit, Phi(-SNR) = fraction of probability cut off
phi_snr = norm.cdf(-snr)
log_phi_correction = np.log(norm.cdf(snr))  # = log(1 - Phi(-SNR))
print(f"Sum of log(Phi(SNR_i)) = {np.sum(log_phi_correction):.2f}")
print(f"  (This is the per-orbit independent truncation correction)")
print(f"  Combined: analytic_full + truncation_correction = {analytic_full + np.sum(log_phi_correction):.2f}")
print(f"\nOrbits with >1% probability cut off (Phi(-SNR) > 0.01, i.e. SNR < 2.33):")
print(f"  Count: {np.sum(snr < 2.33)}")
print(f"  Their truncation correction: {np.sum(log_phi_correction[snr < 2.33]):.2f}")
print(f"\nOrbits with >10% probability cut off (SNR < 1.28):")
print(f"  Count: {np.sum(snr < 1.28)}")
print(f"  Their truncation correction: {np.sum(log_phi_correction[snr < 1.28]):.2f}")
print(f"\nOrbits with >50% probability cut off (SNR < 0):")
print(f"  Count: {np.sum(snr < 0)}")

print(f"\n{'#'*115}")
print(f"logL_best = {logl_best:.2f}")
print(f"Full analytic penalty = {analytic_full:.2f}")
print(f"Full analytic + independent truncation = {analytic_full + np.sum(log_phi_correction):.2f}")
