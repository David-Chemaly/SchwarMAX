"""Test analytic marginalization: determinism and speed.
Uses the orbital library directly (same as test_qmc_convergence.py).
"""
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from model_bar import solve_nnls_admm, _compute_logl_from_weights, EIG_THRESHOLD
from constants import EPSILON

path_data = '/Users/hanyuan/Desktop/PhD_projects/SchwarMAX_data/'
lib_file = 'orbital_library_adaptive_Nmax1e4.pkl'

with open(path_data + lib_file, 'rb') as f:
    orb_lib = pickle.load(f)

(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = orb_lib[:18]

# ---- Analytic marginalization function (mirrors model_bar.py implementation) ----
@jax.jit
def compute_logl_with_analytic_marg(weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                                     y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                                     sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4):
    eps = 1e-8
    # logL_best
    logl_best = _compute_logl_from_weights(
        weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

    # Build design matrix (same as solve_nnls_admm)
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)
    U_rz  = A_Rzphi / (sig_Rzphi[:, None] + eps)
    U_xy_ = A_xy / (sig_xy[:, None] + eps)
    U_h1_ = (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps)
    U_h2_ = (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps)
    U_h3_ = (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps)
    U_h4_ = (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps)
    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])

    n_orb = U.shape[1]
    reg = 1.0 / n_orb
    Q = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)

    _, log_det_Q = jnp.linalg.slogdet(Q)
    analytic_penalty = -0.5 * log_det_Q
    return logl_best, analytic_penalty, logl_best + analytic_penalty


# ---- Test 1: Determinism ----
print("=" * 70)
print("Test 1: Determinism (10 calls, same weights)")
print("=" * 70)

weights = solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
)
jax.block_until_ready(weights)

# Warmup
_ = compute_logl_with_analytic_marg(
    weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

logl_bests = []
penalties = []
logl_margs = []
times = []
for i in range(10):
    t0 = time.time()
    lb, pen, lm = compute_logl_with_analytic_marg(
        weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)
    jax.block_until_ready(lm)
    dt = time.time() - t0
    logl_bests.append(float(lb))
    penalties.append(float(pen))
    logl_margs.append(float(lm))
    times.append(dt)
    print(f"  call {i}: logL_best={float(lb):.4f}  penalty={float(pen):.4f}  logL_marg={float(lm):.4f}  time={dt:.3f}s")

print(f"\nlogL_best range: {max(logl_bests)-min(logl_bests):.6f}")
print(f"penalty range:   {max(penalties)-min(penalties):.6f}")
print(f"logL_marg range: {max(logl_margs)-min(logl_margs):.6f}")
print(f"Mean time: {np.mean(times):.3f}s")

# ---- Test 2: Full pipeline stochasticity (re-solve weights each time) ----
print("\n" + "=" * 70)
print("Test 2: Full pipeline stochasticity (re-solve weights + marg each time)")
print("=" * 70)

logl_margs_full = []
penalties_full = []
for i in range(10):
    w = solve_nnls_admm(
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        lambda_reg=1, maxiter=200,
    )
    lb, pen, lm = compute_logl_with_analytic_marg(
        w, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)
    jax.block_until_ready(lm)
    logl_margs_full.append(float(lm))
    penalties_full.append(float(pen))
    print(f"  call {i}: logL_best={float(lb):.4f}  penalty={float(pen):.4f}  logL_marg={float(lm):.4f}")

print(f"\nlogL_marg range (full pipeline): {max(logl_margs_full)-min(logl_margs_full):.6f}")
print(f"penalty range (full pipeline):   {max(penalties_full)-min(penalties_full):.6f}")
