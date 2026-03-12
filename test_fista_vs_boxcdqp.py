"""
Full downstream comparison: ADMM vs BoxCDQP on REAL orbital library data.
Computes the actual observables (density, V, sigma, h3, h4) and log-likelihood.
"""
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from model import solve_qp_boxcdqp, solve_nnls_admm, solve_fista_nnls
from test_weights_optimisation import get_dict_data
from constants import EPSILON

# ---- Load data ----
path = os.path.dirname(os.path.abspath(__file__)) + '/'

# Monkey-patch path for get_dict_data
import test_weights_optimisation as twm
twm.path = path

dict_data = get_dict_data(path)

with open(path + 'orbital_library_bar_1.pkl', 'rb') as f:
    orb_lib = pickle.load(f)

(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4) = orb_lib

v0 = dict_data['v0']
s  = dict_data['s']

def compute_observables(weights):
    """Compute downstream observables from weights, matching get_weights logic."""
    from model import h_to_V_sigma
    A_h1_, A_h2_, A_h3_, A_h4_ = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1_ @ weights) / y_xy
    h2_model = (A_h2_ @ weights) / y_xy
    h3_model = (A_h3_ @ weights) / y_xy
    h4_model = (A_h4_ @ weights) / y_xy

    clip_val = 10.0
    for arr in [h1_model, h2_model, h3_model, h4_model]:
        arr = jnp.clip(arr, -clip_val, clip_val)

    h1_model = jnp.clip(h1_model, -clip_val, clip_val)
    h2_model = jnp.clip(h2_model, -clip_val, clip_val)
    h3_model = jnp.clip(h3_model, -clip_val, clip_val)
    h4_model = jnp.clip(h4_model, -clip_val, clip_val)

    V_model, sigma_model = h_to_V_sigma(h1_model, h2_model, v0, s)
    return density_2DXY, V_model, sigma_model, h1_model, h2_model, h3_model, h4_model

def compute_loglik(density_2DXY, V_model, sigma_model, h1_model, h2_model, h3_model, h4_model):
    """Compute log-likelihood matching test_weights_optimisation.py logic."""
    h3_m = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_m = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)

    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON))**2
    res_h1 = ((h1_model - y_h1) / (sig_A1 + 1e-3))**2
    res_h2 = ((h2_model - y_h2) / (sig_A2 + 1e-3))**2
    res_h3 = ((h3_m - y_h3) / (sig_A3 + 1e-3))**2
    res_h4 = ((h4_m - y_h4) / (sig_A4 + 1e-3))**2

    res_h1 = jnp.where((h1_model < 9.9), res_h1, 0)
    res_h2 = jnp.where((h2_model < 9.9), res_h2, 0)
    res_h3 = jnp.where((h3_m < 9.9), res_h3, 0)
    res_h4 = jnp.where((h4_m < 9.9), res_h4, 0)

    val1 = jnp.nansum(-0.5 * res_density) / len(density_2DXY)
    val4 = jnp.nansum(-0.5 * res_h1) / len(h1_model)
    val5 = jnp.nansum(-0.5 * res_h2) / len(h2_model)
    val6 = jnp.nansum(-0.5 * res_h3) / len(h3_m)
    val7 = jnp.nansum(-0.5 * res_h4) / len(h4_m)
    return val1 + val4 + val5 + val6 + val7

args = (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

def test_solver(name, solver_fn, **kwargs):
    t0 = time.time()
    w = solver_fn(*args, **kwargs)
    w.block_until_ready()
    dt = time.time() - t0

    obs = compute_observables(w)
    ll = compute_loglik(*obs)

    print(f"\n  {name}")
    print(f"    Time: {dt:.2f}s")
    print(f"    Log-likelihood: {ll:.4f}")
    print(f"    Sparsity: {jnp.sum(w < 1e-6)}/{len(w)}")
    print(f"    Weight stats: min={jnp.min(w):.2e} max={jnp.max(w):.2e} mean={jnp.mean(w):.2e}")
    return w, ll

print("===== Full downstream comparison =====")

w_qp, ll_qp = test_solver("BoxCDQP (ground truth)", solve_qp_boxcdqp,
                            lambda_reg=1, maxiter=100, tol=1e-1)

w_admm200, ll_admm200 = test_solver("ADMM (maxiter=200)", solve_nnls_admm,
                                      lambda_reg=1, maxiter=200)

w_admm500, ll_admm500 = test_solver("ADMM (maxiter=500)", solve_nnls_admm,
                                      lambda_reg=1, maxiter=500)

w_fista, ll_fista = test_solver("FISTA (maxiter=500)", solve_fista_nnls,
                                 lambda_reg=1, maxiter=500)

print(f"\n===== Summary =====")
print(f"  BoxCDQP:    LL={ll_qp:.4f} (reference)")
print(f"  ADMM(200):  LL={ll_admm200:.4f}  (delta: {ll_admm200-ll_qp:+.4f})")
print(f"  ADMM(500):  LL={ll_admm500:.4f}  (delta: {ll_admm500-ll_qp:+.4f})")
print(f"  FISTA(500): LL={ll_fista:.4f}  (delta: {ll_fista-ll_qp:+.4f})")
