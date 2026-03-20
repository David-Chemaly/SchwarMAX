import sys
sys.path.append('/content/drive/MyDrive/SchwarMAX')

import pickle

import jax
import jaxopt
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

import time as tt
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

import os
import corner
import numpy as np
import multiprocessing as mp

from potentials import NFW_potential, NFW_acceleration, MiyamotoNagai_potential, MiyamotoNagai_acceleration
from integrants_with_binning import *
from integrants_with_binning import _integrate_barred_vmap, _integrate_batch_vmap, _integrate_adaptive_batch_vmap
from ghMoments import *
from utils import *
from constants import EPSILON, G

from densities import MiyamotoNagai_density
from dehnen_bar import T3_density, T3_acceleration, T3_potential, V4_density, V4_acceleration, V4_potential

path = '../SchwarMAX/'

# ---- Fixed V4 bulge parameters ----
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0

# ---- Marginalization constants ----
N_ACTIVE_MAX = 1000   # pad active set to this fixed size for JIT
N_QMC = 64            # number of QMC weight samples
EIG_THRESHOLD = 0.01  # eigenvalue threshold for keeping directions
W_THRESHOLD = 1e-6    # weight threshold for active set

# Pre-generate QMC samples at import time (Sobol sequence, fixed seed)
from scipy.stats import qmc as _qmc
_sampler = _qmc.Sobol(d=N_ACTIVE_MAX, scramble=True, seed=42)
_U_sobol = np.clip(_sampler.random(N_QMC), 1e-6, 1 - 1e-6)
Z_QMC = jnp.array(jax.scipy.stats.norm.ppf(jnp.array(_U_sobol, dtype=jnp.float32)))


@jax.jit
def density_func(x, y, z, params):
    """
    Returns Stellar Density nu(R, z) using:
      MiyamotoNagai disc + T3 bar + V4 bulge
    """
    # MN disc density
    mn_params = {
        'logM_disc': params['logM_disc'],
        'Rs_disc': params['Rs_disc'],
        'Hs_disc': params['Hs_disc'],
        'x_origin': params['x_origin'],
        'y_origin': params['y_origin'],
        'z_origin': params['z_origin'],
        'dirx': params['dirx'],
        'diry': params['diry'],
        'dirz': params['dirz'],
    }
    rho_mn = MiyamotoNagai_density(x, y, z, mn_params)

    # T3 bar density
    M_bar = 10.0 ** params['logM_bar']
    L_bar = params['L_bar']
    a_bar = params['a_bar']
    b_bar = params['b_bar']
    rho_t3 = T3_density(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)

    # V4 bulge density (M = M_bar, fixed shape)
    rho_v4 = V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)

    return rho_mn + rho_t3 + rho_v4


@jax.jit
def potential_func(x, y, z, params_baryon, params_halo):
    """Returns Phi(x,y,z) = NFW + MiyamotoNagai + T3 + V4"""
    phi_halo = NFW_potential(x, y, z, params_halo)

    mn_params = {
        'logM_disc': params_baryon['logM_disc'],
        'Rs_disc': params_baryon['Rs_disc'],
        'Hs_disc': params_baryon['Hs_disc'],
        'x_origin': params_baryon['x_origin'],
        'y_origin': params_baryon['y_origin'],
        'z_origin': params_baryon['z_origin'],
        'dirx': params_baryon['dirx'],
        'diry': params_baryon['diry'],
        'dirz': params_baryon['dirz'],
    }
    phi_mn = MiyamotoNagai_potential(x, y, z, mn_params)

    M_bar = 10.0 ** params_baryon['logM_bar']
    L_bar = params_baryon['L_bar']
    a_bar = params_baryon['a_bar']
    b_bar = params_baryon['b_bar']
    phi_t3 = T3_potential(x, y, z, M_bar, a_bar, b_bar, L_bar, GAMMA_BAR)
    phi_v4 = V4_potential(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA)

    return phi_halo + phi_mn + phi_t3 + phi_v4


@jax.jit
def dPhi_dz(x, y, z, params_baryon, params_halo):
    # Numerical derivative of Potential w.r.t z
    d = 5e-3
    return (potential_func(x, y, z+d, params_baryon, params_halo) - potential_func(x, y, z-d, params_baryon, params_halo)) / (2*d)

@jax.jit
def dPhi_dR(x, y, z, params_baryon, params_halo):
    # Numerical derivative of Potential w.r.t R
    d = 5e-3
    R = jnp.sqrt(x**2 + y**2)
    return (potential_func(R+d, 0, z, params_baryon, params_halo) - potential_func(R-d, 0, z, params_baryon, params_halo)) / (2*d)

@jax.jit
def get_jeans_moments(x_star, y_star, z_star, params_baryon, params_disc, params_halo, anisotropy_b=1.0):
    """
    Computes (v_mean, sigma_R, sigma_z, sigma_phi) for a star at (R, z).
    """
    R_star = jnp.sqrt(x_star**2 + y_star**2)

    # --- Step 1: Compute Sigma_z (Vertical Integration) ---
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, params_baryon, params_halo)

    pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
    dx = pts[1] - pts[0]
    integrand_val = jax.vmap(integrand, in_axes = (0))(pts)
    integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)

    nu_val = density_func(x_star, y_star, z_star, params_disc)

    sigma_z2 = (1.0 / nu_val) * integral_val
    sigma_z2 = jnp.maximum(sigma_z2, 0.0)
    sigma_z = jnp.sqrt(sigma_z2)

    # --- Step 2: Compute Sigma_R (Anisotropy assumption) ---
    sigma_R2 = anisotropy_b * sigma_z2
    sigma_R = jnp.sqrt(sigma_R2)

    # --- Step 3: Compute v_phi_total^2 (Radial Equation) ---
    def vertical_pressure(r_in):
        def integrand_r(z_prime):
            return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, params_baryon, params_halo)

        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        dx = pts[1] - pts[0]
        integrand_val = jax.vmap(integrand_r, in_axes = (0))(pts)
        integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)
        return integral_val

    dR = 5e-3
    Pzz_plus = vertical_pressure(R_star + dR)
    Pzz_minus = vertical_pressure(R_star - dR)
    d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

    term1 = sigma_R2
    term2 = (R_star / nu_val) * d_nu_sigR2_dR
    term3 = R_star * dPhi_dR(x_star, y_star, z_star, params_baryon, params_halo)

    v_phi_total_sq = term1 + term2 + term3

    # --- Step 4: Separate Rotation vs Dispersion ---
    sigma_phi = sigma_R
    v_streaming_sq = v_phi_total_sq - sigma_phi**2
    v_streaming_sq = jnp.maximum(v_streaming_sq, 0.0)
    v_mean_phi = jnp.sqrt(v_streaming_sq)

    output = jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))

    return output


@jax.jit
def _nll_z(z,
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, l2):

    x = jnn.softplus(z)  # strictly positive
    r_3dRzphi = (A_Rzphi @ x - y_Rzphi) / sig_Rzphi

    mass_tot = jnp.sum(y_Rzphi)
    N_orb = len(x)
    wi = ( mass_tot / N_orb ) * jnp.ones_like(x)

    density_2DXY = A_xy @ x
    r_2dXY = (density_2DXY - y_xy) / sig_xy

    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)

    y_h1_model = (A_h1 @ x) / y_xy
    y_h2_model = (A_h2 @ x) / y_xy
    y_h3_model = (A_h3 @ x) / y_xy
    y_h4_model = (A_h4 @ x) / y_xy

    clip_val = 10.0
    y_h1_model = jnp.where(y_h1_model > clip_val, clip_val, y_h1_model)
    y_h2_model = jnp.where(y_h2_model > clip_val, clip_val, y_h2_model)
    y_h3_model = jnp.where(y_h3_model > clip_val, clip_val, y_h3_model)
    y_h4_model = jnp.where(y_h4_model > clip_val, clip_val, y_h4_model)
    y_h1_model = jnp.where(y_h1_model < -clip_val, -clip_val, y_h1_model)
    y_h2_model = jnp.where(y_h2_model < -clip_val, -clip_val, y_h2_model)
    y_h3_model = jnp.where(y_h3_model < -clip_val, -clip_val, y_h3_model)
    y_h4_model = jnp.where(y_h4_model < -clip_val, -clip_val, y_h4_model)

    r_h1 = ( y_h1_model - y_h1 ) / sig_A1
    r_h2 = ( y_h2_model - y_h2 ) / sig_A2
    r_h3 = ( y_h3_model - y_h3 ) / sig_A3
    r_h4 = ( y_h4_model - y_h4 ) / sig_A4

    val1 = 0.5 * jnp.dot(r_3dRzphi, r_3dRzphi) / len(r_3dRzphi) * 5
    val2 = 0.5 * jnp.dot(r_2dXY, r_2dXY) / len(r_2dXY) * 5
    val3 = 0.5 * jnp.dot(r_h1, r_h1) / len(r_h1)
    val4 = 0.5 * jnp.dot(r_h2, r_h2) / len(r_h2)
    val5 = 0.5 * jnp.dot(r_h3, r_h3) / len(r_h3)
    val6 = 0.5 * jnp.dot(r_h4, r_h4) / len(r_h4)

    x_renormalised = x / wi
    regularisation_factor = (l2 / N_orb) * jnp.sum(x_renormalised * jnp.log(x_renormalised + EPSILON))

    return val1 + val2 + val3 + val4 + val5 + val6 + regularisation_factor
_nll_z = jax.value_and_grad(_nll_z)

@jax.jit
def solve_lbfgs_softplus(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                        l2=1, maxiter=500, tol=1e-6):

    z0 = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])
    solver = jaxopt.LBFGS(fun=_nll_z, value_and_grad=True, maxiter=maxiter, tol=tol, implicit_diff=True)
    res = solver.run(z0,
                    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, l2)
    x_hat = jnn.softplus(res.params)
    x_hat = jax.lax.stop_gradient(x_hat)
    return x_hat

@jax.jit
def solve_two_stage(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                    l2=1, maxiter=300):

    def density_only_loss(z):
        x = jnn.softplus(z)
        r_3dRzphi = (A_Rzphi @ x - y_Rzphi) / sig_Rzphi
        r_2dXY = (A_xy @ x - y_xy) / sig_xy
        return 0.5 * jnp.dot(r_3dRzphi, r_3dRzphi) + 0.5 * jnp.dot(r_2dXY, r_2dXY)

    z0 = jnp.ones(A_Rzphi.shape[1]) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])
    solver1 = jaxopt.LBFGS(fun=density_only_loss, maxiter=maxiter//2, tol = 1e-3)
    res1 = solver1.run(z0)

    solver2 = jaxopt.LBFGS(fun=_nll_z, value_and_grad=True, maxiter=maxiter//2, tol = 1e-3)
    res2 = solver2.run(res1.params, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, l2)

    x_hat = jnn.softplus(res2.params)
    return x_hat

@partial(jax.jit, static_argnames=("maxiter", "method"))
def solve_nonnegative_nonlinear_cg(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    l2=1.0, maxiter=1000, method="polak-ribiere",
):
    z0 = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])
    solver = jaxopt.NonlinearCG(
        fun=_nll_z,
        value_and_grad=True,
        maxiter=maxiter,
        tol=1e-6,
        method=method,
    )
    res = solver.run(
        z0,
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, l2,
    )
    x_hat = jnn.softplus(res.params)
    return jax.lax.stop_gradient(x_hat)


@partial(jax.jit, static_argnames=("maxiter", "tol"))
def solve_qp_boxcdqp(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=100, tol=1e-1,
):
    eps = 1e-8

    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    U_rz = A_Rzphi / (sig_Rzphi[:, None] + eps)
    y_rz = y_Rzphi / (sig_Rzphi + eps)

    U_xy = A_xy / (sig_xy[:, None] + eps)
    y_xy_obs = y_xy / (sig_xy + eps)

    U_h1 = (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps)
    U_h2 = (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps)
    U_h3 = (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps)
    U_h4 = (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps)

    y_h1_obs = y_h1 / (sig_A1 + eps)
    y_h2_obs = y_h2 / (sig_A2 + eps)
    y_h3_obs = y_h3 / (sig_A3 + eps)
    y_h4_obs = y_h4 / (sig_A4 + eps)

    U = jnp.vstack([U_rz, U_xy, U_h1, U_h2, U_h3, U_h4])
    y = jnp.concatenate([y_rz, y_xy_obs, y_h1_obs, y_h2_obs, y_h3_obs, y_h4_obs])

    n_orb = U.shape[1]
    Q = U.T @ U + (lambda_reg / n_orb) * jnp.eye(n_orb, dtype=U.dtype)
    c = -(U.T @ y)

    lb = jnp.zeros((n_orb,), dtype=U.dtype)
    ub = jnp.full((n_orb,), jnp.inf, dtype=U.dtype)

    w0 = jnp.ones((n_orb,), dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)

    solver = jaxopt.BoxCDQP(maxiter=maxiter, tol=tol, verbose=False, implicit_diff=False)
    sol = solver.run(w0, params_obj=(Q, c), params_ineq=(lb, ub))
    return jax.lax.stop_gradient(sol.params)


@partial(jax.jit, static_argnames=("maxiter", "power_iters"))
def solve_fista_nnls(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=500, power_iters=50,
):
    eps = 1e-8
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    U_rz  = A_Rzphi / (sig_Rzphi[:, None] + eps)
    b_rz  = y_Rzphi / (sig_Rzphi + eps)

    U_xy_ = A_xy / (sig_xy[:, None] + eps)
    b_xy  = y_xy / (sig_xy + eps)

    U_h1_ = (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps)
    U_h2_ = (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps)
    U_h3_ = (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps)
    U_h4_ = (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps)
    b_h1  = y_h1 / (sig_A1 + eps)
    b_h2  = y_h2 / (sig_A2 + eps)
    b_h3  = y_h3 / (sig_A3 + eps)
    b_h4  = y_h4 / (sig_A4 + eps)

    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])
    b = jnp.concatenate([b_rz, b_xy, b_h1, b_h2, b_h3, b_h4])

    n_orb = U.shape[1]
    reg   = jnp.array(lambda_reg, dtype=U.dtype) / n_orb

    v0 = jnp.ones(n_orb, dtype=U.dtype) / jnp.sqrt(float(n_orb))

    def power_step(v, _):
        v = U.T @ (U @ v)
        return v / (jnp.linalg.norm(v) + 1e-30), None

    v_eig, _ = jax.lax.scan(power_step, v0, xs=None, length=power_iters)
    Uv = U @ v_eig
    L  = 1.05 * jnp.dot(Uv, Uv) / (jnp.dot(v_eig, v_eig) + 1e-30) + reg
    lr = 1.0 / L

    w_init = jnp.ones(n_orb, dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)

    UTb = U.T @ b

    def fista_step(carry, _):
        w, z, t = carry

        Uz       = U @ z
        grad     = U.T @ Uz - UTb + reg * z
        w_new    = jnp.maximum(0.0, z - lr * grad)

        restart  = jnp.dot(grad, w_new - w) > 0.0
        t_eff    = jnp.where(restart, 1.0, t)

        t_new    = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_eff * t_eff))
        beta     = (t_eff - 1.0) / t_new

        z_new    = jnp.maximum(0.0, w_new + beta * (w_new - w))

        return (w_new, z_new, t_new), None

    (w_final, _, _), _ = jax.lax.scan(
        fista_step,
        (w_init, w_init, jnp.ones((), dtype=U.dtype)),
        xs=None,
        length=maxiter,
    )
    return jax.lax.stop_gradient(w_final)


@jax.jit
def _compute_logl_from_weights(w, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                                y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                                sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4):
    """
    Compute logL for a weight vector. Matches the NNLS objective
    (no /N_bins, no extra weighting factors).
    """
    eps = 1e-8
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    res_Rzphi = ((A_Rzphi @ w - y_Rzphi) / (sig_Rzphi + EPSILON))**2
    res_density = ((A_xy @ w - y_xy) / (sig_xy + EPSILON))**2

    h1_model = (A_h1 * A_xy) @ w / y_xy_safe
    h2_model = (A_h2 * A_xy) @ w / y_xy_safe
    h3_model = (A_h3 * A_xy) @ w / y_xy_safe
    h4_model = (A_h4 * A_xy) @ w / y_xy_safe

    clip_val = 10.0
    h1_model = jnp.clip(h1_model, -clip_val, clip_val)
    h2_model = jnp.clip(h2_model, -clip_val, clip_val)
    h3_model = jnp.clip(h3_model, -clip_val, clip_val)
    h4_model = jnp.clip(h4_model, -clip_val, clip_val)

    res_h1 = jnp.where(h1_model < 9.9, ((h1_model - y_h1) / (sig_A1 + EPSILON))**2, 0)
    res_h2 = jnp.where(h2_model < 9.9, ((h2_model - y_h2) / (sig_A2 + EPSILON))**2, 0)
    res_h3 = jnp.where(h3_model < 9.9, ((h3_model - y_h3) / (sig_A3 + EPSILON))**2, 0)
    res_h4 = jnp.where(h4_model < 9.9, ((h4_model - y_h4) / (sig_A4 + EPSILON))**2, 0)

    return -0.5 * (jnp.nansum(res_Rzphi) + jnp.nansum(res_density) +
                   jnp.nansum(res_h1) + jnp.nansum(res_h2) +
                   jnp.nansum(res_h3) + jnp.nansum(res_h4))


@jax.jit
def marginalize_weights(weights,
                        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                        lambda_reg):
    """
    Marginalize logL over orbital weight uncertainty using
    eigendecomposition-based QMC sampling in the active subspace.

    TODO: sampling currently disabled for determinism testing.
    Returns logL_best (no marginalization penalty).
    """
    # --- TEMPORARILY DISABLED: just return logL_best ---
    return _compute_logl_from_weights(
        weights, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

    # --- Full marginalization (re-enable later) ---
    # n_orb = weights.shape[0]
    # reg = lambda_reg / n_orb
    # eps = 1e-8

    # y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    # # Build stacked design matrix U (same as NNLS objective)
    # U_rz  = A_Rzphi / (sig_Rzphi[:, None] + eps)
    # U_xy  = A_xy / (sig_xy[:, None] + eps)
    # U_h1  = (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps)
    # U_h2  = (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps)
    # U_h3  = (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps)
    # U_h4  = (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps)
    # U = jnp.vstack([U_rz, U_xy, U_h1, U_h2, U_h3, U_h4])

    # # Active set: top N_ACTIVE_MAX weights
    # sorted_idx = jnp.argsort(-weights)[:N_ACTIVE_MAX]
    # active_weights = weights[sorted_idx]
    # active_mask = (active_weights > W_THRESHOLD).astype(jnp.float32)
    # U_active = U[:, sorted_idx] * active_mask[None, :]

    # # Q_active = U_active^T U_active + reg * I
    # Q_active = U_active.T @ U_active + reg * jnp.eye(N_ACTIVE_MAX)

    # # Eigendecomposition
    # eigvals, eigvecs = jnp.linalg.eigh(Q_active)

    # # Keep well-constrained directions
    # keep_mask = (eigvals > EIG_THRESHOLD).astype(jnp.float32)
    # eigvals_safe = jnp.where(eigvals > EIG_THRESHOLD, eigvals, 1e10)
    # sigma_eig = 1.0 / jnp.sqrt(eigvals_safe)

    # # QMC perturbations in eigenbasis → weight space
    # z_masked = Z_QMC * keep_mask[None, :]          # (N_QMC, N_ACTIVE_MAX)
    # scaled_z = z_masked * sigma_eig[None, :]
    # delta_w_active = scaled_z @ eigvecs.T           # (N_QMC, N_ACTIVE_MAX)
    # w_active_samples = jnp.maximum(active_weights[None, :] + delta_w_active, 0.0)

    # # Scatter into full weight arrays
    # w_full = jnp.tile(weights[None, :], (N_QMC, 1))
    # w_full = w_full.at[:, sorted_idx].set(w_active_samples)

    # # Compute logL for each sample using vmap
    # def _logl_single(w_i):
    #     return _compute_logl_from_weights(
    #         w_i, A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #         y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #         sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

    # logl_samples = jax.vmap(_logl_single)(w_full)   # (N_QMC,)

    # # Log-mean-exp
    # logl_max = jnp.max(logl_samples)
    # logl_marg = logl_max + jnp.log(jnp.mean(jnp.exp(logl_samples - logl_max)))
    # return logl_marg


@partial(jax.jit, static_argnames=("maxiter",))
def solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
):
    eps = 1e-8
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    U_rz  = A_Rzphi / (sig_Rzphi[:, None] + eps)
    y_rz  = y_Rzphi / (sig_Rzphi + eps)
    U_xy_ = A_xy / (sig_xy[:, None] + eps)
    y_xy_ = y_xy / (sig_xy + eps)
    U_h1_ = (A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps)
    U_h2_ = (A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps)
    U_h3_ = (A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps)
    U_h4_ = (A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps)
    y_h1_ = y_h1 / (sig_A1 + eps)
    y_h2_ = y_h2 / (sig_A2 + eps)
    y_h3_ = y_h3 / (sig_A3 + eps)
    y_h4_ = y_h4 / (sig_A4 + eps)

    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])
    y = jnp.concatenate([y_rz, y_xy_, y_h1_, y_h2_, y_h3_, y_h4_])

    n_orb = U.shape[1]
    reg   = lambda_reg / n_orb

    Q = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)
    c = -(U.T @ y)

    rho = jnp.trace(Q) / n_orb

    L_chol = jnp.linalg.cholesky(Q + rho * jnp.eye(n_orb, dtype=U.dtype))

    w_init = jnp.ones(n_orb, dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)
    z_init = w_init.copy()
    u_init = jnp.zeros(n_orb, dtype=U.dtype)

    alpha = 1.6

    def admm_step(carry, _):
        w, z, u = carry
        rhs = rho * (z - u) - c
        w_new = jax.scipy.linalg.cho_solve((L_chol, True), rhs)
        w_hat = alpha * w_new + (1.0 - alpha) * z
        z_new = jnp.maximum(0.0, w_hat + u)
        u_new = u + w_hat - z_new
        return (w_new, z_new, u_new), None

    (_, z_final, _), _ = jax.lax.scan(
        admm_step,
        (w_init, z_init, u_init),
        xs=None,
        length=maxiter,
    )
    return jax.lax.stop_gradient(z_final)


# @jax.jit
@partial(jax.jit, static_argnames=('num_Vbin'))
def model(params_halo_pot, params_disk_rho, dict_data, num_Vbin):

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']
    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    sigma_density_model = params_disk_rho['sigma_density_model']
    sigma_kine_model = params_disk_rho['sigma_kine_model']

    #=========================================== BUILD BARYON PARAMS =====================================================
    # Derive bar parameters
    M_bar = 10.0 ** params_disk_rho['logM_bar']
    L_bar = params_disk_rho['L_bar']
    a_bar = params_disk_rho['a_bar']
    b_bar = params_disk_rho['b_bar']

    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc': params_disk_rho['Rs_disc'],
        'Hs_disc': params_disk_rho['Hs_disc'],
        'x_origin': params_disk_rho['x_origin'],
        'y_origin': params_disk_rho['y_origin'],
        'z_origin': params_disk_rho['z_origin'],
        'dirx': params_disk_rho['dirx'],
        'diry': params_disk_rho['diry'],
        'dirz': params_disk_rho['dirz'],
        'logM_bar': params_disk_rho['logM_bar'],
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
    }

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], params_baryon, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R
        vz = g2 * sig_z
        vphi = v_rot + g3 * sig_phi
        x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
        return jnp.array([x, y, w0[:,2], vx, vy, vz]).T
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    n_realizations = 4
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Integrate orbits =======================================================
    # Single combined potential + grad for acceleration — one forward + one backward pass
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
        return -grad_phi

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    N_step_per_orb = 100
    N_dynamical_time = 50
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-7, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = _integrate_adaptive_batch_vmap(
                        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
                        dt_init_batch, -Omega_bar,
                        atol, rtol,
                        dt_min, dt_max,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #================================== Preprocess the obtained matrices ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    y_xy = y_xy / params_disk_rho['light_to_mass_ratio']

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    h_err_min = 0.03
    sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
    sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
    sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
    sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

    sig_xy = jnp.sqrt(sig_xy**2 + (sigma_density_model*y_xy)**2)
    sig_A1 = jnp.sqrt(sig_A1**2 + (sigma_kine_model)**2)
    sig_A2 = jnp.sqrt(sig_A2**2 + (sigma_kine_model)**2)
    sig_A3 = jnp.sqrt(sig_A3**2 + (sigma_kine_model)**2)
    sig_A4 = jnp.sqrt(sig_A4**2 + (sigma_kine_model)**2)

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Orbital weights optimisation ===========================================

    weights = solve_nnls_admm(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=1, maxiter=200,
    )

    #===================================== Calculate the net kinematics of the model =========================================

    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / y_xy
    h2_model = (A_h2 @ weights) / y_xy
    h3_model = (A_h3 @ weights) / y_xy
    h4_model = (A_h4 @ weights) / y_xy

    clip_val = 10.0
    h1_model = jnp.where(h1_model > clip_val, clip_val, h1_model)
    h2_model = jnp.where(h2_model > clip_val, clip_val, h2_model)
    h3_model = jnp.where(h3_model > clip_val, clip_val, h3_model)
    h4_model = jnp.where(h4_model > clip_val, clip_val, h4_model)
    h1_model = jnp.where(h1_model < -clip_val, -clip_val, h1_model)
    h2_model = jnp.where(h2_model < -clip_val, -clip_val, h2_model)
    h3_model = jnp.where(h3_model < -clip_val, -clip_val, h3_model)
    h4_model = jnp.where(h4_model < -clip_val, -clip_val, h4_model)

    #===================================== Compute logL_best from already-computed quantities ==============================
    # TODO: temporarily bypassing marginalization for determinism testing — reuse matmuls above
    res_Rzphi = ((A_Rzphi @ weights - y_Rzphi) / (sig_Rzphi + EPSILON))**2
    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON))**2
    res_h1 = jnp.where(h1_model < 9.9, ((h1_model - y_h1) / (sig_A1 + EPSILON))**2, 0)
    res_h2 = jnp.where(h2_model < 9.9, ((h2_model - y_h2) / (sig_A2 + EPSILON))**2, 0)
    res_h3 = jnp.where(h3_model < 9.9, ((h3_model - y_h3) / (sig_A3 + EPSILON))**2, 0)
    res_h4 = jnp.where(h4_model < 9.9, ((h4_model - y_h4) / (sig_A4 + EPSILON))**2, 0)
    logl_marg = -0.5 * (jnp.nansum(res_density) +
                        jnp.nansum(res_h1) + jnp.nansum(res_h2) +
                        jnp.nansum(res_h3) + jnp.nansum(res_h4)) - 2 * (jnp.sum(jnp.log(sig_xy)) + 
                        jnp.sum(jnp.log(sig_A1)) + jnp.sum(jnp.log(sig_A2)) + 
                        jnp.sum(jnp.log(sig_A3)) + jnp.sum(jnp.log(sig_A4)))

    V_model, sigma_model = h_to_V_sigma(h1_model, h2_model, v0, s)

    density_set = (density_2DXY, y_xy, sig_xy)
    h1_set = (h1_model, y_h1, sig_A1)
    h2_set = (h2_model, y_h2, sig_A2)
    h3_set = (h3_model, y_h3, sig_A3)
    h4_set = (h4_model, y_h4, sig_A4)

    return density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, weights, logl_marg


@partial(jax.jit, static_argnames=('num_Vbin'))
def model_for_plotting(params_halo_pot, params_disk_rho, dict_data, num_Vbin):

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']
    Omega_bar = params_disk_rho['Omega_bar']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== BUILD BARYON PARAMS =====================================================
    M_bar = 10.0 ** params_disk_rho['logM_bar']
    L_bar = params_disk_rho['L_bar']
    a_bar = params_disk_rho['a_bar']
    b_bar = params_disk_rho['b_bar']

    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc': params_disk_rho['Rs_disc'],
        'Hs_disc': params_disk_rho['Hs_disc'],
        'x_origin': params_disk_rho['x_origin'],
        'y_origin': params_disk_rho['y_origin'],
        'z_origin': params_disk_rho['z_origin'],
        'dirx': params_disk_rho['dirx'],
        'diry': params_disk_rho['diry'],
        'dirz': params_disk_rho['dirz'],
        'logM_bar': params_disk_rho['logM_bar'],
        'L_bar': L_bar,
        'a_bar': a_bar,
        'b_bar': b_bar,
    }

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], params_baryon, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R
        vz = g2 * sig_z
        vphi = v_rot + g3 * sig_phi
        x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
        return jnp.array([x, y, w0[:,2], vx, vy, vz]).T
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    n_realizations = 4
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = 0.1 * _Vc
    v_scale = jnp.clip(v_scale, a_min=1, a_max = 15)
    noise_x = (jax.random.uniform(keys[0], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y = (jax.random.uniform(keys[1], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z = (jax.random.uniform(keys[2], (n_particles, n_realizations,)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations,)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :]
    w0_new_batch = w0_new_batch + jnp.stack([noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    #=========================================== Integrate orbits =======================================================
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
        return -grad_phi

    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    N_step_per_orb = 100
    N_dynamical_time = 100
    N_max = N_step_per_orb * N_dynamical_time
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / N_step_per_orb
    atol, rtol = 1e-6, 1e-4
    dt_min, dt_max = 1e-5, 0.3

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = _integrate_adaptive_batch_vmap(
                        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
                        dt_init_batch, -Omega_bar,
                        atol, rtol,
                        dt_min, dt_max,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #================================== Preprocess the obtained matrices ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    y_xy = y_xy / params_disk_rho['light_to_mass_ratio']

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    h_err_min = 0.03
    sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
    sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
    sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
    sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Orbital weights optimisation ===========================================

    weights = solve_nnls_admm(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=0.1, maxiter=200,
    )
    weights_unity = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])

    #===================================== Calculate the net kinematics of the model =========================================

    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / y_xy
    h2_model = (A_h2 @ weights) / y_xy
    h3_model = (A_h3 @ weights) / y_xy
    h4_model = (A_h4 @ weights) / y_xy

    clip_val = 10.0
    h1_model = jnp.where(h1_model > clip_val, clip_val, h1_model)
    h2_model = jnp.where(h2_model > clip_val, clip_val, h2_model)
    h3_model = jnp.where(h3_model > clip_val, clip_val, h3_model)
    h4_model = jnp.where(h4_model > clip_val, clip_val, h4_model)
    h1_model = jnp.where(h1_model < -clip_val, -clip_val, h1_model)
    h2_model = jnp.where(h2_model < -clip_val, -clip_val, h2_model)
    h3_model = jnp.where(h3_model < -clip_val, -clip_val, h3_model)
    h4_model = jnp.where(h4_model < -clip_val, -clip_val, h4_model)

    V_model, sigma_model = h_to_V_sigma(h1_model, h2_model, v0, s)

    density_set = (density_2DXY, y_xy, sig_xy)
    h1_set = (h1_model, y_h1, sig_A1)
    h2_set = (h2_model, y_h2, sig_A2)
    h3_set = (h3_model, y_h3, sig_A3)
    h4_set = (h4_model, y_h4, sig_A4)

    density_2DXY_unity = A_xy @ weights_unity
    h1_model_unity = (A_h1 @ weights_unity) / y_xy
    h2_model_unity = (A_h2 @ weights_unity) / y_xy
    h3_model_unity = (A_h3 @ weights_unity) / y_xy
    h4_model_unity = (A_h4 @ weights_unity) / y_xy

    clip_val = 10.0
    h1_model_unity = jnp.where(h1_model_unity > clip_val, clip_val, h1_model_unity)
    h2_model_unity = jnp.where(h2_model_unity > clip_val, clip_val, h2_model_unity)
    h3_model_unity = jnp.where(h3_model_unity > clip_val, clip_val, h3_model_unity)
    h4_model_unity = jnp.where(h4_model_unity > clip_val, clip_val, h4_model_unity)
    h1_model_unity = jnp.where(h1_model_unity < -clip_val, -clip_val, h1_model_unity)
    h2_model_unity = jnp.where(h2_model_unity < -clip_val, -clip_val, h2_model_unity)
    h3_model_unity = jnp.where(h3_model_unity < -clip_val, -clip_val, h3_model_unity)
    h4_model_unity = jnp.where(h4_model_unity < -clip_val, -clip_val, h4_model_unity)

    V_model_unity, sigma_model_unity = h_to_V_sigma(h1_model_unity, h2_model_unity, v0, s)

    density_unity_set = (density_2DXY_unity, y_xy, sig_xy)
    h1_unity_set = (h1_model_unity, y_h1, sig_A1)
    h2_unity_set = (h2_model_unity, y_h2, sig_A2)
    h3_unity_set = (h3_model_unity, y_h3, sig_A3)
    h4_unity_set = (h4_model_unity, y_h4, sig_A4)

    return density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
        density_unity_set, V_model_unity, sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
              weights


@partial(jax.jit, static_argnames=('num_Vbin'))
def get_the_orbital_library(params_halo_pot, params_disk_rho, dict_data, num_Vbin):

    w0 = dict_data['w0']
    n_particles = w0.shape[0]
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== BUILD BARYON PARAMS =====================================================
    params_baryon = {
        'logM_disc': params_disk_rho['logM_disc'],
        'Rs_disc': params_disk_rho['Rs_disc'],
        'Hs_disc': params_disk_rho['Hs_disc'],
        'x_origin': params_disk_rho['x_origin'],
        'y_origin': params_disk_rho['y_origin'],
        'z_origin': params_disk_rho['z_origin'],
        'dirx': params_disk_rho['dirx'],
        'diry': params_disk_rho['diry'],
        'dirz': params_disk_rho['dirz'],
        'logM_bar': params_disk_rho['logM_bar'],
        'L_bar': params_disk_rho['L_bar'],
        'a_bar': params_disk_rho['a_bar'],
        'b_bar': params_disk_rho['b_bar'],
    }

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], params_baryon, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R
        vz = g2 * sig_z
        vphi = v_rot + g3 * sig_phi
        x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
        return jnp.array([x, y, w0[:,2], vx, vy, vz]).T
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    w0_new = get_w0_new(w0, key1, key2, key3, n_particles)

    #======================================== Calculate orbital timescale =====================================================
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]

    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R,
        potential_func,
        (params_baryon, params_halo_pot),
        _z
    )

    N_step_per_orb = 100
    N_dynamical_time = 20
    dt = T_orb / N_step_per_orb
    time_integrate = T_orb * N_dynamical_time
    N_steps = N_step_per_orb * N_dynamical_time
    #=========================================== Integrate orbits =======================================================
    @jax.jit
    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        grad_phi = jax.grad(_pot)(jnp.array([x, y, z]))
        return -grad_phi

    _integrate_vmap = jax.vmap(integrate_leapfrog_rot,
                            in_axes=(0, None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None, None))

    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-12.,12.],[-4.,4.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    time = time_integrate
    n_steps = N_steps
    dt = dt
    unroll = False
    initial_time = 0.0
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4 = _integrate_vmap(
                        w0_new, acc_fn, n_steps, dt, initial_time, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    A_Rzphi = Rzphi_bin_counts.T / n_steps
    A_xy = surface_density.T / n_steps
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    #=========================================== Orbital weights optimisation ============================================

    @jax.jit
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_mass(R_grid, z_grid, phi_grid, rho_fct, dict_params, dR, dz, dphi, sample):
        R_samples = R_grid + (sample[:,0] - 0.5) * dR
        z_samples = z_grid + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid + (sample[:,2] - 0.5) * dphi
        density_samples = rho_fct(R_samples, z_samples, phi_samples, dict_params)
        mass_tot = jnp.sum(density_samples * R_samples) / sample.shape[0]
        return mass_tot

    R_grid, dR = dict_data['R_grid'], dict_data['dR']
    z_grid, dz = dict_data['z_grid'], dict_data['dz']
    phi_grid, dphi = dict_data['phi_grid'], dict_data['dphi']
    y_Rzphi = jax.vmap(get_mass, in_axes=[0, 0, 0, None, None, None, None, None, None])(
                R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    y_xy = y_xy / params_disk_rho['light_to_mass_ratio']

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    h_err_min = 0.03
    sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
    sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
    sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
    sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

    y_xy = y_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    return (A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4, \
           y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4, \
           sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4)

@partial(
    jax.jit,
    static_argnames=("solver", "solver_maxiter", "solver_power_iters", "solver_cg_maxiter"),
)
def get_weights(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
           y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
           sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, dict_data,
           solver="lbfgs", solver_maxiter=1000, solver_power_iters=12, solver_cg_maxiter=8):

    v0 = dict_data['v0']
    s = dict_data['s']

    if solver == "nnls":
        weights = solve_fista_nnls(
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
            maxiter=solver_maxiter, power_iters=solver_power_iters,
        )
    elif solver == "admm":
        weights = solve_nnls_admm(
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
            maxiter=solver_maxiter,
        )
    elif solver == "qp":
        weights = solve_qp_boxcdqp(
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        )
    elif solver == "lbfgs":
        weights = solve_lbfgs_softplus(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                                       y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                                       sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                                       l2=1, maxiter=1000)
    else:
        weights = solve_lbfgs_softplus(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                                       y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                                       sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                                       l2=10, maxiter=1000)

    weights_unity = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])

    #===================================== Calculate the net kinematics of the model =========================================

    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / y_xy
    h2_model = (A_h2 @ weights) / y_xy
    h3_model = (A_h3 @ weights) / y_xy
    h4_model = (A_h4 @ weights) / y_xy

    clip_val = 10.0
    h1_model = jnp.where(h1_model > clip_val, clip_val, h1_model)
    h2_model = jnp.where(h2_model > clip_val, clip_val, h2_model)
    h3_model = jnp.where(h3_model > clip_val, clip_val, h3_model)
    h4_model = jnp.where(h4_model > clip_val, clip_val, h4_model)
    h1_model = jnp.where(h1_model < -clip_val, -clip_val, h1_model)
    h2_model = jnp.where(h2_model < -clip_val, -clip_val, h2_model)
    h3_model = jnp.where(h3_model < -clip_val, -clip_val, h3_model)
    h4_model = jnp.where(h4_model < -clip_val, -clip_val, h4_model)

    V_model, sigma_model = h_to_V_sigma(h1_model, h2_model, v0, s)

    density_set = (density_2DXY, y_xy, sig_xy)
    h1_set = (h1_model, y_h1, sig_A1)
    h2_set = (h2_model, y_h2, sig_A2)
    h3_set = (h3_model, y_h3, sig_A3)
    h4_set = (h4_model, y_h4, sig_A4)

    density_2DXY_unity = A_xy @ weights_unity
    h1_model_unity = (A_h1 @ weights_unity) / y_xy
    h2_model_unity = (A_h2 @ weights_unity) / y_xy
    h3_model_unity = (A_h3 @ weights_unity) / y_xy
    h4_model_unity = (A_h4 @ weights_unity) / y_xy

    clip_val = 10.0
    h1_model_unity = jnp.where(h1_model_unity > clip_val, clip_val, h1_model_unity)
    h2_model_unity = jnp.where(h2_model_unity > clip_val, clip_val, h2_model_unity)
    h3_model_unity = jnp.where(h3_model_unity > clip_val, clip_val, h3_model_unity)
    h4_model_unity = jnp.where(h4_model_unity > clip_val, clip_val, h4_model_unity)
    h1_model_unity = jnp.where(h1_model_unity < -clip_val, -clip_val, h1_model_unity)
    h2_model_unity = jnp.where(h2_model_unity < -clip_val, -clip_val, h2_model_unity)
    h3_model_unity = jnp.where(h3_model_unity < -clip_val, -clip_val, h3_model_unity)
    h4_model_unity = jnp.where(h4_model_unity < -clip_val, -clip_val, h4_model_unity)

    V_model_unity, sigma_model_unity = h_to_V_sigma(h1_model_unity, h2_model_unity, v0, s)

    density_unity_set = (density_2DXY_unity, y_xy, sig_xy)
    h1_unity_set = (h1_model_unity, y_h1, sig_A1)
    h2_unity_set = (h2_model_unity, y_h2, sig_A2)
    h3_unity_set = (h3_model_unity, y_h3, sig_A3)
    h4_unity_set = (h4_model_unity, y_h4, sig_A4)

    return density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set,\
        density_unity_set, V_model_unity, sigma_model_unity, h1_unity_set, h2_unity_set, h3_unity_set, h4_unity_set,\
              weights


@partial(jax.jit, static_argnames=('num_Vbin'))
def projection(density_param, dict_data, num_Vbin):

    alpha, beta, gamma = density_param['alpha'], density_param['beta'], density_param['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    @jax.jit
    def density_func_rot(X, Y, Z, rotation_matrix, density_param):
        pos = jnp.stack([X, Y, Z], axis=-1)
        x, y, z = (rotation_matrix.T @ pos.T)
        return density_func(x, y, z, density_param)

    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_surface_density(X_grid, Y_grid, rho_fct, dict_params, dX, dY, sample):
        x_samples = X_grid + (sample[1:,0] - 0.5) * dX
        y_samples = 0      + (sample[1:,1] - 0.5) * 40  # +/-20 kpc
        z_samples = Y_grid + (sample[1:,2] - 0.5) * dY

        density_samples = rho_fct(x_samples, y_samples, z_samples, rotation_matrix, dict_params)
        mass_tot = jnp.sum(density_samples * dX * dY * 40) / x_samples.shape[0]
        surface_density = mass_tot / (dX*dY*1e6)
        return surface_density

    X_grid, dX = dict_data['X_regular_grid'], dict_data['dX']
    Y_grid, dY = dict_data['Y_regular_grid'], dict_data['dY']
    sample_for_integration = dict_data['sample_for_integration_XY']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    surface_density_model = jax.vmap(get_surface_density, in_axes=[0, 0, None, None, None, None, None])(
                X_grid, Y_grid, density_func_rot, density_param, dX, dY, sample_for_integration
    )
    surface_density_model = jnp.array(surface_density_model)

    surface_density_Vbin = jax.ops.segment_sum(surface_density_model, bin_mapping[:-1], num_segments=num_Vbin)
    surface_density_Vbin = surface_density_Vbin / num_per_bin

    surace_luminosity_Vbin = surface_density_Vbin * density_param['light_to_mass_ratio']

    return surace_luminosity_Vbin
