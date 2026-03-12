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

from potentials import *
from integrants_with_binning import *
from integrants_with_binning import _integrate_barred_vmap
from ghMoments import *
from utils import *
from constants import EPSILON

from densities import MiyamotoNagai_density, DoubleExponentialDisk_density, Hernquist_density, Dehnen_density

from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

path = '../SchwarMAX/'
# path = '/content/drive/MyDrive/SchwarMAX/'

@jax.jit
def potential_func(x, y, z, dict_phi, params_halo):
    """ Returns Phi(R, z) """
    phi_halo = NFW_potential(x, y, z, params_halo)
    phi_disk = evaluate_phi_axisymmetric(x, y, z, dict_phi)
    return phi_halo + phi_disk

@jax.jit
def density_func(x, y, z, params):
    """ Returns Stellar Density nu(R, z) """
    # Double Exponential Disk
    val = DoubleExponentialDisk_density(x, y, z, params) + Dehnen_density(x, y, z, params)
    return val

@jax.jit
def dPhi_dz(x, y, z, dict_phi, params_halo):
    # Numerical derivative of Potential w.r.t z
    d = 5e-3
    return (potential_func(x, y, z+d, dict_phi, params_halo) - potential_func(x, y, z-d, dict_phi, params_halo)) / (2*d)

@jax.jit
def dPhi_dR(x, y, z, dict_phi, params_halo):
    # Numerical derivative of Potential w.r.t R
    d = 5e-3
    R = jnp.sqrt(x**2 + y**2)
    return (potential_func(R+d, 0, z, dict_phi, params_halo) - potential_func(R-d, 0, z, dict_phi, params_halo)) / (2*d)

@jax.jit
def get_jeans_moments(x_star, y_star, z_star, dict_phi, params_disc, params_halo, anisotropy_b=1.0):
    """
    Computes (v_mean, sigma_R, sigma_z, sigma_phi) for a star at (R, z).

    Parameters:
    -----------
    x_star, y_star, z_star : float
        Coordinates of the particle.
    density_fn : function
        Function returning stellar density nu(R, z).
    anisotropy_b : float
        sigma_R^2 / sigma_z^2. Default 1.0 (isotropic).
    """
    R_star = jnp.sqrt(x_star**2 + y_star**2)

    # --- Step 1: Compute Sigma_z (Vertical Integration) ---

    # Integrand: nu(R, z) * dPhi/dz
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, dict_phi, params_halo)

    # Integrate from z_star to infinity (e.g., 20 kpc)
    # Note: We assume symmetry, so we integrate |z| to infinity
    pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
    dx = pts[1] - pts[0]
    # integrand_val = integrand(pts)
    integrand_val = jax.vmap(integrand, in_axes = (0))(pts)
    integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)

    nu_val = density_func(x_star, y_star, z_star, params_disc)
    # if nu_val <= 0: return 0, 0, 0, 0

    sigma_z2 = (1.0 / nu_val) * integral_val

    sigma_z2 = jnp.maximum(sigma_z2, 0.0)

    sigma_z = jnp.sqrt(sigma_z2)

    # --- Step 2: Compute Sigma_R (Anisotropy assumption) ---
    sigma_R2 = anisotropy_b * sigma_z2
    sigma_R = jnp.sqrt(sigma_R2)

    # --- Step 3: Compute v_phi_total^2 (Radial Equation) ---
    # We need d(nu * sigma_R^2) / dR
    # Since sigma_R^2 = b * sigma_z^2, we need d(b * Integral) / dR

    # Define a helper to calculate the "Vertical Pressure" P_zz = nu * sigma_z^2 at any R
    def vertical_pressure(r_in):
        # We need to re-integrate at radius r_in to get the pressure there
        def integrand_r(z_prime):
            return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, dict_phi, params_halo)

        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        dx = pts[1] - pts[0]
        # integrand_val = integrand(pts)
        integrand_val = jax.vmap(integrand_r, in_axes = (0))(pts)
        integral_val = jsp.integrate.trapezoid(integrand_val, pts, dx)
        return integral_val # This is (nu * sigma_z^2)

    # Calculate derivative w.r.t R using central difference
    dR = 5e-3
    Pzz_plus = vertical_pressure(R_star + dR)
    Pzz_minus = vertical_pressure(R_star - dR)
    d_nu_sigR2_dR = anisotropy_b * (Pzz_plus - Pzz_minus) / (2*dR)

    # Radial Jeans Equation:
    # v_phi_sq = sigma_R^2 + (R/nu) * d(nu*sigR^2)/dR + R * dPhi/dR
    term1 = sigma_R2
    term2 = (R_star / nu_val) * d_nu_sigR2_dR
    term3 = R_star * dPhi_dR(x_star, y_star, z_star, dict_phi, params_halo)

    v_phi_total_sq = term1 + term2 + term3

    # --- Step 4: Separate Rotation vs Dispersion ---
    # Assumption: sigma_phi approx sigma_R (Round velocity ellipsoid in plane)
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
    # x = jnp.exp(z)  # strictly positive
    # jax.debug.print("A_Rzphi's shape: {x_norm}", x_norm=A_Rzphi.shape)
    # jax.debug.print("x's shape: {x_norm}", x_norm=x.shape)
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
    val3 = 0.5 * jnp.dot(r_h1, r_h1) / len(r_h1)# * 5
    val4 = 0.5 * jnp.dot(r_h2, r_h2) / len(r_h2)# * 5
    val5 = 0.5 * jnp.dot(r_h3, r_h3) / len(r_h3)# * 5
    val6 = 0.5 * jnp.dot(r_h4, r_h4) / len(r_h4)# * 5

    x_renormalised = x / wi
    # regularisation_factor = (l2 / N_orb) * jnp.dot(x_renormalised, x_renormalised)
    regularisation_factor = (l2 / N_orb) * jnp.sum(x_renormalised * jnp.log(x_renormalised + EPSILON))

    # jax.debug.print("z={val1}", 
    #                 val1=z,)
    # jax.debug.print("NLL components: Rzphi={val1}, XY={val2}, h1={val3}, h2={val4}, h3={val5}, h4={val6}, reg={reg}, tot={tot}", 
    #                 val1=val1, val2=val2, val3=val3, val4=val4, val5=val5, val6=val6, reg=regularisation_factor, 
    #                 tot=val1 + val2 + val3 + val4 + val5 + val6 + regularisation_factor)


    return val1 + val2 + val3 + val4 + val5 + val6 + regularisation_factor#
_nll_z = jax.value_and_grad(_nll_z)

@jax.jit
def solve_lbfgs_softplus(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                        y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                        l2=1, maxiter=500, tol=1e-6):
    
    # jax.debug.print("A_Rzphi shape: {x_norm}", x_norm=A_Rzphi.shape)
    # jax.debug.print("total mass in Rzphi data: {x_norm}", x_norm=jnp.sum(y_Rzphi))
    # jax.debug.print("initial guess: {x_norm}", x_norm=(jnp.sum(y_Rzphi) / A_Rzphi.shape[1]))
    z0 = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])  # initial guess
    # z0 = jnp.zeros(A_Rzphi.shape[1], A_Rzphi.dtype)
    solver = jaxopt.LBFGS(fun=_nll_z, value_and_grad=True, maxiter=maxiter, tol=tol, implicit_diff=True)
    res = solver.run(z0, 
                    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4, l2)
    x_hat = jnn.softplus(res.params)
    # x_hat = jnp.exp(res.params)
    x_hat = jax.lax.stop_gradient(x_hat)
    return x_hat

@jax.jit
def solve_two_stage(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                    l2=1, maxiter=300):
    
    # Stage 1: Fit density only
    def density_only_loss(z):
        x = jnn.softplus(z)
        r_3dRzphi = (A_Rzphi @ x - y_Rzphi) / sig_Rzphi
        r_2dXY = (A_xy @ x - y_xy) / sig_xy
        return 0.5 * jnp.dot(r_3dRzphi, r_3dRzphi) + 0.5 * jnp.dot(r_2dXY, r_2dXY)


    
    z0 = jnp.ones(A_Rzphi.shape[1]) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])  # initial guess    
    # z0 = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1]) + (jax.random.normal(jax.random.PRNGKey(45678), (A_Rzphi.shape[1],)) * 0.5)
    solver1 = jaxopt.LBFGS(fun=density_only_loss, maxiter=maxiter//2, tol = 1e-3)
    res1 = solver1.run(z0)
    # x_hat = jnn.softplus(res1.params)

    # Stage 2: Refine with all constraints, starting from Stage 1 solution
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
    """
    Deterministic non-LBFGS optimizer for non-negative orbital weights.

    We optimize the exact clipped objective (_nll_z) with a softplus
    parametrization and nonlinear conjugate gradient.
    """
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
    """
    Box-constrained QP solver for non-negative orbital weights.

    Objective:
        min_w 0.5 * ||U w - y||^2 + 0.5 * lambda_reg * ||w||^2
        s.t.  w >= 0

    The weighted design matrix U follows the same relative term weighting
    as _nll_z for Rzphi / XY / h1-h4 blocks.
    """
    eps = 1e-8

    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    w_rzphi = jnp.sqrt(5.0 / A_Rzphi.shape[0])
    w_xy = jnp.sqrt(5.0 / A_xy.shape[0])
    w_h = jnp.sqrt(1.0 / A_h1.shape[0])

    U_rz = w_rzphi * (A_Rzphi / (sig_Rzphi[:, None] + eps))
    y_rz = w_rzphi * (y_Rzphi / (sig_Rzphi + eps))

    U_xy = w_xy * (A_xy / (sig_xy[:, None] + eps))
    y_xy_obs = w_xy * (y_xy / (sig_xy + eps))

    U_h1 = w_h * ((A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps))
    U_h2 = w_h * ((A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps))
    U_h3 = w_h * ((A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps))
    U_h4 = w_h * ((A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps))

    y_h1_obs = w_h * (y_h1 / (sig_A1 + eps))
    y_h2_obs = w_h * (y_h2 / (sig_A2 + eps))
    y_h3_obs = w_h * (y_h3 / (sig_A3 + eps))
    y_h4_obs = w_h * (y_h4 / (sig_A4 + eps))

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
    """
    Non-negative orbital weight estimation via FISTA with adaptive restart.

    Solves the same problem as solve_qp_boxcdqp:
        min_{w >= 0}  0.5 ||U w - b||^2 + 0.5 (lambda_reg / n_orb) ||w||^2

    Never forms the n_orb x n_orb Gram matrix Q = U.T @ U.
    Each iteration uses two O(m * n_orb) matvecs with U.

    Three correctness fixes vs the original broken FISTA:
      1. Extrapolated point z is projected onto the non-negative orthant.
         Without this, z goes negative, the gradient at z is meaningless for
         the constrained problem, and the iterates diverge.
      2. Gradient restart (O'Donoghue & Candes 2015): momentum is reset when
         grad(z) . (w_new - w) > 0, i.e. the gradient opposes the momentum.
         Costs only one O(n) dot product — no extra matvec.
      3. Lipschitz safety margin (×1.05) prevents step-size overestimation
         from causing oscillations.
    """
    eps = 1e-8
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    # Identical weighting/scaling as solve_qp_boxcdqp
    w_rzphi = jnp.sqrt(5.0 / A_Rzphi.shape[0])
    w_xy    = jnp.sqrt(5.0 / A_xy.shape[0])
    w_h     = jnp.sqrt(1.0 / A_h1.shape[0])

    U_rz  = w_rzphi * (A_Rzphi / (sig_Rzphi[:, None] + eps))
    b_rz  = w_rzphi * (y_Rzphi / (sig_Rzphi + eps))

    U_xy_ = w_xy * (A_xy / (sig_xy[:, None] + eps))
    b_xy  = w_xy * (y_xy / (sig_xy + eps))

    U_h1_ = w_h * ((A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps))
    U_h2_ = w_h * ((A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps))
    U_h3_ = w_h * ((A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps))
    U_h4_ = w_h * ((A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps))
    b_h1  = w_h * (y_h1 / (sig_A1 + eps))
    b_h2  = w_h * (y_h2 / (sig_A2 + eps))
    b_h3  = w_h * (y_h3 / (sig_A3 + eps))
    b_h4  = w_h * (y_h4 / (sig_A4 + eps))

    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])
    b = jnp.concatenate([b_rz, b_xy, b_h1, b_h2, b_h3, b_h4])

    n_orb = U.shape[1]
    reg   = jnp.array(lambda_reg, dtype=U.dtype) / n_orb

    # ------------------------------------------------------------------
    # Lipschitz constant L = max_eigenvalue(U.T @ U) + reg
    # via power iteration, with 5% safety margin.
    # ------------------------------------------------------------------
    v0 = jnp.ones(n_orb, dtype=U.dtype) / jnp.sqrt(float(n_orb))

    def power_step(v, _):
        v = U.T @ (U @ v)
        return v / (jnp.linalg.norm(v) + 1e-30), None

    v_eig, _ = jax.lax.scan(power_step, v0, xs=None, length=power_iters)
    Uv = U @ v_eig
    L  = 1.05 * jnp.dot(Uv, Uv) / (jnp.dot(v_eig, v_eig) + 1e-30) + reg
    lr = 1.0 / L

    # Same initial guess as solve_qp_boxcdqp
    w_init = jnp.ones(n_orb, dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)

    # Precompute U.T @ b once (reused every iteration)
    UTb = U.T @ b

    # ------------------------------------------------------------------
    # FISTA with gradient restart (O'Donoghue & Candes 2015).
    #
    # carry = (w, z, t)
    #   w: current iterate (non-negative)
    #   z: extrapolated point (also projected non-negative)
    #   t: momentum scalar (reset to 1 on restart)
    #
    # Per iteration: 2 matvecs with U + O(n) restart check.
    # ------------------------------------------------------------------
    def fista_step(carry, _):
        w, z, t = carry

        # Gradient at z (z is already non-negative from projection)
        Uz       = U @ z
        grad     = U.T @ Uz - UTb + reg * z
        w_new    = jnp.maximum(0.0, z - lr * grad)

        # Gradient restart: reset momentum when the gradient at the
        # extrapolated point opposes the step direction (w_new - w).
        # This is the O'Donoghue & Candes criterion; costs only O(n).
        restart  = jnp.dot(grad, w_new - w) > 0.0
        t_eff    = jnp.where(restart, 1.0, t)

        t_new    = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_eff * t_eff))
        beta     = (t_eff - 1.0) / t_new

        # Extrapolated point, projected onto non-negative orthant.
        # This is the critical fix: without projection z can go very
        # negative, making the gradient at z meaningless for the
        # constrained problem and causing convergence to a wrong point.
        z_new    = jnp.maximum(0.0, w_new + beta * (w_new - w))

        return (w_new, z_new, t_new), None

    (w_final, _, _), _ = jax.lax.scan(
        fista_step,
        (w_init, w_init, jnp.ones((), dtype=U.dtype)),
        xs=None,
        length=maxiter,
    )
    return jax.lax.stop_gradient(w_final)


@partial(jax.jit, static_argnames=("maxiter",))
def solve_nnls_admm(
    A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    lambda_reg=1, maxiter=200,
):
    """
    Non-negative QP via ADMM (Alternating Direction Method of Multipliers).

    Solves the same problem as solve_qp_boxcdqp:
        min_{w >= 0}  0.5 w^T Q w + c^T w

    Algorithm:
      1. Form Q = U.T @ U + reg*I  (one GEMM)
      2. Cholesky-factor  Q + rho*I  (done once)
      3. ADMM loop (each step = one triangular solve + elementwise max):
           w = cho_solve(L, rho*(z - u) - c)
           z = max(0, w + u)
           u = u + w - z

    On GPU (T4), expected:
      Q + Cholesky:  ~0.5-1s  (GEMM + Cholesky on 10000 x 10000)
      200 iters:     ~0.5-1s  (triangular solves are O(n^2), parallelizable)
      Total:         ~1-2s    vs ~15s for jaxopt.BoxCDQP
    """
    eps = 1e-8
    y_xy_safe = jnp.where(jnp.abs(y_xy) > eps, y_xy, 1.0)

    w_rzphi = jnp.sqrt(5.0 / A_Rzphi.shape[0])
    w_xy    = jnp.sqrt(5.0 / A_xy.shape[0])
    w_h     = jnp.sqrt(1.0 / A_h1.shape[0])

    U_rz  = w_rzphi * (A_Rzphi / (sig_Rzphi[:, None] + eps))
    y_rz  = w_rzphi * (y_Rzphi / (sig_Rzphi + eps))
    U_xy_ = w_xy * (A_xy / (sig_xy[:, None] + eps))
    y_xy_ = w_xy * (y_xy / (sig_xy + eps))
    U_h1_ = w_h * ((A_h1 * A_xy) / y_xy_safe[:, None] / (sig_A1[:, None] + eps))
    U_h2_ = w_h * ((A_h2 * A_xy) / y_xy_safe[:, None] / (sig_A2[:, None] + eps))
    U_h3_ = w_h * ((A_h3 * A_xy) / y_xy_safe[:, None] / (sig_A3[:, None] + eps))
    U_h4_ = w_h * ((A_h4 * A_xy) / y_xy_safe[:, None] / (sig_A4[:, None] + eps))
    y_h1_ = w_h * (y_h1 / (sig_A1 + eps))
    y_h2_ = w_h * (y_h2 / (sig_A2 + eps))
    y_h3_ = w_h * (y_h3 / (sig_A3 + eps))
    y_h4_ = w_h * (y_h4 / (sig_A4 + eps))

    U = jnp.vstack([U_rz, U_xy_, U_h1_, U_h2_, U_h3_, U_h4_])
    y = jnp.concatenate([y_rz, y_xy_, y_h1_, y_h2_, y_h3_, y_h4_])

    n_orb = U.shape[1]
    reg   = lambda_reg / n_orb

    # Step 1: Form Q and c — identical to solve_qp_boxcdqp
    Q = U.T @ U + reg * jnp.eye(n_orb, dtype=U.dtype)
    c = -(U.T @ y)

    # ADMM penalty parameter — standard heuristic: rho = trace(Q) / n
    rho = jnp.trace(Q) / n_orb

    # Step 2: Cholesky factorization of (Q + rho*I) — done once
    L_chol = jnp.linalg.cholesky(Q + rho * jnp.eye(n_orb, dtype=U.dtype))

    # Initial state
    w_init = jnp.ones(n_orb, dtype=U.dtype) * (jnp.sum(y_Rzphi) / n_orb)
    z_init = w_init.copy()
    u_init = jnp.zeros(n_orb, dtype=U.dtype)

    # Over-relaxation parameter (Boyd et al. 2011, Section 3.4.3)
    # alpha in [1.5, 1.8] typically accelerates ADMM convergence.
    alpha = 1.6

    # Step 3: ADMM iterations with over-relaxation
    def admm_step(carry, _):
        w, z, u = carry
        # w-update: solve (Q + rho*I) w = rho*(z - u) - c
        rhs = rho * (z - u) - c
        w_new = jax.scipy.linalg.cho_solve((L_chol, True), rhs)
        # Over-relaxation: blend w_new toward z
        w_hat = alpha * w_new + (1.0 - alpha) * z
        # z-update: proximal operator for non-negativity
        z_new = jnp.maximum(0.0, w_hat + u)
        # u-update: dual variable
        u_new = u + w_hat - z_new
        return (w_new, z_new, u_new), None

    (_, z_final, _), _ = jax.lax.scan(
        admm_step,
        (w_init, z_init, u_init),
        xs=None,
        length=maxiter,
    )
    # z is the non-negative solution
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

    #=========================================== GET DISC POTENTIAL =====================================================

    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-3, 30.0, 1e-3, 15.0, 8.
    Nphi = 300
    N_int = 10_000
    dict_phi = get_phi_m(density_func, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    # jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R # 2 sigma dispersion
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
        (dict_phi, params_halo_pot),
        _z
    )

    E_pot = jax.vmap(potential_func, in_axes=(0, 0, 0, None, None))(w0_new[:,0], w0_new[:,1], w0_new[:,2], dict_phi, params_halo_pot)
    E_kin = 0.5 * (w0_new[:,3]**2 + w0_new[:,4]**2 + w0_new[:,5]**2)
    E_J = E_pot + E_kin - Omega_bar * (w0_new[:,3] * w0_new[:,1] - w0_new[:,4] * w0_new[:,0])

    w0_new = w0_new[jnp.argsort(E_J)]
    w0_lowE = w0_new[:5000, :]
    w0_highE = w0_new[5000:, :]
    T_orb_lowE = T_orb[jnp.argsort(E_J)][:5000]
    T_orb_highE = T_orb[jnp.argsort(E_J)][5000:]

    #=========================================== Integrate orbits =======================================================
    @jax.jit
    def acc_fn(x, y, z):
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk
    
    @jax.jit
    def pot_fn(x, y, z):
        return potential_func(x, y, z, dict_phi, params_halo_pot)
    pot_fn = jax.vmap(pot_fn, in_axes=(0, 0, 0))


    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-12.,12.],[-4.,4.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    N_step_per_orb = 200
    N_dynamical_time = 25
    dt = T_orb_lowE / N_step_per_orb
    time_integrate = T_orb_lowE * N_dynamical_time
    N_steps = N_step_per_orb * N_dynamical_time

    time = time_integrate #Gyr
    n_steps = N_steps
    dt = dt
    unroll = False
    initial_time = 0.0
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = _integrate_barred_vmap(
                        w0_lowE, acc_fn, pot_fn, n_steps, dt, initial_time, -Omega_bar, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    A_Rzphi_1 = Rzphi_bin_counts.T / n_steps
    A_xy_1 = surface_density.T / n_steps
    A_h1_1 = h1.T
    A_h2_1 = h2.T
    A_h3_1 = h3.T
    A_h4_1 = h4.T


    N_step_per_orb = 100
    N_dynamical_time = 50
    dt = T_orb_highE / N_step_per_orb
    time_integrate = T_orb_highE * N_dynamical_time
    N_steps = N_step_per_orb * N_dynamical_time

    time = time_integrate #Gyr
    n_steps = N_steps
    dt = dt
    unroll = False
    initial_time = 0.0
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = _integrate_barred_vmap(
                        w0_highE, acc_fn, pot_fn, n_steps, dt, initial_time, -Omega_bar, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        Rzphi_lim_grid, xy_lim_grid,
                        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                        v0, s, rotation_matrix)
    A_Rzphi_2 = Rzphi_bin_counts.T / n_steps
    A_xy_2 = surface_density.T / n_steps
    A_h1_2 = h1.T
    A_h2_2 = h2.T
    A_h3_2 = h3.T
    A_h4_2 = h4.T

    A_Rzphi = jnp.concatenate([A_Rzphi_1, A_Rzphi_2], axis=1)
    A_xy = jnp.concatenate([A_xy_1, A_xy_2], axis=1)
    A_h1 = jnp.concatenate([A_h1_1, A_h1_2], axis=1)
    A_h2 = jnp.concatenate([A_h2_1, A_h2_2], axis=1)
    A_h3 = jnp.concatenate([A_h3_1, A_h3_2], axis=1)
    A_h4 = jnp.concatenate([A_h4_1, A_h4_2], axis=1)

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
    # y_Rzphi = dict_data['Rzphi_density_data'].astype(jnp.float32)

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    y_xy = y_xy / params_disk_rho['light_to_mass_ratio'] # convert from light to mass

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    # frac_err_min = 0.1
    h_err_min = 0.03
    sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
    sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
    sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
    sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

    y_xy = y_xy / mean_mass_per_orb
    # A_xy = A_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    # A_Rzphi = A_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb

    #=========================================== Orbital weights optimisation ===========================================

    ############################## LBFGS solver ###############################
    # weights = solve_lbfgs_softplus(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=10, maxiter=1000)
    
    ############################## QP solver ###############################

    weights = solve_nnls_admm(
                            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                            y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                            lambda_reg=1, maxiter=200,
    )

    #===================================== Calculate the net kinematics of the model =========================================


    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / y_xy # density_2DXY
    h2_model = (A_h2 @ weights) / y_xy # density_2DXY
    h3_model = (A_h3 @ weights) / y_xy # density_2DXY
    h4_model = (A_h4 @ weights) / y_xy # density_2DXY

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

    return density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, weights


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

    #=========================================== GET DISC POTENTIAL =====================================================

    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-3, 30.0, 1e-3, 15.0, 8.
    Nphi = 300
    N_int = 10_000
    dict_phi = get_phi_m(density_func, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    # jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R # 2 sigma dispersion
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
        (dict_phi, params_halo_pot),
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
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk

    _integrate_vmap = jax.vmap(integrate_leapfrog_barred, 
                        in_axes=(
                                 0, None, None, 0, None, None, None, 
                                 None, None, None, 
                                 None, None, 
                                 None, None, None, 
                                 None, None, None))

    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-12.,12.],[-4.,4.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    time = time_integrate #Gyr
    n_steps = N_steps
    dt = dt
    unroll = False
    initial_time = 0.0
    Rzphi_bin_counts, surface_density, h1, h2, h3, h4 = _integrate_vmap(
                        w0_new, acc_fn, n_steps, dt, initial_time, -Omega_bar, unroll,
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
    # y_Rzphi = dict_data['Rzphi_density_data'].astype(jnp.float32)

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    y_xy = y_xy / params_disk_rho['light_to_mass_ratio'] # convert from light to mass

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    # frac_err_min = 0.1
    h_err_min = 0.03
    sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
    sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
    sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
    sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

    y_xy = y_xy / mean_mass_per_orb
    # A_xy = A_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    # A_Rzphi = A_Rzphi / mean_mass_per_orb
    sig_Rzphi = sig_Rzphi / mean_mass_per_orb


    weights = solve_lbfgs_softplus(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                                    y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                                    sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                                    l2=10, maxiter=1000)
    # weights = solve_two_stage(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=1, maxiter=500)
    # weights = solve_qp(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=1e-3, maxiter=500)
    # weights = solve_fista(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=1e-3, maxiter=500)

    # weights = jax.lax.stop_gradient(weights)

    weights_unity = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])

    #===================================== Calculate the net kinematics of the model =========================================


    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / y_xy # density_2DXY
    h2_model = (A_h2 @ weights) / y_xy # density_2DXY
    h3_model = (A_h3 @ weights) / y_xy # density_2DXY
    h4_model = (A_h4 @ weights) / y_xy # density_2DXY

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
    h1_model_unity = (A_h1 @ weights_unity) / y_xy # density_2DXY
    h2_model_unity = (A_h2 @ weights_unity) / y_xy # density_2DXY
    h3_model_unity = (A_h3 @ weights_unity) / y_xy # density_2DXY
    h4_model_unity = (A_h4 @ weights_unity) / y_xy # density_2DXY

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
    # num_Vbin = dict_data['total_bins']
    alpha, beta, gamma = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

    #=========================================== GET DISC POTENTIAL =====================================================

    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-2, 30.0, 1e-2, 15.0, 8.
    Nphi = 200
    N_int = 10_000
    dict_phi = get_phi_m(density_func, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    # jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
    def get_w0_new(w0, key1, key2, key3, n_particles):
        jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_disk_rho, params_halo_pot, 1.)
        v_rot, sig_R, sig_z, sig_phi = jeans_moments
        g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
        vR = g1 * sig_R # 2 sigma dispersion
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
        (dict_phi, params_halo_pot),
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
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk

    _integrate_vmap = jax.vmap(integrate_leapfrog_rot, 
                            in_axes=(0, None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None, None))

    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-12.,12.],[-4.,4.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])
    Rzphi_n_tot = 360

    time = time_integrate #Gyr
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
    # y_Rzphi = dict_data['Rzphi_density_data'].astype(jnp.float32)

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    y_xy = y_xy / params_disk_rho['light_to_mass_ratio'] # convert from light to mass

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = 0.01 * y_xy + 1e-10
    # frac_err_min = 0.1
    h_err_min = 0.03
    sig_A1 = jnp.where(h_err_min > dict_data['h1_data_err'], h_err_min, dict_data['h1_data_err']) + EPSILON
    sig_A2 = jnp.where(h_err_min > dict_data['h2_data_err'], h_err_min, dict_data['h2_data_err']) + EPSILON
    sig_A3 = jnp.where(h_err_min > dict_data['h3_data_err'], h_err_min, dict_data['h3_data_err']) + EPSILON
    sig_A4 = jnp.where(h_err_min > dict_data['h4_data_err'], h_err_min, dict_data['h4_data_err']) + EPSILON

    mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]

    y_xy = y_xy / mean_mass_per_orb
    # A_xy = A_xy / mean_mass_per_orb
    sig_xy = sig_xy / mean_mass_per_orb
    y_Rzphi = y_Rzphi / mean_mass_per_orb
    # A_Rzphi = A_Rzphi / mean_mass_per_orb
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
    # weights = solve_two_stage(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=1, maxiter=500)
    # weights = solve_qp(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=1e-3, maxiter=500)
    # weights = solve_fista(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
    #                                 y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
    #                                 sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
    #                                 l2=1e-3, maxiter=500)

    # weights = jax.lax.stop_gradient(weights)

    weights_unity = jnp.ones(A_Rzphi.shape[1], A_Rzphi.dtype) * (jnp.sum(y_Rzphi) / A_Rzphi.shape[1])

    #===================================== Calculate the net kinematics of the model =========================================


    A_h1, A_h2, A_h3, A_h4 = (A_h1 * A_xy), (A_h2 * A_xy), (A_h3 * A_xy), (A_h4 * A_xy)
    density_2DXY = A_xy @ weights
    h1_model = (A_h1 @ weights) / y_xy # density_2DXY
    h2_model = (A_h2 @ weights) / y_xy # density_2DXY
    h3_model = (A_h3 @ weights) / y_xy # density_2DXY
    h4_model = (A_h4 @ weights) / y_xy # density_2DXY

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
    h1_model_unity = (A_h1 @ weights_unity) / y_xy # density_2DXY
    h2_model_unity = (A_h2 @ weights_unity) / y_xy # density_2DXY
    h3_model_unity = (A_h3 @ weights_unity) / y_xy # density_2DXY
    h4_model_unity = (A_h4 @ weights_unity) / y_xy # density_2DXY

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

    density_param_disc = {
        'rho0_disc': density_param['rho0_disc'],
        'Rd_disc': density_param['Rd_disc'],
        'hz_disc': density_param['hz_disc'],
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    density_param_bulge = {
        'logM_bar': density_param['logM_bar'],
        'Rs_bar': density_param['Rs_bar'],
        'q_bar': 0.3,
        'p_bar': 0.3,
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
    }

    @jax.jit
    def density_func_rot(X, Y, Z, rotation_matrix, param_disc, param_bulge):
        pos = jnp.stack([X, Y, Z], axis=-1)
        x, y, z = (rotation_matrix.T @ pos.T)
        return DoubleExponentialDisk_density(x, y, z, param_disc) + Dehnen_density(x, y, z, param_bulge)
    
    @partial(jax.jit, static_argnames=['rho_fct'])
    def get_surface_density(X_grid, Y_grid, rho_fct, dict_params_disc, dict_params_bulge, dX, dY, sample):
        x_samples = X_grid + (sample[1:,0] - 0.5) * dX
        y_samples = 0      + (sample[1:,1] - 0.5) * 40  # +/-20 kpc
        z_samples = Y_grid + (sample[1:,2] - 0.5) * dY

        density_samples = rho_fct(x_samples, y_samples, z_samples, rotation_matrix, dict_params_disc, dict_params_bulge)
        mass_tot = jnp.sum(density_samples * dX * dY * 40) / x_samples.shape[0]
        surface_density = mass_tot / (dX*dY*1e6)
        return surface_density

    X_grid, dX = dict_data['X_regular_grid'], dict_data['dX']
    Y_grid, dY = dict_data['Y_regular_grid'], dict_data['dY']
    sample_for_integration = dict_data['sample_for_integration_XY']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']

    surface_density_model = jax.vmap(get_surface_density, in_axes=[0, 0, None, None, None, None, None, None])(
                X_grid, Y_grid, density_func_rot, density_param_disc, density_param_bulge, dX, dY, sample_for_integration
    )
    surface_density_model = jnp.array(surface_density_model)

    surface_density_Vbin = jax.ops.segment_sum(surface_density_model, bin_mapping[:-1], num_segments=num_Vbin)
    surface_density_Vbin = surface_density_Vbin / num_per_bin

    surace_luminosity_Vbin = surface_density_Vbin * density_param['light_to_mass_ratio']

    return surace_luminosity_Vbin

###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    df_ic = pd.read_csv(path + 'mock_initial_conditions_xyz.csv')
    df_ic = df_ic[np.sqrt(df_ic['x']**2 + df_ic['y']**2) < 15.0]
    df_ic = df_ic[np.fabs(df_ic['z']) < 4.0]

    n_particles =  20_000

    print(n_particles)
    np.random.seed(42)
    index = np.random.choice(len(df_ic['x']), size=n_particles, replace=False)
    df_ic = df_ic.iloc[index]
    # w0 = jnp.array([df_ic['x'], df_ic['y'], df_ic['z'], df_ic['vx'], df_ic['vy'], df_ic['vz']]).T
    w0 = jnp.array(df_ic[['x','y','z']].to_numpy())


    with open(path + 'mock_axisymmetric_disc_XY_withRot.pkl', 'rb') as f:
        bin_dict = pickle.load(f)

    # regular grid parameters
    nX, nY = bin_dict['nX_nY']
    X_min, X_max = bin_dict['X_minmax']
    Y_min, Y_max = bin_dict['Y_minmax']
    X_regular_grid = bin_dict['X_regular_grid']
    Y_regular_grid = bin_dict['Y_regular_grid']

    # voronoi binning mapping and data
    xy_vbins = bin_dict['voronoi_bins_xy']
    num_per_bin = jnp.array(bin_dict['num_per_bin'])
    total_bins = jnp.array(bin_dict['total_bins'])
    bin_mapping = jnp.array(bin_dict['bin_mapping'])
    surface_density = jnp.array(bin_dict['surface_density'])
    V_data = jnp.array(bin_dict['V_mean'])
    sigma_data = jnp.array(bin_dict['V_sigma'])
    h1_data = jnp.array(bin_dict['h1'])
    h2_data = jnp.array(bin_dict['h2'])
    h3_data = jnp.array(bin_dict['h3'])
    h4_data = jnp.array(bin_dict['h4'])
    v0 = jnp.array(bin_dict['v0'])
    s = jnp.array(bin_dict['s'])
    alpha, beta, gamma = bin_dict['orientation']
    # alpha, beta, gamma = 0,0,0

    V_data_err = jnp.where(0.1 * jnp.fabs(V_data) < 10, 10, 0.1 * V_data)
    sigma_data_err = jnp.where(0.1 * jnp.fabs(sigma_data) < 5, 5, 0.1 * sigma_data)
    h1_data_err = jnp.where(0.1 * jnp.fabs(h1_data) < 0.03, 0.03, 0.1 * jnp.fabs(h1_data))
    h2_data_err = jnp.where(0.1 * jnp.fabs(h2_data) < 0.03, 0.03, 0.1 * jnp.fabs(h2_data))
    h3_data_err = jnp.where(0.1 * jnp.fabs(h3_data) < 0.03, 0.03, 0.1 * jnp.fabs(h3_data))
    h4_data_err = jnp.where(0.1 * jnp.fabs(h4_data) < 0.03, 0.03, 0.1 * jnp.fabs(h4_data))

    df_Rzphi_data = pd.read_csv(path + 'mock_axisymmetric_disc_Rzphi.csv')
    Rzphi_density_data = jnp.array(df_Rzphi_data['mass'].to_numpy()).astype(jnp.float32)
    with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
        Rzphi_density_data_load = pickle.load(f)

    R_grid, z_grid, phi_grid = Rzphi_density_data_load['R_grid'], Rzphi_density_data_load['z_grid'], Rzphi_density_data_load['phi_grid']
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    sample_for_integration = Rzphi_density_data_load['sample_for_integration']
    # Rzphi_density_data = jnp.array([
    #         get_mass(R_grid[i], z_grid[i], phi_grid[i], dR, dz, dphi,
    #                 dict_data['sample_for_integration']) for i in range(len(R_grid))
    #     ]).astype(jnp.float32)
    dict_data = {
        'w0': w0,
        'v0': v0,
        's': s,
        'Rzphi_density_data': Rzphi_density_data,
        'XY_density_data': surface_density,
        'V_data': V_data,
        'V_data_err': V_data_err,
        'sigma_data': sigma_data,
        'sigma_data_err': sigma_data_err,
        'h1_data': h1_data,
        'h1_data_err': h1_data_err,
        'h2_data': h2_data,
        'h2_data_err': h2_data_err,
        'h3_data': h3_data,
        'h3_data_err': h3_data_err,
        'h4_data': h4_data,
        'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin,
        'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),
        'R_grid': R_grid,
        'z_grid': z_grid,
        'phi_grid': phi_grid,
        'dR': dR,   
        'dz': dz,
        'dphi': dphi,
        'sample_for_integration': sample_for_integration
    }


    with open(path + '/mock_axisymmetric_disc_potential_params.pkl', 'rb') as f:
        gt_params = pickle.load(f)

    N = 20
    i = 9
    ground_truth = {
            'logM_halo': gt_params['halo_params']['logM'].item() + (i-N/2)/10,
            'logRs_halo': jnp.log10(gt_params['halo_params']['scaleRadius']).item(),
            'logM_disk': gt_params['disc_params']['logM'].item(),
            'logRs_disk': jnp.log10(gt_params['disc_params']['scaleRadius']).item(),
            'logHs_disk': jnp.log10(gt_params['disc_params']['scaleHeight']).item(),
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
    }

    logL = logl(ground_truth, dict_data, dict_data['total_bins'])
    print(logL)
