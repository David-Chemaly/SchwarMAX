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
from ghMoments import *
from utils import *
from constants import EPSILON

from potentials import NFW_acceleration
from densities import MiyamotoNagai_density

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
    val = MiyamotoNagai_density(x, y, z, params)
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
    val4 = 0.5 * jnp.dot(r_h1, r_h1) / len(r_h1)# * 5
    val5 = 0.5 * jnp.dot(r_h2, r_h2) / len(r_h2)# * 5
    val3 = 0.5 * jnp.dot(r_h3, r_h3) / len(r_h3)# * 5
    val6 = 0.5 * jnp.dot(r_h4, r_h4) / len(r_h4)# * 5

    x_renormalised = x / wi
    # regularisation_factor = (l2 / N_orb) * jnp.dot(x_renormalised, x_renormalised)
    regularisation_factor = (l2 / N_orb) * jnp.sum(x_renormalised * jnp.log(x_renormalised + EPSILON))

    # jax.debug.print("z={val1}", 
    #                 val1=z,)
    # jax.debug.print("NLL components: Rzphi={val1}, XY={val2}, h1={val3}, h2={val4}, h3={val5}, h4={val6}, reg={reg}, tot={tot}", 
    #                 val1=val1, val2=val2, val3=val3, val4=val4, val5=val5, val6=val6, reg=regularisation_factor, 
    #                 tot=val1 + val2 + val3 + val4 + val5 + val6 + regularisation_factor)


    return val1 + val2 + val3 + val4 + val5 + val6 + regularisation_factor
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

from jaxopt import BoxOSQP
from functools import partial

# @partial(jax.jit, static_argnames=('maxiter',))
def solve_qp(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
             y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
             sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
             l2=1e-5, maxiter=500):
    """
    Quadratic programming for Schwarzschild modeling using BoxOSQP.
    """
    n_orbits = A_Rzphi.shape[1]

    # Linearized kinematic matrices
    A_h1_lin = (A_h1 * A_xy) / y_xy[:, None]
    A_h2_lin = (A_h2 * A_xy) / y_xy[:, None]
    A_h3_lin = (A_h3 * A_xy) / y_xy[:, None]
    A_h4_lin = (A_h4 * A_xy) / y_xy[:, None]

    # Stack all constraints
    A = jnp.vstack([
        A_Rzphi / sig_Rzphi[:, None],
        A_xy / sig_xy[:, None],
        A_h1_lin / sig_A1[:, None],
        A_h2_lin / sig_A2[:, None],
        A_h3_lin / sig_A3[:, None],
        A_h4_lin / sig_A4[:, None],
    ])

    y = jnp.concatenate([
        y_Rzphi / sig_Rzphi,
        y_xy / sig_xy,
        y_h1 / sig_A1,
        y_h2 / sig_A2,
        y_h3 / sig_A3,
        y_h4 / sig_A4,
    ])

    # QP matrices: min 0.5 * x^T Q x + c^T x
    Q = A.T @ A + l2 * jnp.eye(n_orbits)
    c = -A.T @ y

    # Constraint: l <= A_constraint @ x <= u
    # For x >= 0, use A_constraint = I, l = 0, u = inf
    A_constraint = jnp.eye(n_orbits)
    l = jnp.zeros(n_orbits)
    u = jnp.full(n_orbits, jnp.inf)

    qp = BoxOSQP(maxiter=maxiter, tol=1e-3)
    sol = qp.run(params_obj=(Q, c), params_eq=A_constraint, params_ineq=(l, u)).params

    return sol.primal[0]

from functools import partial

# @partial(jax.jit, static_argnames=('maxiter',))
def solve_fista(A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
                y_Rzphi, y_xy, y_h1, y_h2, y_h3, y_h4,
                sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
                l2=1e-5, maxiter=2000):
    """
    FISTA for non-negative least squares.

    Solves: min 0.5 * ||A @ x - y||^2 + 0.5 * l2 * ||x||^2
            s.t. x >= 0
    """
    n_orbits = A_Rzphi.shape[1]

    # Linearized kinematic matrices
    A_h1_lin = (A_h1 * A_xy) / y_xy[:, None]
    A_h2_lin = (A_h2 * A_xy) / y_xy[:, None]
    A_h3_lin = (A_h3 * A_xy) / y_xy[:, None]
    A_h4_lin = (A_h4 * A_xy) / y_xy[:, None]

    # Stack all constraints
    A = jnp.vstack([
        A_Rzphi / sig_Rzphi[:, None],
        A_xy / sig_xy[:, None],
        A_h1_lin / sig_A1[:, None],
        A_h2_lin / sig_A2[:, None],
        A_h3_lin / sig_A3[:, None],
        A_h4_lin / sig_A4[:, None],
    ])

    y = jnp.concatenate([
        y_Rzphi / sig_Rzphi,
        y_xy / sig_xy,
        y_h1 / sig_A1,
        y_h2 / sig_A2,
        y_h3 / sig_A3,
        y_h4 / sig_A4,
    ])

    # Precompute normal equations
    AtA = A.T @ A + l2 * jnp.eye(n_orbits)
    Aty = A.T @ y

    # Lipschitz constant
    L = jnp.linalg.norm(AtA, ord=2)
    step_size = 1.0 / L

    def fista_step(carry, _):
        x, x_old, t = carry

        # Gradient step
        grad = AtA @ x - Aty
        x_new = jnp.clip(x - step_size * grad, 0, None)

        # FISTA momentum
        t_new = (1 + jnp.sqrt(1 + 4 * t**2)) / 2
        x_accel = jnp.clip(x_new + ((t - 1) / t_new) * (x_new - x_old), 0, None)

        return (x_accel, x_new, t_new), None

    x0 = jnp.ones(n_orbits) / n_orbits
    (x_final, _, _), _ = jax.lax.scan(fista_step, (x0, x0, 1.0), None, length=maxiter)

    return x_final

# @jax.jit
@partial(jax.jit, static_argnames=('num_Vbin'))
def model(params_halo_pot, params_disk_rho, dict_data, num_Vbin):

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

    NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-2, 30.0, 1e-3, 15.0, 8.
    Nphi = 200
    N_int = 10_000
    dict_phi = get_phi_m(MiyamotoNagai_density, params_disk_rho, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)

    #=========================================== GET INITIAL VELOCITY ===================================================

    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    # jeans_moments = get_jeans_moments(x_p, y_p, z_p, dict_phi, params_disk_rho,params_halo_pot, anisotropy_b=1.0)
    jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_disk_rho, params_halo_pot, 1.)

    v_rot, sig_R, sig_z, sig_phi = jeans_moments
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    # g1, g2, g3 = jax.random.normal(key1, (n_particles,)), jax.random.normal(key2, (n_particles,)), jax.random.normal(key3, (n_particles,))
    g1, g2, g3 = (jax.random.uniform(key1, (n_particles,))-0.5)*2, (jax.random.uniform(key2, (n_particles,))-0.5)*2, (jax.random.uniform(key3, (n_particles,))-0.5)*2
    vR = g1 * sig_R * 2 # 2 sigma dispersion
    vz = g2 * sig_z * 2
    vphi = v_rot + g3 * sig_phi * 2

    x, y, vx, vy = getCartesianFromCylindrical_clockwise(jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)

    w0_new = jnp.array([x, y, w0[:,2], vx, vy, vz]).T

    #=========================================== Integrate orbits =======================================================
    @jax.jit
    def acc_fn(x, y, z):
        a_halo = NFW_acceleration(x, y, z,  params_halo_pot)
        a_disk = get_acc(x, y, z, dict_phi)
        return a_halo + a_disk

    time = 10. #Gyr
    n_steps = 2500
    dt = time / n_steps
    unroll = False
    initial_time = 0.0

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4 = jax.vmap(integrate_leapfrog_rot, 
                                                     in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None))\
                        (w0_new, acc_fn, n_steps, dt, initial_time, unroll,
                        num_Vbin, bin_mapping, num_per_bin,
                        jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]]), jnp.array([[-12.,12.],[-4.,4.]]),
                        jnp.array([10,6,6]), jnp.array([60,40]), 360,
                        v0, s, rotation_matrix)
    # jax.debug.print("h1 shape: {x_norm}", x_norm=h1.shape)
    # jax.debug.print("h1 nan: {x_norm}", x_norm=jnp.isnan(h1).sum())
    # print("h1 shape: {x_norm}",h1.shape)
    # print("h1 nan: {x_norm}",jnp.isnan(h1).sum())
    # print("h1 non zero: {x_norm}",(h1!=0).sum())
    # print("h2 shape: {x_norm}",h2.shape)
    # print("h2 nan: {x_norm}",jnp.isnan(h2).sum())
    # print("h2 non zero: {x_norm}",(h2!=0).sum())
    # print("h3 shape: {x_norm}",h3.shape)
    # print("h3 nan: {x_norm}",jnp.isnan(h3).sum())
    # print("h3 non zero: {x_norm}",(h3!=0).sum())
    # print("h4 shape: {x_norm}",h4.shape)
    # print("h4 nan: {x_norm}",jnp.isnan(h4).sum())
    # print("h4 non zero: {x_norm}",(h4!=0).sum())
    # print("Nxy nan: {x_norm}",jnp.isnan(Nxy).sum())
    # print("Nxy non zero: {x_norm}",(Nxy!=0).sum())
    #=========================================== Orbital weights optimisation ============================================
    A_Rzphi = Rzphi_bin_counts.T / n_steps
    A_xy = surface_density.T / n_steps
    A_h1 = h1.T
    A_h2 = h2.T
    A_h3 = h3.T
    A_h4 = h4.T

    @jax.jit
    def MiyamotoNagaiDisk(R, z, phi, params_MN):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return MiyamotoNagai_density(x, y, z, params_MN)

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
                R_grid, z_grid, phi_grid, MiyamotoNagaiDisk, params_disk_rho, dR, dz, dphi, dict_data['sample_for_integration']
    )
    # y_Rzphi = dict_data['Rzphi_density_data'].astype(jnp.float32)

    y_xy = dict_data['XY_density_data'].astype(jnp.float32)
    y_h1 = dict_data['h1_data'].astype(jnp.float32)
    y_h2 = dict_data['h2_data'].astype(jnp.float32)
    y_h3 = dict_data['h3_data'].astype(jnp.float32)
    y_h4 = dict_data['h4_data'].astype(jnp.float32)

    sig_Rzphi = 0.02 * y_Rzphi + 1.0
    sig_xy = 0.01 * y_xy + 1.0
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
                                    l2=1, maxiter=500)
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

    # weights = jnp.ones(A_Rzphi.shape[1]) / A_Rzphi.shape[1]

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


# @jax.jit
@partial(jax.jit, static_argnames=('num_Vbin'))
def logl(params, dict_data, num_Vbin):


    params_halo_pot = {
        'logM': params['logM_halo'],
        'Rs':10 ** params['logRs_halo'],
        'a':1.0,
        'b':1.0,
        'c':1.0,
        'x_origin':0.0,
        'y_origin':0.0,
        'z_origin':0.0,
        'dirx':0.0,
        'diry':0.0,
        'dirz':1.0
    }

    params_disk_rho = {
        'logM': params['logM_disk'],
        'Rs': 10 ** params['logRs_disk'],
        'Hs': 10 ** params['logHs_disk'],
        'x_origin': 0.0,
        'y_origin': 0.0,
        'z_origin': 0.0,
        'dirx': 0.0,
        'diry': 0.0,
        'dirz': 1.0,
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma']
    }

    density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, weights = model(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    density_2DXY, y_xy, sig_xy = density_set
    h1_model, y_h1, sig_A1 = h1_set
    h2_model, y_h2, sig_A2 = h2_set
    h3_model, y_h3, sig_A3 = h3_set
    h4_model, y_h4, sig_A4 = h4_set
    # jax.debug.print("`V_model` shape: {x_norm}", x_norm=V_model.shape)
    # jax.debug.print("V_model with nan: {x_norm}", x_norm=jnp.isnan(V_model).sum())
    # jax.debug.print("`V_model`: {x_norm}", x_norm=V_model)
    # jax.debug.print("`sigma_model`: {x_norm}", x_norm=sigma_model)
    # jax.debug.print("`weights`: {x_norm}", x_norm=weights)
    # jax.debug.print("`weights 16th`: {x_norm}", x_norm=jnp.percentile(weights, 16))
    # jax.debug.print("`weights 84th`: {x_norm}", x_norm=jnp.percentile(weights, 84))
    # jax.debug.print("weights with nan: {x_norm}", x_norm=jnp.isnan(weights).sum())



    V_model = jnp.where(jnp.isnan(V_model), 0.0, V_model)
    sigma_model = jnp.where(jnp.isnan(sigma_model), 0.0, sigma_model)
    h3_model = jnp.where(jnp.isnan(h3_model), 0.0, h3_model)
    h4_model = jnp.where(jnp.isnan(h4_model), 0.0, h4_model)
    V_data, V_data_err = dict_data['V_data'], dict_data['V_data_err']
    sigma_data, sigma_data_err = dict_data['sigma_data'], dict_data['sigma_data_err']
    h1_data, h1_data_err = y_h1, sig_A1
    h2_data, h2_data_err = y_h2, sig_A2
    h3_data, h3_data_err = y_h3, sig_A3
    h4_data, h4_data_err = y_h4, sig_A4


    # jax.debug.print("V diff mean sigma_model: {x_norm}, {sigma}",
    #                 x_norm=jnp.nanmean(jnp.fabs(V_model - V_data)), sigma=jnp.nanstd(jnp.fabs(V_model - V_data)))

    # jax.debug.print("sigma diff mean sigma_model: {x_norm}, {sigma}",
    #                 x_norm=jnp.nanmean(jnp.fabs(sigma_model - sigma_data)), sigma=jnp.nanstd(jnp.fabs(sigma_model - sigma_data)))

    # jax.debug.print("h3 diff mean sigma_model: {x_norm}, {sigma}",
    #                 x_norm=jnp.nanmean(jnp.fabs(h3_model - h3_data)), sigma=jnp.nanstd(jnp.fabs(h3_model - h3_data)))

    # jax.debug.print("h4 diff mean sigma_model: {x_norm}, {sigma}",
    #                 x_norm=jnp.nanmean(jnp.fabs(h4_model - h4_data)), sigma=jnp.nanstd(jnp.fabs(h4_model - h4_data)))


    # frac_err_min = 0.1
    V_err_min = 10
    sigma_err_min = 5
    V_data_err = jnp.where(V_err_min > V_data_err, V_err_min, V_data_err)
    sigma_data_err = jnp.where(sigma_err_min > sigma_data_err, sigma_err_min, sigma_data_err)

    res_density = ((density_2DXY - y_xy) / (sig_xy + EPSILON))**2
    res_V = ((V_model - V_data) / (V_data_err + 1e-3))**2
    res_sigma = ((sigma_model - sigma_data) / (sigma_data_err + 1e-3))**2
    res_h1 = ((h1_model - h1_data) / (h1_data_err + 1e-3))**2
    res_h2 = ((h2_model - h2_data) / (h2_data_err + 1e-3))**2
    res_h3 = ((h3_model - h3_data) / (h3_data_err + 1e-3))**2
    res_h4 = ((h4_model - h4_data) / (h4_data_err + 1e-3))**2    

    res_density = jnp.where(res_density<jnp.percentile(res_density, 98.0), res_density, 0)
    res_h1 = jnp.where(res_h1<jnp.percentile(res_h1, 98.0), res_h1, 0)
    res_h2 = jnp.where(res_h2<jnp.percentile(res_h2, 98.0), res_h2, 0)
    res_h3 = jnp.where(res_h3<jnp.percentile(res_h3, 98.0), res_h3, 0)
    res_h4 = jnp.where(res_h4<jnp.percentile(res_h4, 98.0), res_h4, 0)

    val1 = jnp.nansum( -0.5 * res_density ) / len(density_2DXY)
    val2 = jnp.nansum( -0.5 * res_V ) / len(V_model)
    val3 = jnp.nansum( -0.5 * res_sigma ) / len(sigma_model)
    val4 = jnp.nansum( -0.5 * res_h1 ) / len(h1_model)
    val5 = jnp.nansum( -0.5 * res_h2 ) / len(h2_model)
    val6 = jnp.nansum( -0.5 * res_h3 ) / len(h3_model)
    val7 = jnp.nansum( -0.5 * res_h4 ) / len(h4_model)

    log_likelihood = 0
    log_likelihood += val1 + val4 + val5 + val6 + val7
    # jax.debug.print("Log-likelihood components: h1={val4}, h2={val5}, h3={val6}, h4={val7}, tot={tot}",
    #                 val4=val4, val5=val5, val6=val6, val7=val7, tot=log_likelihood)

    return log_likelihood

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
