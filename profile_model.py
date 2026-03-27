"""
Profile the time budget of model_bootstrap().

Breaks the function into its major blocks and times each one.
Run on GPU for meaningful results (JAX will JIT-compile on first call).

Usage:
    python profile_model.py
"""

from google.colab import drive
drive.mount('/content/drive')

# ! pip install numpyro
# # ! pip install jax_cosmo
# ! pip install arviz
! pip install jaxopt
! pip install corner
! pip install emcee

path = '/content/drive/MyDrive/SchwarMAX-analytic/'

import sys
sys.path.append(path)

from model_bar import *
from likelihoods_bar import *
from utils import *
from sample_from_density import sample_from_density_grid

import os
# os.environ["JAX_ENABLE_X64"] = "True"

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.numpy.linalg as jnn
import jax.scipy.optimize as jso
import pandas as pd
import numpy as np
import scipy as sp
import pickle

import emcee
import corner
import matplotlib.pyplot as plt

from constants import EPSILON



path = '/content/drive/MyDrive/SchwarMAX-analytic/'
sys.path.append(path)

import jax
import jax.numpy as jnp
from functools import partial

from model_bar import (
    get_jeans_moments, potential_func, density_func,
    get_rotation_curve, estimate_orbital_timescale,
    _integrate_adaptive_batch_vmap,
    solve_nnls_admm_bootstrap,
    compute_model_and_logl_bootstrap,
    makeRotationMatrix,
    getCartesianFromCylindrical_clockwise,
)
from likelihoods_bar import get_dict_data_bootstrap, logl_density
from constants import EPSILON

# ============================================================
# Load data + set up params (same as schwarmax_nautilus.py)
# ============================================================
data_path = path
filename = 'mock_Nbody_bar_XY_withRot_Nbins1000.pkl'
dict_data = get_dict_data_bootstrap(data_path, filename)

# Use a reasonable set of parameters (near the mode)
params_halo_pot = {
    'logM': 11.8, 'Rs': 10**1.24, 'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}

logM_disc, logM_bar = 10.71, 9.94
logRs_disk, logHs_disk, logL_bar = 0.62, 0.02, 0.56
alpha, beta, gamma = 0.58 * 180/np.pi, 0.36 * 180/np.pi, 2.47 * 180/np.pi
L_bar = 10.0**logL_bar
Hs_disc = 10.0**logHs_disk

params_disk_rho = {
    'logM_disc': logM_disc, 'Rs_disc': 10**logRs_disk, 'Hs_disc': Hs_disc,
    'logM_bar': logM_bar, 'L_bar': L_bar, 'a_bar': L_bar/5.0, 'b_bar': Hs_disc,
    'light_to_mass_ratio': 1.0, 'Omega_bar': 10**1.5,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'alpha': alpha, 'beta': beta, 'gamma': gamma,
    'sigma_density_model': 0.0, 'sigma_kine_model': 0.0,
}

num_Vbin = dict_data['total_bins']

# ============================================================
# Warm up JIT (first call compiles everything)
# ============================================================
print("JIT warmup (first call, will be slow)...")
t0 = time.time()
from model_bar import model_bootstrap
result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
jax.block_until_ready(result)
t_warmup = time.time() - t0
print(f"Warmup: {t_warmup:.1f}s", 'logL = ', result[1])

# ============================================================
# Full call timing (post-JIT)
# ============================================================
print("\nFull model_bootstrap() call (post-JIT)...")
t0 = time.time()
result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
jax.block_until_ready(result)
t_full = time.time() - t0
print(f"Full call: {t_full:.2f}s")

# ============================================================
# Profile individual blocks
# ============================================================
print("\n" + "=" * 60)
print("Profiling individual blocks")
print("=" * 60)

timings = {}

# --- Unpack data ---
w0 = dict_data['w0']
n_particles = w0.shape[0]
v0 = dict_data['v0']
s = dict_data['s']
num_per_bin = dict_data['num_per_bin']
bin_mapping = dict_data['bin_mapping']
Omega_bar = params_disk_rho['Omega_bar']
rotation_matrix = makeRotationMatrix(alpha, beta, gamma)

M_bar = 10.0 ** params_disk_rho['logM_bar']
params_baryon = {
    'logM_disc': params_disk_rho['logM_disc'],
    'Rs_disc': params_disk_rho['Rs_disc'],
    'Hs_disc': params_disk_rho['Hs_disc'],
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'logM_bar': params_disk_rho['logM_bar'],
    'L_bar': L_bar, 'a_bar': L_bar/5.0, 'b_bar': Hs_disc,
}

# --- Block 1: Jeans moments + initial velocities ---
get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))

def run_jeans():
    jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2],
                                            params_baryon, params_disk_rho, params_halo_pot, 1.)
    v_rot, sig_R, sig_z, sig_phi = jeans_moments
    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    g1 = jax.random.normal(key1, (n_particles,))
    g2 = jax.random.normal(key2, (n_particles,))
    g3 = jax.random.normal(key3, (n_particles,))
    vR = g1 * sig_R
    vz = g2 * sig_z
    vphi = v_rot + g3 * sig_phi
    x, y, vx, vy = getCartesianFromCylindrical_clockwise(
        jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
    return jnp.array([x, y, w0[:,2], vx, vy, vz]).T

# Warmup call (JIT compile)
w0_new = run_jeans()
jax.block_until_ready(w0_new)
# Timed call
t0 = time.time()
w0_new = run_jeans()
jax.block_until_ready(w0_new)
timings['1_jeans_moments'] = time.time() - t0

# --- Block 2: Orbital timescale + noise batch ---
def run_timescale():
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]
    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z)

    n_real = 4
    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = jnp.clip(0.1 * _Vc, a_min=1, a_max=15)
    noise = [(jax.random.uniform(keys[i], (n_particles, n_real)) - 0.5) *
             (d_scale if i < 3 else v_scale)[:, jnp.newaxis] for i in range(6)]

    batch = w0_new[:, jnp.newaxis, :] + jnp.stack(noise, axis=-1)
    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z)
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_real, axis=1)
    return _R, _z, _Vc, batch, T_orb, T_orb_batch

# Warmup
_R, _z, _Vc, w0_new_batch, T_orb, T_orb_batch = run_timescale()
jax.block_until_ready(T_orb_batch)
# Timed
t0 = time.time()
_R, _z, _Vc, w0_new_batch, T_orb, T_orb_batch = run_timescale()
jax.block_until_ready(T_orb_batch)
timings['2_orbital_timescale'] = time.time() - t0

n_realizations = 4

# --- Block 3: Orbit integration ---
@jax.jit
def acc_fn(x, y, z):
    def _pot(pos):
        return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
    return -jax.grad(_pot)(jnp.array([x, y, z]))

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

def run_integration():
    return _integrate_adaptive_batch_vmap(
        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
        dt_init_batch, -Omega_bar,
        atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim_grid, xy_lim_grid,
        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
        v0, s, rotation_matrix)

# Warmup (already compiled from model_bootstrap, but acc_fn/pot_fn are new)
res = run_integration()
jax.block_until_ready(res[-1])
# Timed
t0 = time.time()
Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = run_integration()
jax.block_until_ready(h4)
timings['3_orbit_integration'] = time.time() - t0

A_Rzphi = Rzphi_bin_counts.T
A_xy = surface_density.T
A_h1, A_h2, A_h3, A_h4 = h1.T, h2.T, h3.T, h4.T

# --- Block 4: Expected density (get_mass vmap) ---
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

def run_density():
    return jax.vmap(get_mass, in_axes=[0,0,0,None,None,None,None,None,None])(
        R_grid, z_grid, phi_grid, density_func_Rz, params_disk_rho, dR, dz, dphi,
        dict_data['sample_for_integration'])

# Warmup
y_Rzphi = run_density()
jax.block_until_ready(y_Rzphi)
# Timed
t0 = time.time()
y_Rzphi = run_density()
jax.block_until_ready(y_Rzphi)
timings['4_expected_density'] = time.time() - t0

# --- Block 5: Preprocessing + bootstrap generation ---
t0 = time.time()
y_xy = dict_data['XY_density_data'].astype(jnp.float32) / params_disk_rho['light_to_mass_ratio']
y_h1, y_h2 = dict_data['h1_data'], dict_data['h2_data']
y_h3, y_h4 = dict_data['h3_data'], dict_data['h4_data']

sig_Rzphi = 0.02 * y_Rzphi + 1e-10
sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / params_disk_rho['light_to_mass_ratio']
sig_A1 = dict_data['h1_data_err'] + EPSILON
sig_A2 = dict_data['h2_data_err'] + EPSILON
sig_A3 = dict_data['h3_data_err'] + EPSILON
sig_A4 = dict_data['h4_data_err'] + EPSILON

mean_mass_per_orb = jnp.sum(y_Rzphi) / A_Rzphi.shape[1]
y_xy = y_xy / mean_mass_per_orb
sig_xy = sig_xy / mean_mass_per_orb
y_Rzphi = y_Rzphi / mean_mass_per_orb
sig_Rzphi = sig_Rzphi / mean_mass_per_orb

y_xy_boot = y_xy[None, :] + dict_data['XY_standard_normal'] * sig_xy[None, :]
y_h1_boot = y_h1[None, :] + dict_data['h1_standard_normal'] * sig_A1[None, :]
y_h2_boot = y_h2[None, :] + dict_data['h2_standard_normal'] * sig_A2[None, :]
y_h3_boot = y_h3[None, :] + dict_data['h3_standard_normal'] * sig_A3[None, :]
y_h4_boot = y_h4[None, :] + dict_data['h4_standard_normal'] * sig_A4[None, :]
jax.block_until_ready(y_h4_boot)
timings['5_preprocessing_bootstrap'] = time.time() - t0

# --- Block 6: NNLS solver (vmapped over bootstraps) ---
def run_nnls():
    return solve_nnls_admm_bootstrap(
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy,
        y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        lambda_reg=1, maxiter=250,
    )

# Warmup
weights_all = run_nnls()
jax.block_until_ready(weights_all)
# Timed
t0 = time.time()
weights_all = run_nnls()
jax.block_until_ready(weights_all)
timings['6_nnls_solver'] = time.time() - t0

# --- Block 7: Compute model + logL ---
def run_logl():
    return compute_model_and_logl_bootstrap(
        weights_all,
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi,
        y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        v0, s,
    )

# Warmup
res7 = run_logl()
jax.block_until_ready(res7[0])
# Timed
t0 = time.time()
logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, m_eff = run_logl()
jax.block_until_ready(logl_marg)
timings['7_model_logl'] = time.time() - t0

# ============================================================
# Summary
# ============================================================
print()
print(f"{'Block':>30} | {'Time (s)':>10} | {'% of total':>10}")
print(f"{'-'*56}")
t_sum = sum(timings.values())
for name, t in timings.items():
    pct = 100 * t / t_sum
    print(f"{name:>30} | {t:>10.3f} | {pct:>9.1f}%")
print(f"{'-'*56}")
print(f"{'Sum of blocks':>30} | {t_sum:>10.3f} |")
print(f"{'Full call (for reference)':>30} | {t_full:>10.3f} |")
print()
print(f"N particles: {n_particles}")
print(f"N Voronoi bins: {num_Vbin}")
print(f"N bootstrap: {y_xy_boot.shape[0]}")
print(f"N orbits (realizations): {n_realizations}")
print(f"N_max steps: {N_max}")
print(f"logL result: {float(logl_marg):.2f}")
