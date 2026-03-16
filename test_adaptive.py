"""
Compare adaptive RK4 integrator vs fixed-step leapfrog (v2).
Tests: JIT compilation, vmap, speed, and E_J conservation.
"""
import agama
agama.setUnits(mass=1, length=1, velocity=1)

from constants import *
from integrants_with_binning import (
    integrate_leapfrog_barred_v2, _integrate_barred_v2_vmap,
    integrate_adaptive_barred, _integrate_adaptive_vmap,
    _rk4_step, _compute_EJ,
)
from sample_from_density import sample_from_density_grid
from densities import *
from potentials import *
from utils import *
from model import *
from CylindricalSpline import get_phi_m, get_acc, evaluate_phi_axisymmetric

import numpy as np
import jax
import jax.numpy as jnp
import pickle
import time

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

# ── Load mock data ──
with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
    bin_dict = pickle.load(f)

num_per_bin = jnp.array(bin_dict['num_per_bin'])
total_bins = jnp.array(bin_dict['total_bins'])
bin_mapping = jnp.array(bin_dict['bin_mapping'])
v0 = jnp.array(bin_dict['v0'])
s = jnp.array(bin_dict['s'])

# ── Parameters (same as test_integration.py) ──
logMhalo, logrho0, logM_bar, logRh_disk = 11.8, 8.8, 10.4, 1.2
logRs_disk, logHs_disk, logRs_bar = 0.45, -0.24, 0.3
alpha_deg, beta_deg, gamma_deg = 30.0, 20.0, 130.0
logLM, logOmega_bar = 0, 1.6

alpha = alpha_deg * np.pi / 180
beta = beta_deg * np.pi / 180
gamma = gamma_deg * np.pi / 180

params_halo_pot = {
    'logM': logMhalo, 'Rs': 10**logRh_disk,
    'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0
}

params_dict = {
    'rho0_disc': 10**logrho0, 'Rd_disc': 10**logRs_disk, 'hz_disc': 10**logHs_disk,
    'light_to_mass_ratio': 10**logLM,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'alpha': alpha_deg, 'beta': beta_deg, 'gamma': gamma_deg,
    'logM_bar': logM_bar, 'Rs_bar': 10**logRs_bar,
    'Omega_bar': 10**logOmega_bar,
    'p_bar': 0.3, 'q_bar': 0.3,
}

def Dehnen_density(x, y, z, params):
    p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
    M = 10.0 ** params['logM_bar']
    r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2) + EPSILON
    val = M / (4 * jnp.pi * p * q * Rs**3) * (r / Rs)**(-2) * (1 + r / Rs)**(-2) * jnp.exp(-(z / 3)**4) * jnp.exp(-(r / 10)**4)
    return val

def density_func(x, y, z, params):
    return Dehnen_density(x, y, z, params) + DoubleExponentialDisk_density(x, y, z, params)

# ── Build disk potential spline ──
NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-3, 30.0, 1e-3, 15.0, 8.
Nphi, N_int = 300, 10_000
print("Building disk potential spline...")
dict_phi = get_phi_m(density_func, params_dict, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)
print("Done.")

# ── Acceleration and potential functions ──
@jax.jit
def acc_fn(x, y, z):
    a_halo = NFW_acceleration(x, y, z, params_halo_pot)
    a_disk = get_acc(x, y, z, dict_phi)
    return a_halo + a_disk

@jax.jit
def potential_func(x, y, z, dict_phi, params_halo):
    phi_halo = NFW_potential(x, y, z, params_halo)
    phi_disk = evaluate_phi_axisymmetric(x, y, z, dict_phi)
    return phi_halo + phi_disk

@jax.jit
def pot_fn_single(x, y, z):
    return potential_func(x, y, z, dict_phi, params_halo_pot)

pot_fn = jax.vmap(pot_fn_single, in_axes=(0, 0, 0))

# ── Sample initial conditions ──
n_particles = 200  # small batch for testing
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

x_grid = np.linspace(0, 15, 1000)
logP_xexp = XexpX_pdf_log(x_grid, 4.0)
R_samples = sample_from_logP(x_grid, logP_xexp, n_particles, keys[0])
phi_samples = jax.random.uniform(keys[1], (n_particles,)) * 2 * np.pi

x_samples = R_samples * jnp.cos(phi_samples)
y_samples = R_samples * jnp.sin(phi_samples)

x_grid_z = np.linspace(0, 4, 1000)
logP_exp = expX_pdf_log(x_grid_z, 1.5)
z_samples = sample_from_logP(x_grid_z, logP_exp, n_particles, keys[2])

# Simple velocity ICs: circular + small dispersion
R_vals = jnp.sqrt(x_samples**2 + y_samples**2)
Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
    R_vals, potential_func, (dict_phi, params_halo_pot), z_samples
)
vx_ic = -Vc * jnp.sin(phi_samples) + jax.random.normal(keys[3], (n_particles,)) * 5.0
vy_ic =  Vc * jnp.cos(phi_samples) + jax.random.normal(keys[4], (n_particles,)) * 5.0
vz_ic = jax.random.normal(keys[5], (n_particles,)) * 3.0

w0_all = jnp.stack([x_samples, y_samples, z_samples, vx_ic, vy_ic, vz_ic], axis=-1)
print(f"Initial conditions: {w0_all.shape}")

# ── Grid / binning parameters ──
rotation_matrix = makeRotationMatrix(alpha_deg, beta_deg, gamma_deg)
num_Vbin = total_bins.item()
Rzphi_lim = jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]])
xy_lim = jnp.array([[-12., 12.], [-4., 4.]])
nRzphi = jnp.array([10, 6, 6])
nXY = jnp.array([60, 40])
Rzphi_n_tot = 360
Omega_bar = params_dict['Omega_bar']

# ── Leapfrog parameters ──
N_step_per_orb = 200
N_dynamical_time = 20
T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
    R_vals, potential_func, (dict_phi, params_halo_pot), z_samples
)
dt_lf = T_orb / N_step_per_orb
n_steps = N_step_per_orb * N_dynamical_time

# ── Adaptive parameters ──
T_total_per_orbit = T_orb * N_dynamical_time  # per-orbit total integration time
dt_init_adaptive = dt_lf  # per-orbit initial dt
N_max = n_steps  # same budget as leapfrog (can be tuned)

print(f"\n{'='*60}")
print(f"Leapfrog: n_steps={n_steps}, dt per orbit varies")
print(f"Adaptive: N_max={N_max}, T_total per orbit (median={jnp.median(T_total_per_orbit):.2f} Gyr), EJ_tol=0.01")
print(f"Omega_bar = {Omega_bar:.2f}")
print(f"{'='*60}\n")

# ══════════════════════════════════════════════════════════════
# Test 1: Leapfrog v2 (warmup + timed)
# ══════════════════════════════════════════════════════════════
print("── Leapfrog v2 ──")
print("  Warmup (JIT compile)...")
t0 = time.time()
lf_result = _integrate_barred_v2_vmap(
    w0_all, acc_fn, pot_fn, n_steps, dt_lf, 0.0, -Omega_bar, False,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim,
    nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
lf_result[0].block_until_ready()
t1 = time.time()
print(f"  Warmup: {t1-t0:.2f}s")

print("  Timed run...")
t0 = time.time()
lf_result = _integrate_barred_v2_vmap(
    w0_all, acc_fn, pot_fn, n_steps, dt_lf, 0.0, -Omega_bar, False,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim,
    nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
lf_result[0].block_until_ready()
t1 = time.time()
lf_time = t1 - t0
print(f"  Time: {lf_time:.3f}s")

lf_Rzphi, lf_sd, lf_h1, lf_h2, lf_h3, lf_h4, lf_valid = lf_result
print(f"  Valid orbits: {lf_valid.sum():.0f} / {n_particles}")
print(f"  Rzphi counts range: [{lf_Rzphi.min():.1f}, {lf_Rzphi.max():.1f}]")

# ══════════════════════════════════════════════════════════════
# Test 2: Adaptive RK4 (warmup + timed)
# ══════════════════════════════════════════════════════════════
print("\n── Adaptive RK4 ──")
print("  Warmup (JIT compile)...")
t0 = time.time()
ad_result = _integrate_adaptive_vmap(
    w0_all, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-8, 1e-6, 1e-5, 0.1,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim,
    nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result[0].block_until_ready()
t1 = time.time()
print(f"  Warmup: {t1-t0:.2f}s")

print("  Timed run...")
t0 = time.time()
ad_result = _integrate_adaptive_vmap(
    w0_all, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-8, 1e-6, 1e-5, 0.1,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim,
    nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result[0].block_until_ready()
t1 = time.time()
ad_time = t1 - t0
print(f"  Time: {ad_time:.3f}s")

ad_Rzphi, ad_sd, ad_h1, ad_h2, ad_h3, ad_h4, ad_valid, ad_n_accepted = ad_result
print(f"  Valid orbits: {ad_valid.sum():.0f} / {n_particles}")
print(f"  Accepted steps: median={jnp.median(ad_n_accepted):.0f}, "
      f"min={ad_n_accepted.min():.0f}, max={ad_n_accepted.max():.0f}")
print(f"  Rzphi counts range: [{ad_Rzphi.min():.1f}, {ad_Rzphi.max():.1f}]")

# ── Test with looser tolerance ──
print("\n── Adaptive BS23 (rtol=1e-4) ──")
print("  Warmup...")
t0 = time.time()
ad_result2 = _integrate_adaptive_vmap(
    w0_all, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-6, 1e-4, 1e-5, 0.1,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim,
    nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result2[0].block_until_ready()
t1 = time.time()
print(f"  Warmup: {t1-t0:.2f}s")
print("  Timed run...")
t0 = time.time()
ad_result2 = _integrate_adaptive_vmap(
    w0_all, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-6, 1e-4, 1e-5, 0.1,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim,
    nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result2[0].block_until_ready()
t1 = time.time()
ad_time2 = t1 - t0
ad2_valid = ad_result2[6]
ad2_n_accepted = ad_result2[7]
print(f"  Time: {ad_time2:.3f}s")
print(f"  Valid orbits: {ad2_valid.sum():.0f} / {n_particles}")
print(f"  Accepted steps: median={jnp.median(ad2_n_accepted):.0f}, "
      f"min={ad2_n_accepted.min():.0f}, max={ad2_n_accepted.max():.0f}")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"  Leapfrog v2:  {lf_time:.3f}s  |  valid: {lf_valid.sum():.0f}/{n_particles}")
print(f"  Adaptive RK4: {ad_time:.3f}s  |  valid: {ad_valid.sum():.0f}/{n_particles}")
print(f"  Speedup: {lf_time/ad_time:.2f}x" if ad_time > 0 else "  Speedup: N/A")
print(f"{'='*60}")
