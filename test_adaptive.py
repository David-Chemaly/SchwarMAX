"""
Realistic comparison: adaptive BS23 vs fixed-step leapfrog v2.
Uses the full 5k particle setup from test_integration.py (no realisations).
"""
import agama
agama.setUnits(mass=1, length=1, velocity=1)

from constants import *
from integrants_with_binning import (
    _integrate_barred_v2_vmap, _integrate_barred_vmap,
    _integrate_adaptive_vmap,
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
import jax.scipy as jsp
import pickle
import time

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

# ══════════════════════════════════════════════════════════════
# Setup (copied from test_integration.py)
# ══════════════════════════════════════════════════════════════

def get_dict_data(path):
    with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
        bin_dict = pickle.load(f)
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
    V_data_err = jnp.array(bin_dict['V_mean_err'])
    sigma_data_err = jnp.array(bin_dict['V_sigma_err'])
    h1_data_err = jnp.array(bin_dict['h1_err'])
    h2_data_err = jnp.array(bin_dict['h2_err'])
    h3_data_err = jnp.array(bin_dict['h3_err'])
    h4_data_err = jnp.array(bin_dict['h4_err'])
    with open(path + 'mock_axisymmetric_disc_Rzphi.pkl', 'rb') as f:
        Rzphi_density_data = pickle.load(f)
    R_grid, z_grid, phi_grid = Rzphi_density_data['R_grid'], Rzphi_density_data['z_grid'], Rzphi_density_data['phi_grid']
    dR = np.unique(R_grid)[1] - np.unique(R_grid)[0]
    dz = np.unique(z_grid)[1] - np.unique(z_grid)[0]
    dphi = np.unique(phi_grid)[1] - np.unique(phi_grid)[0]
    sample_for_integration = Rzphi_density_data['sample_for_integration']
    from scipy.stats import qmc
    X_regular_grid, Y_regular_grid = bin_dict['X_regular_grid'], bin_dict['Y_regular_grid']
    dX = jnp.unique(X_regular_grid)[1] - jnp.unique(X_regular_grid)[0]
    dY = jnp.unique(Y_regular_grid)[1] - jnp.unique(Y_regular_grid)[0]
    sampler = qmc.Sobol(d=3, scramble=False)
    sample = sampler.random_base2(m=10)
    return {
        'v0': v0, 's': s,
        'XY_density_data': surface_density,
        'V_data': V_data, 'V_data_err': V_data_err,
        'sigma_data': sigma_data, 'sigma_data_err': sigma_data_err,
        'h1_data': h1_data, 'h1_data_err': h1_data_err,
        'h2_data': h2_data, 'h2_data_err': h2_data_err,
        'h3_data': h3_data, 'h3_data_err': h3_data_err,
        'h4_data': h4_data, 'h4_data_err': h4_data_err,
        'num_per_bin': num_per_bin, 'bin_mapping': bin_mapping,
        'total_bins': total_bins.item(),
        'R_grid': R_grid, 'z_grid': z_grid, 'phi_grid': phi_grid,
        'dR': dR, 'dz': dz, 'dphi': dphi,
        'sample_for_integration': sample_for_integration,
        'X_regular_grid': X_regular_grid, 'Y_regular_grid': Y_regular_grid,
        'dX': dX, 'dY': dY, 'sample_for_integration_XY': sample,
    }

dict_data = get_dict_data(path)

# Parameters
logMhalo, logrho0, logM_bar, logRh_disk = 11.8, 8.8, 10.4, 1.2
logRs_disk, logHs_disk, logRs_bar = 0.45, -0.24, 0.3
alpha_best_fit, beta_best_fit, gamma_best_fit = 30*np.pi/180, 20*np.pi/180, 130*np.pi/180
logLM, logOmega_bar = 0, 1.6

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
    'alpha': alpha_best_fit * 180/np.pi, 'beta': beta_best_fit * 180/np.pi,
    'gamma': gamma_best_fit * 180/np.pi,
    'logM_bar': logM_bar, 'Rs_bar': 10**logRs_bar,
    'Omega_bar': 10**logOmega_bar,
    'p_bar': 0.3, 'q_bar': 0.3,
}

@jax.jit
def Dehnen_density(x, y, z, params):
    p, q, Rs = params['p_bar'], params['q_bar'], params['Rs_bar']
    M = 10.0 ** params['logM_bar']
    r = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2) + EPSILON
    return M / (4 * jnp.pi * p * q * Rs**3) * (r / Rs)**(-2) * (1 + r / Rs)**(-2) * jnp.exp(-(z / 3)**4) * jnp.exp(-(r / 10)**4)

def density_func(x, y, z, params):
    return Dehnen_density(x, y, z, params) + DoubleExponentialDisk_density(x, y, z, params)

# Build disk potential spline
print("Building disk potential spline...")
NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-3, 30.0, 1e-3, 15.0, 8.
Nphi, N_int = 300, 10_000
dict_phi = get_phi_m(density_func, params_dict, NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax, Nphi, N_int)
print("Done.")

# Potential and acceleration
@jax.jit
def potential_func(x, y, z, dict_phi, params_halo):
    return NFW_potential(x, y, z, params_halo) + evaluate_phi_axisymmetric(x, y, z, dict_phi)

@jax.jit
def acc_fn(x, y, z):
    return NFW_acceleration(x, y, z, params_halo_pot) + get_acc(x, y, z, dict_phi)

@jax.jit
def pot_fn_single(x, y, z):
    return potential_func(x, y, z, dict_phi, params_halo_pot)

pot_fn = jax.vmap(pot_fn_single, in_axes=(0, 0, 0))

# ══════════════════════════════════════════════════════════════
# Sample 5k initial conditions (same as test_integration.py)
# ══════════════════════════════════════════════════════════════
n_samples = 5_000
x_grid = np.linspace(0, 15, 1000)
logP_xexp = XexpX_pdf_log(x_grid, 4.0)
key = jax.random.PRNGKey(10086)
R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)
x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)

x_grid_z = np.linspace(0, 4, 1000)
logP_exp = expX_pdf_log(x_grid_z, 1.5)
key = jax.random.PRNGKey(10010)
z_samples = sample_from_logP(x_grid_z, logP_exp, n_samples, key)
w0 = jnp.array(np.array([x_samples, y_samples, z_samples]).T)
n_particles = w0.shape[0]

# Jeans moments for initial velocities
@jax.jit
def dPhi_dz(x, y, z, dict_phi, params_halo):
    d = 5e-3
    return (potential_func(x, y, z+d, dict_phi, params_halo) - potential_func(x, y, z-d, dict_phi, params_halo)) / (2*d)

@jax.jit
def dPhi_dR(x, y, z, dict_phi, params_halo):
    d = 5e-3
    R = jnp.sqrt(x**2 + y**2)
    return (potential_func(R+d, 0, z, dict_phi, params_halo) - potential_func(R-d, 0, z, dict_phi, params_halo)) / (2*d)

@jax.jit
def get_jeans_moments(x_star, y_star, z_star, dict_phi, params_disc, params_halo, anisotropy_b=1.0):
    R_star = jnp.sqrt(x_star**2 + y_star**2)
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, dict_phi, params_halo)
    pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
    integrand_val = jax.vmap(integrand)(pts)
    integral_val = jsp.integrate.trapezoid(integrand_val, pts)
    nu_val = density_func(x_star, y_star, z_star, params_disc)
    sigma_z2 = jnp.maximum((1.0 / nu_val) * integral_val, 0.0)
    sigma_z = jnp.sqrt(sigma_z2)
    sigma_R2 = anisotropy_b * sigma_z2
    sigma_R = jnp.sqrt(sigma_R2)
    def vertical_pressure(r_in):
        def integrand_r(z_prime):
            return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, dict_phi, params_halo)
        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        integrand_val = jax.vmap(integrand_r)(pts)
        return jsp.integrate.trapezoid(integrand_val, pts)
    dR = 5e-3
    d_nu_sigR2_dR = anisotropy_b * (vertical_pressure(R_star + dR) - vertical_pressure(R_star - dR)) / (2*dR)
    v_phi_total_sq = sigma_R2 + (R_star / nu_val) * d_nu_sigR2_dR + R_star * dPhi_dR(x_star, y_star, z_star, dict_phi, params_halo)
    sigma_phi = sigma_R
    v_streaming_sq = jnp.maximum(v_phi_total_sq - sigma_phi**2, 0.0)
    v_mean_phi = jnp.sqrt(v_streaming_sq)
    return jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))

print("Computing Jeans moments for 5k particles...")
get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
jeans_moments = get_jeans_moments_vmap(w0[:,0], w0[:,1], w0[:,2], dict_phi, params_dict, params_halo_pot, 1.)
v_rot, sig_R, sig_z, sig_phi = jeans_moments
key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
g1 = jax.random.normal(key1, (n_particles,))
g2 = jax.random.normal(key2, (n_particles,))
g3 = jax.random.normal(key3, (n_particles,))
vR = g1 * sig_R
vz = g2 * sig_z
vphi = v_rot + g3 * sig_phi
x_ic, y_ic, vx_ic, vy_ic = getCartesianFromCylindrical_clockwise(
    jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
w0_new = jnp.array([x_ic, y_ic, w0[:,2], vx_ic, vy_ic, vz]).T
print(f"Initial conditions ready: {w0_new.shape}")

# Orbital timescales
_R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
_z = w0_new[:,2]
T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
    _R, potential_func, (dict_phi, params_halo_pot), _z)

# ══════════════════════════════════════════════════════════════
# Grid / binning parameters
# ══════════════════════════════════════════════════════════════
alpha_deg = params_dict['alpha']
beta_deg = params_dict['beta']
gamma_deg = params_dict['gamma']
rotation_matrix = makeRotationMatrix(alpha_deg, beta_deg, gamma_deg)
num_Vbin = dict_data['total_bins']
bin_mapping = dict_data['bin_mapping']
num_per_bin = dict_data['num_per_bin']
v0 = dict_data['v0']
s = dict_data['s']

Rzphi_lim = jnp.array([[0, 10.], [-3, 3], [-jnp.pi, jnp.pi]])
xy_lim = jnp.array([[-12., 12.], [-4., 4.]])
nRzphi = jnp.array([10, 6, 6])
nXY = jnp.array([60, 40])
Rzphi_n_tot = 360
Omega_bar = params_dict['Omega_bar']

N_step_per_orb = 100
N_dynamical_time = 50
dt_lf = T_orb / N_step_per_orb
n_steps = N_step_per_orb * N_dynamical_time

# Adaptive parameters
T_total_per_orbit = T_orb * N_dynamical_time
dt_init_adaptive = dt_lf
N_max = 6000 #n_steps  # same budget

print(f"\n{'='*70}")
print(f"5000 particles, no realisations")
print(f"Leapfrog: n_steps={n_steps}, N_step_per_orb={N_step_per_orb}")
print(f"Adaptive BS23: N_max={N_max}, rtol=1e-6")
print(f"Omega_bar = {Omega_bar:.2f}")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════
# Test 1: Leapfrog v2
# ══════════════════════════════════════════════════════════════
print("── Leapfrog v2 (5k orbits) ──")
print("  Warmup (JIT compile)...")
t0 = time.time()
lf_result = _integrate_barred_v2_vmap(
    w0_new, acc_fn, pot_fn, n_steps, dt_lf, 0.0, -Omega_bar, False,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
lf_result[0].block_until_ready()
t1 = time.time()
print(f"  Warmup: {t1-t0:.2f}s")

print("  Timed run...")
t0 = time.time()
lf_result = _integrate_barred_v2_vmap(
    w0_new, acc_fn, pot_fn, n_steps, dt_lf, 0.0, -Omega_bar, False,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
lf_result[0].block_until_ready()
t1 = time.time()
lf_time = t1 - t0
lf_valid = lf_result[6]
print(f"  Time (integrate + bin): {lf_time:.3f}s")
print(f"  Valid orbits: {lf_valid.sum():.0f} / {n_particles}")

# ══════════════════════════════════════════════════════════════
# Test 2: Adaptive BS23 (rtol=1e-6)
# ══════════════════════════════════════════════════════════════
print("\n── Adaptive BS23 rtol=1e-6 (5k orbits) ──")
print("  Warmup (JIT compile)...")
t0 = time.time()
ad_result = _integrate_adaptive_vmap(
    w0_new, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-8, 1e-6, 1e-5, 0.5,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result[0].block_until_ready()
t1 = time.time()
print(f"  Warmup: {t1-t0:.2f}s")

print("  Timed run...")
t0 = time.time()
ad_result = _integrate_adaptive_vmap(
    w0_new, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-8, 1e-6, 1e-5, 0.5,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result[0].block_until_ready()
t1 = time.time()
ad_time = t1 - t0
ad_valid = ad_result[6]
ad_n_accepted = ad_result[7]
print(f"  Time (integrate + dt-weighted bin): {ad_time:.3f}s")
print(f"  Valid orbits: {ad_valid.sum():.0f} / {n_particles}")
print(f"  Accepted steps: median={jnp.median(ad_n_accepted):.0f}, "
      f"min={ad_n_accepted.min():.0f}, max={ad_n_accepted.max():.0f}")

# ══════════════════════════════════════════════════════════════
# Test 3: Adaptive BS23 (rtol=1e-4, looser)
# ══════════════════════════════════════════════════════════════
print("\n── Adaptive BS23 rtol=1e-4 (5k orbits) ──")
print("  Warmup...")
t0 = time.time()
ad_result2 = _integrate_adaptive_vmap(
    w0_new, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-6, 1e-4, 1e-5, 0.3,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result2[0].block_until_ready()
t1 = time.time()
print(f"  Warmup: {t1-t0:.2f}s")

print("  Timed run...")
t0 = time.time()
ad_result2 = _integrate_adaptive_vmap(
    w0_new, acc_fn, pot_fn, N_max, T_total_per_orbit,
    dt_init_adaptive, -Omega_bar, 1e-6, 1e-4, 1e-5, 0.3,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix)
ad_result2[0].block_until_ready()
t1 = time.time()
ad_time2 = t1 - t0
ad2_valid = ad_result2[6]
ad2_n_accepted = ad_result2[7]
print(f"  Time (integrate + dt-weighted bin): {ad_time2:.3f}s")
print(f"  Valid orbits: {ad2_valid.sum():.0f} / {n_particles}")
print(f"  Accepted steps: median={jnp.median(ad2_n_accepted):.0f}, "
      f"min={ad2_n_accepted.min():.0f}, max={ad2_n_accepted.max():.0f}")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — 5000 orbits, N_max={N_max}")
print(f"{'='*70}")
print(f"  Method                  Time     Valid    Evals/step  Notes")
print(f"  ─────────────────────   ─────    ─────    ──────────  ─────")
print(f"  Leapfrog v2             {lf_time:.3f}s   {lf_valid.sum():.0f}/{n_particles}   2 acc_fn    fixed dt")
print(f"  BS23 rtol=1e-6          {ad_time:.3f}s   {ad_valid.sum():.0f}/{n_particles}   3 acc_fn    dt-weighted bin")
print(f"  BS23 rtol=1e-4          {ad_time2:.3f}s   {ad2_valid.sum():.0f}/{n_particles}   3 acc_fn    dt-weighted bin")
print(f"  LF/BS23(1e-6) ratio: {lf_time/ad_time:.2f}x")
print(f"{'='*70}")
print(f"\nNote: dt-weighting replaces interpolation — each orbit point is")
print(f"weighted by its dt in segment sums, equivalent to regular time-sampling.")
print(f"No extra interpolation cost needed.")
