"""
Benchmark: monolithic JIT vs split-JIT blocks.

The monolithic @jax.jit on model_bootstrap() compiles the entire function
into one XLA program (4.3s per call). Individual blocks profiled separately
sum to only 1.6s.

Strategy: split into 3 JIT blocks, each compiled independently:
  Block A: Jeans moments + orbital timescale
  Block B: Orbit integration (defines acc_fn/pot_fn inside its JIT boundary)
  Block C: Expected density
  Blocks D,E (NNLS + logL) are already @jax.jit at module level.
  Preprocessing (block 5) is trivial array math, no JIT needed.

The wrapper calls A → B → C → preprocess → D → E from plain Python.

Usage:
    python benchmark_nojit.py
"""
import sys
import time
import numpy as np

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp
from functools import partial

from model_bar import (
    model_bootstrap,
    get_jeans_moments, potential_func, density_func,
    get_rotation_curve, estimate_orbital_timescale,
    _integrate_adaptive_batch_vmap,
    solve_nnls_admm_bootstrap,
    compute_model_and_logl_bootstrap,
    makeRotationMatrix,
    getCartesianFromCylindrical_clockwise,
)
from likelihoods_bar import get_dict_data_bootstrap
from constants import EPSILON

# ============================================================
# Load data + params
# ============================================================
data_path = path
filename = 'mock_Nbody_bar_XY_withRot_Nbins1000.pkl'
dict_data = get_dict_data_bootstrap(data_path, filename)

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
# Define separate JIT blocks
# ============================================================

# --- Block A: Jeans moments + orbital timescale ---
# params are traced abstractly → no recompilation when values change
@jax.jit
def block_A_init(w0, params_baryon, params_disk_rho, params_halo_pot):
    n_particles = w0.shape[0]
    n_realizations = 4

    # Jeans moments → initial velocities
    get_jeans_moments_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
    jeans_moments = get_jeans_moments_vmap(
        w0[:,0], w0[:,1], w0[:,2], params_baryon, params_disk_rho, params_halo_pot, 1.)
    v_rot, sig_R, sig_z, sig_phi = jeans_moments

    key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
    g1 = jax.random.normal(key1, (n_particles,))
    g2 = jax.random.normal(key2, (n_particles,))
    g3 = jax.random.normal(key3, (n_particles,))
    vR = g1 * sig_R
    vz = g2 * sig_z
    vphi = v_rot + g3 * sig_phi
    R = jnp.sqrt(w0[:,0]**2 + w0[:,1]**2)
    phi = jnp.arctan2(w0[:,1], w0[:,0])
    x, y, vx, vy = getCartesianFromCylindrical_clockwise(R, phi, vR, vphi)
    w0_new = jnp.array([x, y, w0[:,2], vx, vy, vz]).T

    # Orbital timescale
    _R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
    _z = w0_new[:,2]
    _Vc = jax.vmap(get_rotation_curve, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z)

    key = jax.random.PRNGKey(911)
    keys = jax.random.split(key, 6)
    d_scale = 0.1 * jnp.ones(_R.shape)
    v_scale = jnp.clip(0.1 * _Vc, a_min=1, a_max=15)
    noise_x  = (jax.random.uniform(keys[0], (n_particles, n_realizations)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_y  = (jax.random.uniform(keys[1], (n_particles, n_realizations)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_z  = (jax.random.uniform(keys[2], (n_particles, n_realizations)) - 0.5) * d_scale[:, jnp.newaxis]
    noise_vx = (jax.random.uniform(keys[3], (n_particles, n_realizations)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vy = (jax.random.uniform(keys[4], (n_particles, n_realizations)) - 0.5) * v_scale[:, jnp.newaxis]
    noise_vz = (jax.random.uniform(keys[5], (n_particles, n_realizations)) - 0.5) * v_scale[:, jnp.newaxis]

    w0_new_batch = w0_new[:, jnp.newaxis, :] + jnp.stack(
        [noise_x, noise_y, noise_z, noise_vx, noise_vy, noise_vz], axis=-1)

    T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
        _R, potential_func, (params_baryon, params_halo_pot), _z)
    T_orb_batch = T_orb[:, jnp.newaxis].repeat(n_realizations, axis=1)

    return w0_new_batch, T_orb_batch


# --- Block B: Orbit integration ---
# acc_fn and pot_fn defined INSIDE the JIT boundary → no closure recompilation
@partial(jax.jit, static_argnames=('N_max', 'num_Vbin', 'Rzphi_n_tot'))
def block_B_integrate(w0_new_batch, params_baryon, params_halo_pot,
                      T_orb_batch, Omega_bar,
                      N_max, num_Vbin, bin_mapping, num_per_bin,
                      Rzphi_lim_grid, xy_lim_grid,
                      Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
                      v0, s, rotation_matrix):

    N_dynamical_time = 50
    T_total_batch = T_orb_batch * N_dynamical_time
    dt_init_batch = T_orb_batch / 100  # N_step_per_orb = 100

    def acc_fn(x, y, z):
        def _pot(pos):
            return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
        return -jax.grad(_pot)(jnp.array([x, y, z]))

    def pot_fn(x, y, z):
        return potential_func(x, y, z, params_baryon, params_halo_pot)

    return _integrate_adaptive_batch_vmap(
        w0_new_batch, acc_fn, pot_fn, N_max, T_total_batch,
        dt_init_batch, -Omega_bar,
        1e-7, 1e-4,   # atol, rtol
        1e-5, 0.3,    # dt_min, dt_max
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim_grid, xy_lim_grid,
        Rzphi_n_grid, xy_n_grid, Rzphi_n_tot,
        v0, s, rotation_matrix)


# --- Block C: Expected density ---
@jax.jit
def block_C_density(R_grid, z_grid, phi_grid, params_disk_rho, dR, dz, dphi, sample):
    def density_func_Rz(R, z, phi, params):
        x = R * jnp.cos(phi)
        y = R * jnp.sin(phi)
        return density_func(x, y, z, params)

    def get_mass(R_grid_i, z_grid_i, phi_grid_i):
        R_samples = R_grid_i + (sample[:,0] - 0.5) * dR
        z_samples = z_grid_i + (sample[:,1] - 0.5) * dz
        phi_samples = phi_grid_i + (sample[:,2] - 0.5) * dphi
        density_samples = density_func_Rz(R_samples, z_samples, phi_samples, params_disk_rho)
        return jnp.sum(density_samples * R_samples) / sample.shape[0]

    return jax.vmap(get_mass)(R_grid, z_grid, phi_grid)


# ============================================================
# Split-JIT wrapper (no @jax.jit on this function)
# ============================================================
def model_bootstrap_split(params_halo_pot, params_disk_rho, dict_data, num_Vbin):
    """Same as model_bootstrap but with separate JIT blocks instead of one monolithic JIT."""

    w0 = dict_data['w0']
    v0 = dict_data['v0']
    s = dict_data['s']
    num_per_bin = dict_data['num_per_bin']
    bin_mapping = dict_data['bin_mapping']
    Omega_bar = params_disk_rho['Omega_bar']
    alpha_l, beta_l, gamma_l = params_disk_rho['alpha'], params_disk_rho['beta'], params_disk_rho['gamma']
    rotation_matrix = makeRotationMatrix(alpha_l, beta_l, gamma_l)
    sigma_density_model = params_disk_rho['sigma_density_model']
    sigma_kine_model = params_disk_rho['sigma_kine_model']

    L_bar_l = params_disk_rho['L_bar']
    a_bar_l = params_disk_rho['a_bar']
    b_bar_l = params_disk_rho['b_bar']

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
        'L_bar': L_bar_l,
        'a_bar': a_bar_l,
        'b_bar': b_bar_l,
    }

    # Block A: Jeans + timescale
    w0_new_batch, T_orb_batch = block_A_init(w0, params_baryon, params_disk_rho, params_halo_pot)

    # Block B: Orbit integration
    Rzphi_lim_grid = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
    xy_lim_grid = jnp.array([[-10.,10.],[-3.,3.]])
    Rzphi_n_grid = jnp.array([10,6,6])
    xy_n_grid = jnp.array([60,40])

    Rzphi_bin_counts, surface_density, h1, h2, h3, h4, _ = block_B_integrate(
        w0_new_batch, params_baryon, params_halo_pot,
        T_orb_batch, Omega_bar,
        5000, num_Vbin, bin_mapping, num_per_bin,  # N_max = 5000
        Rzphi_lim_grid, xy_lim_grid,
        Rzphi_n_grid, xy_n_grid, 360,
        v0, s, rotation_matrix)

    A_Rzphi = Rzphi_bin_counts.T
    A_xy = surface_density.T
    A_h1, A_h2, A_h3, A_h4 = h1.T, h2.T, h3.T, h4.T

    # Block C: Expected density
    y_Rzphi = block_C_density(
        dict_data['R_grid'], dict_data['z_grid'], dict_data['phi_grid'],
        params_disk_rho,
        dict_data['dR'], dict_data['dz'], dict_data['dphi'],
        dict_data['sample_for_integration'])

    # Preprocessing (plain array math — fast, no JIT needed)
    y_xy = dict_data['XY_density_data'].astype(jnp.float32) / params_disk_rho['light_to_mass_ratio']
    y_h1 = dict_data['h1_data']
    y_h2 = dict_data['h2_data']
    y_h3 = dict_data['h3_data']
    y_h4 = dict_data['h4_data']

    sig_Rzphi = 0.02 * y_Rzphi + 1e-10
    sig_xy = (dict_data['XY_density_data_err'] + EPSILON) / params_disk_rho['light_to_mass_ratio']
    sig_A1 = dict_data['h1_data_err'] + EPSILON
    sig_A2 = dict_data['h2_data_err'] + EPSILON
    sig_A3 = dict_data['h3_data_err'] + EPSILON
    sig_A4 = dict_data['h4_data_err'] + EPSILON

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

    # Bootstrap observations
    y_xy_boot = y_xy[None, :] + dict_data['XY_standard_normal'] * sig_xy[None, :]
    y_h1_boot = y_h1[None, :] + dict_data['h1_standard_normal'] * sig_A1[None, :]
    y_h2_boot = y_h2[None, :] + dict_data['h2_standard_normal'] * sig_A2[None, :]
    y_h3_boot = y_h3[None, :] + dict_data['h3_standard_normal'] * sig_A3[None, :]
    y_h4_boot = y_h4[None, :] + dict_data['h4_standard_normal'] * sig_A4[None, :]

    # Block D: NNLS solver (already @jax.jit at module level)
    weights_all = solve_nnls_admm_bootstrap(
        A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
        y_Rzphi, y_xy,
        y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
        sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
        lambda_reg=1, maxiter=250)

    # Block E: Model + logL (already @jax.jit at module level)
    logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, m_eff = \
        compute_model_and_logl_bootstrap(
            weights_all,
            A_Rzphi, A_xy, A_h1, A_h2, A_h3, A_h4,
            y_Rzphi,
            y_xy_boot, y_h1_boot, y_h2_boot, y_h3_boot, y_h4_boot,
            sig_Rzphi, sig_xy, sig_A1, sig_A2, sig_A3, sig_A4,
            v0, s)

    return weights_all, logl_marg, density_all, h1_all, h2_all, h3_all, h4_all, V_all, sigma_all, logl_all, m_eff


# ============================================================
# Benchmark
# ============================================================
print("=" * 60)
print("Warming up monolithic JIT version...")
print("=" * 60)
t0 = time.time()
result_jit = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
jax.block_until_ready(result_jit)
print(f"Monolithic JIT warmup: {time.time() - t0:.1f}s")

print("\nWarming up split-JIT version (each block compiles separately)...")
t0 = time.time()
result_split = model_bootstrap_split(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
jax.block_until_ready(result_split)
print(f"Split-JIT warmup: {time.time() - t0:.1f}s")

# Verify same result
logl_jit = float(result_jit[1])
logl_split = float(result_split[1])
print(f"\nlogL check: monolithic={logl_jit:.4f}, split={logl_split:.4f}, diff={abs(logl_jit - logl_split):.6f}")

# ============================================================
# Timing: 5 calls each
# ============================================================
print("\n" + "=" * 60)
print("Timing: monolithic @jax.jit model_bootstrap()")
print("=" * 60)
times_jit = []
for i in range(5):
    t0 = time.time()
    result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    jax.block_until_ready(result)
    dt = time.time() - t0
    times_jit.append(dt)
    print(f"  Call {i+1}: {dt:.3f}s")
print(f"  Mean: {np.mean(times_jit):.3f}s +/- {np.std(times_jit):.3f}s")

print("\n" + "=" * 60)
print("Timing: split-JIT model_bootstrap_split()")
print("=" * 60)
times_split = []
for i in range(5):
    t0 = time.time()
    result = model_bootstrap_split(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    jax.block_until_ready(result)
    dt = time.time() - t0
    times_split.append(dt)
    print(f"  Call {i+1}: {dt:.3f}s")
print(f"  Mean: {np.mean(times_split):.3f}s +/- {np.std(times_split):.3f}s")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
mean_jit = np.mean(times_jit)
mean_split = np.mean(times_split)
print(f"  Monolithic JIT:  {mean_jit:.3f}s")
print(f"  Split-JIT:       {mean_split:.3f}s")
print(f"  Speedup:         {mean_jit / mean_split:.2f}x")
print(f"  Time saved:      {mean_jit - mean_split:.3f}s per call")
if mean_split < mean_jit:
    n_calls = 26000
    hours_jit = n_calls * mean_jit / 3600
    hours_split = n_calls * mean_split / 3600
    print(f"\n  For emcee ({n_calls:,} calls):")
    print(f"    Monolithic JIT: {hours_jit:.1f} hours")
    print(f"    Split-JIT:      {hours_split:.1f} hours")
    print(f"    Saved:          {hours_jit - hours_split:.1f} hours")
