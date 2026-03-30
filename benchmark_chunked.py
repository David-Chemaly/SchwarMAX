"""
Benchmark: chunked binning vs original post-scan binning in adaptive integrator.

Compares:
  1. Original: store full (N_max, 6) trajectory, bin at end
  2. Chunked: bin every chunk_size steps vectorially, accumulate

Tests accuracy (per-orbit and per-bin) and CPU/GPU timing.

Usage:
    python benchmark_chunked.py
"""
import sys
import time
import numpy as np

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from constants import EPSILON, KPCGYR_TO_KMS, G
from integrants_with_binning import (
    _integrate_adaptive_vmap,
    _integrate_adaptive_chunked_vmap,
)
from densities import MiyamotoNagai_density
from potentials import NFW_potential, MiyamotoNagai_potential
from dehnen_bar import T3_density, T3_potential, V4_density, V4_potential
from utils import (
    makeRotationMatrix, getCartesianFromCylindrical_clockwise,
    estimate_orbital_timescale, get_rotation_curve,
    sample_from_logP, XexpX_pdf_log, expX_pdf_log,
)
import pickle

# ══════════════════════════════════════════════════════════════
# Model setup (same as test_integration_convergence.py)
# ══════════════════════════════════════════════════════════════
V4_A, V4_B, V4_L, V4_GAMMA = 0.5, 0.5, 0.1, 0.0
GAMMA_BAR = 1.0

logM_halo, logM_disc, logM_bar, logRs_halo = 11.8, 10.7, 10.1, 1.2
logRs_disc, logHs_disc, logL_bar = 0.8, -0.24, 0.5
alpha_best_fit, beta_best_fit, gamma_best_fit = 30, 20, 140

L_bar = 10.0 ** logL_bar
a_bar = L_bar / 5.0
Hs_disc = 10.0 ** logHs_disc
b_bar = Hs_disc

params_baryon = {
    'logM_disc': logM_disc, 'Rs_disc': 10.0**logRs_disc, 'Hs_disc': Hs_disc,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'logM_bar': logM_bar, 'L_bar': L_bar, 'a_bar': a_bar, 'b_bar': b_bar,
}
params_halo_pot = {
    'logM': logM_halo, 'Rs': 10**logRs_halo,
    'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}
params_disk_rho = {**params_baryon, 'alpha': alpha_best_fit, 'beta': beta_best_fit,
                   'gamma': gamma_best_fit, 'light_to_mass_ratio': 1.0,
                   'Omega_bar': 10**1.6}

@jax.jit
def density_func(x, y, z, params):
    mn_params = {k: params[k] for k in ['logM_disc','Rs_disc','Hs_disc',
                 'x_origin','y_origin','z_origin','dirx','diry','dirz']}
    M_bar = 10.0 ** params['logM_bar']
    return (MiyamotoNagai_density(x, y, z, mn_params)
            + T3_density(x, y, z, M_bar, params['a_bar'], params['b_bar'], params['L_bar'], GAMMA_BAR)
            + V4_density(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA))

@jax.jit
def potential_func(x, y, z, params_baryon, params_halo):
    mn_params = {k: params_baryon[k] for k in ['logM_disc','Rs_disc','Hs_disc',
                 'x_origin','y_origin','z_origin','dirx','diry','dirz']}
    M_bar = 10.0 ** params_baryon['logM_bar']
    return (NFW_potential(x, y, z, params_halo)
            + MiyamotoNagai_potential(x, y, z, mn_params)
            + T3_potential(x, y, z, M_bar, params_baryon['a_bar'], params_baryon['b_bar'], params_baryon['L_bar'], GAMMA_BAR)
            + V4_potential(x, y, z, M_bar, V4_A, V4_B, V4_L, V4_GAMMA))

@jax.jit
def dPhi_dz(x, y, z, pb, ph):
    d = 5e-3
    return (potential_func(x,y,z+d,pb,ph) - potential_func(x,y,z-d,pb,ph)) / (2*d)

@jax.jit
def dPhi_dR(x, y, z, pb, ph):
    d = 5e-3
    R = jnp.sqrt(x**2 + y**2)
    return (potential_func(R+d,0,z,pb,ph) - potential_func(R-d,0,z,pb,ph)) / (2*d)

@jax.jit
def get_jeans_moments(x_star, y_star, z_star, params_baryon, params_disc, params_halo, anisotropy_b=1.0):
    R_star = jnp.sqrt(x_star**2 + y_star**2)
    def integrand(z_prime):
        return density_func(x_star, y_star, z_prime, params_disc) * dPhi_dz(x_star, y_star, z_prime, params_baryon, params_halo)
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
            return density_func(r_in, 0, z_prime, params_disc) * dPhi_dz(r_in, 0, z_prime, params_baryon, params_halo)
        pts = jnp.linspace(jnp.abs(z_star), 10.0, 1000)
        return jsp.integrate.trapezoid(jax.vmap(integrand_r)(pts), pts)
    dR = 5e-3
    d_nu_sigR2_dR = anisotropy_b * (vertical_pressure(R_star + dR) - vertical_pressure(R_star - dR)) / (2*dR)
    v_phi_total_sq = sigma_R2 + (R_star / nu_val) * d_nu_sigR2_dR + R_star * dPhi_dR(x_star, y_star, z_star, params_baryon, params_halo)
    sigma_phi = sigma_R
    v_streaming_sq = jnp.maximum(v_phi_total_sq - sigma_phi**2, 0.0)
    v_mean_phi = jnp.sqrt(v_streaming_sq)
    return jax.lax.cond(nu_val<=0, lambda: (0.0, 0.0, 0.0, 0.0), lambda: (v_mean_phi, sigma_R, sigma_z, sigma_phi))


# ══════════════════════════════════════════════════════════════
# Load data and build initial conditions
# ══════════════════════════════════════════════════════════════
print("Loading data...")
with open(path + 'mock_Nbody_bar_XY_withRot.pkl', 'rb') as f:
    bin_dict = pickle.load(f)

num_per_bin = jnp.array(bin_dict['num_per_bin'])
bin_mapping = jnp.array(bin_dict['bin_mapping'])
num_Vbin = jnp.array(bin_dict['total_bins']).item()
v0 = jnp.array(bin_dict['v0'])
s = jnp.array(bin_dict['s'])
rotation_matrix = makeRotationMatrix(alpha_best_fit, beta_best_fit, gamma_best_fit)
Omega_bar = params_disk_rho['Omega_bar']

# Sample initial positions
n_samples = 5000
x_grid = np.linspace(0, 12, 1000)
logP_xexp = XexpX_pdf_log(x_grid, 4.0)
key = jax.random.PRNGKey(10086)
R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
phi_samples = np.random.RandomState(42).uniform(0, 2*np.pi, size=n_samples)
x_samples = R_samples * np.cos(phi_samples)
y_samples = R_samples * np.sin(phi_samples)

x_grid_z = np.linspace(0, 4, 1000)
logP_exp = expX_pdf_log(x_grid_z, 1.5)
key = jax.random.PRNGKey(10010)
z_samples = sample_from_logP(x_grid_z, logP_exp, n_samples, key)
w0 = jnp.array(np.array([x_samples, y_samples, z_samples]).T)

# Jeans moments for initial velocities
print("Computing Jeans moments...")
get_jeans_vmap = jax.vmap(get_jeans_moments, in_axes=(0,0,0,None,None,None,None))
jeans = get_jeans_vmap(w0[:,0], w0[:,1], w0[:,2], params_baryon, params_disk_rho, params_halo_pot, 1.)
v_rot, sig_R, sig_z, sig_phi = jeans

key1, key2, key3 = jax.random.PRNGKey(42), jax.random.PRNGKey(109), jax.random.PRNGKey(2026)
g1 = jax.random.normal(key1, (n_samples,))
g2 = jax.random.normal(key2, (n_samples,))
g3 = jax.random.normal(key3, (n_samples,))
vR = g1 * sig_R
vz = g2 * sig_z
vphi = v_rot + g3 * sig_phi
x_ic, y_ic, vx_ic, vy_ic = getCartesianFromCylindrical_clockwise(
    jnp.sqrt(w0[:,0]**2 + w0[:,1]**2), jnp.arctan2(w0[:,1], w0[:,0]), vR, vphi)
w0_new = jnp.array([x_ic, y_ic, w0[:,2], vx_ic, vy_ic, vz]).T
print(f"Initial conditions: {w0_new.shape}")

# Orbital timescales
_R = jnp.sqrt(w0_new[:,0]**2 + w0_new[:,1]**2)
_z = w0_new[:,2]
T_orb = jax.vmap(estimate_orbital_timescale, in_axes=(0, None, None, 0))(
    _R, potential_func, (params_baryon, params_halo_pot), _z)

# Acceleration and potential closures
@jax.jit
def acc_fn(x, y, z):
    def _pot(pos):
        return potential_func(pos[0], pos[1], pos[2], params_baryon, params_halo_pot)
    return -jax.grad(_pot)(jnp.array([x, y, z]))

@jax.jit
def pot_fn(x, y, z):
    return potential_func(x, y, z, params_baryon, params_halo_pot)

# ══════════════════════════════════════════════════════════════
# Integration parameters
# ══════════════════════════════════════════════════════════════
Rzphi_lim = jnp.array([[0,10.],[-3,3],[-jnp.pi, jnp.pi]])
xy_lim = jnp.array([[-10.,10.],[-3.,3.]])
nRzphi = jnp.array([10,6,6])
nXY = jnp.array([60,40])
Rzphi_n_tot = 360

N_step_per_orb = 100
N_dynamical_time = 50
N_max = N_step_per_orb * N_dynamical_time  # 5000
atol, rtol = 1e-7, 1e-4
dt_min, dt_max = 1e-5, 0.3

# Test subset
N_test = 200
w0_test = w0_new[:N_test]
T_total_test = (T_orb * N_dynamical_time)[:N_test]
dt_init_test = (T_orb / N_step_per_orb)[:N_test]

print(f"\nN_test = {N_test}, N_max = {N_max}")
print(f"T_orb range: [{float(T_orb[:N_test].min()):.4f}, {float(T_orb[:N_test].max()):.4f}] Gyr")
print(f"Memory per orbit: original = {N_max*6*4/1024:.1f} KB, chunked(500) = {500*6*4/1024:.1f} KB")

# ══════════════════════════════════════════════════════════════
# 1. Original integrator (reference)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ORIGINAL (full N_max trajectory, bin at end)")
print("="*70)
print("JIT compiling...")
t0 = time.time()
res_orig = _integrate_adaptive_vmap(
    w0_test, acc_fn, pot_fn, N_max, T_total_test,
    dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
    num_Vbin, bin_mapping, num_per_bin,
    Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
    v0, s, rotation_matrix,
)
jax.block_until_ready(res_orig)
print(f"  JIT compile: {time.time()-t0:.1f}s")

orig_valid = np.array(res_orig[6])
orig_Rzphi = np.array(res_orig[0])
orig_sd = np.array(res_orig[1])
orig_h1 = np.array(res_orig[2])
orig_h2 = np.array(res_orig[3])
orig_h3 = np.array(res_orig[4])
orig_h4 = np.array(res_orig[5])
orig_T = np.array(res_orig[8])
print(f"  Valid orbits: {int(orig_valid.sum())} / {N_test}")

# ══════════════════════════════════════════════════════════════
# 2. Chunked integrator — accuracy for each chunk size
# ══════════════════════════════════════════════════════════════
chunk_sizes = [100, 250, 500, 1000, 2500]

for cs in chunk_sizes:
    if N_max % cs != 0:
        print(f"\nSkipping chunk_size={cs} (N_max={N_max} not divisible)")
        continue

    print(f"\n{'='*70}")
    print(f"CHUNKED (chunk_size={cs}, {N_max//cs} chunks)")
    print(f"{'='*70}")
    print("JIT compiling...")
    t0 = time.time()
    res_chunk = _integrate_adaptive_chunked_vmap(
        w0_test, acc_fn, pot_fn, N_max, T_total_test,
        dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
        v0, s, rotation_matrix,
        cs,
    )
    jax.block_until_ready(res_chunk)
    print(f"  JIT compile: {time.time()-t0:.1f}s")

    chunk_valid = np.array(res_chunk[6])
    chunk_Rzphi = np.array(res_chunk[0])
    chunk_sd = np.array(res_chunk[1])
    chunk_h1 = np.array(res_chunk[2])
    chunk_h2 = np.array(res_chunk[3])
    chunk_h3 = np.array(res_chunk[4])
    chunk_h4 = np.array(res_chunk[5])

    # --- Per-orbit accuracy (only orbits valid in both) ---
    mask = (orig_valid > 0.5) & (chunk_valid > 0.5)
    n_both = mask.sum()
    n_only_orig = ((orig_valid > 0.5) & (chunk_valid < 0.5)).sum()
    n_only_chunk = ((orig_valid < 0.5) & (chunk_valid > 0.5)).sum()
    print(f"  Valid: orig={int(orig_valid.sum())}, chunk={int(chunk_valid.sum())}, both={n_both}, "
          f"only_orig={n_only_orig}, only_chunk={n_only_chunk}")

    if n_both > 0:
        # Per-orbit Rzphi density comparison
        rz_o = orig_Rzphi[mask]
        rz_c = chunk_Rzphi[mask]
        rz_nz = np.sum(np.abs(rz_o), axis=1) > 1e-8
        if rz_nz.any():
            rz_rel = np.max(np.abs(rz_o[rz_nz] - rz_c[rz_nz]) / (np.abs(rz_o[rz_nz]) + 1e-10))
            rz_med = np.median(np.abs(rz_o[rz_nz] - rz_c[rz_nz]) / (np.abs(rz_o[rz_nz]) + 1e-10))
            print(f"  Rzphi density: max rel diff = {rz_rel:.4e}, median = {rz_med:.4e}")

        # Per-orbit surface density comparison
        sd_o, sd_c = orig_sd[mask], chunk_sd[mask]
        sd_nz = np.sum(np.abs(sd_o), axis=1) > 1e-8
        if sd_nz.any():
            sd_rel = np.max(np.abs(sd_o[sd_nz] - sd_c[sd_nz]) / (np.abs(sd_o[sd_nz]) + 1e-10))
            sd_med = np.median(np.abs(sd_o[sd_nz] - sd_c[sd_nz]) / (np.abs(sd_o[sd_nz]) + 1e-10))
            print(f"  Surface density: max rel diff = {sd_rel:.4e}, median = {sd_med:.4e}")

        # Per-orbit GH moment comparison
        for name, ho, hc in [('h1', orig_h1, chunk_h1), ('h2', orig_h2, chunk_h2),
                              ('h3', orig_h3, chunk_h3), ('h4', orig_h4, chunk_h4)]:
            h_o, h_c = ho[mask], hc[mask]
            h_nz = np.abs(h_o) > 0.01
            if h_nz.any():
                h_abs = np.max(np.abs(h_o[h_nz] - h_c[h_nz]))
                h_rel = np.max(np.abs(h_o[h_nz] - h_c[h_nz]) / (np.abs(h_o[h_nz]) + 1e-10))
                print(f"  {name}: max abs diff = {h_abs:.4e}, max rel diff = {h_rel:.4e}")

        # --- Aggregate bin-level comparison (sum over orbits with unit weights) ---
        # This is what matters for the model — the combined orbital library
        w_orig = np.where(orig_valid > 0.5, 1.0, 0.0)
        w_chunk = np.where(chunk_valid > 0.5, 1.0, 0.0)

        agg_sd_orig = orig_sd.T @ w_orig  # (n_Vbin,)
        agg_sd_chunk = chunk_sd.T @ w_chunk
        agg_nz = np.abs(agg_sd_orig) > 1e-8
        if agg_nz.any():
            agg_rel = np.max(np.abs(agg_sd_orig[agg_nz] - agg_sd_chunk[agg_nz]) / np.abs(agg_sd_orig[agg_nz]))
            print(f"  Aggregate surface density max rel diff: {agg_rel:.4e}")

        agg_h1_orig = (orig_h1 * orig_sd).T @ w_orig / (agg_sd_orig + 1e-10)
        agg_h1_chunk = (chunk_h1 * chunk_sd).T @ w_chunk / (agg_sd_chunk + 1e-10)
        agg_h1_nz = np.abs(agg_h1_orig) > 0.01
        if agg_h1_nz.any():
            agg_h1_rel = np.max(np.abs(agg_h1_orig[agg_h1_nz] - agg_h1_chunk[agg_h1_nz]) / (np.abs(agg_h1_orig[agg_h1_nz]) + 1e-10))
            print(f"  Aggregate h1 max rel diff: {agg_h1_rel:.4e}")

# ══════════════════════════════════════════════════════════════
# 3. Timing comparison
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TIMING COMPARISON")
print("="*70)

n_warmup = 1
n_iter = 3

# Original
for _ in range(n_warmup):
    r = _integrate_adaptive_vmap(
        w0_test, acc_fn, pot_fn, N_max, T_total_test,
        dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
        v0, s, rotation_matrix)
    jax.block_until_ready(r)

times_orig = []
for _ in range(n_iter):
    t0 = time.perf_counter()
    r = _integrate_adaptive_vmap(
        w0_test, acc_fn, pot_fn, N_max, T_total_test,
        dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
        num_Vbin, bin_mapping, num_per_bin,
        Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
        v0, s, rotation_matrix)
    jax.block_until_ready(r)
    times_orig.append(time.perf_counter() - t0)
t_orig = np.median(times_orig)
print(f"  Original (N_max={N_max}):         {t_orig:.3f}s  (1.00x)")

for cs in chunk_sizes:
    if N_max % cs != 0:
        continue

    # Already JIT'd from accuracy section above
    for _ in range(n_warmup):
        r = _integrate_adaptive_chunked_vmap(
            w0_test, acc_fn, pot_fn, N_max, T_total_test,
            dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
            v0, s, rotation_matrix, cs)
        jax.block_until_ready(r)

    times_chunk = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        r = _integrate_adaptive_chunked_vmap(
            w0_test, acc_fn, pot_fn, N_max, T_total_test,
            dt_init_test, -Omega_bar, atol, rtol, dt_min, dt_max,
            num_Vbin, bin_mapping, num_per_bin,
            Rzphi_lim, xy_lim, nRzphi, nXY, Rzphi_n_tot,
            v0, s, rotation_matrix, cs)
        jax.block_until_ready(r)
        times_chunk.append(time.perf_counter() - t0)
    t_chunk = np.median(times_chunk)
    ratio = t_orig / t_chunk
    mem_ratio = N_max / cs
    print(f"  Chunked (chunk={cs:>4}, {N_max//cs:>2} chunks): {t_chunk:.3f}s  ({ratio:.2f}x)  [{mem_ratio:.0f}x less memory]")

print(f"\nNote: memory savings are per-orbit trajectory buffer.")
print(f"Original stores ({N_max}, 6) = {N_max*6*4/1024:.0f} KB per orbit.")
print(f"With {N_test} orbits vmapped, total trajectory = {N_test*N_max*6*4/1024/1024:.0f} MB.")
