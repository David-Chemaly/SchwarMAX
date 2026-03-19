"""Quick test: verify model() runs with the adaptive integrator."""
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import time
import sys
sys.path.insert(0, '.')

from model import model
from utils import XexpX_pdf_log, expX_pdf_log, sample_from_logP
from test_integration import get_dict_data

path = './'
dict_data = get_dict_data(path)

# Generate w0 (position samples) — same as test_integration.py
n_samples = 5_000
x_grid = np.linspace(0, 15, 1000)
logP_xexp = XexpX_pdf_log(x_grid, 4.0)
key = jax.random.PRNGKey(10086)
R_samples = sample_from_logP(x_grid, logP_xexp, n_samples, key)
phi_samples = np.random.uniform(0, 2*np.pi, size=n_samples)
x_samples, y_samples = R_samples * np.cos(phi_samples), R_samples * np.sin(phi_samples)
x_grid = np.linspace(0, 4, 1000)
logP_exp = expX_pdf_log(x_grid, 1.5)
key = jax.random.PRNGKey(10010)
z_samples = sample_from_logP(x_grid, logP_exp, n_samples, key)
w0 = jnp.array(np.array([x_samples, y_samples, z_samples]).T)
dict_data['w0'] = w0

num_Vbin = int(dict_data['total_bins'])

# Best-fit parameters (angles in radians, then converted to degrees inside)
logMhalo, logrho0, logM_bar, logRh_disk, logRs_disk, logHs_disk, logRs_bar, \
    alpha_rad, beta_rad, gamma_rad, logLM, logOmega_bar = (
        11.8, 8.8, 10.4, 1.2, 0.45, -0.24, 0.3,
        30*np.pi/180, 20*np.pi/180, 130*np.pi/180, 0, 1.6
    )

alpha = alpha_rad * 180 / np.pi
beta = beta_rad * 180 / np.pi
gamma = gamma_rad * 180 / np.pi

params_halo_pot = {
    'logM': logMhalo,
    'Rs': 10 ** logRh_disk,
    'a': 1.0, 'b': 1.0, 'c': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
}

params_disk_rho = {
    'rho0_disc': 10 ** logrho0,
    'Rd_disc': 10 ** logRs_disk,
    'hz_disc': 10 ** logHs_disk,
    'light_to_mass_ratio': 10 ** logLM,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'alpha': alpha, 'beta': beta, 'gamma': gamma,
    'logM_bar': logM_bar,
    'Rs_bar': 10 ** logRs_bar,
    'p_bar': 0.3, 'q_bar': 0.3,
    'Omega_bar': 10 ** logOmega_bar,
}

print(f"n_particles = {w0.shape[0]}, num_Vbin = {num_Vbin}")
print(f"Omega_bar = {params_disk_rho['Omega_bar']:.2f}")
print()

print("Running model() with adaptive integrator...")
t0 = time.time()
result = model(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
# Block until computation completes
density_set = result[0]
density_set[0].block_until_ready()
t1 = time.time()
print(f"model() completed in {t1 - t0:.1f}s (includes JIT)")

density_set, V_model, sigma_model, h1_set, h2_set, h3_set, h4_set, weights = result
print(f"  density shape: {density_set[0].shape}")
print(f"  V_model shape: {V_model.shape}")
print(f"  weights shape: {weights.shape}, sum={float(weights.sum()):.4f}")
print(f"  weights min={float(weights.min()):.6f}, max={float(weights.max()):.6f}")

# Second run (no JIT overhead)
print("\nSecond run (warm)...")
t0 = time.time()
result2 = model(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
result2[0][0].block_until_ready()
t1 = time.time()
print(f"model() completed in {t1 - t0:.1f}s")
print("Done.")
