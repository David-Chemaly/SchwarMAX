"""
Profile model_bootstrap() using JAX's built-in profiler.

This captures the actual XLA execution timeline, showing exactly
where time is spent inside the single JIT-compiled function.

Usage:
    python profile_model_jax.py

Then open the trace in Chrome: chrome://tracing
or TensorBoard: tensorboard --logdir=./jax_profile
"""
import sys
import time
import numpy as np

path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp

from model_bar import model_bootstrap
from likelihoods_bar import get_dict_data_bootstrap
from constants import EPSILON

# Load data
data_path = path
filename = 'mock_Nbody_bar_XY_withRot_Nbins1000.pkl'
dict_data = get_dict_data_bootstrap(data_path, filename)

# Parameters near the mode
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

# Warmup
print("JIT warmup...")
result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
jax.block_until_ready(result)
print("Warmup done.")

# Baseline timing
t0 = time.time()
result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
jax.block_until_ready(result)
t_base = time.time() - t0
print(f"Baseline: {t_base:.3f}s")

# Profile with JAX profiler
print("\nProfiling with JAX tracer...")
profile_dir = path + 'jax_profile'
jax.profiler.start_trace(profile_dir)

for i in range(3):
    result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    jax.block_until_ready(result)

jax.profiler.stop_trace()
print(f"Profile saved to: {profile_dir}")
print("View with: tensorboard --logdir=./jax_profile")
print("Or open the .trace.json.gz in chrome://tracing")

# Also try: what if we call the lowered/compiled version directly?
print("\n--- Timing experiments ---")

# Test: multiple calls to see consistency
times = []
for i in range(5):
    t0 = time.time()
    result = model_bootstrap(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    jax.block_until_ready(result)
    dt = time.time() - t0
    times.append(dt)
    print(f"  Call {i+1}: {dt:.3f}s")

print(f"\n  Mean: {np.mean(times):.3f}s ± {np.std(times):.3f}s")

# Test: what fraction is Python overhead vs XLA execution?
# Lower the function to see compiled execution time
print("\nCompiled (lowered) execution:")
lowered = jax.jit(model_bootstrap, static_argnames=('num_Vbin')).lower(
    params_halo_pot, params_disk_rho, dict_data, num_Vbin)
compiled = lowered.compile()

# Print the compiled cost analysis if available
try:
    cost = compiled.cost_analysis()
    if cost:
        print(f"  XLA cost analysis: {cost}")
except Exception:
    pass

times_compiled = []
for i in range(5):
    t0 = time.time()
    result = compiled(params_halo_pot, params_disk_rho, dict_data, num_Vbin)
    jax.block_until_ready(result)
    dt = time.time() - t0
    times_compiled.append(dt)

print(f"  Compiled mean: {np.mean(times_compiled):.3f}s ± {np.std(times_compiled):.3f}s")
print(f"  vs jit mean:   {np.mean(times):.3f}s")
print(f"  Difference (Python overhead): {np.mean(times) - np.mean(times_compiled):.3f}s")
