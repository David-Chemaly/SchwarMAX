# -*- coding: utf-8 -*-
"""
SchwarMAX inference with Nautilus sampler.

Pipeline:
  1. Density-only emcee (8D) → best-fit density params
  2. Nelder-Mead on full logL (13D) → find the mode
  3. Nautilus with tight priors around the mode

Usage on Colab:
  1. Mount Google Drive
  2. pip install nautilus-sampler jaxopt corner
  3. Run this script
"""

# ============================================================
# Setup
# ============================================================
import sys
import os
import time
import pickle
import numpy as np

# path = '/content/drive/MyDrive/SchwarMAX-analytic/'
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'
sys.path.append(path)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from model_bar import *
from likelihoods_bar import *
from utils import *
from constants import EPSILON

# ============================================================
# Load data
# ============================================================
data_path = path
filename = 'mock_Nbody_bar_XY_withRot_Nbins1000.pkl'
dict_data = get_dict_data_bootstrap(data_path, filename)

# ============================================================
# Step 1: Density-only emcee → best-fit + logl_density_max
# ============================================================
print("=" * 70)
print("Step 1: Density-only MCMC")
print("=" * 70)

import emcee

def log_prior_density(theta):
    if (7 < theta[0] < 12) and (8 < theta[1] < 12) and \
       (-1 < theta[2] < 2) and (-1 < theta[3] < 1) and (-1 < theta[4] < 1) and \
       (0 <= theta[5] < jnp.pi) and (0 <= theta[6] < jnp.pi/2) and \
       (0 <= theta[7] < jnp.pi):
        return 0.0
    return -np.inf

def log_prob_density(theta):
    lp = log_prior_density(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl_density(theta, dict_data, dict_data['total_bins'])
    return float(ll) + lp

ndim_density = 8
nwalkers_density = 16
p0_density = np.array([10.5, 10, 0.8, 0., 0.5, jnp.pi/4, jnp.pi/4, 3.5*jnp.pi/4])
np.random.seed(42)
initial_pos_density = p0_density + np.random.uniform(-0.3, 0.3, (nwalkers_density, ndim_density))

sampler_density = emcee.EnsembleSampler(nwalkers_density, ndim_density, log_prob_density)
sampler_density.run_mcmc(initial_pos_density, 500, progress=True)

samples_density = sampler_density.get_chain(discard=200, flat=True)
params_bestfit = np.percentile(samples_density, axis=0, q=50)
logl_val = logl_density(params_bestfit, dict_data, dict_data['total_bins'])
print(f'Best-fit density logL: {float(logl_val):.4f}')
dict_data['logl_density_max'] = logl_val

logM_disc_bf, logM_bulge_bf, logRd_disc_bf, logHs_disc_bf, logRs_bulge_bf, \
    alpha_bf, beta_bf, gamma_bf = params_bestfit

print(f'logM_disc:  {logM_disc_bf:.4f}')
print(f'logM_bar:   {logM_bulge_bf:.4f}')
print(f'logRd_disc: {logRd_disc_bf:.4f}')
print(f'logHs_disc: {logHs_disc_bf:.4f}')
print(f'logL_bar:   {logRs_bulge_bf:.4f}')
print(f'alpha:      {alpha_bf * 180 / np.pi:.1f} deg')
print(f'beta:       {beta_bf * 180 / np.pi:.1f} deg')
print(f'gamma:      {gamma_bf * 180 / np.pi:.1f} deg')

# ============================================================
# Step 2: Nelder-Mead on full 13D logL → find the mode
# ============================================================
print()
print("=" * 70)
print("Step 2: Nelder-Mead minimisation of full logL (13D)")
print("=" * 70)

from scipy.optimize import minimize

x0 = np.array([
    11.6,             # logM_halo
    logM_disc_bf,     # logM_disc
    logM_bulge_bf,    # logM_bar
    jnp.log10(19).item(),  # logRs_halo
    logRd_disc_bf,    # logRs_disk
    logHs_disc_bf,    # logHs_disk
    logRs_bulge_bf,   # logL_bar
    alpha_bf,         # alpha (rad)
    beta_bf,          # beta (rad)
    gamma_bf,         # gamma (rad)
    0.,               # log_L2M
    1.5,              # log_Omega
    -2.,              # log_sigma
])

param_names = [
    'logM_halo', 'logM_disc', 'logM_bar', 'logRs_halo',
    'logRs_disk', 'logHs_disk', 'logL_bar',
    'alpha', 'beta', 'gamma',
    'log_L2M', 'log_Omega', 'log_sigma',
]

n_eval = [0]
def neg_logl(params):
    n_eval[0] += 1
    ll = logl_angular_input_bootstrap(params, dict_data, dict_data['total_bins'])
    val = -float(ll)
    if not np.isfinite(val):
        return 1e100
    if n_eval[0] % 20 == 0:
        print(f"  eval {n_eval[0]:>4}: logL = {-val:.2f}", flush=True)
    return val

# Sanity check
logL_x0 = -neg_logl(x0)
print(f"logL at x0: {logL_x0:.2f}")

t0 = time.time()
res = minimize(neg_logl, x0, method='Nelder-Mead',
               options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1.0, 'adaptive': True})
t_min = time.time() - t0

x_mode = res.x
logL_mode = -res.fun

print(f"\nNelder-Mead done in {t_min:.0f}s ({n_eval[0]} evals)")
print(f"logL at mode: {logL_mode:.2f}")
print(f"\nMode:")
for i, name in enumerate(param_names):
    print(f"  {name:>12}: {x_mode[i]:.4f}  (started at {x0[i]:.4f})")

# ============================================================
# Step 3: Nautilus with tight priors around the mode
# ============================================================
print()
print("=" * 70)
print("Step 3: Nautilus nested sampling (13D)")
print("=" * 70)

from nautilus import Sampler

ndim = 13

# Prior: mode ± half_width for each parameter
# Tune these widths: wide enough to contain the posterior,
# narrow enough to keep Nautilus efficient.
half_widths = np.array([
    0.15,   # logM_halo
    0.15,   # logM_disc
    0.15,   # logM_bar
    0.15,   # logRs_halo
    0.15,   # logRs_disk
    0.15,   # logHs_disk
    0.15,   # logL_bar
    0.10,   # alpha (rad) ~ ±6 deg
    0.10,   # beta  (rad) ~ ±6 deg
    0.10,   # gamma (rad) ~ ±6 deg
    0.15,   # log_L2M
    0.15,   # log_Omega
    0.5,    # log_sigma (less constrained)
])

prior_low = x_mode - half_widths
prior_high = x_mode + half_widths

# Clip angles to physical range
prior_low[7] = max(prior_low[7], 0.0)            # alpha >= 0
prior_high[7] = min(prior_high[7], float(jnp.pi)) # alpha <= pi
prior_low[8] = max(prior_low[8], 0.0)            # beta >= 0
prior_high[8] = min(prior_high[8], float(jnp.pi/2))  # beta <= pi/2
prior_low[9] = max(prior_low[9], 0.0)            # gamma >= 0
prior_high[9] = min(prior_high[9], float(jnp.pi)) # gamma <= pi

def prior_transform(u):
    return prior_low + (prior_high - prior_low) * u

print(f"Prior bounds (mode ± half_width):")
for i, name in enumerate(param_names):
    print(f"  {name:>12}: [{prior_low[i]:.4f}, {prior_high[i]:.4f}]  (mode={x_mode[i]:.4f})")
print()

# --- Progress tracker ---
progress_log = path + 'nautilus_progress.log'

class ProgressTracker:
    def __init__(self, log_file):
        self.n_calls = 0
        self.n_density_rejected = 0
        self.t_start = time.time()
        self.t_last_print = 0
        self.best_ll = -np.inf
        self.log_file = log_file
        header = f"{'calls':>8} | {'rejected':>8} | {'elapsed':>10} | {'best logL':>12} | {'avg s/call':>10}"
        sep = "-" * 62
        print(header)
        print(sep)
        with open(self.log_file, 'w') as f:
            f.write(header + "\n")
            f.write(sep + "\n")

    def __call__(self, params):
        ll = logl_angular_input_bootstrap(params, dict_data, dict_data['total_bins'])
        ll_val = float(ll)
        if not np.isfinite(ll_val):
            ll_val = -1e100

        self.n_calls += 1
        if ll_val <= -1e99:
            self.n_density_rejected += 1
        if ll_val > self.best_ll:
            self.best_ll = ll_val

        t_now = time.time()
        if t_now - self.t_last_print > 60 or self.n_calls <= 5:
            elapsed = t_now - self.t_start
            avg = elapsed / self.n_calls
            rej_pct = 100 * self.n_density_rejected / self.n_calls
            line = (f"{self.n_calls:>8d} | {rej_pct:>6.1f}% | {elapsed/3600:>9.2f}h | "
                    f"{self.best_ll:>12.2f} | {avg:>9.2f}s")
            print(line, flush=True)
            with open(self.log_file, 'a') as f:
                f.write(line + "\n")
            self.t_last_print = t_now

        return ll_val

tracker = ProgressTracker(progress_log)

checkpoint_file = path + 'nautilus_checkpoint.hdf5'

sampler = Sampler(
    prior_transform,
    tracker,
    n_dim=ndim,
    n_live=500,
    n_networks=4,
    filepath=checkpoint_file,
    seed=42,
)

print(f"Checkpoint: {checkpoint_file}")
print(f"(Delete this file to start fresh)")
print()

t0 = time.time()
sampler.run(
    verbose=True,
    n_eff=2000,
    n_like_max=20000,
    f_live=0.05,
)
t_total = time.time() - t0

print(f"\nSampling complete in {t_total:.0f}s ({t_total/3600:.1f} hours)")
print(f"Density-rejected: {tracker.n_density_rejected}/{tracker.n_calls} "
      f"({100*tracker.n_density_rejected/max(1,tracker.n_calls):.1f}%)")

# ============================================================
# Step 4: Results
# ============================================================
print()
print("=" * 70)
print("Results")
print("=" * 70)

points, log_w, log_l = sampler.posterior()
log_Z = sampler.log_z

weights = np.exp(log_w - np.max(log_w))
weights /= weights.sum()
post_mean = np.average(points, weights=weights, axis=0)
post_std = np.sqrt(np.average((points - post_mean)**2, weights=weights, axis=0))

print(f"log Z = {log_Z:.4f}")
print(f"N_eff = {1.0 / np.sum(weights**2):.0f}")
print(f"Total logL calls: {tracker.n_calls:,}")
print()
print(f"  {'param':>12} | {'mean':>10} | {'std':>10} | {'mode':>10}")
print(f"  {'-'*42}")
for i, name in enumerate(param_names):
    print(f"  {name:>12} | {post_mean[i]:>10.4f} | {post_std[i]:>10.4f} | {x_mode[i]:>10.4f}")

# ============================================================
# Step 5: Save
# ============================================================
import pandas as pd

n_resample = 10000
idx = np.random.choice(len(points), size=n_resample, p=weights, replace=True)
equal_weight_samples = points[idx]

df = pd.DataFrame(equal_weight_samples, columns=param_names)
df['log_likelihood'] = log_l[idx]
df.to_csv(path + 'nautilus_posterior.csv', index=False)

with open(path + 'nautilus_posterior_full.pkl', 'wb') as f:
    pickle.dump({
        'points': points, 'log_w': log_w, 'log_l': log_l,
        'log_Z': log_Z, 'param_names': param_names,
        'prior_low': prior_low, 'prior_high': prior_high,
        'x_mode': x_mode, 'logL_mode': logL_mode,
    }, f)
print(f"Posterior saved to: {path}nautilus_posterior.csv")

# ============================================================
# Step 6: Corner plot
# ============================================================
try:
    import corner
    fig = corner.corner(
        equal_weight_samples, labels=param_names, truths=x_mode,
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3f',
    )
    fig.savefig(path + 'nautilus_corner.png', dpi=150, bbox_inches='tight')
    print(f"Corner plot saved to: {path}nautilus_corner.png")
    plt.close(fig)
except ImportError:
    print("Install corner: pip install corner")
