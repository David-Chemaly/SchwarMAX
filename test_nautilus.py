"""
Test Nautilus sampler on a 10D toy problem.

Ground truth: 10D correlated Gaussian with known mean, covariance, and evidence.
This mimics the structure of Schwarzschild modeling:
  - Some parameters are correlated (like mass-scale degeneracies)
  - Moderate dimensionality (10D, our real problem is 13D)
  - We test recovery of means, widths, and log-evidence

We also compare wall-clock time and number of likelihood calls against Dynesty.
"""
import time
import numpy as np
from scipy.stats import multivariate_normal

# ============================================================
# 1. Build the toy problem: 10D correlated Gaussian
# ============================================================
ndim = 10
rng = np.random.default_rng(42)

# True parameter values (centre of the Gaussian likelihood)
true_means = np.array([0.5, -0.3, 1.2, 0.0, -0.7, 0.8, 0.1, -0.5, 0.3, 0.6])

# Build a random correlation matrix (positive-definite)
A = rng.normal(size=(ndim, ndim)) * 0.3
cov_true = A @ A.T + 0.1 * np.eye(ndim)  # ensures positive-definite
# Scale so that marginal σ ~ 0.2–0.5 (realistic posterior width)
scales = np.sqrt(np.diag(cov_true))
cov_true = cov_true / np.outer(scales, scales) * 0.3**2

# Precompute inverse and log-normalisation for speed
cov_inv = np.linalg.inv(cov_true)
sign, log_det = np.linalg.slogdet(cov_true)
log_norm = -0.5 * (ndim * np.log(2 * np.pi) + log_det)

# Analytic log-evidence for Gaussian likelihood × uniform prior
# log Z = log_norm + log ∫ exp(-0.5 (x-μ)^T Σ^{-1} (x-μ)) dx over prior box
# For wide enough prior, ≈ log_norm + 0.5*ndim*log(2π) + 0.5*log|Σ|
# = 0  (if prior volume normalisation cancels perfectly)
# More precisely: log Z = log(prior_volume) + log_norm + integral
# For uniform prior on [-5, 5]^10, prior_volume = 10^10
prior_low, prior_high = -5.0, 5.0
prior_volume = (prior_high - prior_low) ** ndim
# Since the Gaussian is well within the prior box, virtually all probability is captured
log_Z_analytic = np.log(1.0 / prior_volume) + 0.0  # likelihood integrates to 1 for normalised Gaussian
# Actually: ∫ L(x) dx = 1 for normalised Gaussian, so Z = 1/prior_volume
log_Z_analytic = -ndim * np.log(prior_high - prior_low)

print(f"Toy problem: {ndim}D correlated Gaussian")
print(f"True means: {true_means}")
print(f"Marginal σ:  {np.sqrt(np.diag(cov_true))}")
print(f"Analytic log Z = {log_Z_analytic:.4f}")
print(f"Prior: uniform [{prior_low}, {prior_high}]^{ndim}")
print()

# ============================================================
# Likelihood (called by both samplers)
# ============================================================
n_calls = 0

def log_likelihood(params):
    global n_calls
    n_calls += 1
    diff = params - true_means
    return float(log_norm - 0.5 * diff @ cov_inv @ diff)

# Add a fake delay to simulate expensive likelihood (optional)
FAKE_DELAY = 0.0  # set to e.g. 0.01 to simulate 10ms per call

def log_likelihood_slow(params):
    if FAKE_DELAY > 0:
        time.sleep(FAKE_DELAY)
    return log_likelihood(params)


# ============================================================
# 2. Run Nautilus
# ============================================================
print("=" * 70)
print("NAUTILUS")
print("=" * 70)

from nautilus import Sampler

# Use array-based prior transform (like Dynesty) instead of named parameters
def nautilus_prior(u):
    """Map unit cube to prior bounds."""
    return prior_low + (prior_high - prior_low) * u

n_calls = 0
t0 = time.time()

sampler = Sampler(nautilus_prior, log_likelihood_slow, n_dim=ndim, n_live=1000)
sampler.run(verbose=True)

t_nautilus = time.time() - t0
n_calls_nautilus = n_calls

# Extract posterior
points, log_w, log_l = sampler.posterior()
log_Z_nautilus = sampler.log_z

# Compute weighted mean and std
weights = np.exp(log_w - np.max(log_w))
weights /= weights.sum()
post_mean = np.average(points, weights=weights, axis=0)
post_std = np.sqrt(np.average((points - post_mean)**2, weights=weights, axis=0))

print(f"\nNautilus results:")
print(f"  Wall time:    {t_nautilus:.1f}s")
print(f"  logL calls:   {n_calls_nautilus:,}")
print(f"  log Z:        {log_Z_nautilus:.4f}  (analytic: {log_Z_analytic:.4f})")
print(f"  log Z error:  {log_Z_nautilus - log_Z_analytic:.4f}")
print()
print(f"  {'param':>6} | {'true':>8} | {'mean':>8} | {'std':>8} | {'true σ':>8} | {'|bias|/σ':>8}")
print(f"  {'-'*60}")
for i in range(ndim):
    bias = abs(post_mean[i] - true_means[i]) / np.sqrt(cov_true[i, i])
    print(f"  x{i:>4} | {true_means[i]:>8.4f} | {post_mean[i]:>8.4f} | {post_std[i]:>8.4f} | {np.sqrt(cov_true[i,i]):>8.4f} | {bias:>8.2f}")


# ============================================================
# 3. Run Dynesty for comparison
# ============================================================
print()
print("=" * 70)
print("DYNESTY")
print("=" * 70)

import dynesty

def prior_transform(u):
    return prior_low + (prior_high - prior_low) * u

n_calls = 0
t0 = time.time()

dsampler = dynesty.NestedSampler(
    log_likelihood_slow, prior_transform, ndim,
    nlive=1000, sample='rslice',
)
dsampler.run_nested(print_progress=True)

t_dynesty = time.time() - t0
n_calls_dynesty = n_calls

res = dsampler.results
log_Z_dynesty = res.logz[-1]
log_Z_err_dynesty = res.logzerr[-1]

# Weighted posterior
d_weights = np.exp(res.logwt - np.max(res.logwt))
d_weights /= d_weights.sum()
d_mean = np.average(res.samples, weights=d_weights, axis=0)
d_std = np.sqrt(np.average((res.samples - d_mean)**2, weights=d_weights, axis=0))

print(f"\nDynesty results:")
print(f"  Wall time:    {t_dynesty:.1f}s")
print(f"  logL calls:   {n_calls_dynesty:,}")
print(f"  log Z:        {log_Z_dynesty:.4f} ± {log_Z_err_dynesty:.4f}  (analytic: {log_Z_analytic:.4f})")
print(f"  log Z error:  {log_Z_dynesty - log_Z_analytic:.4f}")
print()
print(f"  {'param':>6} | {'true':>8} | {'mean':>8} | {'std':>8} | {'true σ':>8} | {'|bias|/σ':>8}")
print(f"  {'-'*60}")
for i in range(ndim):
    bias = abs(d_mean[i] - true_means[i]) / np.sqrt(cov_true[i, i])
    print(f"  x{i:>4} | {true_means[i]:>8.4f} | {d_mean[i]:>8.4f} | {d_std[i]:>8.4f} | {np.sqrt(cov_true[i,i]):>8.4f} | {bias:>8.2f}")


# ============================================================
# 4. Summary comparison
# ============================================================
print()
print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"  {'':>20} | {'Nautilus':>12} | {'Dynesty':>12} | {'Speedup':>10}")
print(f"  {'-'*60}")
print(f"  {'Wall time (s)':>20} | {t_nautilus:>12.1f} | {t_dynesty:>12.1f} | {t_dynesty/t_nautilus:>10.1f}x")
print(f"  {'logL calls':>20} | {n_calls_nautilus:>12,} | {n_calls_dynesty:>12,} | {n_calls_dynesty/n_calls_nautilus:>10.1f}x")
print(f"  {'log Z':>20} | {log_Z_nautilus:>12.4f} | {log_Z_dynesty:>12.4f} | {'':>10}")
print(f"  {'log Z error':>20} | {log_Z_nautilus - log_Z_analytic:>12.4f} | {log_Z_dynesty - log_Z_analytic:>12.4f} | {'':>10}")
print()
print(f"  Analytic log Z = {log_Z_analytic:.4f}")
print()

if FAKE_DELAY > 0:
    print(f"  With {FAKE_DELAY}s/call delay:")
    print(f"    Nautilus would take: {n_calls_nautilus * FAKE_DELAY / 3600:.1f} hours")
    print(f"    Dynesty would take:  {n_calls_dynesty * FAKE_DELAY / 3600:.1f} hours")
else:
    print(f"  Projected for 4s/call (SchwarMAX):")
    print(f"    Nautilus: {n_calls_nautilus * 4 / 3600:.1f} hours  ({n_calls_nautilus * 4 / 86400:.1f} days)")
    print(f"    Dynesty:  {n_calls_dynesty * 4 / 3600:.1f} hours  ({n_calls_dynesty * 4 / 86400:.1f} days)")
