"""
Test emcee ensemble sampler on the same 10D correlated Gaussian.
Track convergence of mean, std, and autocorrelation over iterations.
Compare total logL calls to Nautilus and Dynesty.
"""
import time
import numpy as np
import emcee

# ============================================================
# 1. Same toy problem as test_nautilus.py
# ============================================================
ndim = 10
rng = np.random.default_rng(42)

true_means = np.array([0.5, -0.3, 1.2, 0.0, -0.7, 0.8, 0.1, -0.5, 0.3, 0.6])

A = rng.normal(size=(ndim, ndim)) * 0.3
cov_true = A @ A.T + 0.1 * np.eye(ndim)
scales = np.sqrt(np.diag(cov_true))
cov_true = cov_true / np.outer(scales, scales) * 0.3**2

cov_inv = np.linalg.inv(cov_true)
sign, log_det = np.linalg.slogdet(cov_true)
log_norm = -0.5 * (ndim * np.log(2 * np.pi) + log_det)

prior_low, prior_high = -5.0, 5.0

print(f"Toy problem: {ndim}D correlated Gaussian")
print(f"True means: {true_means}")
print(f"Marginal σ:  {np.sqrt(np.diag(cov_true))}")
print()

# ============================================================
# 2. Define log-probability for emcee (log-prior + log-likelihood)
# ============================================================
n_calls = 0

def log_prior(params):
    if np.all((params > prior_low) & (params < prior_high)):
        return 0.0  # uniform prior
    return -np.inf

def log_probability(params):
    global n_calls
    n_calls += 1
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    diff = params - true_means
    return lp + float(log_norm - 0.5 * diff @ cov_inv @ diff)

# ============================================================
# 3. Run emcee with convergence tracking
# ============================================================
nwalkers = 50  # emcee recommends >= 2*ndim, use 5*ndim for robustness
max_steps = 20000

# Initialize walkers: small ball around a random point (NOT the true answer)
# This tests burn-in behaviour
init_center = rng.uniform(prior_low + 1, prior_high - 1, size=ndim)
p0 = init_center + 0.5 * rng.normal(size=(nwalkers, ndim))

print(f"emcee: {nwalkers} walkers, up to {max_steps} steps")
print(f"Initial center (NOT true): {np.round(init_center, 2)}")
print()

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

# Run in chunks and track convergence
chunk_size = 500
checkpoints = []  # (step, mean_err, std_err, tau, n_calls_at_step)

print(f"{'step':>6} | {'mean |err|':>10} | {'std |err|':>10} | {'max τ':>8} | {'converged':>10} | {'logL calls':>12}")
print(f"{'-'*70}")

t0 = time.time()
state = None

for chunk_start in range(0, max_steps, chunk_size):
    state = sampler.run_mcmc(state if state else p0, chunk_size, progress=False)
    step = chunk_start + chunk_size

    # Get chain so far
    chain = sampler.get_chain(flat=False)  # (steps, nwalkers, ndim)
    total_steps = chain.shape[0]

    # Discard first half as burn-in for statistics
    burn = total_steps // 2
    flat_chain = chain[burn:].reshape(-1, ndim)

    # Mean and std errors
    post_mean = np.mean(flat_chain, axis=0)
    post_std = np.std(flat_chain, axis=0)
    mean_err = np.mean(np.abs(post_mean - true_means))
    std_err = np.mean(np.abs(post_std - np.sqrt(np.diag(cov_true))))

    # Autocorrelation time (may not be reliable for short chains)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        max_tau = np.max(tau)
        # Convergence criterion: chain length > 50 * tau, and tau estimate stable
        converged = total_steps > 50 * max_tau
    except Exception:
        max_tau = np.nan
        converged = False

    checkpoints.append((step, mean_err, std_err, max_tau, n_calls))

    status = "YES" if converged else "no"
    print(f"{step:>6} | {mean_err:>10.4f} | {std_err:>10.4f} | {max_tau:>8.1f} | {status:>10} | {n_calls:>12,}")

    if converged and step >= 5000:
        print(f"\n  Converged at step {step}!")
        break

t_emcee = time.time() - t0
n_calls_emcee = n_calls

# ============================================================
# 4. Final results
# ============================================================
# Use the autocorrelation time for proper thinning
try:
    tau = sampler.get_autocorr_time()
    burn_final = int(2 * np.max(tau))
    thin_final = max(1, int(0.5 * np.min(tau)))
except Exception:
    burn_final = total_steps // 2
    thin_final = 1

flat_samples = sampler.get_chain(discard=burn_final, thin=thin_final, flat=True)

post_mean = np.mean(flat_samples, axis=0)
post_std = np.std(flat_samples, axis=0)

print(f"\nemcee final results:")
print(f"  Wall time:      {t_emcee:.1f}s")
print(f"  logL calls:     {n_calls_emcee:,}")
print(f"  Total steps:    {sampler.get_chain().shape[0]}")
print(f"  Burn-in:        {burn_final}")
print(f"  Thin:           {thin_final}")
print(f"  Effective samples: {len(flat_samples)}")
try:
    print(f"  Max τ:          {np.max(tau):.1f}")
except:
    pass
print()
print(f"  {'param':>6} | {'true':>8} | {'mean':>8} | {'std':>8} | {'true σ':>8} | {'|bias|/σ':>8}")
print(f"  {'-'*60}")
for i in range(ndim):
    bias = abs(post_mean[i] - true_means[i]) / np.sqrt(cov_true[i, i])
    print(f"  x{i:>4} | {true_means[i]:>8.4f} | {post_mean[i]:>8.4f} | {post_std[i]:>8.4f} | {np.sqrt(cov_true[i,i]):>8.4f} | {bias:>8.2f}")

# ============================================================
# 5. Comparison with Nautilus/Dynesty from previous test
# ============================================================
print()
print("=" * 70)
print("COMPARISON (emcee vs previous results)")
print("=" * 70)
print(f"  {'':>20} | {'emcee':>12} | {'Nautilus':>12} | {'Dynesty':>12}")
print(f"  {'-'*55}")
print(f"  {'logL calls':>20} | {n_calls_emcee:>12,} | {'~46,000':>12} | {'~1,611,000':>12}")
print(f"  {'Wall time (s)':>20} | {t_emcee:>12.1f} | {'~76':>12} | {'~19':>12}")
print(f"  {'Evidence (log Z)':>20} | {'N/A':>12} | {'-23.10':>12} | {'-23.19':>12}")
print()
print(f"  Projected for 4s/call (SchwarMAX):")
print(f"    emcee:    {n_calls_emcee * 4 / 3600:.1f} hours  ({n_calls_emcee * 4 / 86400:.1f} days)")
print(f"    Nautilus: {46000 * 4 / 3600:.1f} hours  ({46000 * 4 / 86400:.1f} days)")
print(f"    Dynesty:  {1611000 * 4 / 3600:.1f} hours  ({1611000 * 4 / 86400:.1f} days)")
print()
print("  Note: emcee does NOT provide evidence (log Z) for model comparison.")
print("  Note: emcee logL calls = nwalkers × nsteps (all walkers evaluated each step).")
