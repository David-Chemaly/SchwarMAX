"""
Sanity checks for the JAX ensemble (stretch move) sampler.

Tests against distributions with known analytic properties:
  1. Isotropic Gaussian (2D) — recover mean and std
  2. Correlated Gaussian (5D) — recover full covariance matrix
  3. Bimodal Gaussian (2D) — known limitation of stretch move
  4. Heavy-tailed Student-t (2D, df=3) — recover scale and heavier tails
  5. Banana (Rosenbrock) distribution (2D) — curved degeneracy

Run:  python test_ensemble_sampler.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# ═══════════════════════════════════════════════════════════════════════
# Generic stretch-move sampler (extracted from run_ensemble_mcmc.py)
# ═══════════════════════════════════════════════════════════════════════

def _sample_z(rng_key, n, a=2.0):
    u = jax.random.uniform(rng_key, shape=(n,))
    return ((a - 1.0) * u + 1.0) ** 2 / a


def _stretch_propose(rng_key, active, complement, ndim, a=2.0):
    n_half = active.shape[0]
    k1, k2 = jax.random.split(rng_key)
    idx = jax.random.randint(k1, (n_half,), 0, complement.shape[0])
    companions = complement[idx]
    z = _sample_z(k2, n_half, a)
    proposals = companions + z[:, None] * (active - companions)
    log_factors = (ndim - 1) * jnp.log(z)
    return proposals, log_factors


def _de_propose(rng_key, active, complement, ndim, sigma=1e-5):
    """Differential Evolution move (ter Braak 2006)."""
    n_half = active.shape[0]
    n_comp = complement.shape[0]
    k1, k2, k3 = jax.random.split(rng_key, 3)
    gamma = 2.38 / jnp.sqrt(2.0 * ndim)
    r1 = jax.random.randint(k1, (n_half,), 0, n_comp)
    r2 = jax.random.randint(k2, (n_half,), 0, n_comp - 1)
    r2 = jnp.where(r2 >= r1, r2 + 1, r2)
    diff = complement[r1] - complement[r2]
    noise = sigma * jax.random.normal(k3, active.shape)
    proposals = active + gamma * diff + noise
    log_factors = jnp.zeros(n_half)  # symmetric
    return proposals, log_factors


def _snooker_propose(rng_key, active, complement, ndim, gamma=1.7):
    """DE-Snooker move (ter Braak & Vrugt 2008)."""
    n_half = active.shape[0]
    n_comp = complement.shape[0]
    k1, k2 = jax.random.split(rng_key)
    r0 = jax.random.randint(k1, (n_half,), 0, n_comp)
    r12 = jax.random.randint(k2, (n_half, 2), 0, n_comp)
    z0 = complement[r0]
    direction = active - z0
    dist = jnp.linalg.norm(direction, axis=1, keepdims=True)
    dist_safe = jnp.maximum(dist, 1e-30)
    d_hat = direction / dist_safe
    diff = complement[r12[:, 0]] - complement[r12[:, 1]]
    proj = jnp.sum(diff * d_hat, axis=1, keepdims=True)
    proposals = active + gamma * proj * d_hat
    new_dist = jnp.linalg.norm(proposals - z0, axis=1)
    log_factors = (ndim - 1) * jnp.log(new_dist / dist_safe.squeeze())
    return proposals, log_factors


def _mixed_move_half(rng_key, active, active_logp, complement,
                     ndim, logdensity_vmap, move_weights):
    """Mixed move: per-walker random selection of DE/Snooker/Stretch."""
    n_half = active.shape[0]
    k_sel, k_de, k_snk, k_str, k_acc = jax.random.split(rng_key, 5)

    prop_de, lf_de = _de_propose(k_de, active, complement, ndim)
    prop_snk, lf_snk = _snooker_propose(k_snk, active, complement, ndim)
    prop_str, lf_str = _stretch_propose(k_str, active, complement, ndim)

    u_sel = jax.random.uniform(k_sel, (n_half,))
    use_de = u_sel < move_weights[0]
    use_snk = (u_sel >= move_weights[0]) & (u_sel < move_weights[1])

    proposals = jnp.where(use_de[:, None], prop_de,
                    jnp.where(use_snk[:, None], prop_snk, prop_str))
    log_factors = jnp.where(use_de, lf_de,
                    jnp.where(use_snk, lf_snk, lf_str))

    prop_logp = logdensity_vmap(proposals)
    log_accept = log_factors + prop_logp - active_logp
    log_u = jnp.log(jax.random.uniform(k_acc, (n_half,)))
    accept = log_u < log_accept

    new_active = jnp.where(accept[:, None], proposals, active)
    new_logp = jnp.where(accept, prop_logp, active_logp)
    return new_active, new_logp, jnp.sum(accept)


def _stretch_only_half(rng_key, active, active_logp, complement,
                       ndim, logdensity_vmap, a=2.0):
    """Stretch-only move for backward compatibility."""
    n_half = active.shape[0]
    k1, k_acc = jax.random.split(rng_key)
    proposals, log_factors = _stretch_propose(k1, active, complement, ndim, a)
    prop_logp = logdensity_vmap(proposals)
    log_accept = log_factors + prop_logp - active_logp
    log_u = jnp.log(jax.random.uniform(k_acc, (n_half,)))
    accept = log_u < log_accept
    new_active = jnp.where(accept[:, None], proposals, active)
    new_logp = jnp.where(accept, prop_logp, active_logp)
    return new_active, new_logp, jnp.sum(accept)


def ensemble_step(rng_key, positions, logp, n_walkers, ndim,
                  logdensity_vmap, move='mixed',
                  move_weights=jnp.array([0.7, 0.9, 1.0])):
    """One full ensemble step with configurable move strategy.

    move: 'mixed' (DE 70% + Snooker 20% + Stretch 10%) or 'stretch' (stretch only)
    """
    n_half = n_walkers // 2
    k1, k2 = jax.random.split(rng_key)

    s0, s1 = positions[:n_half], positions[n_half:]
    lp0, lp1 = logp[:n_half], logp[n_half:]

    if move == 'mixed':
        s0, lp0, acc0 = _mixed_move_half(k1, s0, lp0, s1, ndim, logdensity_vmap, move_weights)
        s1, lp1, acc1 = _mixed_move_half(k2, s1, lp1, s0, ndim, logdensity_vmap, move_weights)
    else:
        s0, lp0, acc0 = _stretch_only_half(k1, s0, lp0, s1, ndim, logdensity_vmap)
        s1, lp1, acc1 = _stretch_only_half(k2, s1, lp1, s0, ndim, logdensity_vmap)

    new_positions = jnp.concatenate([s0, s1], axis=0)
    new_logp = jnp.concatenate([lp0, lp1], axis=0)
    return new_positions, new_logp, acc0 + acc1


def run_sampler(logdensity_fn, ndim, n_walkers, n_steps, init_positions,
                a=2.0, seed=42, burnin_frac=0.5, verbose=True,
                move='mixed'):
    """Run the ensemble sampler and return flat post-burnin samples.

    move: 'mixed' (DE 70% + Snooker 20% + Stretch 10%) or 'stretch' (stretch only)
    """
    logdensity_vmap = jax.vmap(logdensity_fn)
    rng_key = jax.random.PRNGKey(seed)

    positions = init_positions
    logp = logdensity_vmap(positions)

    chain = np.zeros((n_steps, n_walkers, ndim))
    logp_chain = np.zeros((n_steps, n_walkers))
    acc_rates = []

    for step in range(n_steps):
        rng_key, step_key = jax.random.split(rng_key)
        positions, logp, n_acc = ensemble_step(
            step_key, positions, logp, n_walkers, ndim, logdensity_vmap,
            move=move)
        chain[step] = np.array(positions)
        logp_chain[step] = np.array(logp)
        acc_rates.append(float(n_acc) / n_walkers)

    burnin = int(n_steps * burnin_frac)
    flat = chain[burnin:].reshape(-1, ndim)
    mean_acc = np.mean(acc_rates[burnin:])

    if verbose:
        print(f"  Move strategy: {move}")
        print(f"  Acceptance rate (post-burnin): {mean_acc:.3f}")
        print(f"  Samples: {flat.shape[0]} (burnin={burnin}, kept={n_steps-burnin})")

    return flat, chain, logp_chain, mean_acc


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Isotropic 2D Gaussian
# ═══════════════════════════════════════════════════════════════════════

def test_isotropic_gaussian():
    print("\n" + "="*70)
    print("TEST 1: Isotropic 2D Gaussian  N(mu=[3, -2], sigma=[1, 0.5])")
    print("="*70)

    mu = jnp.array([3.0, -2.0])
    sigma = jnp.array([1.0, 0.5])

    def logp(x):
        return -0.5 * jnp.sum(((x - mu) / sigma) ** 2)

    ndim, n_walkers, n_steps = 2, 20, 2000
    key = jax.random.PRNGKey(0)
    init = mu[None, :] + 0.1 * jax.random.normal(key, (n_walkers, ndim))

    flat, _, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init)

    mean_est = np.mean(flat, axis=0)
    std_est = np.std(flat, axis=0)

    print(f"\n  {'':>10s} {'True':>10s} {'Estimated':>10s} {'Error':>10s}")
    print(f"  {'mean[0]':>10s} {3.0:10.4f} {mean_est[0]:10.4f} {abs(mean_est[0]-3.0):10.4f}")
    print(f"  {'mean[1]':>10s} {-2.0:10.4f} {mean_est[1]:10.4f} {abs(mean_est[1]+2.0):10.4f}")
    print(f"  {'std[0]':>10s} {1.0:10.4f} {std_est[0]:10.4f} {abs(std_est[0]-1.0):10.4f}")
    print(f"  {'std[1]':>10s} {0.5:10.4f} {std_est[1]:10.4f} {abs(std_est[1]-0.5):10.4f}")

    ok_mean = np.allclose(mean_est, np.array([3.0, -2.0]), atol=0.1)
    ok_std = np.allclose(std_est, np.array([1.0, 0.5]), atol=0.1)
    status = "PASS" if (ok_mean and ok_std) else "FAIL"
    print(f"\n  Result: {status}  (tolerance: 0.1 on mean and std)")
    return ok_mean and ok_std


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Correlated 5D Gaussian
# ═══════════════════════════════════════════════════════════════════════

def test_correlated_gaussian():
    print("\n" + "="*70)
    print("TEST 2: Correlated 5D Gaussian (condition number ~100)")
    print("="*70)

    ndim = 5
    mu = jnp.array([1.0, -1.0, 2.0, 0.0, -0.5])

    # Build a covariance with non-trivial correlations
    rng = np.random.RandomState(123)
    A = rng.randn(ndim, ndim) * 0.5
    cov_true = A @ A.T + 0.1 * np.eye(ndim)
    cov_true = np.array(cov_true)
    prec = jnp.array(np.linalg.inv(cov_true))
    print(f"  Condition number: {np.linalg.cond(cov_true):.1f}")

    def logp(x):
        d = x - mu
        return -0.5 * d @ prec @ d

    n_walkers, n_steps = 40, 4000
    key = jax.random.PRNGKey(1)
    init = mu[None, :] + 0.01 * jax.random.normal(key, (n_walkers, ndim))

    flat, _, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init)

    mean_est = np.mean(flat, axis=0)
    cov_est = np.cov(flat.T)

    print(f"\n  Mean errors: {np.abs(mean_est - np.array(mu))}")
    ok_mean = np.allclose(mean_est, np.array(mu), atol=0.15)

    # Compare covariance: relative Frobenius norm
    cov_err = np.linalg.norm(cov_est - cov_true) / np.linalg.norm(cov_true)
    print(f"  Covariance relative Frobenius error: {cov_err:.4f}")
    ok_cov = cov_err < 0.15

    # Check individual variances
    std_true = np.sqrt(np.diag(cov_true))
    std_est = np.std(flat, axis=0)
    print(f"  Std true:      {std_true}")
    print(f"  Std estimated: {std_est}")

    # Check off-diagonal correlations
    corr_true = cov_true / np.outer(std_true, std_true)
    corr_est = cov_est / np.outer(std_est, std_est)
    corr_err = np.max(np.abs(corr_est - corr_true))
    print(f"  Max correlation error: {corr_err:.4f}")
    ok_corr = corr_err < 0.15

    status = "PASS" if (ok_mean and ok_cov and ok_corr) else "FAIL"
    print(f"\n  Result: {status}")
    return ok_mean and ok_cov and ok_corr


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Bimodal Gaussian (2D)
# ═══════════════════════════════════════════════════════════════════════

def test_bimodal():
    print("\n" + "="*70)
    print("TEST 3: Bimodal 2D Gaussian — two modes separated by 6 sigma")
    print("  NOTE: Stretch move CANNOT cross probability valleys.")
    print("  This test documents the known limitation.")
    print("="*70)

    mu1 = jnp.array([-3.0, 0.0])
    mu2 = jnp.array([3.0, 0.0])
    sigma = 0.8

    def logp(x):
        lp1 = -0.5 * jnp.sum(((x - mu1) / sigma) ** 2)
        lp2 = -0.5 * jnp.sum(((x - mu2) / sigma) ** 2)
        return jnp.logaddexp(lp1, lp2)  # equal weight mixture

    ndim, n_walkers, n_steps = 2, 40, 3000

    # Initialize walkers spread across BOTH modes
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    init_mode1 = mu1[None, :] + 0.1 * jax.random.normal(k1, (n_walkers // 2, ndim))
    init_mode2 = mu2[None, :] + 0.1 * jax.random.normal(k2, (n_walkers // 2, ndim))
    init = jnp.concatenate([init_mode1, init_mode2], axis=0)

    flat, chain, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init, burnin_frac=0.3)

    # Check: are samples in both modes?
    in_mode1 = np.sum(flat[:, 0] < 0)
    in_mode2 = np.sum(flat[:, 0] > 0)
    frac1 = in_mode1 / len(flat)
    frac2 = in_mode2 / len(flat)
    print(f"\n  Fraction in mode 1 (x<0): {frac1:.3f} (true: 0.500)")
    print(f"  Fraction in mode 2 (x>0): {frac2:.3f} (true: 0.500)")

    # Check if walkers CROSS between modes (count transitions)
    transitions = 0
    for w in range(n_walkers):
        walker_x = chain[:, w, 0]
        sign_changes = np.sum(np.diff(np.sign(walker_x)) != 0)
        transitions += sign_changes
    print(f"  Total mode transitions across all walkers: {transitions}")

    # With both-mode init and no transitions, the fractions should be ~50/50
    # but individual walkers never cross
    ok_both = min(frac1, frac2) > 0.3  # at least 30% in each mode
    ok_std = True

    if ok_both:
        # Check std within each mode
        mode1_samples = flat[flat[:, 0] < 0]
        mode2_samples = flat[flat[:, 0] > 0]
        if len(mode1_samples) > 100 and len(mode2_samples) > 100:
            std1 = np.std(mode1_samples[:, 0])
            std2 = np.std(mode2_samples[:, 0])
            print(f"  Std within mode 1: {std1:.4f} (true: {sigma})")
            print(f"  Std within mode 2: {std2:.4f} (true: {sigma})")
            ok_std = abs(std1 - sigma) < 0.15 and abs(std2 - sigma) < 0.15

    if transitions > 5:
        print(f"\n  Result: PASS (walkers DO cross modes — surprising for stretch move)")
    elif ok_both and ok_std:
        print(f"\n  Result: PARTIAL PASS")
        print(f"    Both modes sampled (due to init), width correct,")
        print(f"    but walkers don't cross modes (expected limitation).")
    else:
        print(f"\n  Result: EXPECTED FAILURE (stretch move can't cross valleys)")

    return ok_both and ok_std


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Heavy-tailed Student-t (2D, df=3)
# ═══════════════════════════════════════════════════════════════════════

def test_student_t():
    print("\n" + "="*70)
    print("TEST 4: 2D Student-t (df=3) — heavy tails")
    print("  True std = sqrt(df/(df-2)) = sqrt(3) ≈ 1.732")
    print("="*70)

    df = 3.0
    true_std = np.sqrt(df / (df - 2))  # 1.732

    def logp(x):
        # Multivariate t: log p(x) ∝ -(df+d)/2 * log(1 + x^T x / df)
        d = 2.0
        return -(df + d) / 2.0 * jnp.log(1.0 + jnp.sum(x ** 2) / df)

    ndim, n_walkers, n_steps = 2, 30, 5000
    key = jax.random.PRNGKey(3)
    init = 0.5 * jax.random.normal(key, (n_walkers, ndim))

    flat, _, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init)

    mean_est = np.mean(flat, axis=0)
    std_est = np.std(flat, axis=0)

    print(f"\n  {'':>10s} {'True':>10s} {'Estimated':>10s} {'Error':>10s}")
    print(f"  {'mean[0]':>10s} {0.0:10.4f} {mean_est[0]:10.4f} {abs(mean_est[0]):10.4f}")
    print(f"  {'mean[1]':>10s} {0.0:10.4f} {mean_est[1]:10.4f} {abs(mean_est[1]):10.4f}")
    print(f"  {'std[0]':>10s} {true_std:10.4f} {std_est[0]:10.4f} {abs(std_est[0]-true_std):10.4f}")
    print(f"  {'std[1]':>10s} {true_std:10.4f} {std_est[1]:10.4f} {abs(std_est[1]-true_std):10.4f}")

    # Check kurtosis (excess kurtosis of t(3) is infinite, but finite-sample should be > Gaussian's 0)
    from scipy.stats import kurtosis as scipy_kurtosis
    kurt = scipy_kurtosis(flat, axis=0, fisher=True)
    print(f"  Excess kurtosis: [{kurt[0]:.2f}, {kurt[1]:.2f}] (Gaussian=0, t(3)=∞)")

    ok_mean = np.allclose(mean_est, 0.0, atol=0.2)
    ok_std = np.allclose(std_est, true_std, atol=0.4)  # wider tolerance for heavy tails
    ok_kurt = np.all(kurt > 1.0)  # should be clearly non-Gaussian

    status = "PASS" if (ok_mean and ok_std and ok_kurt) else "FAIL"
    print(f"\n  Result: {status}")
    if not ok_kurt:
        print(f"    Kurtosis too low — sampler may be under-exploring tails")
    return ok_mean and ok_std and ok_kurt


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Banana (Rosenbrock) distribution (2D)
# ═══════════════════════════════════════════════════════════════════════

def test_banana():
    print("\n" + "="*70)
    print("TEST 5: Banana (Rosenbrock) distribution (2D)")
    print("  Tests curved degeneracy: y ~ N(x^2, 0.5^2), x ~ N(0, 1)")
    print("="*70)

    sigma_x = 1.0
    sigma_y = 0.5
    # p(x,y) ∝ exp(-x^2/2) * exp(-(y - x^2)^2 / (2*sigma_y^2))

    def logp(theta):
        x, y = theta[0], theta[1]
        return -0.5 * (x / sigma_x) ** 2 - 0.5 * ((y - x ** 2) / sigma_y) ** 2

    ndim, n_walkers, n_steps = 2, 30, 5000
    key = jax.random.PRNGKey(4)
    init = 0.1 * jax.random.normal(key, (n_walkers, ndim))

    flat, _, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init)

    # Analytic: E[x] = 0, Var[x] = 1
    #           E[y] = E[x^2] = 1, Var[y] = Var[x^2] + sigma_y^2 = 2 + 0.25 = 2.25
    mean_est = np.mean(flat, axis=0)
    std_est = np.std(flat, axis=0)

    true_mean_x, true_std_x = 0.0, 1.0
    true_mean_y = 1.0  # E[x^2] = sigma_x^2
    true_std_y = np.sqrt(2 * sigma_x**4 + sigma_y**2)  # sqrt(2 + 0.25) ≈ 1.5

    print(f"\n  {'':>10s} {'True':>10s} {'Estimated':>10s} {'Error':>10s}")
    print(f"  {'E[x]':>10s} {true_mean_x:10.4f} {mean_est[0]:10.4f} {abs(mean_est[0]-true_mean_x):10.4f}")
    print(f"  {'E[y]':>10s} {true_mean_y:10.4f} {mean_est[1]:10.4f} {abs(mean_est[1]-true_mean_y):10.4f}")
    print(f"  {'std[x]':>10s} {true_std_x:10.4f} {std_est[0]:10.4f} {abs(std_est[0]-true_std_x):10.4f}")
    print(f"  {'std[y]':>10s} {true_std_y:10.4f} {std_est[1]:10.4f} {abs(std_est[1]-true_std_y):10.4f}")

    # Check correlation: Cov(x, y) = Cov(x, x^2) = E[x^3] = 0 for symmetric x
    # So corr(x, y) should be near 0 despite the curved relationship
    corr_xy = np.corrcoef(flat[:, 0], flat[:, 1])[0, 1]
    print(f"  corr(x,y): {corr_xy:.4f} (true: 0.0)")

    ok_mean = abs(mean_est[0] - true_mean_x) < 0.15 and abs(mean_est[1] - true_mean_y) < 0.2
    ok_std = abs(std_est[0] - true_std_x) < 0.15 and abs(std_est[1] - true_std_y) < 0.3

    status = "PASS" if (ok_mean and ok_std) else "FAIL"
    print(f"\n  Result: {status}")
    return ok_mean and ok_std


# ═══════════════════════════════════════════════════════════════════════
# Test 6: High-D isotropic Gaussian (13D — same as SchwarMAX)
# ═══════════════════════════════════════════════════════════════════════

def test_high_d_gaussian():
    print("\n" + "="*70)
    print("TEST 6: 13D isotropic Gaussian (same dimensionality as SchwarMAX)")
    print("="*70)

    ndim = 13
    mu = jnp.arange(ndim, dtype=jnp.float32) * 0.1  # [0, 0.1, 0.2, ..., 1.2]
    sigma = jnp.ones(ndim) * 0.5

    def logp(x):
        return -0.5 * jnp.sum(((x - mu) / sigma) ** 2)

    n_walkers, n_steps = 52, 4000  # 4*ndim walkers
    key = jax.random.PRNGKey(5)
    init = mu[None, :] + 0.05 * jax.random.normal(key, (n_walkers, ndim))

    flat, _, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init)

    mean_est = np.mean(flat, axis=0)
    std_est = np.std(flat, axis=0)
    mean_err = np.max(np.abs(mean_est - np.array(mu)))
    std_err = np.max(np.abs(std_est - 0.5))

    print(f"\n  Max mean error across 13 dims: {mean_err:.4f}")
    print(f"  Max std error across 13 dims:  {std_err:.4f}")
    print(f"  Mean of estimated stds: {np.mean(std_est):.4f} (true: 0.5)")

    ok = mean_err < 0.1 and std_err < 0.1
    status = "PASS" if ok else "FAIL"
    print(f"\n  Result: {status}")
    return ok


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Verify detailed balance (reversibility check)
# ═══════════════════════════════════════════════════════════════════════

def test_detailed_balance():
    print("\n" + "="*70)
    print("TEST 7: Detailed balance — compare sampler histogram to analytic PDF")
    print("  Uses 1D Gaussian, thins by autocorrelation time for valid KS test")
    print("="*70)

    mu, sigma = 2.0, 1.5

    def logp(x):
        return -0.5 * ((x[0] - mu) / sigma) ** 2

    ndim, n_walkers, n_steps = 1, 20, 8000
    key = jax.random.PRNGKey(6)
    init = mu + 0.1 * jax.random.normal(key, (n_walkers, ndim))

    flat, chain, _, acc = run_sampler(logp, ndim, n_walkers, n_steps, init)

    # Estimate integrated autocorrelation time per walker, then thin
    burnin = n_steps // 2
    post_chain = chain[burnin:, :, 0]  # (n_steps/2, n_walkers)

    # Simple autocorrelation time estimate (mean across walkers)
    def _autocorr_time(x, max_lag=200):
        n = len(x)
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-15:
            return 1.0
        acf = np.correlate(x, x, mode='full')[n-1:n-1+max_lag] / (var * n)
        # Integrate until ACF drops below 0
        tau = 1.0
        for k in range(1, max_lag):
            if acf[k] < 0:
                break
            tau += 2 * acf[k]
        return tau

    taus = [_autocorr_time(post_chain[:, w]) for w in range(n_walkers)]
    tau_mean = np.mean(taus)
    thin = max(1, int(tau_mean))
    print(f"\n  Estimated autocorrelation time: {tau_mean:.1f} steps")
    print(f"  Thinning by: {thin}")

    # Thin samples for KS test (need approximately independent draws)
    thinned = flat[::thin, 0]
    print(f"  Thinned samples: {len(thinned)} (from {len(flat)})")

    from scipy.stats import kstest, norm
    ks_stat, ks_pval = kstest(thinned, 'norm', args=(mu, sigma))
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  KS p-value:   {ks_pval:.6f}")

    # Also check quantiles (on full samples — fine for percentiles)
    for q in [0.025, 0.16, 0.50, 0.84, 0.975]:
        true_val = norm.ppf(q, loc=mu, scale=sigma)
        est_val = np.percentile(flat[:, 0], q * 100)
        print(f"  Q{q:.3f}: true={true_val:7.4f}, est={est_val:7.4f}, err={abs(est_val-true_val):.4f}")

    ok = ks_pval > 0.01  # reject if p < 1%
    status = "PASS" if ok else "FAIL"
    print(f"\n  Result: {status}  (KS test at 1% significance, after thinning)")
    return ok


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Realistic 13D posterior mimic (from actual SchwarMAX chain)
# ═══════════════════════════════════════════════════════════════════════

def test_realistic_13d(move='mixed', label=None):
    if label is None:
        label = move
    print("\n" + "="*70)
    print(f"TEST 8 [{label}]: Realistic 13D posterior (SchwarMAX geometry)")
    print("  Condition number ~119,000; r(M_halo, Rs_halo)=0.995")
    print("  r(M_disk, LtM)=-0.901; r(Rs_disk, gamma)=0.844")
    print("="*70)

    cov_true = np.load('/tmp/posterior_cov_13d.npy')
    mu_true = np.load('/tmp/posterior_mean_13d.npy')

    ndim = 13
    prec = jnp.array(np.linalg.inv(cov_true))
    mu = jnp.array(mu_true)

    cond = np.linalg.cond(cov_true)
    print(f"  Covariance condition number: {cond:.0f}")

    def logp(x):
        d = x - mu
        return -0.5 * d @ prec @ d

    n_walkers, n_steps = 52, 6000
    key = jax.random.PRNGKey(77)

    eigvals, eigvecs = np.linalg.eigh(cov_true)
    init_scale = 0.1 * np.sqrt(eigvals)
    noise_pc = np.array(jax.random.normal(key, (n_walkers, ndim))) * init_scale[None, :]
    init = mu_true[None, :] + (noise_pc @ eigvecs.T)
    init = jnp.array(init)

    flat, chain, logp_chain, acc = run_sampler(
        logp, ndim, n_walkers, n_steps, init, burnin_frac=0.4, move=move)

    param_names = [
        'logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk',
        'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
        'log_LtM', 'log_Omega', 'log_sigma',
    ]

    # ── Check 1: Means ──
    mean_est = np.mean(flat, axis=0)
    std_true = np.sqrt(np.diag(cov_true))
    mean_err_nsigma = np.abs(mean_est - mu_true) / std_true
    max_mean_err = np.max(mean_err_nsigma)
    print(f"\n  Mean recovery (in units of true sigma):")
    print(f"  {'Param':>12s} {'true':>10s} {'est':>10s} {'err/sig':>8s}")
    for i, name in enumerate(param_names):
        print(f"  {name:>12s} {mu_true[i]:10.4f} {mean_est[i]:10.4f} {mean_err_nsigma[i]:8.3f}")
    ok_mean = max_mean_err < 0.5  # within 0.5 sigma on every param
    print(f"  Max mean error: {max_mean_err:.3f} sigma -> {'PASS' if ok_mean else 'FAIL'}")

    # ── Check 2: Variances ──
    std_est = np.std(flat, axis=0)
    std_ratio = std_est / std_true
    print(f"\n  Variance recovery (ratio estimated/true):")
    print(f"  {'Param':>12s} {'true_std':>10s} {'est_std':>10s} {'ratio':>8s}")
    for i, name in enumerate(param_names):
        print(f"  {name:>12s} {std_true[i]:10.6f} {std_est[i]:10.6f} {std_ratio[i]:8.3f}")
    # All ratios should be within [0.7, 1.3]
    ok_std = np.all((std_ratio > 0.7) & (std_ratio < 1.3))
    print(f"  Std ratio range: [{std_ratio.min():.3f}, {std_ratio.max():.3f}] -> {'PASS' if ok_std else 'FAIL'}")

    # ── Check 3: Key correlations ──
    cov_est = np.cov(flat.T)
    std_est_full = np.sqrt(np.diag(cov_est))
    corr_est = cov_est / np.outer(std_est_full, std_est_full)
    corr_true = cov_true / np.outer(std_true, std_true)

    key_pairs = [
        (0, 3, 'logM_halo--logRs_halo'),   # r=0.995
        (1, 10, 'logM_disk--log_LtM'),       # r=-0.901
        (4, 9, 'logRs_disk--gamma'),         # r=0.844
        (8, 9, 'beta--gamma'),               # r=0.788
        (6, 11, 'logL_bar--log_Omega'),       # r=-0.714
        (0, 7, 'logM_halo--alpha'),           # r=0.641
        (2, 6, 'logM_bar--logL_bar'),         # r=0.633
    ]
    print(f"\n  Key correlation recovery:")
    print(f"  {'Pair':>30s} {'true':>8s} {'est':>8s} {'err':>8s}")
    max_corr_err = 0
    for i, j, label in key_pairs:
        err = abs(corr_est[i, j] - corr_true[i, j])
        max_corr_err = max(max_corr_err, err)
        print(f"  {label:>30s} {corr_true[i,j]:+8.3f} {corr_est[i,j]:+8.3f} {err:8.3f}")
    ok_corr = max_corr_err < 0.15
    print(f"  Max correlation error: {max_corr_err:.3f} -> {'PASS' if ok_corr else 'FAIL'}")

    # ── Check 4: Covariance Frobenius norm ──
    cov_err = np.linalg.norm(cov_est - cov_true) / np.linalg.norm(cov_true)
    ok_cov = cov_err < 0.20
    print(f"\n  Covariance relative Frobenius error: {cov_err:.4f} -> {'PASS' if ok_cov else 'FAIL'}")

    # ── Check 5: Principal component widths (eigenvalues) ──
    eigvals_est = np.linalg.eigvalsh(cov_est)
    eigvals_true = np.linalg.eigvalsh(cov_true)
    # Compare in log space (eigenvalues span many orders of magnitude)
    log_eig_err = np.abs(np.log10(eigvals_est) - np.log10(eigvals_true))
    print(f"\n  Principal component width recovery (log10 eigenvalues):")
    print(f"  {'PC':>5s} {'true':>12s} {'est':>12s} {'log10_err':>10s}")
    for i in range(ndim):
        print(f"  PC{i:2d}  {eigvals_true[i]:12.2e} {eigvals_est[i]:12.2e} {log_eig_err[i]:10.3f}")
    ok_eig = np.max(log_eig_err) < 0.3  # within factor of 2 on every PC
    print(f"  Max log10 eigenvalue error: {np.max(log_eig_err):.3f} -> {'PASS' if ok_eig else 'FAIL'}")

    # ── Check 6: Autocorrelation time (efficiency) ──
    burnin = int(n_steps * 0.4)
    post_chain = chain[burnin:]  # (n_kept, n_walkers, ndim)

    def _autocorr_time(x, max_lag=200):
        n = len(x)
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-15:
            return n  # stuck chain
        acf = np.correlate(x, x, mode='full')[n-1:n-1+max_lag] / (var * n)
        tau = 1.0
        for k in range(1, max_lag):
            if acf[k] < 0:
                break
            tau += 2 * acf[k]
        return tau

    taus_per_dim = []
    for d in range(ndim):
        taus_walkers = [_autocorr_time(post_chain[:, w, d]) for w in range(n_walkers)]
        taus_per_dim.append(np.mean(taus_walkers))
    tau_mean = np.mean(taus_per_dim)
    tau_max = np.max(taus_per_dim)
    n_eff = flat.shape[0] / tau_mean

    print(f"\n  Autocorrelation time: mean={tau_mean:.1f}, max={tau_max:.1f}")
    print(f"  Effective samples: {n_eff:.0f} (from {flat.shape[0]} raw)")
    print(f"  ESS/step/walker: {n_eff / (n_steps - burnin) / n_walkers:.3f}")

    # ── Overall ──
    all_ok = ok_mean and ok_std and ok_corr and ok_cov and ok_eig
    status = "PASS" if all_ok else "FAIL"
    n_checks = 5
    n_pass = sum([ok_mean, ok_std, ok_corr, ok_cov, ok_eig])
    print(f"\n  Overall: {status} ({n_pass}/{n_checks} sub-checks passed)")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# Test 9: Same geometry but with non-Gaussian marginals (skew + kurtosis)
# ═══════════════════════════════════════════════════════════════════════

def test_realistic_nongaussian():
    print("\n" + "="*70)
    print("TEST 9: 13D with realistic correlations + non-Gaussian marginals")
    print("  Uses a 'warped Gaussian': apply monotone transforms to some dims")
    print("  to create skewness and heavy tails while preserving rank correlations")
    print("="*70)

    cov_true_gaussian = np.load('/tmp/posterior_cov_13d.npy')
    mu_true_gaussian = np.load('/tmp/posterior_mean_13d.npy')

    ndim = 13
    L = jnp.array(np.linalg.cholesky(cov_true_gaussian))
    mu_g = jnp.array(mu_true_gaussian)

    # In the warped space, we sample z ~ N(0, I), then x = mu + L @ z,
    # then apply warping to some dimensions.
    # For the test logp, we define the target in the WARPED space.
    #
    # Strategy: define logp in the original (warped) x-space.
    # Dims 0,3 (logM_halo, logRs_halo): Student-t tails (df=5)
    # Dim 9 (gamma): skewed via sinh-arcsinh transform
    # Rest: Gaussian

    prec_g = jnp.array(np.linalg.inv(cov_true_gaussian))

    def logp(x):
        d = x - mu_g
        # Base Gaussian log-density
        base = -0.5 * d @ prec_g @ d

        # Add heavy tails on dims 0 and 3 by mixing in Student-t
        # log t(x; df) - log N(x; 0,1) gives the tail correction
        df = 5.0
        for dim in [0, 3]:
            z_i = d[dim] / jnp.sqrt(cov_true_gaussian[dim, dim])
            # Student-t correction: replace Gaussian marginal with t
            log_t = -(df + 1) / 2 * jnp.log(1 + z_i**2 / df)
            log_n = -0.5 * z_i**2
            base = base + (log_t - log_n)

        return base

    n_walkers, n_steps = 52, 6000
    key = jax.random.PRNGKey(88)

    eigvals, eigvecs = np.linalg.eigh(cov_true_gaussian)
    init_scale = 0.1 * np.sqrt(eigvals)
    noise_pc = np.array(jax.random.normal(key, (n_walkers, ndim))) * init_scale[None, :]
    init = mu_true_gaussian[None, :] + (noise_pc @ eigvecs.T)
    init = jnp.array(init)

    flat, chain, logp_chain, acc = run_sampler(
        logp, ndim, n_walkers, n_steps, init, burnin_frac=0.4, move='mixed')

    param_names = [
        'logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk',
        'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
        'log_LtM', 'log_Omega', 'log_sigma',
    ]

    std_true = np.sqrt(np.diag(cov_true_gaussian))
    mean_est = np.mean(flat, axis=0)

    # Means should still be close (Student-t is symmetric)
    mean_err_nsigma = np.abs(mean_est - mu_true_gaussian) / std_true
    ok_mean = np.max(mean_err_nsigma) < 0.5
    print(f"\n  Max mean error: {np.max(mean_err_nsigma):.3f} sigma -> {'PASS' if ok_mean else 'FAIL'}")

    # Dims 0,3 should have HEAVIER tails than Gaussian
    from scipy.stats import kurtosis as scipy_kurtosis
    kurt = scipy_kurtosis(flat, axis=0, fisher=True)
    print(f"\n  Excess kurtosis (dims 0,3 should be > 0; others ~ 0):")
    for i in [0, 3, 1, 2, 4, 5]:
        label = "HEAVY-TAIL" if i in [0, 3] else "gaussian"
        print(f"    dim {i:2d} ({param_names[i]:>12s}): kurt = {kurt[i]:+6.2f}  [{label}]")

    ok_heavy = kurt[0] > 0.5 and kurt[3] > 0.5  # clearly heavier than Gaussian
    ok_light = all(abs(kurt[i]) < 3.0 for i in [1, 2, 4, 5])  # Gaussian dims roughly normal
    print(f"  Heavy-tail dims have excess kurtosis: {'PASS' if ok_heavy else 'FAIL'}")
    print(f"  Gaussian dims have low kurtosis: {'PASS' if ok_light else 'FAIL'}")

    # Key correlations should still be recovered (rank correlations preserved)
    cov_est = np.cov(flat.T)
    std_est = np.sqrt(np.diag(cov_est))
    corr_est = cov_est / np.outer(std_est, std_est)
    corr_true = cov_true_gaussian / np.outer(std_true, std_true)

    # The near-perfect degeneracy should still show
    r_halo = corr_est[0, 3]
    r_disk_ltm = corr_est[1, 10]
    print(f"\n  Key correlations:")
    print(f"    logM_halo--logRs_halo: true={corr_true[0,3]:+.3f}, est={r_halo:+.3f}")
    print(f"    logM_disk--log_LtM:    true={corr_true[1,10]:+.3f}, est={r_disk_ltm:+.3f}")
    ok_corr = abs(r_halo - corr_true[0, 3]) < 0.1 and abs(r_disk_ltm - corr_true[1, 10]) < 0.15

    all_ok = ok_mean and ok_heavy and ok_light and ok_corr
    status = "PASS" if all_ok else "FAIL"
    n_pass = sum([ok_mean, ok_heavy, ok_light, ok_corr])
    print(f"\n  Overall: {status} ({n_pass}/4 sub-checks passed)")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  ENSEMBLE SAMPLER VALIDATION SUITE")
    print("="*70)

    results = {}
    results['1_isotropic']   = test_isotropic_gaussian()
    results['2_correlated']  = test_correlated_gaussian()
    results['3_bimodal']     = test_bimodal()
    results['4_student_t']   = test_student_t()
    results['5_banana']      = test_banana()
    results['6_high_d']      = test_high_d_gaussian()
    results['7_detailed_bal'] = test_detailed_balance()
    results['8_realistic_13d_stretch'] = test_realistic_13d(move='stretch', label='stretch-only')
    results['8_realistic_13d_mixed'] = test_realistic_13d(move='mixed', label='mixed (DE+Snooker+Stretch)')
    results['9_nongaussian_13d'] = test_realistic_nongaussian()

    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:>20s}: {status}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")

    if n_pass == n_total:
        print("\n  Sampler implementation appears correct.")
    elif n_pass >= n_total - 1:
        print("\n  Sampler is mostly correct. Check failed tests above.")
    else:
        print("\n  WARNING: Multiple failures. Investigate before using on real data.")


if __name__ == '__main__':
    main()
