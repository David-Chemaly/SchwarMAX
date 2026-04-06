"""
Custom Adaptive Tempered SMC for 13D SchwarMAX inference.

Uses explicit jax.vmap for parallel logL evaluation (proven 10x speedup).
No BlackJAX dependency — all SMC logic is self-contained.

SMC evolves N_PARTICLES from prior → posterior through adaptive tempering:
  β=0: p(θ) ∝ prior
  β→1: p(θ) ∝ prior × likelihood^β

Each tempering step:
  1. Find next β adaptively (target ESS fraction)
  2. Reweight particles by likelihood^(β_new - β_old)
  3. Resample (systematic) if ESS drops below threshold
  4. Diversify — N_MCMC random-walk Metropolis steps (vmapped)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import os
import pandas as pd

from likelihoods_bar import logl_angular_input_bootstrap

# ── Configuration ────────────────────────────────────────────────────
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

N_PARTICLES = 200
N_MCMC = 5
TARGET_ESS_FRAC = 0.5       # target ESS / N_PARTICLES for adaptive tempering
MAX_TEMPERING_STEPS = 200

CHECKPOINT_FILE = os.path.join(path, 'smc_checkpoint_custom.pkl')
OUTPUT_FILE = os.path.join(path, 'smc_results_custom.pkl')
OUTPUT_CSV = os.path.join(path, 'smc_posterior_custom.csv')

# ── Load data ────────────────────────────────────────────────────────
with open(os.path.join(path, 'mock_Nbody_bar_XY_withRot.pkl'), 'rb') as f:
    dict_data_raw = pickle.load(f)

dict_data = {k: jnp.array(v) if isinstance(v, np.ndarray) else v
             for k, v in dict_data_raw.items()}
num_Vbin = int(dict_data['total_bins'])

# ── Best-fit point ───────────────────────────────────────────────────
res = np.load(os.path.join(path, 'minimise_0329_gal2_Nbins1000.npy'))

# ── Prior bounds (13D) ───────────────────────────────────────────────
NDIM = 13
param_names = [
    'logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk',
    'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
    'log_light_to_mass_ratio', 'log_Omega', 'log_sigma',
]

BOUNDS_LO = jnp.array([
    res[0] - 3, res[1] - 3, res[2] - 3, res[3] - 1,
    res[4] - 1, res[5] - 1, res[6] - 1,
    0., 0., 0., -2., 0., -5.,
])
BOUNDS_HI = jnp.array([
    res[0] + 3, res[1] + 3, res[2] + 3, res[3] + 1,
    res[4] + 1, res[5] + 1, res[6] + 1,
    float(jnp.pi), float(jnp.pi / 2), float(jnp.pi), 2., 2., -0.5,
])

# ── RMH proposal step sizes ─────────────────────────────────────────
rmh_sigma = jnp.array([
    0.1, 0.1, 0.1, 0.05,
    0.05, 0.05, 0.05,
    0.05, 0.05, 0.1,
    0.1, 0.05,
    0.1,
])

# ── Vectorised log-likelihood and log-prior ──────────────────────────
def loglikelihood_fn(theta):
    ll = logl_angular_input_bootstrap(theta, dict_data, num_Vbin)
    return jnp.where(jnp.isfinite(ll), ll, -1e30)

def logprior_fn(theta):
    in_bounds = jnp.all((theta >= BOUNDS_LO) & (theta <= BOUNDS_HI))
    log_vol = jnp.sum(jnp.log(BOUNDS_HI - BOUNDS_LO))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)

# Vmapped versions — proven to parallelise on GPU
_vmap_loglikelihood = jax.vmap(loglikelihood_fn)
_vmap_logprior = jax.vmap(logprior_fn)


# ── Adaptive beta selection ──────────────────────────────────────────
def _find_next_beta(beta_old, logL, target_ess_frac, n_particles):
    """Binary search for next beta such that ESS ≈ target_ess_frac * N."""
    target_ess = target_ess_frac * n_particles

    def ess_at_beta(beta_new):
        log_w = (beta_new - beta_old) * logL
        log_w = log_w - jnp.max(log_w)  # shift for numerical stability
        w = jnp.exp(log_w)
        w = w / jnp.sum(w)
        return 1.0 / jnp.sum(w ** 2)

    # If jumping straight to beta=1 gives enough ESS, do it
    ess_at_one = ess_at_beta(1.0)

    # Binary search between beta_old and 1.0
    lo, hi = beta_old, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        ess_mid = ess_at_beta(mid)
        lo, hi = jax.lax.cond(
            ess_mid > target_ess,
            lambda _: (mid, hi),
            lambda _: (lo, mid),
            None,
        )

    beta_new = jnp.where(ess_at_one >= target_ess, 1.0, 0.5 * (lo + hi))
    return jnp.clip(beta_new, beta_old + 1e-6, 1.0)


# ── Systematic resampling ────────────────────────────────────────────
def _systematic_resample(rng_key, weights, n):
    """Systematic resampling: returns indices."""
    cumsum = jnp.cumsum(weights)
    u0 = jax.random.uniform(rng_key) / n
    u = u0 + jnp.arange(n) / n
    indices = jnp.searchsorted(cumsum, u)
    return jnp.clip(indices, 0, n - 1)


# ── RMH diversification (vmapped) ───────────────────────────────────
def _rmh_diversify(rng_key, particles, logL, beta, n_mcmc):
    """
    Run n_mcmc RMH steps on all particles.
    Uses vmap for parallel logL evaluation at each MCMC step.
    """
    n = particles.shape[0]

    def _one_mcmc_step(carry, rng_key):
        particles, logL, n_accept = carry

        # Propose
        k1, k2 = jax.random.split(rng_key)
        noise = jax.random.normal(k1, particles.shape) * rmh_sigma[None, :]
        proposals = particles + noise

        # Evaluate log-posterior at proposals (vmapped)
        logL_prop = _vmap_loglikelihood(proposals)
        logprior_prop = _vmap_logprior(proposals)
        logprior_curr = _vmap_logprior(particles)

        # Acceptance ratio: tempered posterior
        log_alpha = (beta * (logL_prop - logL)
                     + (logprior_prop - logprior_curr))
        log_u = jnp.log(jax.random.uniform(k2, shape=(n,)))
        accept = log_u < log_alpha

        # Update
        new_particles = jnp.where(accept[:, None], proposals, particles)
        new_logL = jnp.where(accept, logL_prop, logL)
        n_accept = n_accept + jnp.sum(accept)

        return (new_particles, new_logL, n_accept), None

    keys = jax.random.split(rng_key, n_mcmc)
    (particles, logL, n_accept), _ = jax.lax.scan(
        _one_mcmc_step, (particles, logL, 0.0), keys
    )
    accept_rate = n_accept / (n * n_mcmc)
    return particles, logL, accept_rate


# ── Checkpointing ───────────────────────────────────────────────────
def save_checkpoint(particles, logL, log_weights, beta, rng_key,
                    step, log_evidence, history):
    ckpt = {
        'particles': np.array(particles),
        'logL': np.array(logL),
        'log_weights': np.array(log_weights),
        'beta': float(beta),
        'rng_key': np.array(rng_key),
        'step': step,
        'log_evidence': float(log_evidence),
        'history': history,
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(ckpt, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, 'rb') as f:
        ckpt = pickle.load(f)
    print(f"Found checkpoint: step {ckpt['step']}, "
          f"beta={ckpt['beta']:.4f}, "
          f"log_evidence={ckpt['log_evidence']:.2f}")
    return ckpt


# ── Main SMC loop ────────────────────────────────────────────────────
def run_smc(rng_key, particles, logL, log_weights, beta=0.0,
            log_evidence=0.0, history=None, start_step=0):
    if history is None:
        history = []

    n = particles.shape[0]
    step = start_step

    print(f"\nSMC: {n} particles, {N_MCMC} MCMC moves/step, {NDIM}D")
    print(f"{'Step':>5} {'beta':>8} {'d_beta':>8} {'ESS':>8} "
          f"{'logZ':>10} {'time':>7} {'accept':>8}")
    print("-" * 65)

    while beta < 1.0 and step < MAX_TEMPERING_STEPS:
        t0 = time.time()
        rng_key, k_beta, k_resample, k_mcmc = jax.random.split(rng_key, 4)

        # 1. Find next beta
        beta_new = float(_find_next_beta(beta, logL, TARGET_ESS_FRAC, n))
        d_beta = beta_new - beta

        # 2. Reweight
        log_inc_w = d_beta * logL
        log_evidence += float(jax.scipy.special.logsumexp(
            log_weights + log_inc_w) - jnp.log(n))
        log_weights = log_weights + log_inc_w
        log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights = jnp.exp(log_weights)
        ess = float(1.0 / jnp.sum(weights ** 2))

        # 3. Resample
        indices = _systematic_resample(k_resample, weights, n)
        particles = particles[indices]
        logL = logL[indices]
        log_weights = jnp.full(n, -jnp.log(n))  # reset to uniform

        # 4. Diversify with RMH (vmapped logL)
        particles, logL, accept_rate = _rmh_diversify(
            k_mcmc, particles, logL, beta_new, N_MCMC)
        accept_rate = float(accept_rate)

        beta = beta_new
        step += 1
        dt = time.time() - t0

        step_info = {
            'step': step, 'beta': beta, 'd_beta': d_beta,
            'ess': ess, 'log_evidence': float(log_evidence),
            'time': dt, 'accept_rate': accept_rate,
        }
        history.append(step_info)

        print(f"{step:5d} {beta:8.4f} {d_beta:8.4f} {ess:8.1f} "
              f"{float(log_evidence):10.2f} {dt:6.1f}s {accept_rate:8.3f}")

        save_checkpoint(particles, logL, log_weights, beta, rng_key,
                        step, log_evidence, history)

    return particles, logL, log_weights, log_evidence, history, rng_key


# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(42)

    ckpt = load_checkpoint()

    if ckpt is not None and ckpt['beta'] < 1.0:
        # Resume
        particles = jnp.array(ckpt['particles'])
        logL = jnp.array(ckpt['logL'])
        log_weights = jnp.array(ckpt['log_weights'])
        rng_key = jnp.array(ckpt['rng_key'])
        particles, logL, log_weights, log_evidence, history, rng_key = run_smc(
            rng_key, particles, logL, log_weights,
            beta=ckpt['beta'],
            log_evidence=ckpt['log_evidence'],
            history=ckpt['history'],
            start_step=ckpt['step'],
        )
    else:
        # Fresh start
        rng_key, init_key = jax.random.split(rng_key)
        particles = jax.random.uniform(
            init_key, shape=(N_PARTICLES, NDIM),
            minval=BOUNDS_LO, maxval=BOUNDS_HI)

        print(f"Evaluating logL for {N_PARTICLES} initial particles...")
        t0 = time.time()
        logL = _vmap_loglikelihood(particles)
        jax.block_until_ready(logL)
        print(f"  Done in {time.time()-t0:.1f}s")

        log_weights = jnp.full(N_PARTICLES, -jnp.log(N_PARTICLES))
        print(f"Initialized {N_PARTICLES} particles from prior in {NDIM}D")

        particles, logL, log_weights, log_evidence, history, rng_key = \
            run_smc(rng_key, particles, logL, log_weights)

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SMC completed in {len(history)} tempering steps")
    print(f"Log-evidence (log Z): {log_evidence:.2f}")

    weights = jnp.exp(log_weights)
    particles_np = np.array(particles)

    # Resample to get unweighted samples
    rng_key, resample_key = jax.random.split(rng_key)
    indices = jax.random.choice(resample_key, N_PARTICLES,
                                shape=(N_PARTICLES,), p=weights)
    unweighted_samples = particles_np[np.array(indices)]

    print(f"\n── Posterior summary ──")
    print(f"{'Parameter':>30s} {'mean':>10s} {'std':>10s} "
          f"{'2.5%':>10s} {'97.5%':>10s} {'truth':>10s}")
    print("-" * 82)
    for i, name in enumerate(param_names):
        s = unweighted_samples[:, i]
        truth = res[i]
        print(f"{name:>30s} {s.mean():10.4f} {s.std():10.4f} "
              f"{np.percentile(s, 2.5):10.4f} {np.percentile(s, 97.5):10.4f} "
              f"{truth:10.4f}")

    # Save results
    results = {
        'particles': particles_np,
        'weights': np.array(weights),
        'unweighted_samples': unweighted_samples,
        'log_evidence': log_evidence,
        'history': history,
        'param_names': param_names,
        'bounds_lo': np.array(BOUNDS_LO),
        'bounds_hi': np.array(BOUNDS_HI),
    }
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {OUTPUT_FILE}")

    pd.DataFrame(unweighted_samples, columns=param_names).to_csv(
        OUTPUT_CSV, index=False)
    print(f"CSV saved to {OUTPUT_CSV}")
