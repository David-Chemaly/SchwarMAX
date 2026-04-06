"""
BlackJAX Adaptive Tempered SMC sampler for 13D SchwarMAX inference.

SMC evolves N_PARTICLES from the prior → posterior through adaptive tempering:

  β=0:  p(θ) ∝ prior                     (particles spread across prior volume)
  β→1:  p(θ) ∝ prior × likelihood^β      (particles concentrate onto posterior)

Each tempering step:
  1. Increase β adaptively (keep ~50% of particles "alive")
  2. Reweight particles by likelihood^(β_new - β_old)
  3. Resample — duplicate high-weight particles, kill low-weight ones
  4. Diversify — run N_MCMC random-walk steps to spread duplicates apart

Key parameters:
  N_PARTICLES  — number of posterior samples you get at the end (~200-500 for 13D)
  N_MCMC       — diversification moves per tempering step (5-10 typical)
  Tempering steps — determined automatically (~20-40 for a 13D problem)

Features:
  - Adaptive temperature schedule (ESS-based)
  - Progress printing + checkpointing after each tempering step
  - Resume from checkpoint
  - Evidence (marginal likelihood) computed as byproduct
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from blackjax.smc.resampling import systematic
import pickle
import time
import os
import pandas as pd

from likelihoods_bar import logl_angular_input_bootstrap

# ── Configuration ────────────────────────────────────────────────────
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

N_PARTICLES = 200           # posterior sample count (increase for better coverage)
N_MCMC = 5                  # RMH diversification moves per tempering step
MAX_TEMPERING_STEPS = 200   # safety limit

CHECKPOINT_FILE = os.path.join(path, 'smc_checkpoint_0401_gal2.pkl')
OUTPUT_FILE = os.path.join(path, 'smc_results_0401_gal2.pkl')
OUTPUT_CSV = os.path.join(path, 'smc_posterior_0401_gal2.csv')

# ── Load data ────────────────────────────────────────────────────────
with open(os.path.join(path, 'mock_Nbody_bar_XY_withRot.pkl'), 'rb') as f:
    dict_data_raw = pickle.load(f)

dict_data = {k: jnp.array(v) if isinstance(v, np.ndarray) else v
             for k, v in dict_data_raw.items()}
num_Vbin = int(dict_data['total_bins'])

# ── Best-fit point (for RMH proposal scaling) ────────────────────────
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


# ── Log-prior (uniform) ─────────────────────────────────────────────
def logprior_fn(theta):
    in_bounds = jnp.all((theta >= BOUNDS_LO) & (theta <= BOUNDS_HI))
    log_vol = jnp.sum(jnp.log(BOUNDS_HI - BOUNDS_LO))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)


# ── Log-likelihood ───────────────────────────────────────────────────
def loglikelihood_fn(theta):
    params = [theta[i] for i in range(NDIM)]
    ll = logl_angular_input_bootstrap(params, dict_data, num_Vbin)
    return jnp.where(jnp.isfinite(ll), ll, -1e30)


# ── RMH proposal step sizes ─────────────────────────────────────────
# Scale: ~5% of prior width for mass/scale params, smaller for angles
rmh_sigma = jnp.array([
    0.1, 0.1, 0.1, 0.05,     # logM_halo, logM_disk, logM_bar, logRs_halo
    0.05, 0.05, 0.05,         # logRs_disk, logHs_disk, logL_bar
    0.05, 0.05, 0.1,          # alpha, beta, gamma
    0.1, 0.05,                # log_ltm, log_Omega
    0.1,                      # log_sigma
])

# ── Build SMC kernel ─────────────────────────────────────────────────
# Bind the proposal into the kernel so mcmc_parameters can be empty.
# (BlackJAX 1.4 has a bug where callable values in mcmc_parameters
#  crash in from_mcmc.unshared_parameters_and_step_fn)
_raw_kernel = blackjax.additive_step_random_walk.build_kernel()
_normal_proposal = blackjax.mcmc.random_walk.normal(rmh_sigma)

def _bound_kernel(rng_key, state, logdensity_fn):
    return _raw_kernel(rng_key, state, logdensity_fn, _normal_proposal)

def _bound_init(position, logdensity_fn):
    return blackjax.additive_step_random_walk.init(position, logdensity_fn)

smc_algo = blackjax.adaptive_tempered_smc(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    mcmc_step_fn=_bound_kernel,
    mcmc_init_fn=_bound_init,
    mcmc_parameters={},
    resampling_fn=systematic,
    target_ess=0.5,
    num_mcmc_steps=N_MCMC,
)
smc_init = smc_algo.init
smc_step = smc_algo.step


# ── Sample from prior ───────────────────────────────────────────────
def sample_from_prior(rng_key, n):
    return jax.random.uniform(rng_key, shape=(n, NDIM),
                              minval=BOUNDS_LO, maxval=BOUNDS_HI)


# ── Checkpointing ───────────────────────────────────────────────────
def save_checkpoint(state, rng_key, step, log_evidence, history):
    ckpt = {
        'particles': np.array(state.particles),
        'weights': np.array(state.weights),
        'tempering_param': float(state.tempering_param),
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
          f"beta={ckpt['tempering_param']:.4f}, "
          f"log_evidence={ckpt['log_evidence']:.2f}")
    return ckpt


# ── Main SMC loop ────────────────────────────────────────────────────
def run_smc(rng_key, state, log_evidence=0.0, history=None, start_step=0):
    if history is None:
        history = []

    step = start_step
    beta_prev = float(state.tempering_param)
    print(f"\nSMC: {N_PARTICLES} particles, {N_MCMC} MCMC moves/step, {NDIM}D")
    print(f"{'Step':>5} {'beta':>8} {'d_beta':>8} {'ESS':>8} {'logZ':>10} {'time':>7} {'accept':>8}")
    print("-" * 65)

    while state.tempering_param < 1.0 and step < MAX_TEMPERING_STEPS:
        t0 = time.time()
        rng_key, step_key = jax.random.split(rng_key)

        state, info = smc_step(step_key, state)
        jax.block_until_ready(state)
        dt = time.time() - t0

        log_evidence += info.log_likelihood_increment
        beta_new = float(state.tempering_param)
        d_beta = beta_new - beta_prev
        beta_prev = beta_new
        ess = float(1.0 / jnp.sum(state.weights ** 2))

        # Acceptance rate from the RMH diversification moves
        accept_rate = float(jnp.mean(info.update_info.acceptance_rate))

        step += 1
        step_info = {
            'step': step, 'beta': beta_new, 'd_beta': d_beta,
            'ess': ess, 'log_evidence': float(log_evidence),
            'time': dt, 'accept_rate': accept_rate,
        }
        history.append(step_info)

        accept_str = f"{accept_rate:.3f}"
        print(f"{step:5d} {beta_new:8.4f} {d_beta:8.4f} {ess:8.1f} "
              f"{float(log_evidence):10.2f} {dt:6.1f}s {accept_str:>8}")

        save_checkpoint(state, rng_key, step, log_evidence, history)

    return state, log_evidence, history, rng_key


# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(42)

    ckpt = load_checkpoint()

    if ckpt is not None and ckpt['tempering_param'] < 1.0:
        # Resume
        particles = jnp.array(ckpt['particles'])
        state = blackjax.adaptive_tempered_smc.init(particles)
        state = state._replace(
            tempering_param=ckpt['tempering_param'],
            weights=jnp.array(ckpt['weights']),
        )
        rng_key = jnp.array(ckpt['rng_key'])
        state, log_evidence, history, rng_key = run_smc(
            rng_key, state,
            log_evidence=ckpt['log_evidence'],
            history=ckpt['history'],
            start_step=ckpt['step'],
        )
    else:
        # Fresh start
        rng_key, init_key = jax.random.split(rng_key)
        initial_particles = sample_from_prior(init_key, N_PARTICLES)
        state = smc_init(initial_particles)
        print(f"Initialized {N_PARTICLES} particles from prior in {NDIM}D")

        state, log_evidence, history, rng_key = run_smc(rng_key, state)

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SMC completed in {len(history)} tempering steps")
    print(f"Log-evidence (log Z): {log_evidence:.2f}")
    print(f"Total logL evals: ~{len(history) * N_PARTICLES * N_MCMC}")

    particles = np.array(state.particles)
    weights = np.array(state.weights)

    # Resample to get unweighted samples
    rng_key, resample_key = jax.random.split(rng_key)
    indices = jax.random.choice(resample_key, N_PARTICLES,
                                shape=(N_PARTICLES,), p=state.weights)
    unweighted_samples = particles[np.array(indices)]

    print(f"\n── Posterior summary ──")
    print(f"{'Parameter':>30s} {'mean':>10s} {'std':>10s} {'2.5%':>10s} {'97.5%':>10s} {'truth':>10s}")
    print("-" * 82)
    for i, name in enumerate(param_names):
        s = unweighted_samples[:, i]
        truth = res[i]
        print(f"{name:>30s} {s.mean():10.4f} {s.std():10.4f} "
              f"{np.percentile(s, 2.5):10.4f} {np.percentile(s, 97.5):10.4f} {truth:10.4f}")

    # Save results
    results = {
        'particles': particles,
        'weights': weights,
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

    # Also save as CSV (flat samples)
    pd.DataFrame(unweighted_samples, columns=param_names).to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved to {OUTPUT_CSV}")
