"""
NumPyro ESS (Ensemble Slice Sampling) for the 13D Schwarzschild model.
Drop-in replacement for the emcee sampler with progress tracking and checkpointing.

Usage:
    python run_numpyro_ess.py

Resume from checkpoint:
    Set RESUME = True below.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, ESS
import pickle
import time
import os

# ── Your project imports ──
from likelihoods_bar import logl_angular_input_bootstrap

# ── Configuration ──
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX'

NWALKERS = 30          # must be >= 2 * ndim = 26
NSTEPS = 1000
WARMUP = 0             # ESS doesn't use warmup in the HMC sense, but numpyro requires >= 0
CHECKPOINT_EVERY = 50  # save every N steps
RESUME = False

CHECKPOINT_FILE = os.path.join(path, 'ess_checkpoint_0401_gal2.pkl')
OUTPUT_CSV = os.path.join(path, 'test_posterior_ess_0401_gal2_Nbins1000.csv')
OUTPUT_CHAIN_PKL = os.path.join(path, 'test_posterior_WholeChain_ess_0401_gal2_Nbins1000.pkl')
OUTPUT_LOGPROB_PKL = os.path.join(path, 'test_posterior_logprob_ess_0401_gal2_Nbins1000.pkl')

# ── Load data ──
with open(os.path.join(path, 'mock_Nbody_bar_XY_withRot.pkl'), 'rb') as f:
    dict_data_raw = pickle.load(f)

# Convert dict_data to JAX arrays (do this once)
dict_data = {k: jnp.array(v) if isinstance(v, np.ndarray) else v
             for k, v in dict_data_raw.items()}

num_Vbin = int(dict_data['total_bins'])

# ── Best-fit / ground truth ──
res = np.load(os.path.join(path, 'minimise_0329_gal2_Nbins1000.npy'))

# ── Prior bounds ──
prior_low = jnp.array([
    res[0] - 3, res[1] - 3, res[2] - 3, res[3] - 1,
    res[4] - 1, res[5] - 1, res[6] - 1,
    0., 0., 0., -2., 0., -5.,
])
prior_high = jnp.array([
    res[0] + 3, res[1] + 3, res[2] + 3, res[3] + 1,
    res[4] + 1, res[5] + 1, res[6] + 1,
    float(jnp.pi), float(jnp.pi / 2), float(jnp.pi), 2., 2., -0.5,
])

ndim = 13
param_names = [
    'logM_halo', 'logM_disk', 'logM_bar', 'logRs_halo', 'logRs_disk',
    'logHs_disk', 'logL_bar', 'alpha', 'beta', 'gamma',
    'log_light_to_mass_ratio', 'log_Omega', 'log_sigma',
]


# ── NumPyro model ──
def model():
    theta = numpyro.sample(
        'theta',
        dist.Uniform(prior_low, prior_high).to_event(1),
    )
    params = [theta[i] for i in range(ndim)]
    ll = logl_angular_input_bootstrap(params, dict_data, num_Vbin)
    ll = jnp.where(jnp.isfinite(ll), ll, -1e30)
    numpyro.factor('loglik', ll)


# ── Initial positions (same perturbation scheme as emcee) ──
def make_init_positions(rng_key):
    p0 = res
    pos = np.zeros((NWALKERS, ndim))
    np.random.seed(20256)
    pos[:, :7] = p0[:7] + np.random.uniform(-0.2, 0.2, (NWALKERS, 7))
    pos[:, 7:9] = p0[7:9] + np.random.uniform(-0.1, 0.1, (NWALKERS, 2))
    pos[:, 7:9] = np.clip(pos[:, 7:9], 0, [float(jnp.pi), float(jnp.pi / 2)])
    pos[:, 9] = p0[9] + np.random.uniform(-0.3, 0.3, NWALKERS)
    pos[:, 10] = p0[10] + np.random.uniform(-0.5, 0.5, NWALKERS)
    pos[:, 11] = p0[11] + np.random.uniform(-0.4, 0.4, NWALKERS)
    pos[:, 12] = p0[12] + np.random.uniform(-0.4, 0.4, NWALKERS)
    return jnp.array(pos)


# ── Run in chunks for progress + checkpointing ──
def run_ess():
    rng_key = jax.random.PRNGKey(42)
    init_pos = make_init_positions(rng_key)
    init_params = {'theta': init_pos}

    # Resume from checkpoint if requested
    start_chunk = 0
    all_samples = []
    all_logprob = []

    if RESUME and os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE, 'rb') as f:
            ckpt = pickle.load(f)
        init_params = ckpt['last_state']
        start_chunk = ckpt['chunks_done']
        all_samples = ckpt['all_samples']
        all_logprob = ckpt['all_logprob']
        print(f"  Loaded {start_chunk * CHECKPOINT_EVERY} steps, continuing...")

    n_chunks = NSTEPS // CHECKPOINT_EVERY
    total_time = 0.0

    for chunk_idx in range(start_chunk, n_chunks):
        chunk_start = time.time()

        kernel = ESS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=0 if chunk_idx > 0 else 200,  # warmup only on first chunk
            num_samples=CHECKPOINT_EVERY,
            num_chains=NWALKERS,
            chain_method='vectorized',
            progress_bar=True,
        )

        rng_key, sub_key = jax.random.split(rng_key)
        mcmc.run(sub_key, init_params=init_params)

        # Extract samples: (num_chains, num_samples, ndim)
        samples = mcmc.get_samples(group_by_chain=True)['theta']

        # Get log probability
        extra = mcmc.get_extra_fields(group_by_chain=True)
        logprob = extra.get('potential_energy', None)
        if logprob is not None:
            logprob = -logprob  # numpyro stores negative log prob as potential energy

        all_samples.append(np.array(samples))
        if logprob is not None:
            all_logprob.append(np.array(logprob))

        # Update init_params for next chunk: use last sample from each chain
        init_params = {'theta': samples[:, -1, :]}

        chunk_time = time.time() - chunk_start
        total_time += chunk_time
        steps_done = (chunk_idx + 1) * CHECKPOINT_EVERY
        avg_per_step = total_time / steps_done

        print(f"\n[Chunk {chunk_idx + 1}/{n_chunks}] "
              f"Steps {steps_done}/{NSTEPS} | "
              f"Chunk time: {chunk_time:.1f}s | "
              f"Avg: {avg_per_step:.2f}s/step | "
              f"ETA: {avg_per_step * (NSTEPS - steps_done) / 60:.1f}min")

        # Checkpoint
        ckpt = {
            'last_state': init_params,
            'chunks_done': chunk_idx + 1,
            'all_samples': all_samples,
            'all_logprob': all_logprob,
        }
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"  Checkpoint saved to {CHECKPOINT_FILE}")

    # ── Concatenate all chunks ──
    # all_samples entries: (num_chains, CHECKPOINT_EVERY, ndim)
    full_chain = np.concatenate(all_samples, axis=1)  # (num_chains, total_steps, ndim)
    print(f"\nFull chain shape: {full_chain.shape}")

    if all_logprob:
        full_logprob = np.concatenate(all_logprob, axis=1)  # (num_chains, total_steps)
    else:
        full_logprob = None

    # ── Save outputs (same format as emcee) ──
    # Flat samples (discard 100 steps burn-in)
    discard = 100
    if full_chain.shape[1] > discard:
        flat_samples = full_chain[:, discard:, :].reshape(-1, ndim)
    else:
        flat_samples = full_chain.reshape(-1, ndim)

    import pandas as pd
    pd.DataFrame(flat_samples, columns=param_names).to_csv(OUTPUT_CSV, index=False)
    print(f"Flat samples saved to {OUTPUT_CSV}")

    # Full chain pickle (shape: num_steps, num_chains, ndim — same as emcee)
    chain_emcee_format = np.transpose(full_chain, (1, 0, 2))  # (steps, walkers, ndim)
    with open(OUTPUT_CHAIN_PKL, 'wb') as f:
        pickle.dump(chain_emcee_format, f)
    print(f"Full chain saved to {OUTPUT_CHAIN_PKL}")

    if full_logprob is not None:
        logprob_emcee_format = full_logprob.T  # (steps, walkers)
        with open(OUTPUT_LOGPROB_PKL, 'wb') as f:
            pickle.dump(logprob_emcee_format, f)
        print(f"Log prob saved to {OUTPUT_LOGPROB_PKL}")

    return full_chain, full_logprob


if __name__ == '__main__':
    run_ess()
