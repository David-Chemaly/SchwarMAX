"""
BlackJAX parallel adaptive RMH chains for 13D SchwarMAX inference.

Runs N_CHAINS independent Random Walk Metropolis chains via jax.vmap.
Adapts the proposal covariance periodically using collected samples
(Roberts & Rosenthal 2001: scale = 2.38 / sqrt(D)).

Strategy: use small initial proposal for high acceptance (~30-40%),
delay covariance adaptation until chains have explored enough,
then adapt using only recent samples.

Usage:
    python run_blackjax_mcmc_parallel.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import pickle
import os
import pandas as pd
from tqdm import tqdm

from likelihoods_bar import logl_angular_input_bootstrap

# ── Configuration ────────────────────────────────────────────────────
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

N_CHAINS = 20              # number of parallel chains (must fit in GPU vmap)
N_STEPS = 5000             # total MCMC steps per chain
CHECKPOINT_EVERY = 50
BURNIN = 600
ADAPT_EVERY = 100          # re-estimate covariance every N steps
ADAPT_AFTER = 500          # delay adaptation until chains have explored

CHECKPOINT_FILE = os.path.join(path, 'mcmc_checkpoint_parallel.pkl')
OUTPUT_FILE = os.path.join(path, 'mcmc_results_parallel.pkl')
OUTPUT_CSV = os.path.join(path, 'mcmc_posterior_parallel.csv')

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
    0., 0., 0., -2., 0., -1.,
])
BOUNDS_HI = jnp.array([
    res[0] + 3, res[1] + 3, res[2] + 3, res[3] + 1,
    res[4] + 1, res[5] + 1, res[6] + 1,
    float(jnp.pi), float(jnp.pi / 2), float(jnp.pi), 2., 2., 2.,
])

# ── Log-posterior ────────────────────────────────────────────────────
def logdensity_fn(theta):
    in_bounds = jnp.all((theta >= BOUNDS_LO) & (theta <= BOUNDS_HI))
    log_vol = jnp.sum(jnp.log(BOUNDS_HI - BOUNDS_LO))
    logprior = jnp.where(in_bounds, -log_vol, -jnp.inf)

    ll = logl_angular_input_bootstrap(theta, dict_data, num_Vbin)
    ll = jnp.where(jnp.isfinite(ll), ll, -1e30)

    return logprior + ll

# ── Initial proposal (small for ~30-40% acceptance) ──────────────────
OPTIMAL_SCALE = 2.38 / np.sqrt(NDIM)  # Roberts & Rosenthal 2001

rmh_sigma_init = jnp.array([
    0.02, 0.02, 0.02, 0.01,
    0.01, 0.01, 0.01,
    0.01, 0.01, 0.02,
    0.02, 0.01,
    0.02,
])

# ── Helper: build sampler from proposal ──────────────────────────────
def build_sampler(proposal):
    rw = blackjax.additive_step_random_walk(logdensity_fn, proposal)
    return jax.vmap(rw.step)

# ── Initial positions ────────────────────────────────────────────────
def make_init_positions(rng_key):
    """Start chains near best-fit with small perturbation."""
    p0 = jnp.array(res)
    noise = jax.random.normal(rng_key, shape=(N_CHAINS, NDIM)) * rmh_sigma_init[None, :]
    positions = p0[None, :] + noise
    positions = jnp.clip(positions, BOUNDS_LO[None, :], BOUNDS_HI[None, :])
    return positions


# ── Checkpointing ───────────────────────────────────────────────────
def save_checkpoint(all_samples, all_logprob, step, rng_key, adapt_count=0,
                    proposal_L=None, scale_factor=1.0):
    ckpt = {
        'all_samples': [np.array(s) for s in all_samples],
        'all_logprob': [np.array(lp) for lp in all_logprob],
        'step': step,
        'rng_key': np.array(rng_key),
        'adapt_count': adapt_count,
        'proposal_L': np.array(proposal_L) if proposal_L is not None else None,
        'scale_factor': scale_factor,
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(ckpt, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, 'rb') as f:
        ckpt = pickle.load(f)
    n_steps = len(ckpt['all_samples'])
    n_chains = ckpt['all_samples'][0].shape[0]
    print(f"Found checkpoint: {n_steps} steps, {n_chains} chains, "
          f"adapt_count={ckpt.get('adapt_count', 0)}")
    return ckpt


# ── Main loop ────────────────────────────────────────────────────────
def run_mcmc(resume=True):
    rng_key = jax.random.PRNGKey(42)

    ckpt = load_checkpoint() if resume else None

    if ckpt is not None:
        # ── Resume from checkpoint ────────────────────────────────
        all_positions = ckpt['all_samples']
        all_logprob = ckpt['all_logprob']
        start_step = ckpt['step']
        rng_key = jnp.array(ckpt['rng_key'])
        adapt_count = ckpt.get('adapt_count', 0)
        proposal_L = ckpt.get('proposal_L', None)
        scale_factor = ckpt.get('scale_factor', 1.0)

        # Reconstruct states from last saved positions/logprob
        last_positions = jnp.array(all_positions[-1])
        rw_init = blackjax.additive_step_random_walk(
            logdensity_fn, blackjax.mcmc.random_walk.normal(rmh_sigma_init))
        states = jax.vmap(rw_init.init)(last_positions)

        # Rebuild proposal from saved Cholesky factor
        if proposal_L is not None:
            proposal = blackjax.mcmc.random_walk.normal(
                jnp.array(proposal_L * scale_factor))
        else:
            proposal = blackjax.mcmc.random_walk.normal(rmh_sigma_init)
        vmap_step = build_sampler(proposal)

        print(f"Resumed from step {start_step}, {len(all_positions)} samples, "
              f"adapt_count={adapt_count}, scale_factor={scale_factor:.3f}")
    else:
        # ── Fresh start ───────────────────────────────────────────
        rng_key, init_key = jax.random.split(rng_key)
        positions = make_init_positions(init_key)

        print(f"Initialising {N_CHAINS} chains...")
        rw_init = blackjax.additive_step_random_walk(
            logdensity_fn, blackjax.mcmc.random_walk.normal(rmh_sigma_init))
        states = jax.vmap(rw_init.init)(positions)
        print(f"  Init done.")

        vmap_step = build_sampler(blackjax.mcmc.random_walk.normal(rmh_sigma_init))
        all_positions = []
        all_logprob = []
        start_step = 0
        adapt_count = 0
        proposal_L = None
        scale_factor = 1.0

    # Track recent acceptance (last ADAPT_EVERY steps)
    recent_accepts = []

    print(f"\nAdaptive RMH: {N_CHAINS} chains, {N_STEPS} steps, {NDIM}D")
    print(f"  Adapt covariance every {ADAPT_EVERY} steps after step {ADAPT_AFTER}")
    print(f"  Using all samples from step {ADAPT_AFTER} onward for covariance")
    print(f"  Optimal scale: {OPTIMAL_SCALE:.3f}")
    print(f"  Starting from step {start_step + 1}")

    pbar = tqdm(range(start_step + 1, N_STEPS + 1), desc="RMH", unit="step")
    for step in pbar:
        rng_key, step_key = jax.random.split(rng_key)
        keys = jax.random.split(step_key, N_CHAINS)

        states, infos = vmap_step(keys, states)

        all_positions.append(np.array(states.position))
        all_logprob.append(np.array(states.logdensity))

        step_accept = float(jnp.mean(infos.acceptance_rate))
        recent_accepts.append(step_accept)
        if len(recent_accepts) > ADAPT_EVERY:
            recent_accepts.pop(0)

        # ── Adapt proposal covariance ────────────────────────────
        if step >= ADAPT_AFTER and step % ADAPT_EVERY == 0:
            # Use samples from ADAPT_AFTER onward (skip burn-in)
            samples_post_burnin = np.stack(all_positions[ADAPT_AFTER:], axis=0)
            flat = samples_post_burnin.reshape(-1, NDIM)
            n_used = flat.shape[0]

            if n_used > 2 * NDIM:
                cov = np.cov(flat.T)
                diag = np.diag(np.diag(cov))
                cov_reg = 0.8 * cov + 0.2 * diag
                proposal_cov = OPTIMAL_SCALE**2 * cov_reg
                try:
                    L = np.linalg.cholesky(proposal_cov)
                    proposal_L = L

                    # Adjust scale based on recent acceptance
                    recent_acc = np.mean(recent_accepts)
                    if recent_acc < 0.15:
                        scale_factor *= 0.8
                    elif recent_acc > 0.35:
                        scale_factor *= 1.2
                    scale_factor = np.clip(scale_factor, 0.1, 5.0)

                    proposal = blackjax.mcmc.random_walk.normal(
                        jnp.array(L * scale_factor))
                    vmap_step = build_sampler(proposal)
                    adapt_count += 1

                    tqdm.write(
                        f"  [Adapt #{adapt_count} at step {step}] "
                        f"recent_acc={recent_acc:.3f}, scale={scale_factor:.3f}, "
                        f"n={n_used}, "
                        f"cov diag={np.sqrt(np.diag(proposal_cov))[:3].round(4)}")
                except np.linalg.LinAlgError:
                    tqdm.write(f"  [Adapt #{adapt_count+1} at step {step}] "
                               f"Cholesky failed, keeping current proposal")
            else:
                tqdm.write(f"  [Adapt at step {step}] "
                           f"Too few samples ({n_used}), skipping")

        if step % 10 == 0:
            recent_acc = np.mean(recent_accepts) if recent_accepts else 0.0
            pbar.set_postfix(
                logP=f"{float(jnp.mean(states.logdensity)):.1f}",
                acc=f"{recent_acc:.3f}",
                adapt=adapt_count,
            )

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_positions, all_logprob, step, rng_key,
                            adapt_count, proposal_L, scale_factor)

    # ── Results ──────────────────────────────────────────────────────
    chain = np.stack(all_positions, axis=0)  # (N_STEPS, N_CHAINS, NDIM)
    logprob = np.stack(all_logprob, axis=0)
    print(f"\nChain shape: {chain.shape}")

    # Flat samples after burn-in
    if chain.shape[0] > BURNIN:
        flat_samples = chain[BURNIN:].reshape(-1, NDIM)
    else:
        flat_samples = chain.reshape(-1, NDIM)

    print(f"\n── Posterior summary ({flat_samples.shape[0]} samples) ──")
    print(f"{'Parameter':>30s} {'mean':>10s} {'std':>10s} "
          f"{'2.5%':>10s} {'97.5%':>10s} {'truth':>10s}")
    print("-" * 82)
    for i, name in enumerate(param_names):
        s = flat_samples[:, i]
        truth = res[i]
        print(f"{name:>30s} {s.mean():10.4f} {s.std():10.4f} "
              f"{np.percentile(s, 2.5):10.4f} {np.percentile(s, 97.5):10.4f} "
              f"{truth:10.4f}")

    # Save
    results = {
        'chain': chain,
        'logprob': logprob,
        'flat_samples': flat_samples,
        'param_names': param_names,
    }
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {OUTPUT_FILE}")

    pd.DataFrame(flat_samples, columns=param_names).to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved to {OUTPUT_CSV}")

    return chain, logprob


if __name__ == '__main__':
    run_mcmc()
