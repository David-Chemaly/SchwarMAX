"""
JAX Ensemble MCMC sampler (Goodman & Weare 2010 stretch move)
for 13D SchwarMAX inference.

Uses jax.vmap for parallel logL evaluation within each split-phase.
No covariance estimation needed — affine invariant by construction.

Algorithm per step:
  1. Split L walkers into two halves S0, S1
  2. Phase 1: update S0 using S1 as companions (L/2 parallel evals)
  3. Phase 2: update S1 using updated S0 (L/2 parallel evals)

Usage:
    python run_ensemble_mcmc.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm

from likelihoods_bar import logl_angular_input_bootstrap

# ── Configuration ────────────────────────────────────────────────────
path = '/Users/hanyuan/Dropbox/python_script/SchwarMAX/'

N_WALKERS = 40             # must be even and >= 2*NDIM
N_STEPS = 3000
CHECKPOINT_EVERY = 50
BURNIN = 500
STRETCH_A = 2.0            # stretch move scale parameter

CHECKPOINT_FILE = os.path.join(path, 'ensemble_checkpoint.pkl')
OUTPUT_FILE = os.path.join(path, 'ensemble_results.pkl')
OUTPUT_CSV = os.path.join(path, 'ensemble_posterior.csv')

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

_vmap_logdensity = jax.vmap(logdensity_fn)


# ── Move weights (emcee-style mixture) ──────────────────────────────
# Same mix as the user's emcee config:
#   70% DEMove, 20% DESnookerMove, 10% StretchMove
MOVE_WEIGHTS = jnp.array([0.7, 0.9, 1.0])  # cumulative thresholds

DE_GAMMA = 2.38 / jnp.sqrt(2.0 * NDIM)    # ter Braak (2006) optimal
DE_NOISE = 1e-5                              # small jitter for ergodicity
SNOOKER_GAMMA = 1.7                          # ter Braak & Vrugt (2008)


# ── Stretch move ────────────────────────────────────────────────────
def _sample_z(rng_key, n, a=STRETCH_A):
    """Sample Z from g(z) ∝ 1/sqrt(z) on [1/a, a]."""
    u = jax.random.uniform(rng_key, shape=(n,))
    return ((a - 1.0) * u + 1.0) ** 2 / a


def _stretch_propose(rng_key, active, complement, ndim):
    """Stretch move (Goodman & Weare 2010).
    Returns (proposals, log_factors)."""
    n_half = active.shape[0]
    k1, k2 = jax.random.split(rng_key)

    idx = jax.random.randint(k1, (n_half,), 0, complement.shape[0])
    companions = complement[idx]
    z = _sample_z(k2, n_half)

    proposals = companions + z[:, None] * (active - companions)
    log_factors = (ndim - 1) * jnp.log(z)
    return proposals, log_factors


# ── DE move (ter Braak 2006) ────────────────────────────────────────
def _de_propose(rng_key, active, complement, ndim,
                gamma=None, sigma=DE_NOISE):
    """Differential Evolution move.
    Y = X + gamma * (c[r1] - c[r2]) + noise
    Symmetric proposal → log_factor = 0."""
    n_half = active.shape[0]
    n_comp = complement.shape[0]
    k1, k2, k3 = jax.random.split(rng_key, 3)

    if gamma is None:
        gamma = 2.38 / jnp.sqrt(2.0 * ndim)

    # Pick two distinct companions per walker
    r1 = jax.random.randint(k1, (n_half,), 0, n_comp)
    r2 = jax.random.randint(k2, (n_half,), 0, n_comp - 1)
    # Avoid r1 == r2: shift r2 up by 1 where r2 >= r1
    r2 = jnp.where(r2 >= r1, r2 + 1, r2)

    diff = complement[r1] - complement[r2]
    noise = sigma * jax.random.normal(k3, active.shape)

    proposals = active + gamma * diff + noise
    log_factors = jnp.zeros(n_half)  # symmetric proposal
    return proposals, log_factors


# ── DE-Snooker move (ter Braak & Vrugt 2008) ───────────────────────
def _snooker_propose(rng_key, active, complement, ndim,
                     gamma=SNOOKER_GAMMA):
    """DE-Snooker move: project differential onto line through walker and pivot.
    Has Jacobian factor (||Y-z0|| / ||X-z0||)^(D-1)."""
    n_half = active.shape[0]
    n_comp = complement.shape[0]
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Pick 3 companions: z0 (pivot), z1, z2 (for differential)
    r0 = jax.random.randint(k1, (n_half,), 0, n_comp)
    r12 = jax.random.randint(k2, (n_half, 2), 0, n_comp)
    r1, r2 = r12[:, 0], r12[:, 1]

    z0 = complement[r0]   # pivot
    z1 = complement[r1]
    z2 = complement[r2]

    # Direction: active → pivot
    direction = active - z0                           # (n_half, ndim)
    dist = jnp.linalg.norm(direction, axis=1, keepdims=True)
    dist_safe = jnp.maximum(dist, 1e-30)
    d_hat = direction / dist_safe                     # unit vector

    # Differential projected onto direction
    diff = z1 - z2                                     # (n_half, ndim)
    proj = jnp.sum(diff * d_hat, axis=1, keepdims=True)  # scalar projection

    # Propose along the direction
    proposals = active + gamma * proj * d_hat

    # Jacobian: (||Y - z0|| / ||X - z0||)^(D-1)
    new_dist = jnp.linalg.norm(proposals - z0, axis=1)
    log_factors = (ndim - 1) * jnp.log(new_dist / dist_safe.squeeze())

    return proposals, log_factors


# ── Mixed move: per-walker random selection ─────────────────────────
def _mixed_move_half(rng_key, active, active_logp, complement):
    """Update active walkers using a mixture of DE, Snooker, and Stretch moves.

    Each walker independently draws which move to use, then all proposals
    are evaluated in a single vmapped logL batch.
    """
    n_half = active.shape[0]
    k_sel, k_de, k_snk, k_str, k_acc = jax.random.split(rng_key, 5)

    # Generate proposals from ALL three moves (compute all, select per-walker)
    prop_de, lf_de = _de_propose(k_de, active, complement, NDIM)
    prop_snk, lf_snk = _snooker_propose(k_snk, active, complement, NDIM)
    prop_str, lf_str = _stretch_propose(k_str, active, complement, NDIM)

    # Per-walker move selection
    u_sel = jax.random.uniform(k_sel, (n_half,))
    use_de = u_sel < MOVE_WEIGHTS[0]                                    # 0.0–0.7
    use_snk = (u_sel >= MOVE_WEIGHTS[0]) & (u_sel < MOVE_WEIGHTS[1])   # 0.7–0.9
    # else: stretch                                                      # 0.9–1.0

    # Select proposal and log_factor per walker
    proposals = jnp.where(use_de[:, None], prop_de,
                    jnp.where(use_snk[:, None], prop_snk, prop_str))
    log_factors = jnp.where(use_de, lf_de,
                    jnp.where(use_snk, lf_snk, lf_str))

    # Evaluate all proposals in one vmapped batch
    prop_logp = _vmap_logdensity(proposals)

    # Metropolis acceptance
    log_accept = log_factors + prop_logp - active_logp
    log_u = jnp.log(jax.random.uniform(k_acc, (n_half,)))
    accept = log_u < log_accept

    new_active = jnp.where(accept[:, None], proposals, active)
    new_logp = jnp.where(accept, prop_logp, active_logp)
    n_accepted = jnp.sum(accept)

    return new_active, new_logp, n_accepted


def ensemble_step(rng_key, positions, logp):
    """One full ensemble step: update both halves with mixed moves.

    Args:
        rng_key: JAX PRNG key
        positions: (N_WALKERS, NDIM)
        logp: (N_WALKERS,)

    Returns:
        new_positions, new_logp, n_accepted
    """
    n_half = N_WALKERS // 2
    k1, k2 = jax.random.split(rng_key)

    s0, s1 = positions[:n_half], positions[n_half:]
    lp0, lp1 = logp[:n_half], logp[n_half:]

    # Phase 1: update S0 using S1
    s0, lp0, acc0 = _mixed_move_half(k1, s0, lp0, s1)

    # Phase 2: update S1 using updated S0
    s1, lp1, acc1 = _mixed_move_half(k2, s1, lp1, s0)

    new_positions = jnp.concatenate([s0, s1], axis=0)
    new_logp = jnp.concatenate([lp0, lp1], axis=0)
    n_accepted = acc0 + acc1

    return new_positions, new_logp, n_accepted


# ── Initial positions ────────────────────────────────────────────────
def make_init_positions(rng_key):
    """Start walkers spread across a broad region around the best-fit.

    Uses ~30% of the prior width per parameter so walkers explore widely
    from the start — the starting point may not be the true mode.
    """
    p0 = jnp.array(res)
    # Spread = 30% of prior half-width per parameter
    init_spread = 0.3 * (BOUNDS_HI - BOUNDS_LO) / 2.0
    noise = jax.random.normal(rng_key, shape=(N_WALKERS, NDIM)) * init_spread[None, :]
    positions = p0[None, :] + noise
    positions = jnp.clip(positions, BOUNDS_LO[None, :], BOUNDS_HI[None, :])
    return positions


# ── Checkpointing ───────────────────────────────────────────────────
def save_checkpoint(all_positions, all_logprob, step, rng_key):
    ckpt = {
        'all_samples': [np.array(s) for s in all_positions],
        'all_logprob': [np.array(lp) for lp in all_logprob],
        'step': step,
        'rng_key': np.array(rng_key),
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(ckpt, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, 'rb') as f:
        ckpt = pickle.load(f)
    n_steps = len(ckpt['all_samples'])
    n_walkers = ckpt['all_samples'][0].shape[0]
    print(f"Found checkpoint: {n_steps} steps, {n_walkers} walkers")
    return ckpt


# ── Main loop ────────────────────────────────────────────────────────
def run_ensemble(resume=True):
    rng_key = jax.random.PRNGKey(42)

    ckpt = load_checkpoint() if resume else None

    if ckpt is not None:
        all_positions = ckpt['all_samples']
        all_logprob = ckpt['all_logprob']
        start_step = ckpt['step']
        rng_key = jnp.array(ckpt['rng_key'])

        positions = jnp.array(all_positions[-1])
        logp = jnp.array(all_logprob[-1])

        print(f"Resumed from step {start_step}, {len(all_positions)} samples")
    else:
        rng_key, init_key = jax.random.split(rng_key)
        positions = make_init_positions(init_key)

        print(f"Initialising {N_WALKERS} walkers...")
        n_half = N_WALKERS // 2
        logp = jnp.concatenate([
            _vmap_logdensity(positions[:n_half]),
            _vmap_logdensity(positions[n_half:]),
        ])
        print(f"  Init done. logP: mean={float(jnp.mean(logp)):.1f}, "
              f"max={float(jnp.max(logp)):.1f}")

        all_positions = [np.array(positions)]
        all_logprob = [np.array(logp)]
        start_step = 0

    print(f"\nEnsemble MCMC: {N_WALKERS} walkers, {N_STEPS} steps, {NDIM}D")
    print(f"  Stretch parameter a={STRETCH_A}")
    print(f"  Starting from step {start_step + 1}")

    pbar = tqdm(range(start_step + 1, N_STEPS + 1), desc="Ensemble", unit="step")
    for step in pbar:
        rng_key, step_key = jax.random.split(rng_key)

        positions, logp, n_accepted = ensemble_step(step_key, positions, logp)

        all_positions.append(np.array(positions))
        all_logprob.append(np.array(logp))

        if step % 10 == 0:
            acc = float(n_accepted) / N_WALKERS
            pbar.set_postfix(
                logP=f"{float(jnp.mean(logp)):.1f}",
                acc=f"{acc:.3f}",
                best=f"{float(jnp.max(logp)):.1f}",
            )

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_positions, all_logprob, step, rng_key)

    # ── Results ──────────────────────────────────────────────────────
    chain = np.stack(all_positions, axis=0)  # (N_STEPS+1, N_WALKERS, NDIM)
    logprob = np.stack(all_logprob, axis=0)
    print(f"\nChain shape: {chain.shape}")

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
    run_ensemble()
