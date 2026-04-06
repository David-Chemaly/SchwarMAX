"""
BlackJAX Random Walk Metropolis sampler for SchwarMAX.

2 free parameters: log_light_to_mass, log_Omega_bar
Fixed: logM_halo=11.88, logRs_halo=log10(19.2)

Features:
- Chunked inference loop with progress bar and checkpointing
- Resume from checkpoint if Colab disconnects
"""

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import pickle
import time
import os

from likelihoods import logl_fixed_potential_bootstrap

# ── Load data ────────────────────────────────────────────────────────
with open("mock_bar_disc_XY_withRot.pkl", "rb") as f:
    dict_data = pickle.load(f)

num_Vbin = dict_data["total_bins"]
dict_phi = dict_data["dict_phi"]

# ── Fixed params ─────────────────────────────────────────────────────
FIXED_LOGM_HALO = 11.88
FIXED_LOGRS_HALO = jnp.log10(19.2).item()

# ── Log-probability (matches your log_prob exactly) ──────────────────
def logdensity_fn(theta):
    """
    theta: jnp.array([log_light_to_mass, log_Omega_bar])
    Returns scalar log-probability.
    """
    params = jnp.array([FIXED_LOGM_HALO, FIXED_LOGRS_HALO, theta[0], theta[1]])
    ll, _meff = logl_fixed_potential_bootstrap(params, dict_phi, dict_data, num_Vbin)
    # Replace non-finite with -inf (JAX-compatible, no Python if/else)
    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


# ── Flat priors (reject outside bounds via -inf) ─────────────────────
BOUNDS_LO = jnp.array([-2.0, -1.0])   # [log_ltm_lo, log_Omega_lo]
BOUNDS_HI = jnp.array([ 2.0,  1.0])   # [log_ltm_hi, log_Omega_hi]

def logdensity_with_prior(theta):
    """logdensity + uniform prior (bounds check)."""
    in_bounds = jnp.all((theta >= BOUNDS_LO) & (theta <= BOUNDS_HI))
    return jax.lax.cond(in_bounds, lambda: logdensity_fn(theta), lambda: -jnp.inf)


# ── BlackJAX RMH setup ──────────────────────────────────────────────
# Step sizes for the random walk proposal (tune for ~25-40% acceptance)
sigma = jnp.array([0.05, 0.05])  # [log_ltm, log_Omega_bar]

rmh = blackjax.rmh(logdensity_with_prior, blackjax.mcmc.random_walk.normal(sigma))

# ── Chunked inference with progress + checkpointing ─────────────────
CHECKPOINT_FILE = "blackjax_checkpoint.pkl"

def run_chunk(rng_key, initial_state, num_steps):
    """Run a chunk of MCMC steps using lax.scan (no Python overhead)."""
    @jax.jit
    def one_step(state, rng_key):
        state, info = rmh.step(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_steps)
    final_state, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return final_state, states, infos


def save_checkpoint(state, rng_key, all_samples, all_accepted, step_count, phase):
    """Save checkpoint to disk for crash recovery."""
    checkpoint = {
        'state': jax.tree.map(np.array, state),
        'rng_key': np.array(rng_key),
        'samples': np.array(all_samples),
        'accepted': np.array(all_accepted),
        'step_count': step_count,
        'phase': phase,
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint = pickle.load(f)
    # Convert back to JAX arrays
    checkpoint['state'] = jax.tree.map(jnp.array, checkpoint['state'])
    checkpoint['rng_key'] = jnp.array(checkpoint['rng_key'])
    print(f"Resumed from checkpoint: {checkpoint['step_count']} steps done (phase: {checkpoint['phase']})")
    return checkpoint


def run_with_checkpointing(rng_key, initial_state, num_steps, chunk_size=50, phase="sampling"):
    """
    Run MCMC in chunks of `chunk_size` steps.
    After each chunk: print progress, save checkpoint.
    """
    all_samples = []
    all_accepted = []
    state = initial_state
    steps_done = 0

    while steps_done < num_steps:
        this_chunk = min(chunk_size, num_steps - steps_done)
        rng_key, chunk_key = jax.random.split(rng_key)

        t0 = time.time()
        state, states, infos = run_chunk(chunk_key, state, this_chunk)
        # Force computation to finish before timing
        jax.block_until_ready(state)
        dt = time.time() - t0

        # Collect samples
        positions = np.array(states.position)  # (chunk_size, 2)
        accepted = np.array(infos.is_accepted)
        all_samples.append(positions)
        all_accepted.append(accepted)

        steps_done += this_chunk
        acc_rate = accepted.mean()
        per_step = dt / this_chunk

        print(f"  [{phase}] {steps_done:5d}/{num_steps}  "
              f"acc={acc_rate:.0%}  {per_step:.2f}s/step  "
              f"theta=[{float(state.position[0]):.4f}, {float(state.position[1]):.4f}]  "
              f"logL={float(state.logdensity):.1f}")

        # Checkpoint
        combined_samples = np.concatenate(all_samples, axis=0)
        combined_accepted = np.concatenate(all_accepted, axis=0)
        save_checkpoint(state, rng_key, combined_samples, combined_accepted, steps_done, phase)

    samples = np.concatenate(all_samples, axis=0)
    accepted = np.concatenate(all_accepted, axis=0)
    return state, samples, accepted


# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_WARMUP = 200
    NUM_SAMPLES = 1000
    CHUNK_SIZE = 50      # checkpoint every 50 steps

    # Check for existing checkpoint
    checkpoint = load_checkpoint()

    if checkpoint is not None and checkpoint['phase'] == 'sampling':
        # Resume production sampling
        state = checkpoint['state']
        rng_key = checkpoint['rng_key']
        steps_done = checkpoint['step_count']
        prev_samples = checkpoint['samples']
        prev_accepted = checkpoint['accepted']

        remaining = NUM_SAMPLES - steps_done
        if remaining > 0:
            print(f"Resuming production: {remaining} steps remaining...")
            state, new_samples, new_accepted = run_with_checkpointing(
                rng_key, state, remaining, CHUNK_SIZE, phase="sampling"
            )
            samples = np.concatenate([prev_samples, new_samples], axis=0)
            accepted = np.concatenate([prev_accepted, new_accepted], axis=0)
        else:
            print("Sampling already complete!")
            samples = prev_samples
            accepted = prev_accepted

    else:
        # Start from scratch (or resume warmup)
        rng_key = jax.random.PRNGKey(0)

        if checkpoint is not None and checkpoint['phase'] == 'warmup':
            state = checkpoint['state']
            rng_key = checkpoint['rng_key']
            warmup_remaining = NUM_WARMUP - checkpoint['step_count']
            print(f"Resuming warmup: {warmup_remaining} steps remaining...")
        else:
            # Fresh start
            initial_position = jnp.array([0.0, 0.0])  # [log_ltm, log_Omega_bar]
            state = rmh.init(initial_position)
            warmup_remaining = NUM_WARMUP

        # Warmup
        if warmup_remaining > 0:
            print(f"Running {warmup_remaining} warmup steps...")
            rng_key, warmup_key = jax.random.split(rng_key)
            state, warmup_samples, warmup_accepted = run_with_checkpointing(
                warmup_key, state, warmup_remaining, CHUNK_SIZE, phase="warmup"
            )
            print(f"Warmup acceptance rate: {warmup_accepted.mean():.0%}\n")

        # Production
        print(f"Running {NUM_SAMPLES} production steps...")
        rng_key, sample_key = jax.random.split(rng_key)
        state, samples, accepted = run_with_checkpointing(
            sample_key, state, NUM_SAMPLES, CHUNK_SIZE, phase="sampling"
        )

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\nAcceptance rate: {accepted.mean():.0%}")
    print("\n── Posterior summary ──")
    labels = ['log_light_to_mass', 'log_Omega_bar']
    for i, label in enumerate(labels):
        s = samples[:, i]
        print(f"  {label:25s}: mean={s.mean():.4f}  std={s.std():.4f}  "
              f"95% CI=[{np.percentile(s, 2.5):.4f}, {np.percentile(s, 97.5):.4f}]")

    # Save final samples
    with open("blackjax_samples.pkl", "wb") as f:
        pickle.dump({'samples': samples, 'accepted': accepted, 'labels': labels}, f)
    print("\nSaved to blackjax_samples.pkl")
