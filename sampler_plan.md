# Sampler Plan for SchwarMAX 13D Inference

## Problem Diagnosis

The adaptive RMH chains freeze after ~100 steps because:
1. Initial diagonal proposal gives ~15% acceptance (okay)
2. First covariance adaptation at step 100 uses burn-in contaminated samples with wrong correlations
3. After bad adaptation, acceptance drops to <3%, chains freeze
4. All subsequent adaptations use frozen chain positions — death spiral

Key evidence from the `mcmc_checkpoint_gal2_0402.pkl` run:
- Steps 0-50: 15.4% acceptance, 17% unique positions — chains are exploring
- Steps 50-100: 5.0% acceptance, 6.9% unique — already degrading
- Steps 200-400: 2.6% unique — nearly frozen
- Steps 1000+: <1% unique — completely stuck
- `log_sigma` is worst: early std=0.052 vs converged spread=0.007 (7.8x mismatch)

## Solution 1: Conservative RMH with Delayed Adaptation

**Key changes to `run_blackjax_mcmc_parallel.py`:**

1. **Smaller initial proposal** — reduce `rmh_sigma_init` by ~2x so initial acceptance is ~30-40% instead of 15%. Slower exploration but chains stay mobile longer.

2. **Delay first adaptation** — set `ADAPT_AFTER=500` (was 50). Let chains explore for 500 steps (~2.7 hours) before first covariance update. This gives 500 × 16 = 8000 samples from the actual posterior region.

3. **Use only recent samples for covariance** — sliding window of last `ADAPT_WINDOW=200` steps. After step 500, the chains should be in the converged region and recent samples reflect the true posterior geometry.

4. **Reduce OPTIMAL_SCALE slightly** — try `1.5/sqrt(D)` instead of `2.38/sqrt(D)`. The 2.38 is optimal for Gaussian targets; our posterior has boundaries and non-Gaussian features. More conservative scale → higher acceptance → chains stay mobile.

5. **Acceptance-based scale adjustment** — if recent acceptance < 15%, shrink proposal by 0.8x. If > 35%, grow by 1.2x. Simple feedback to prevent freezing.

**Expected behavior:**
- Steps 0-500: ~30% acceptance with small diagonal proposal, slow but steady exploration
- Step 500: first adaptation with 8000 diverse samples → correct correlations
- Steps 500+: ~20-25% acceptance with adapted proposal, proper sampling

**Implementation**: Modify existing `run_blackjax_mcmc_parallel.py`.

## Solution 2: JAX Ensemble Sampler (Stretch Move)

**Write `run_ensemble_mcmc.py`** — a JAX implementation of the Goodman & Weare (2010) stretch move.

**Algorithm per step:**
1. Split `L` walkers into two groups S0, S1 of size `L/2`
2. Phase 1: For each walker in S0, draw companion from S1, propose stretch move, accept/reject → `L/2` parallel logL evals via vmap
3. Phase 2: Same for S1 using updated S0 → another `L/2` parallel logL evals
4. Total: 2 batches of `L/2` parallel evals per step

**Stretch move proposal:**
```
Z ~ g(z) ∝ 1/sqrt(z),  z ∈ [1/a, a],  a=2
Y = X_j + Z * (X_k - X_j)
Accept with prob min(1, Z^(D-1) * pi(Y) / pi(X_k))
```

**Key design decisions:**
- `L = 32` walkers (>2×D=26 minimum). With 16 per batch, matches our GPU sweet spot.
- `a = 2.0` (standard, single tuning parameter)
- No covariance estimation needed — affine invariant by construction
- Prior bounds enforced by returning `-inf` for out-of-bounds proposals (same as current `logdensity_fn`)

**GPU parallelism:**
- Phase 1: `jax.vmap` over 16 logL evaluations → ~14s
- Phase 2: `jax.vmap` over 16 logL evaluations → ~14s
- Total: ~28s per full step, 32 walker updates
- Compare: current RMH at ~20s/step for 16 chain updates

**Advantages over RMH:**
- No burn-in covariance estimation — works from step 1
- Affine invariant — handles correlated parameters automatically
- Information sharing between walkers — converges faster
- Only 1 tuning parameter (a=2)

**Implementation plan:**
1. Write stretch move kernel in pure JAX (no BlackJAX dependency)
2. Split-merge (red-blue) update for parallelism
3. Checkpoint save/load (same format as RMH for compatibility with `monitor_mcmc.py`)
4. Test on 2D Gaussian first, then 13D correlated Gaussian, then real likelihood

**Verification:**
- Both samplers should converge to the same posterior
- Compare marginal distributions, correlations, and logP distributions
- ESS per wall-time as efficiency metric

## Priority

1. **Solution 1 first** — quick fix, reuse existing code
2. **Solution 2 second** — more robust long-term solution
3. **Cross-check** — run both, compare posteriors
