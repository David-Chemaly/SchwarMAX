# Ensemble Sampler for SchwarMAX

## What Our Sampler Does — Step by Step

We have `L=32` walkers, each holding a position `X_k` in 13D parameter space and its log-posterior `logP(X_k)`. Every step updates all 32 walkers:

### Phase 1: Update the first 16 walkers (S0), holding the other 16 (S1) fixed

For **each** walker `k` in S0 (all 16 in parallel via `jax.vmap`):

1. **Roll a die** to pick which move to use:
   - 70% chance: **DE move**
   - 20% chance: **DE-Snooker move**
   - 10% chance: **Stretch move**

2. **Generate a proposal** `Y` depending on which move was chosen (details below).

3. **Evaluate** `logP(Y)` — this is the expensive part (orbit integration + NNLS + likelihood).

4. **Accept or reject** using the Metropolis rule:
   ```
   log_acceptance_ratio = log_factor + logP(Y) - logP(X_k)
   ```
   where `log_factor` depends on the move type (0 for DE, Jacobian for Snooker/Stretch).
   Draw `u ~ Uniform(0,1)`. If `log(u) < log_acceptance_ratio`, accept: set `X_k = Y`. Otherwise keep `X_k` unchanged.

### Phase 2: Update the second 16 walkers (S1), holding updated S0 fixed

Same procedure, but now S1 walkers use the **already-updated** S0 as companions.

**Total cost per step:** 2 batches of 16 parallel logL evaluations = **~3 seconds on L4 GPU**.

---

## The Three Moves

All moves use the **complementary half** (the 16 walkers held fixed) as companions. No external covariance matrix is ever estimated.

### DE Move (70%) — ter Braak 2006

The workhorse. Proposes along the difference of two randomly chosen companions:

```
Pick two companions c[r1], c[r2] from the other half
gamma = 2.38 / sqrt(2 * 13) ≈ 0.467
Y = X_k + gamma * (c[r1] - c[r2]) + small_noise
```

- **log_factor = 0** (symmetric proposal, no Jacobian needed)
- The differential `c[r1] - c[r2]` automatically points along directions the ensemble has already spread out in — this is how correlations are learned without explicit covariance estimation
- `gamma = 2.38/sqrt(2D)` is the theoretically optimal scale for Gaussian targets (same as the optimal RMH scale, but here it emerges naturally)
- The small noise (`1e-5 * N(0,1)`) ensures ergodicity (the chain can reach any point, not just linear combinations of current walkers)

### DE-Snooker Move (20%) — ter Braak & Vrugt 2008

Projects a differential onto the line connecting the walker to a pivot point. Better at exploring curved degeneracies:

```
Pick three companions c[r0], c[r1], c[r2]
direction = X_k - c[r0]               (line from pivot to walker)
d_hat = direction / ||direction||      (unit vector)
diff = c[r1] - c[r2]                  (differential)
proj = dot(diff, d_hat)               (scalar projection onto line)
gamma = 1.7
Y = X_k + gamma * proj * d_hat        (move along the line)
```

- **log_factor = (D-1) * log(||Y - c[r0]|| / ||X_k - c[r0]||)** — Jacobian correction because the proposal density depends on the distance to the pivot
- Moves along a 1D subspace (the line through walker and pivot), which helps when the posterior has narrow ridges
- The projection ensures the step size adapts to the ensemble spread in that specific direction

### Stretch Move (10%) — Goodman & Weare 2010

The original emcee move. Interpolates/extrapolates between the walker and a companion:

```
Pick one companion c[j]
Z ~ g(z) ∝ 1/sqrt(z),  z ∈ [1/2, 2]    (CDF inversion: Z = ((a-1)*U + 1)^2 / a)
Y = c[j] + Z * (X_k - c[j])
```

- When `Z < 1`: proposal is between `c[j]` and `X_k` (contraction)
- When `Z > 1`: proposal is beyond `X_k` away from `c[j]` (expansion)
- **log_factor = (D-1) * log(Z)** — Jacobian from the stretch transformation
- Affine invariant: performance is unchanged under any linear transformation of the parameter space

---

## Why This Mixture Works

The three moves complement each other:

| Move | Strengths | Weaknesses |
|---|---|---|
| DE | Best acceptance rate; good at following linear degeneracies | Can't explore curved ridges well |
| Snooker | Explores curved degeneracies; adapts step size to local geometry | Lower acceptance; needs 3 companions |
| Stretch | Affine invariant; simple | Degrades in high-D; can't cross valleys |

At 13D, the DE move does most of the work (70%). The Snooker move helps navigate curved features like the halo mass–concentration degeneracy. The Stretch move provides a small amount of diversity in proposal directions.

---

## Why No Covariance Estimation Is Needed

In RMH, you need to estimate a 13×13 covariance matrix from the chain history, which requires O(D^2) = 169 samples to converge — and if those samples are from the wrong region (burn-in), the covariance is wrong and the chain freezes.

The ensemble sampler never estimates a covariance matrix. Instead:
- The **positions of the other walkers** implicitly encode the posterior geometry
- The **differential** `c[r1] - c[r2]` is a random vector drawn from the empirical distribution of the ensemble
- As walkers spread to cover the posterior, these differentials automatically align with the principal axes
- This works from step 1 — no burn-in needed for tuning

This is why the RMH chains froze (bad covariance → low acceptance → stuck chains → worse covariance → death spiral) but the ensemble sampler doesn't.

---

## Detailed Balance

Each move satisfies detailed balance individually:

- **DE move**: symmetric proposal (`Y - X = gamma*(c[r1]-c[r2])` has the same distribution as `X - Y`), so the standard Metropolis ratio `min(1, pi(Y)/pi(X))` is correct.
- **Snooker move**: asymmetric (the proposal depends on `||X - pivot||`), corrected by the Jacobian factor `(||Y-pivot||/||X-pivot||)^(D-1)`.
- **Stretch move**: asymmetric (the stretch factor Z biases toward contraction), corrected by `Z^(D-1)`.

The mixture preserves detailed balance because each walker independently selects a move, and each move individually satisfies detailed balance.

The **red-blue split** (updating S0 then S1) ensures walkers in the same batch don't use each other's updated positions — the complement is held fixed during each phase. This is required because the proposals depend on the companion positions; updating companions mid-batch would break detailed balance.

---

## Validation Results (2026-04-06)

Tested against known distributions. The sampler recovers the correct posterior in all cases.

### Realistic 13D test (SchwarMAX posterior geometry)

Used the actual covariance matrix from the `mcmc_checkpoint_gal2_0402.pkl` chain:
- Condition number: 118,626
- Near-perfect degeneracy: r(logM_halo, logRs_halo) = 0.995
- Strong correlations: r(logM_disk, log_LtM) = -0.901, r(logRs_disk, gamma) = 0.844

**Head-to-head: Stretch-only vs Mixed moves (same 6000 steps, 52 walkers)**

| Metric | Stretch-only | Mixed | Improvement |
|---|---|---|---|
| Max mean error | 0.074σ | 0.033σ | 2.2x |
| Std ratio range | [0.969, 1.021] | [0.987, 1.015] | 3x tighter |
| Cov Frobenius error | 6.1% | 1.2% | 5x |
| Max correlation error | 0.021 | 0.010 | 2x |
| Autocorrelation time | 146 steps | 61 steps | 2.4x faster |
| Effective samples | 1,281 | 3,091 | 2.4x more |

### Other tests (all PASS)

| Test | Description |
|---|---|
| 2D isotropic Gaussian | Mean and std within 0.04 |
| 5D correlated Gaussian | Full covariance recovered, Frobenius error 3.6% |
| 2D bimodal Gaussian | Both modes sampled, widths correct |
| 2D Student-t (df=3) | Heavy tails detected, kurtosis >> 0 |
| 2D Banana/Rosenbrock | Curved degeneracy, E[y]=0.97 (true 1.0) |
| 13D isotropic Gaussian | Max error 0.03 on mean, 0.02 on std |
| 1D KS test | p=0.025 after thinning (detailed balance verified) |

---

## Practical Notes for SchwarMAX Runs

- **32 walkers**, 16 per vmap batch → matches GPU parallelism on L4
- **~3s per step** (2 batches × 1.5s per vmapped logL)
- **ESS/step/walker ≈ 0.017** → ~1 effective sample per 100s
- **No tuning parameters** that need manual adjustment (gamma values are theoretically determined)
- **Checkpoints** every 50 steps; can resume from any checkpoint
- **Known limitation**: cannot cross deep probability valleys (multimodality). For SchwarMAX's unimodal posterior this is not a concern, but initialize walkers near the best-fit point to be safe.

## References

- Goodman & Weare (2010) — stretch move
- ter Braak (2006) — DE move for MCMC
- ter Braak & Vrugt (2008) — DE-Snooker move
- Foreman-Mackey et al. (2013) — emcee implementation (~10,000 citations)
