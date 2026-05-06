# SchwarMAX (bar branch)

> **Environment:** always run this code inside the conda env `astro312`
> (`conda activate astro312`). Every shell command, notebook kernel, and
> script invocation in this repo assumes that env.

JAX-based Schwarzschild dynamical modelling framework for constraining the
gravitational potential and stellar kinematics of barred galaxies. This branch
extends the original axisymmetric pipeline with a **rotating bar**: T3
ellipsoid + V4 bulge baryonic component on top of an NFW halo and
Miyamoto–Nagai disc, integrated in the bar's rotating frame at pattern speed
`Omega_bar`.

The end product is a posterior over halo, disc, bar, and viewing-angle
parameters obtained by running an adaptive random-walk Metropolis MCMC
(BlackJAX) on top of a forward model that, *for every proposed parameter set*,
re-integrates an orbit library and re-solves a non-negative least-squares
problem for orbital weights.

---

## Pipeline (forward model, per likelihood call)

For one parameter vector `theta`, `model_bar.model_bootstrap()` does:

1. **Build potentials** (`potentials.py`, `dehnen_bar.py`)
   - NFW halo (`logM_halo`, `Rs_halo`)
   - Miyamoto–Nagai disc (`logM_disc`, `Rs_disc`, `Hs_disc`)
   - T3 bar ellipsoid + V4 bulge (`logM_bar`, `L_bar`, `a_bar=L/5`,
     `b_bar=Hs_disc`, fixed shape constants `V4_A=V4_B=0.5`, `V4_L=0.1`)
2. **Initial conditions for orbit library** — solves Jeans equations
   (`get_jeans_moments`) at each particle position to set `(v_R, v_z, v_phi)`
   from `(σ_R, σ_z, σ_phi)` plus mean rotation. `n_realizations=4` jittered
   copies per particle for orbit averaging.
3. **Orbit integration** (`integrants_with_binning.py`) — adaptive leapfrog
   in the rotating frame at `-Omega_bar`, run for
   `N_dynamical_time=50` orbital periods with `N_step_per_orb=100`,
   `atol=1e-7`, `rtol=1e-4`. Each orbit's time-averaged contribution is
   accumulated into:
   - 3D Rzphi density grid (default `Rzphi_n_grid=(10,6,10)`, `Rzphi_n_tot=600`
     used by likelihood; `model_bar.model()` defaults to `(10,6,6)`/360)
   - projected XY surface-density grid (after rotation by α, β, γ)
   - Gauss–Hermite kinematic moments h1–h4 per Voronoi bin
   This produces the orbital-library matrices `A_Rzphi, A_xy, A_h1..A_h4`.
4. **Solve for non-negative orbital weights** with **ADMM-NNLS**
   (`solve_nnls_admm` / `solve_nnls_admm_bootstrap`):
   - Stacks the rescaled design matrix
     `U = [A_Rzphi/σ; A_xy/σ; (A_hk·A_xy)/y_xy/σ_hk]_k=1..4`
   - Builds `Q = UᵀU + (λ/n_orb) I` once, with `λ=1`, `maxiter=200`.
   - Single Cholesky of `Q + ρI`; `vmap` over bootstrap RHSs `c_i = -Uᵀ y_i`.
   - Returns `weights_all : (N_boot, n_orb)`, all ≥ 0.
5. **Marginalise over data noise (parametric bootstrap)** —
   `dict_data['XY_standard_normal']` etc. carry frozen `N(0,1)` draws
   (seed 42, first row zeroed = unperturbed). Each sample becomes
   `y_i = y + z_i · σ`. Per-sample chi² + log-σ normalization is computed in
   `compute_model_and_logl_bootstrap`, then combined as
   `logL_marg = log(mean(exp(logL_i)))`. The `m_eff` term computes a
   model-flexibility correction (currently not subtracted by default).
6. **Density gate** — A cheap MN+T3+V4 surface-density-only chi² is
   compared to `dict_data['logl_density_max']`; if the proposal is more
   than `1000` worse, the expensive orbit step is skipped and `-inf` is
   returned. This is what makes large MCMC runs tractable.

`likelihoods_bar.logl_angular_input_bootstrap()` is the public likelihood
wrapper: unpacks the parameter vector, computes the gate, and calls
`model_bootstrap`.

---

## Parameter space (13-D, `logl_angular_input_bootstrap`)

| idx | name | meaning |
|---|---|---|
| 0 | `logM_10kpc` | log10 NFW mass inside 10 kpc (converted internally to `logM_halo, logRs_halo` via `logMenc_logc_to_logM_logRs`) |
| 1 | `logM_disk` | MN total disc mass |
| 2 | `logM_bar` | T3 bar mass = V4 bulge mass |
| 3 | `logC_halo` | NFW concentration (`Δ=200`, `ρ_crit=277.54`) |
| 4 | `logRs_disk` | MN scale length |
| 5 | `logHs_disk` | MN scale height (= T3 `b_bar`) |
| 6 | `logL_bar` | T3 half-length; `a_bar = L_bar/5` |
| 7 | `alpha` | viewing angle (rad) |
| 8 | `beta`  | viewing angle (rad) |
| 9 | `gamma` | viewing angle (rad) |
| 10 | `log_light_to_mass_ratio` | applied to `XY_density_data` |
| 11 | `log_Omega_bar` | bar pattern speed |
| 12 | `log_sigma_amplifier` | multiplicative inflation of XY/h_k errors in the logL |

Other likelihoods in `likelihoods_bar.py`:
- `logl()` — 8-param axisymmetric (no bar), legacy.
- `logl_angular_input` / `logl_angular_input_marg` — 12-param variants without bootstrap; `_marg` uses Hessian-determinant marginalisation via `model_marg`.
- `logl_angular_input_bootstrap_psf` — same as `_bootstrap` but uses `model_bootstrap_psf` and expects `dict_data['P_psf']`.
- `logl_angular_input_bootstrap_test` — for integration-step convergence experiments via `model_test_convergence`.
- `jackknife_error_wrapper` — delete-d jackknife on chi² (see `model_jackknife_chi2`).
- `logl_density` — 8-param surface-density-only chi² (used to set `logl_density_max` via emcee warm-up).

---

## Main entry: `SchwarMAX_blackjax.ipynb` / `schwarmax_blackjax.py`

The notebook is a Colab driver. Order of operations:

1. **Load** mock data via `get_dict_data_bootstrap(path, filename, N_BOOTSTRAP=100)` from `likelihoods_bar.py`. This populates `dict_data` with the Voronoi binning, kinematics, errors, the frozen `*_standard_normal` arrays, and Rzphi / Sobol integration grids.
2. **Warm-up** with emcee on the 8-param `log_prob` (`logl_density`) to find a density-only best fit and store `dict_data['logl_density_max']` — this anchors the gate in step 6 above.
3. **Test the bar likelihood** along 1-D slices (L/M, Ω, σ-amplifier) and time one `logl_angular_input_bootstrap` call.
4. **Run BlackJAX adaptive RMH** on the 13-D log-posterior:
   - `N_CHAINS=16` (or 40 split into two halves to fit GPU), `N_STEPS=3000`,
     `BURNIN=600`.
   - Initial proposal `rmh_sigma_init` is per-parameter scalar; from
     `ADAPT_AFTER=300` onward, every `ADAPT_EVERY=100` steps the empirical
     covariance is shrunk (0.8·full + 0.2·diag) and Cholesky-factored, scaled
     by the Roberts–Rosenthal optimum `2.38/√D`.
   - Checkpoints to `mcmc_checkpoint_*.pkl` every 10 steps; final results to
     `mcmc_results_*.pkl` and `mcmc_posterior_*.csv`.
   - The `res = np.load('minimise_*.npy')` file holds a previously-found
     best fit used to seed chains and centre prior bounds (±3 in masses,
     ±1 in scales).

Other samplers / drivers in the repo:
- `fit_dynesty.py`, `main_dynesty.py` — nested sampling.
- `run_blackjax_mcmc.py`, `run_blackjax_mcmc_parallel.py`, `run_blackjax_smc.py`, `run_custom_smc.py` — script versions.
- `run_ensemble_mcmc.py`, `run_numpyro_ess.py`, `schwarmax_emceejax.py` — alternative samplers.
- `SchwarMAX_fixedpot.ipynb`, `get_best_orbital_library_fixed_potential.py` — fix the potential and only fit weights / kinematic post-processing.

---

## Key files

| File | Role |
|---|---|
| `model_bar.py` | Core forward model: density/potential funcs, Jeans, NNLS solvers (LBFGS, BoxCDQP, FISTA, ADMM, ADMM-bootstrap), `model()`, `model_marg()`, **`model_bootstrap()`**, `model_bootstrap_psf()`, `model_jackknife_chi2()`, `model_test_convergence()` |
| `likelihoods_bar.py` | Public log-likelihood wrappers + `get_dict_data_bootstrap()` data loader (this is the bar/Nbody version of the likelihood) |
| `model.py`, `likelihoods.py` | Older axisymmetric versions still imported by some scripts |
| `model_bulge.py` | Variant with separate bulge component |
| `potentials.py` | NFW + MN potentials and accelerations (analytic) |
| `dehnen_bar.py` | T3 ellipsoidal bar + V4 ovoid bulge density/potential/acceleration (Dehnen 2023 form) |
| `dehnen_2023_bar_pairs.py` | Multi-component bar variants |
| `densities.py` | Stand-alone density functions (MN etc.) |
| `integrants_with_binning.py` | Adaptive leapfrog integrator with on-the-fly Voronoi + 3D Rzphi binning. Provides `_integrate_adaptive_batch_vmap`, `_integrate_adaptive_batch_chunked_vmap` (chunk size 100 used by `model_bootstrap`), `assign_regular_grid` |
| `ghMoments.py` | Gauss–Hermite weights and `h_to_V_sigma` conversion |
| `utils.py` | Rotation matrices (`makeRotationMatrix`), Cylindrical↔Cartesian transforms, `estimate_orbital_timescale`, `get_rotation_curve`, `XexpX_pdf_log`/`expX_pdf_log`/`sample_from_logP` (used to draw `w0`), `compute_transfer_matrix` (PSF), `logM_logRs_to_logMenc_logc` and inverse |
| `CylindricalSpline*.py` | JAX-friendly cylindrical spline interpolators (used when caching potential as a grid) |
| `prior.py` | Prior transforms (used by Dynesty) |
| `sample_from_density.py` | Helper for sampling initial particle positions from a density grid |
| `constants.py` | Units (kpc, M☉, Gyr, km/s) and Legendre-Q tabulations for AGAMA-style multipoles |

---

## Mock data files

`mock_Nbody_bar_XY_withRot_*.pkl` are the primary inputs. The naming
encodes the configuration; e.g. `gal2_Nbins1000_Bar45deg.pkl` =
galaxy 2, 1000 Voronoi bins, bar at 45°.
`get_dict_data_bootstrap()` expects each pickle to contain at minimum:

`X_minmax`, `Y_minmax`, `nX_nY`, `num_per_bin`, `total_bins`, `bin_mapping`,
`surface_density`, `V_mean`, `V_sigma`, `h1..h4` and their `_err`,
`v0`, `s` (Gauss–Hermite normalization), `orientation = (α, β, γ)`,
`X_regular_grid`, `Y_regular_grid`.

The Rzphi integration grid is built deterministically inside
`get_dict_data_bootstrap` (R∈[0,10], z∈[-3,3], φ∈[-π,π], `n_R=10, n_z=6, n_phi=10`).

`XY_standard_normal`, `h{1..4}_standard_normal`, `V/sigma_standard_normal`
are pre-generated `N(0,1)` matrices of shape `(N_BOOTSTRAP, n_bins)` with
the first row zeroed — these are the bootstrap noise realisations; they are
multiplied by the model-time-varying σ rather than by the static data σ, so
the bootstrap reflects the *model's* current noise budget.

---

## Units

- Length kpc · Mass M☉ · Time Gyr · Velocity km/s (factor `KPCGYR_TO_KMS` in `constants.py`).
- Log parameters are `log10`; angles in `params[7..9]` are radians.

---

## Running

No build system. JAX with CUDA is required for any non-trivial run; an A100 with `XLA_PYTHON_CLIENT_MEM_FRACTION≈0.95` was used for the 1000-bin / 5000-particle / 16-chain configuration.

```bash
# Optional axisymmetric Dynesty fit
python fit_dynesty.py

# 13-D adaptive RMH with checkpointing (set paths inside the script first)
python schwarmax_blackjax.py
```

The notebook variants (`SchwarMAX_blackjax.ipynb`, `SchwarMAX_MCMC.ipynb`,
`SchwarMAX_emceejax.ipynb`, `SchwarMAX_fixedpot.ipynb`) are the canonical
form of how the pipeline is exercised end-to-end and where plots are made
(`plot_data_vs_model.py`, `plot_posterior*.ipynb`, etc.).

---

## Reading order for a fresh agent

1. `likelihoods_bar.get_dict_data_bootstrap` — what the data dict looks like.
2. `likelihoods_bar.logl_angular_input_bootstrap` — parameter unpacking + density gate.
3. `model_bar.model_bootstrap` — the forward model and what it returns.
4. `model_bar.solve_nnls_admm_bootstrap` + `compute_model_and_logl_bootstrap` — the inner NNLS + bootstrap-marginalised logL.
5. `schwarmax_blackjax.py` (or the notebook) — how the likelihood is wrapped into BlackJAX adaptive RMH.

After that, `model_bar.model()` (single-shot, density-only `m_eff` not used)
and `model_bar.model_jackknife_chi2()` are easier to read because they reuse
the same orbit-integration block as `model_bootstrap`.
