# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SchwarMAX** is a Schwarzschild dynamical modeling framework for constraining galaxy kinematics. It finds the combination of stellar orbits that best reproduces observed surface density and kinematic moments (V, σ, h1–h4), then uses Bayesian nested sampling to infer the underlying gravitational potential parameters.

## Development Environment

The project is developed primarily in Jupyter notebooks and plain Python scripts. JAX is used for GPU-accelerated computation throughout.

```bash
# Launch notebooks
jupyter notebook

# Run Dynesty inference directly
python fit_dynesty.py

# Generate mock observational data
python generate_mock_data.py
```

There is no build system, test runner, or linting configuration. Validate changes by running cells in `main.ipynb` or `test_model.ipynb`.

## Architecture

The pipeline runs in sequence:

1. **Potential** (`potentials.py`) — NFW dark matter halo + Miyamoto-Nagai stellar disk. The disk potential is precomputed on a grid and stored as a `CylindricalSpline` (`CylindricalSpline.py`) for fast JAX-compatible evaluation during orbit integration.

2. **Jeans solver** (`main.py`) — Given the combined potential, solves the Jeans equations to get velocity dispersions (σ_R, σ_z, σ_φ) and rotation velocity v_rot at each particle position. This seeds the initial conditions for orbit integration.

3. **Orbit integration** (`integrants_with_binning.py`) — Leapfrog integrator runs N particles for 10 Gyr (2500 steps) in the combined potential. Each orbit's time-averaged contributions to 3D density, projected surface density, and Gauss-Hermite kinematic moments (h1–h4) are accumulated into Voronoi bins.

4. **Orbital weight optimization** (`main.py: model()`) — Solves a non-negative least-squares problem to find orbit weights that best reproduce the observed kinematics. Supports three solvers: LBFGS with softplus reparameterization, BoxOSQP (QP), and FISTA. Default is LBFGS with `maxiter=500` and L2 regularization.

5. **Likelihood** (`main.py: logl()`, `likelihoods.py`) — Computes χ² log-likelihood comparing model predictions to data. Applies error floors (V ≥ 10 km/s, σ ≥ 5 km/s, h_i ≥ 0.03) and rejects outlier bins above the 98th percentile residual.

6. **Bayesian inference** (`fit_dynesty.py`) — Wraps the likelihood in Dynesty nested sampling (`nlive=2000`, `sample='rslice'`). Parameter space is typically 8D: `logM_halo`, `logRs_halo`, `logM_disk`, `logRs_disk`, `logHs_disk`, and orientation angles (α, β, γ).

## Key Modules

| File | Role |
|------|------|
| `main.py` | Central module: `model()` runs the full forward model; `logl()` is the likelihood |
| `likelihoods.py` | Alternative likelihood variants (angular parameterization) |
| `potentials.py` | NFW halo, Miyamoto-Nagai disk, bar perturbation potentials |
| `integrants_with_binning.py` | Leapfrog orbit integrator with Voronoi binning |
| `ghMoments.py` | Gauss-Hermite moment computation (h1–h4) |
| `utils.py` | Coordinate transforms (Cartesian ↔ Cylindrical), rotation matrices, histogram utilities |
| `constants.py` | Physical constants and unit conversions (kpc, M☉, Gyr, km/s) |
| `CylindricalSpline.py` | JAX-compatible spline interpolation for disk potential on cylindrical grid |
| `prior.py` | Prior transforms for Dynesty |
| `generate_mock_data.py` | Creates mock observational data with Voronoi binning |

## Physical Units

- Length: kpc
- Mass: M☉
- Time: Gyr
- Velocity: km/s (conversion via `KPCGYR_TO_KMS` in `constants.py`)

Parameters like `logM_halo` are log₁₀ solar masses; `logRs_disk` is log₁₀ kpc.

## Data Format

The model expects a `dict_data` dictionary containing:
- `w0`: initial particle positions (N, 3) in kpc (R, z, φ)
- `Rzphi_density_data`, `XY_density_data`: 3D and projected density observations
- `h1_data` through `h4_data`: Gauss-Hermite kinematic moments per Voronoi bin
- `V_data`, `V_data_err`, `sigma_data`, `sigma_data_err`: line-of-sight kinematics
- Voronoi bin assignments and bin area metadata

Mock data is stored as `.pkl` files (e.g., `mock_axisymmetric_disc_XY_withRot.pkl`) and loaded with `pickle`.
